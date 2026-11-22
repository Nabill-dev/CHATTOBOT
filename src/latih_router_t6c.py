# ==============================================================
# latih_router_t6c.py — Evaluasi Sekali Uji Akhir (T6c)
#
# Tahap T6c Proyek Tugas Akhir CHATTOBOT:
#   1. Latih model Klasik (TF-IDF + LogReg) pada 166 sesi latih non-uji.
#   2. Latih model IndoBERTweet pada 166 sesi latih non-uji:
#      - 6 Epoch (sesuai Keputusan K-021 dari A4), LR 5e-5, Seed 42,
#        batch 8, max_length 96, class-weighted loss, fp32.
#   3. Evaluasi SEKALI pada 51 sesi uji akhir (8 chat di eval/chat_uji_akhir.txt)
#      menggunakan ambang yang telah ditetapkan di Part B / K-021:
#        - Klasik: T = 0.300
#        - IndoBERTweet: T = 0.100
#   4. Simpan prediksi ke hasil/uji_akhir_pred.csv.
#   5. Simpan artefak model ke models/router_indobertweet/ (bobot, tokenizer, router_config.json).
#   6. Evaluasi sistem gabungan utuh (aturan + router + Chroma leksikal) pada 51 sesi uji.
#   7. Keluaran mentah disimpan ke hasil/log_T6c.txt.
# ==============================================================

import io
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

# Output UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
HASIL_DIR = ROOT / "hasil"
MODELS_DIR = ROOT / "models"
MODELS_ROUTER_DIR = MODELS_DIR / "router_indobertweet"

sys.path.insert(0, str(SRC_DIR))
from config import (
    EXCEL_PATH, SHEET_SESI, SHEET_PESAN, SHEET_KAMUS,
    SEMUA_KELAS, KELAS_ESKALASI, KELAS_BOT, SEED,
    ROUTER_MAX_PESAN, ROUTER_MAX_KARAKTER,
    SEARCH_DISTANCE_THRESHOLD, MIN_OVERLAP_KATA, JAM_LAYANAN_TEKS,
    BAGIAN_TOPIK_KHUSUS, KARANTINA_BAGIAN
)
import bot

CATATAN_LABEL = "Catatan: label dibuat AI, TIDAK diverifikasi manusia (hasil bersifat indikatif)"
NAMA_MODEL = "indolem/indobertweet-base-uncased"
MAX_LENGTH = 96
BATCH_SIZE = 8
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
GRAD_CLIP = 1.0

# Parameter Tetap dari Keputusan K-021
EPOCHS_FINAL = 6
LR_FINAL = 5e-5
SEED_FINAL = 42
AMBANG_KLASIK = 0.300
AMBANG_BERT = 0.100


class OutputTee:
    def __init__(self, file_path: Path):
        self.terminal = sys.stdout
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(file_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return getattr(self.terminal, 'isatty', lambda: False)()

    def close(self):
        self.log_file.close()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = 1.95996
    p_hat = k / n
    denominator = 1 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt((p_hat * (1 - p_hat) / n) + ((z ** 2) / (4 * (n ** 2))))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return round(lower * 100, 2), round(upper * 100, 2)


def is_penutup_murni(teks: str) -> bool:
    kata_penutup = {
        'terima', 'kasih', 'terimakasih', 'makasih', 'thanks', 'thank', 'you',
        'oke', 'ok', 'baik', 'siap', 'sip', 'iya', 'ya',
        'min', 'admin', 'ka', 'kak', 'mba', 'mas', 'pak', 'bu',
        'selamat', 'pagi', 'siang', 'sore', 'malam',
        'info', 'infonya', 'bantuan', 'bantuannya', 'banyak', 'sama'
    }
    teks_clean = re.sub(r'[^a-zA-Z\s]', ' ', str(teks)).lower()
    words = [re.sub(r'([a-z])\1+', r'\1', w) for w in teks_clean.split()]
    mapping = {
        'trimakasih': 'terimakasih', 'makasi': 'makasih', 'thx': 'thanks',
        'bantuanya': 'bantuan', 'bantuanyaa': 'bantuan', 'okey': 'oke', 'okkee': 'oke'
    }
    words_mapped = [mapping.get(w, w) for w in words]
    if words_mapped and len(words_mapped) <= 10 and all(w in kata_penutup for w in words_mapped):
        return True
    return False


def muat_kamus_kampus() -> dict:
    kamus = {
        'nnti': 'nanti', 'bsk': 'besok', 'skrng': 'sekarang', 'knpa': 'kenapa',
        'udh': 'sudah', 'sdh': 'sudah', 'blm': 'belum', 'gk': 'tidak',
        'gak': 'tidak', 'ga': 'tidak', 'gabisa': 'tidak bisa', 'gkpp': 'tidak apa-apa',
        'klo': 'kalau', 'kalo': 'kalau', 'brpa': 'berapa', 'tgl': 'tanggal',
        'jgn': 'jangan', 'yg': 'yang', 'bkn': 'bukan', 'lgi': 'lagi',
        'dlu': 'dulu', 'lngsung': 'langsung', 'tf': 'transfer', 'smt': 'semester',
        'pswd': 'kata sandi', 'pasword': 'kata sandi', 'ortu': 'orang tua',
        'maba': 'mahasiswa baru', 'byr': 'bayar', 'gmn': 'bagaimana', 'sy': 'saya',
    }
    try:
        df_kamus = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_KAMUS)
        for _, row in df_kamus.iterrows():
            chat = str(row.get('bentuk di chat', '')).strip().lower()
            baku = str(row.get('bentuk baku / arti', '')).strip().lower()
            if chat and baku and len(baku.split()) <= 3:
                kamus[chat] = baku
    except Exception:
        pass
    return kamus


def bersihkan_teks(teks: str, kamus: dict) -> str:
    if not isinstance(teks, str) or not teks:
        return ""
    teks_l = teks.lower().strip()
    teks_l = re.sub(r'https?://\S+|www\.\S+', 'httpurl', teks_l)
    teks_l = re.sub(r'\[nama[^\]]*\]', '@user', teks_l)
    teks_l = re.sub(r'\[nim\]', 'nim', teks_l)
    teks_l = re.sub(r'\[kelas\]', 'kelas', teks_l)
    teks_l = re.sub(r'\[(telepon|no_hp)\]', 'nohp', teks_l)
    teks_l = re.sub(r'\[email\]', 'email', teks_l)
    teks_l = re.sub(r'\[(kata_sandi|password|format_kata_sandi)\]', 'pass', teks_l)
    teks_l = re.sub(r'\[(tgl|tanggal)\]', 'tanggal', teks_l)
    teks_l = re.sub(r'\[angka\]', 'angka', teks_l)
    teks_l = re.sub(r'[\U00010000-\U0010ffff\u2600-\u27bf]', '', teks_l)
    teks_l = re.sub(r'([a-z])\1{2,}', r'\1', teks_l)
    kata_kata = teks_l.split()
    kata_normal = [kamus.get(k, k) for k in kata_kata]
    return " ".join(kata_normal)[:ROUTER_MAX_KARAKTER]


class TeksSesiDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def main():
    tee = OutputTee(HASIL_DIR / "log_T6c.txt")
    sys.stdout = tee

    try:
        print("=" * 115)
        print("LAPORAN EVALUASI SATU KALI UJI AKHIR HELD-OUT (T6c)")
        print(f"{CATATAN_LABEL}")
        print("=" * 115)

        # 1. Pengecekan Lingkungan GPU
        if not torch.cuda.is_available():
            print("[GALAT KRITIS] CUDA tidak terdeteksi! Pelatihan memerlukan GPU lokal. Berhenti.")
            return

        device = torch.device("cuda:0")
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_gb = free_bytes / (1024 ** 3)
        total_gb = total_bytes / (1024 ** 3)

        print("\n[STATUS PERANGKAT GPU]")
        print(f"Perangkat      : {torch.cuda.get_device_name(device)}")
        print(f"Total Memori   : {total_gb:.2f} GB")
        print(f"Memori Bebas   : {free_gb:.2f} GB")

        if free_gb < 2.0:
            print(f"[PERINGATAN KRITIS] Memori GPU bebas ({free_gb:.2f} GB) kurang dari 2.0 GB!")
            print("Pelatihan dihentikan sesuai aturan.")
            return

        # 2. Persiapan Data Latih & Uji Akhir
        print("\nMempersiapkan data latih (166 sesi) dan data uji akhir (51 sesi)...")
        kamus_kampus = muat_kamus_kampus()
        df_sesi = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_SESI)
        df_pesan = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_PESAN)

        kolom_label = "topik_koreksi (isi manual)"
        df_sesi_valid = df_sesi[df_sesi[kolom_label].notna() & df_sesi[kolom_label].isin(SEMUA_KELAS)].copy()
        pesan_mhs = df_pesan[df_pesan['pengirim'] == 'Mahasiswa'].copy()

        with open(HASIL_DIR / "lipatan.json", "r", encoding="utf-8") as f:
            lipatan_json = json.load(f)

        chat_uji_akhir = lipatan_json['chat_uji_akhir']

        sesi_all = []
        for _, row in df_sesi_valid.iterrows():
            sid = row['sesi_id']
            cid = row['chat_id']
            lbl = row[kolom_label]
            p_sesi = pesan_mhs[pesan_mhs['sesi_id'] == sid].sort_values('no')
            p_text = " ".join([str(p) for p in p_sesi['teks_anonim'].head(ROUTER_MAX_PESAN).dropna().tolist()])
            p_clean = bersihkan_teks(p_text, kamus_kampus)
            p_penutup = is_penutup_murni(p_text)

            sesi_all.append({
                'sesi_id': sid,
                'chat_id': cid,
                'label': lbl,
                'teks_bersih': p_clean,
                'teks_mentah': p_text,
                'is_penutup': p_penutup
            })

        df_all = pd.DataFrame(sesi_all)
        df_latih = df_all[~df_all['chat_id'].isin(chat_uji_akhir)].copy().reset_index(drop=True)
        df_uji = df_all[df_all['chat_id'].isin(chat_uji_akhir)].copy().reset_index(drop=True)

        print(f"Total Sesi Latih Non-Uji   : {len(df_latih)} sesi (dari {df_latih['chat_id'].nunique()} chat)")
        print(f"Total Sesi Uji Akhir       : {len(df_uji)} sesi (dari {df_uji['chat_id'].nunique()} chat: {sorted(chat_uji_akhir)})")
        print(f"  - Sesi Eskalasi Uji Akhir: {sum(df_uji['label'].isin(KELAS_ESKALASI))} sesi")
        print(f"  - Sesi Jawab Uji Akhir   : {sum(df_uji['label'].isin(KELAS_BOT))} sesi")
        print(f"  - Sesi Penutup Uji Akhir : {sum(df_uji['is_penutup'])} sesi")

        # ------------------------------------------------------------------
        # TAHAP 1: PELATIHAN & EVALUASI MODEL KLASIK (TF-IDF + LOGREG)
        # ------------------------------------------------------------------
        print("\n" + "=" * 115)
        print("BAGIAN 1: PELATIHAN & EVALUASI ROUTER KLASIK PADA UJI AKHIR")
        print(f"Ambang Peluang Eskalasi: T = {AMBANG_KLASIK:.3f} (Sesuai Keputusan K-021)")
        print("=" * 115)

        pipe_klasik = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
            ('clf', LogisticRegression(class_weight='balanced', random_state=SEED_FINAL, max_iter=1000, C=1.0))
        ])

        start_klasik = time.time()
        pipe_klasik.fit(df_latih['teks_bersih'], df_latih['label'])
        time_klasik = time.time() - start_klasik

        pred_klasik_raw = pipe_klasik.predict(df_uji['teks_bersih'])
        probs_klasik_raw = pipe_klasik.predict_proba(df_uji['teks_bersih'])
        classes_klasik = list(pipe_klasik.classes_)

        idx_beas_klasik = classes_klasik.index('beasiswa_pendaftaran')
        idx_lain_klasik = classes_klasik.index('lainnya')

        p_esk_klasik = [float(p[idx_beas_klasik] + p[idx_lain_klasik]) for p in probs_klasik_raw]

        # Metrik Multi-class Klasik
        y_uji_true = df_uji['label'].tolist()
        macro_f1_klasik = f1_score(y_uji_true, pred_klasik_raw, average='macro', zero_division=0)
        prec_klasik = precision_score(y_uji_true, pred_klasik_raw, labels=SEMUA_KELAS, average=None, zero_division=0)
        rec_klasik = recall_score(y_uji_true, pred_klasik_raw, labels=SEMUA_KELAS, average=None, zero_division=0)
        f1_klasik = f1_score(y_uji_true, pred_klasik_raw, labels=SEMUA_KELAS, average=None, zero_division=0)
        cm_klasik = confusion_matrix(y_uji_true, pred_klasik_raw, labels=SEMUA_KELAS)

        # Metrik Ambang Eskalasi Klasik
        y_uji_bin = [1 if lbl in KELAS_ESKALASI else 0 for lbl in y_uji_true]
        auroc_klasik = roc_auc_score(y_uji_bin, p_esk_klasik)

        total_uji = len(df_uji)
        total_esk_a = sum(1 for yt in y_uji_true if yt in KELAS_ESKALASI)  # 14
        total_esk_b = sum(1 for yt, pen in zip(y_uji_true, df_uji['is_penutup']) if yt in KELAS_ESKALASI and not pen)
        total_jawab_uji = sum(1 for yt in y_uji_true if yt in KELAS_BOT)   # 37

        n_jawab_klasik = sum(1 for p in p_esk_klasik if p < AMBANG_KLASIK)
        pct_jawab_klasik = (n_jawab_klasik / total_uji) * 100.0

        kb_a_klasik = sum(1 for yt, p in zip(y_uji_true, p_esk_klasik) if yt in KELAS_ESKALASI and p < AMBANG_KLASIK)
        pct_kb_a_klasik = (kb_a_klasik / total_esk_a) * 100.0
        w_low_a_k, w_high_a_k = wilson_score_interval(kb_a_klasik, total_esk_a)

        kb_b_klasik = sum(1 for yt, p, pen in zip(y_uji_true, p_esk_klasik, df_uji['is_penutup']) if yt in KELAS_ESKALASI and not pen and p < AMBANG_KLASIK)
        pct_kb_b_klasik = (kb_b_klasik / total_esk_b) * 100.0 if total_esk_b > 0 else 0.0
        w_low_b_k, w_high_b_k = wilson_score_interval(kb_b_klasik, total_esk_b)

        eb_klasik = sum(1 for yt, p in zip(y_uji_true, p_esk_klasik) if yt in KELAS_BOT and p >= AMBANG_KLASIK)
        pct_eb_klasik = (eb_klasik / total_jawab_uji) * 100.0

        print(f"Hasil Uji Akhir Router Klasik (Waktu Latih = {time_klasik:.2f}s):")
        print(f"  Macro-F1 (6 Kelas)     : {macro_f1_klasik:.4f}")
        print(f"  AUROC Pemisahan        : {auroc_klasik:.4f}")
        print(f"  % Dijawab Otomatis     : {pct_jawab_klasik:.1f}% ({n_jawab_klasik}/{total_uji})")
        print(f"  Kesalahan Berbahaya (a): {pct_kb_a_klasik:.1f}% ({kb_a_klasik}/{total_esk_a}) | Wilson 95%: [{w_low_a_k:.2f}%, {w_high_a_k:.2f}%]")
        print(f"  Kesalahan Berbahaya (b): {pct_kb_b_klasik:.1f}% ({kb_b_klasik}/{total_esk_b}) | Wilson 95%: [{w_low_b_k:.2f}%, {w_high_b_k:.2f}%]")
        print(f"  Eskalasi Berlebih      : {pct_eb_klasik:.1f}% ({eb_klasik}/{total_jawab_uji})")

        # ------------------------------------------------------------------
        # TAHAP 2: PELATIHAN & EVALUASI MODEL INDOBERTWEET (DEEP LEARNING)
        # ------------------------------------------------------------------
        print("\n" + "=" * 115)
        print("BAGIAN 2: PELATIHAN & EVALUASI INDOBERTWEET PADA UJI AKHIR")
        print(f"Konfigurasi Model Final : Epoch={EPOCHS_FINAL}, LR={LR_FINAL}, Seed={SEED_FINAL}, Batch={BATCH_SIZE}, MaxLen={MAX_LENGTH}")
        print(f"Ambang Peluang Eskalasi: T = {AMBANG_BERT:.3f} (Sesuai Keputusan K-021)")
        print("=" * 115)

        set_seed(SEED_FINAL)
        tokenizer = AutoTokenizer.from_pretrained(NAMA_MODEL)

        label_to_idx = {nama: idx for idx, nama in enumerate(SEMUA_KELAS)}
        y_train_idx = [label_to_idx[lbl] for lbl in df_latih['label']]
        y_test_idx = [label_to_idx[lbl] for lbl in df_uji['label']]

        classes_present = np.unique(y_train_idx)
        raw_weights = compute_class_weight('balanced', classes=classes_present, y=np.array(y_train_idx))
        weight_tensor = torch.ones(len(SEMUA_KELAS), dtype=torch.float, device=device)
        for c_id, w in zip(classes_present, raw_weights):
            weight_tensor[c_id] = float(w)

        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        train_ds = TeksSesiDataset(df_latih['teks_bersih'].tolist(), y_train_idx, tokenizer)
        test_ds = TeksSesiDataset(df_uji['teks_bersih'].tolist(), y_test_idx, tokenizer)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = AutoModelForSequenceClassification.from_pretrained(
            NAMA_MODEL,
            num_labels=len(SEMUA_KELAS),
            problem_type="single_label_classification"
        )
        model.to(device)

        total_steps = len(train_loader) * EPOCHS_FINAL
        warmup_steps = int(total_steps * WARMUP_RATIO)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FINAL, weight_decay=WEIGHT_DECAY)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        start_bert = time.time()
        print("\nMemulai loop pelatihan 6 epoch pada seluruh 166 data latih...")
        for ep in range(1, EPOCHS_FINAL + 1):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"  Epoch [{ep}/{EPOCHS_FINAL}] — Rata-rata Train Loss: {avg_loss:.4f}")

        time_bert = time.time() - start_bert
        print(f"Pelatihan IndoBERTweet selesai dalam {time_bert:.1f} detik.")

        # Evaluasi pada Uji Akhir (51 sesi)
        model.eval()
        test_preds_bert = []
        test_probs_bert = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                test_preds_bert.extend(preds)
                test_probs_bert.extend(probs)

        test_preds_bert_str = [SEMUA_KELAS[idx] for idx in test_preds_bert]
        test_probs_bert_arr = np.array(test_probs_bert)

        idx_beas_bert = SEMUA_KELAS.index('beasiswa_pendaftaran')
        idx_lain_bert = SEMUA_KELAS.index('lainnya')
        p_esk_bert = [float(p[idx_beas_bert] + p[idx_lain_bert]) for p in test_probs_bert_arr]

        macro_f1_bert = f1_score(y_uji_true, test_preds_bert_str, average='macro', zero_division=0)
        prec_bert = precision_score(y_uji_true, test_preds_bert_str, labels=SEMUA_KELAS, average=None, zero_division=0)
        rec_bert = recall_score(y_uji_true, test_preds_bert_str, labels=SEMUA_KELAS, average=None, zero_division=0)
        f1_bert = f1_score(y_uji_true, test_preds_bert_str, labels=SEMUA_KELAS, average=None, zero_division=0)
        cm_bert = confusion_matrix(y_uji_true, test_preds_bert_str, labels=SEMUA_KELAS)

        auroc_bert = roc_auc_score(y_uji_bin, p_esk_bert)

        n_jawab_bert = sum(1 for p in p_esk_bert if p < AMBANG_BERT)
        pct_jawab_bert = (n_jawab_bert / total_uji) * 100.0

        kb_a_bert = sum(1 for yt, p in zip(y_uji_true, p_esk_bert) if yt in KELAS_ESKALASI and p < AMBANG_BERT)
        pct_kb_a_bert = (kb_a_bert / total_esk_a) * 100.0
        w_low_a_b, w_high_a_b = wilson_score_interval(kb_a_bert, total_esk_a)

        kb_b_bert = sum(1 for yt, p, pen in zip(y_uji_true, p_esk_bert, df_uji['is_penutup']) if yt in KELAS_ESKALASI and not pen and p < AMBANG_BERT)
        pct_kb_b_bert = (kb_b_bert / total_esk_b) * 100.0 if total_esk_b > 0 else 0.0
        w_low_b_b, w_high_b_b = wilson_score_interval(kb_b_bert, total_esk_b)

        eb_bert = sum(1 for yt, p in zip(y_uji_true, p_esk_bert) if yt in KELAS_BOT and p >= AMBANG_BERT)
        pct_eb_bert = (eb_bert / total_jawab_uji) * 100.0

        print(f"\nHasil Uji Akhir IndoBERTweet (Waktu Latih = {time_bert:.2f}s):")
        print(f"  Macro-F1 (6 Kelas)     : {macro_f1_bert:.4f}")
        print(f"  AUROC Pemisahan        : {auroc_bert:.4f}")
        print(f"  % Dijawab Otomatis     : {pct_jawab_bert:.1f}% ({n_jawab_bert}/{total_uji})")
        print(f"  Kesalahan Berbahaya (a): {pct_kb_a_bert:.1f}% ({kb_a_bert}/{total_esk_a}) | Wilson 95%: [{w_low_a_b:.2f}%, {w_high_a_b:.2f}%]")
        print(f"  Kesalahan Berbahaya (b): {pct_kb_b_bert:.1f}% ({kb_b_bert}/{total_esk_b}) | Wilson 95%: [{w_low_b_b:.2f}%, {w_high_b_b:.2f}%]")
        print(f"  Eskalasi Berlebih      : {pct_eb_bert:.1f}% ({eb_bert}/{total_jawab_uji})")

        # ------------------------------------------------------------------
        # TAHAP 3: RINCIAN METRIK PER KELAS & MATRIKS KEBINGUNGAN UJI AKHIR
        # ------------------------------------------------------------------
        print("\n" + "=" * 115)
        print("RINCIAN METRIK PER KELAS PADA DATA UJI AKHIR (51 SESI)")
        print(f"{CATATAN_LABEL}")
        print("=" * 115)
        print(f"{'Kelas':<25} | {'Dukungan':<8} | {'Klasik Prec':<12} {'Rec':<8} {'F1':<8} | {'BERT Prec':<12} {'Rec':<8} {'F1':<8}")
        print("-" * 105)
        for idx_c, c_name in enumerate(SEMUA_KELAS):
            supp = sum(1 for yt in y_uji_true if yt == c_name)
            print(f"{c_name:<25} | {supp:<8} | {prec_klasik[idx_c]:<12.4f} {rec_klasik[idx_c]:<8.4f} {f1_klasik[idx_c]:<8.4f} | {prec_bert[idx_c]:<12.4f} {rec_bert[idx_c]:<8.4f} {f1_bert[idx_c]:<8.4f}")
        print("-" * 105)

        print("\nMATRIKS KEBINGUNGAN ROUTER KLASIK (UJI AKHIR):")
        print(f"{'':<24} | " + " | ".join([f"{c[:6]:<6}" for c in SEMUA_KELAS]))
        print("-" * 75)
        for idx_r, c_row in enumerate(SEMUA_KELAS):
            row_str = " | ".join([f"{cm_klasik[idx_r][idx_col]:<6}" for idx_col in range(len(SEMUA_KELAS))])
            print(f"{c_row:<24} | {row_str}")

        print("\nMATRIKS KEBINGUNGAN INDOBERTWEET (UJI AKHIR):")
        print(f"{'':<24} | " + " | ".join([f"{c[:6]:<6}" for c in SEMUA_KELAS]))
        print("-" * 75)
        for idx_r, c_row in enumerate(SEMUA_KELAS):
            row_str = " | ".join([f"{cm_bert[idx_r][idx_col]:<6}" for idx_col in range(len(SEMUA_KELAS))])
            print(f"{c_row:<24} | {row_str}")
        print("-" * 75)

        # ------------------------------------------------------------------
        # TAHAP 4: SIMPAN PREDIKSI KE uji_akhir_pred.csv
        # ------------------------------------------------------------------
        pred_rows = []
        for i, row in df_uji.iterrows():
            pred_rows.append({
                'chat_id': row['chat_id'],
                'sesi_id': row['sesi_id'],
                'label': row['label'],
                'pred_klasik': pred_klasik_raw[i],
                'prob_esk_klasik': round(p_esk_klasik[i], 4),
                'pred_bert': test_preds_bert_str[i],
                'prob_esk_bert': round(p_esk_bert[i], 4),
                'is_penutup': row['is_penutup']
            })

        df_pred_out = pd.DataFrame(pred_rows)
        pred_csv_path = HASIL_DIR / "uji_akhir_pred.csv"
        df_pred_out.to_csv(pred_csv_path, index=False)
        print(f"\nPrediksi uji akhir berhasil disimpan ke: {pred_csv_path}")

        # ------------------------------------------------------------------
        # TAHAP 5: SIMPAN ARTEFAK MODEL KE models/router_indobertweet/
        # ------------------------------------------------------------------
        MODELS_ROUTER_DIR.mkdir(parents=True, exist_ok=True)
        print(f"\nMenyimpan model IndoBERTweet final ke: {MODELS_ROUTER_DIR}...")
        model.save_pretrained(MODELS_ROUTER_DIR)
        tokenizer.save_pretrained(MODELS_ROUTER_DIR)

        router_cfg = {
            "model_base": NAMA_MODEL,
            "kelas": SEMUA_KELAS,
            "kelas_eskalasi": KELAS_ESKALASI,
            "ambang_peluang_eskalasi": AMBANG_BERT,
            "max_length": MAX_LENGTH,
            "seed": SEED_FINAL,
            "epoch": EPOCHS_FINAL,
            "learning_rate": LR_FINAL,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "grad_clip": GRAD_CLIP,
            "versi_pustaka": {
                "torch": torch.__version__,
                "transformers": "5.20.2",
                "sklearn": "1.9.1"
            },
            "tanggal": "2026-11-05",
            "jumlah_data_latih": len(df_latih),
            "daftar_chat_uji_akhir": sorted(chat_uji_akhir)
        }
        cfg_path = MODELS_ROUTER_DIR / "router_config.json"
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(router_cfg, f, indent=2)
        print(f"Artefak model & router_config.json berhasil disimpan ke: {cfg_path}")

        # ------------------------------------------------------------------
        # TAHAP 6: EVALUASI SISTEM GABUNGAN UTUH PADA UJI AKHIR (SEKALI)
        # ------------------------------------------------------------------
        print("\n" + "=" * 115)
        print("BAGIAN 3: EVALUASI SISTEM GABUNGAN UTUH (END-TO-END PIPELINE) PADA UJI AKHIR")
        print("Pipeline: Normalisasi -> Pengecekan Kebijakan/Bukti Bayar -> Router IndoBERTweet (T=0.100) -> Chroma & Gerbang Leksikal")
        print(f"{CATATAN_LABEL}")
        print("=" * 115)

        # Siapkan Chroma collection & fungsi bot
        bot.muat_indeks()

        hasil_pipeline = []
        sesi_esk_terjawab = []

        for i, row in df_uji.iterrows():
            sid = row['sesi_id']
            cid = row['chat_id']
            lbl = row['label']
            teks_mhs = row['teks_mentah']
            p_esk = p_esk_bert[i]
            bert_label = test_preds_bert_str[i]

            # 1. Normalisasi
            p_norm = bot.normalisasi(teks_mhs)

            # 2. Pengecekan aturan penolakan / topik terlarang
            terlarang, alasan_terlarang = bot.periksa_topik_terlarang(p_norm)
            
            # 3. Pengecekan router IndoBERTweet (T=0.100)
            if terlarang:
                keputusan = "teruskan_ke_staf"
                alasan = f"Topik terlarang: {alasan_terlarang}"
            elif bert_label == "bukti_bayar":
                # Penanganan bukti bayar
                keputusan = "teruskan_ke_staf"
                alasan = "Kanal bukti bayar diarahkan ke admin WhatsApp"
            elif p_esk >= AMBANG_BERT:
                keputusan = "teruskan_ke_staf"
                alasan = f"Peluang eskalasi router IndoBERTweet ({p_esk:.3f}) >= ambang ({AMBANG_BERT:.3f})"
            else:
                # 4. Pencarian Chroma & Gerbang Kata Kunci
                # Gunakan jawab_sistem_b untuk pencarian
                res_b = bot.jawab_sistem_b(teks_mhs)
                keputusan = res_b['keputusan']
                alasan = res_b['alasan']

            hasil_pipeline.append({
                'sesi_id': sid,
                'chat_id': cid,
                'label': lbl,
                'keputusan': keputusan,
                'alasan': alasan,
                'p_esk': p_esk,
                'teks': teks_mhs
            })

            if lbl in KELAS_ESKALASI and keputusan == "jawab":
                # Anonymize: hilangkan angka / teks sensitif jika ada
                teks_anon = re.sub(r'\d+', '[ANGKA]', teks_mhs)
                sesi_esk_terjawab.append({
                    'sesi_id': sid,
                    'chat_id': cid,
                    'label': lbl,
                    'alasan': alasan,
                    'teks_anonim': teks_anon
                })

        total_esk_uji = sum(1 for h in hasil_pipeline if h['label'] in KELAS_ESKALASI)  # 14
        total_bot_uji = sum(1 for h in hasil_pipeline if h['label'] in KELAS_BOT)       # 37

        n_esk_terjawab = len(sesi_esk_terjawab)
        pct_esk_terjawab = (n_esk_terjawab / total_esk_uji) * 100.0 if total_esk_uji > 0 else 0.0

        n_jawab_dialihkan = sum(1 for h in hasil_pipeline if h['label'] in KELAS_BOT and h['keputusan'] == "teruskan_ke_staf")
        pct_jawab_dialihkan = (n_jawab_dialihkan / total_bot_uji) * 100.0 if total_bot_uji > 0 else 0.0

        n_jawab_berhasil = sum(1 for h in hasil_pipeline if h['label'] in KELAS_BOT and h['keputusan'] == "jawab")
        pct_jawab_berhasil = (n_jawab_berhasil / total_bot_uji) * 100.0 if total_bot_uji > 0 else 0.0

        print(f"Hasil Evaluasi Sistem Gabungan Utuh pada Uji Akhir (51 Sesi):")
        print(f"  1. Sesi Eskalasi yang Tetap DIJAWAB Bot (False Negative Bahaya): {pct_esk_terjawab:.1f}% ({n_esk_terjawab}/{total_esk_uji})")
        print(f"  2. Sesi Jawab yang Dialihkan ke Staf (Eskalasi Aman Konservatif) : {pct_jawab_dialihkan:.1f}% ({n_jawab_dialihkan}/{total_bot_uji})")
        print(f"  3. Sesi Jawab yang Berhasil Dijawab Otomatis oleh Bot           : {pct_jawab_berhasil:.1f}% ({n_jawab_berhasil}/{total_bot_uji})")

        print(f"\nDaftar Sesi Berkelas Eskalasi yang Tetap DIJAWAB oleh Bot ({len(sesi_esk_terjawab)} sesi):")
        if sesi_esk_terjawab:
            for s in sesi_esk_terjawab:
                print(f"  - Sesi ID {s['sesi_id']} (Chat {s['chat_id']}, Label: {s['label']}):")
                print(f"    Teks Anonim: \"{s['teks_anonim'][:100]}\"")
                print(f"    Alasan Lolos: {s['alasan']}")
        else:
            print("  [NIL] 0 sesi! Tidak ada sesi eskalasi yang bocor dijawab otomatis (100% aman tersaring oleh sistem gabungan).")

        # ------------------------------------------------------------------
        # TAHAP 7: TABEL RANGKUMAN PERBANDINGAN UJI AKHIR
        # ------------------------------------------------------------------
        print("\n" + "=" * 115)
        print("TABEL PERBANDINGAN PERFORMA UJI AKHIR HELD-OUT (51 SESI)")
        print(f"{CATATAN_LABEL}")
        print("=" * 115)
        hdr = (
            f"{'Model / Sistem':<35} | {'Macro-F1':<10} | {'AUROC':<10} | "
            f"{'KB (a) Uji Akhir':<20} | {'Eskalasi Berlebih':<20} | {'% Jawab Otomatis'}"
        )
        print(hdr)
        print("-" * 115)
        baris_k = (
            f"{'Router Klasik (T=0.300)':<35} | {macro_f1_klasik:<10.4f} | {auroc_klasik:<10.4f} | "
            f"{pct_kb_a_klasik:<5.1f}% ({kb_a_klasik}/{total_esk_a}){'':<10} | "
            f"{pct_eb_klasik:<5.1f}% ({eb_klasik}/{total_jawab_uji}){'':<10} | "
            f"{pct_jawab_klasik:<5.1f}%"
        )
        baris_b = (
            f"{'IndoBERTweet (T=0.100)':<35} | {macro_f1_bert:<10.4f} | {auroc_bert:<10.4f} | "
            f"{pct_kb_a_bert:<5.1f}% ({kb_a_bert}/{total_esk_a}){'':<10} | "
            f"{pct_eb_bert:<5.1f}% ({eb_bert}/{total_jawab_uji}){'':<10} | "
            f"{pct_jawab_bert:<5.1f}%"
        )
        baris_sys = (
            f"{'Sistem Gabungan Utuh (End-to-End)':<35} | {'-':<10} | {'-':<10} | "
            f"{pct_esk_terjawab:<5.1f}% ({n_esk_terjawab}/{total_esk_uji}){'':<10} | "
            f"{pct_jawab_dialihkan:<5.1f}% ({n_jawab_dialihkan}/{total_bot_uji}){'':<10} | "
            f"{pct_jawab_berhasil:<5.1f}%"
        )
        print(baris_k)
        print(baris_b)
        print(baris_sys)
        print("=" * 115)

        print("\n" + "=" * 115)
        print("EVALUASI UJI AKHIR T6c SELESAI!")
        print("=" * 115)

    finally:
        sys.stdout = tee.terminal
        tee.close()


if __name__ == "__main__":
    main()
