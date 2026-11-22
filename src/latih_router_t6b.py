# ==============================================================
# latih_router_t6b.py — Pelatihan Deep Learning Penuh (T6b)
#
# Tahap T6b Proyek Tugas Akhir CHATTOBOT:
#   - Model: indolem/indobertweet-base-uncased (BERT lokal)
#   - Pengecekan memori GPU bebas (> 2.0 GB)
#   - Loop pelatihan PyTorch standar (tanpa Trainer / tanpa accelerate)
#   - fp32, batch 8, max_length 96, weight decay 0.01, warmup 10%,
#     gradient clipping 1.0, loss CrossEntropy berbobot kelas
#   - Pembersihan cache GPU antar lipatan
#   - Lipatan: hasil\lipatan.json (5 lipatan yang sama persis dengan T5)
#   - Evaluasi dua konfigurasi pada Seed 42:
#       (A) Epoch 8, LR 3e-5
#       (B) Epoch 15, LR 5e-5
#   - Konfigurasi terbaik dipilih hanya dari validasi, lalu diulang pada
#     Seed 43 dan Seed 44
#   - Pencatatan kurva belajar ke hasil\kurva_belajar.csv
#   - Penyimpanan probabilitas OOF ke hasil\oof_indobertweet.csv
#   - Analisis ambang peluang eskalasi (AUROC, Wilson 95% CI, versi a dan b)
#   - Tabel perbandingan berdampingan: Aturan vs Klasik vs IndoBERTweet
#   - Keluaran mentah dicatat ke hasil\log_T6b.txt
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
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

# Set terminal UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
HASIL_DIR = ROOT / "hasil"

sys.path.insert(0, str(SRC_DIR))
from config import (
    EXCEL_PATH, SHEET_SESI, SHEET_PESAN, SHEET_KAMUS,
    KELAS_BOT, KELAS_ESKALASI, SEMUA_KELAS, SEED,
    ROUTER_MAX_PESAN, ROUTER_MAX_KARAKTER
)

CATATAN_LABEL = "Catatan: label dibuat AI, TIDAK diverifikasi manusia (hasil bersifat indikatif)"
NAMA_MODEL = "indolem/indobertweet-base-uncased"
MAX_LENGTH = 96
BATCH_SIZE = 8
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
GRAD_CLIP = 1.0

# Kategori penutup untuk analisis substantif Versi (b)
KATA_PENUTUP_SET = {
    'terima', 'kasih', 'terimakasih', 'makasih', 'thanks', 'thank', 'you',
    'oke', 'ok', 'baik', 'siap', 'sip', 'iya', 'ya',
    'min', 'admin', 'ka', 'kak', 'mba', 'mas', 'pak', 'bu',
    'selamat', 'pagi', 'siang', 'sore', 'malam',
    'info', 'infonya', 'bantuan', 'bantuannya', 'banyak', 'sama'
}


class OutputTee:
    """Menduplikasi keluaran ke terminal dan berkas log."""
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
    """Menghitung selang kepercayaan Wilson 95% untuk proporsi k / n."""
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
    """Aturan sederhana: pesan hanya ucapan terima kasih / sapaan / penutup singkat."""
    teks_clean = re.sub(r'[^a-zA-Z\s]', ' ', teks).lower()
    words = [re.sub(r'([a-z])\1+', r'\1', w) for w in teks_clean.split()]
    mapping = {
        'trimakasih': 'terimakasih', 'makasi': 'makasih', 'thx': 'thanks',
        'bantuanya': 'bantuan', 'bantuanyaa': 'bantuan', 'okey': 'oke', 'okkee': 'oke'
    }
    words_mapped = [mapping.get(w, w) for w in words]
    if words_mapped and len(words_mapped) <= 10 and all(w in KATA_PENUTUP_SET for w in words_mapped):
        return True
    return False


def muat_kamus_kampus() -> dict:
    """Memuat kamus singkatan lokal kampus (sheet Kamus Excel + fallback bawaan)."""
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
    """Normalisasi teks masukan menggunakan kamus kampus saja."""
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
    """Dataset PyTorch untuk sesi obrolan mahasiswa."""
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


def latih_satu_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer,
    device,
    epochs: int,
    lr: float,
    seed: int,
    config_name: str,
    kurva_records: list
):
    """Melatih satu lipatan model BERT dengan PyTorch standar."""
    # Hitung bobot kelas seimbang
    label_to_idx = {nama: idx for idx, nama in enumerate(SEMUA_KELAS)}
    y_train_idx = [label_to_idx[lbl] for lbl in train_df['label']]
    y_val_idx = [label_to_idx[lbl] for lbl in val_df['label']]

    classes_present = np.unique(y_train_idx)
    raw_weights = compute_class_weight('balanced', classes=classes_present, y=np.array(y_train_idx))
    weight_tensor = torch.ones(len(SEMUA_KELAS), dtype=torch.float, device=device)
    for c_id, w in zip(classes_present, raw_weights):
        weight_tensor[c_id] = float(w)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    train_ds = TeksSesiDataset(train_df['teks'].tolist(), y_train_idx, tokenizer)
    val_ds = TeksSesiDataset(val_df['teks'].tolist(), y_val_idx, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Inisialisasi model
    model = AutoModelForSequenceClassification.from_pretrained(
        NAMA_MODEL,
        num_labels=len(SEMUA_KELAS),
        problem_type="single_label_classification"
    )
    model.to(device)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    peak_mem_mb = 0.0
    start_time = time.time()
    train_losses_per_epoch = []
    val_f1_per_epoch = []

    for ep in range(1, epochs + 1):
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

            cur_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            if cur_mem > peak_mem_mb:
                peak_mem_mb = cur_mem

        avg_train_loss = total_loss / len(train_loader)
        train_losses_per_epoch.append(avg_train_loss)

        # Evaluasi validasi akhir epoch
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_trues = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_trues.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_macro_f1 = f1_score(val_trues, val_preds, average='macro', zero_division=0)
        val_f1_per_epoch.append(val_macro_f1)

        kurva_records.append({
            'seed': seed,
            'konfigurasi': config_name,
            'fold': fold_idx,
            'epoch': ep,
            'train_loss': round(avg_train_loss, 4),
            'val_loss': round(avg_val_loss, 4),
            'val_macro_f1': round(val_macro_f1, 4)
        })

    elapsed_sec = time.time() - start_time

    # Evaluasi final & ekstrak probabilitas softmax OOF
    model.eval()
    val_preds_final = []
    val_probs_final = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            val_preds_final.extend(preds)
            val_probs_final.extend(probs)

    final_val_f1 = f1_score(y_val_idx, val_preds_final, average='macro', zero_division=0)
    loss_turun = "Ya" if len(train_losses_per_epoch) >= 2 and train_losses_per_epoch[-1] < train_losses_per_epoch[0] else "Stabil/Naik"

    # Bersihkan model & cache GPU
    del model
    del optimizer
    del scheduler
    torch.cuda.empty_cache()

    return {
        'fold': fold_idx,
        'val_f1': final_val_f1,
        'elapsed_sec': elapsed_sec,
        'peak_mem_mb': peak_mem_mb,
        'loss_turun': loss_turun,
        'train_loss_start': train_losses_per_epoch[0],
        'train_loss_end': train_losses_per_epoch[-1],
        'val_preds': val_preds_final,
        'val_probs': val_probs_final,
        'val_trues': y_val_idx,
        'sesi_ids': val_df['sesi_id'].tolist(),
        'chat_ids': val_df['chat_id'].tolist(),
        'labels_str': val_df['label'].tolist()
    }


def evaluasi_5_lipatan(
    config_name: str,
    epochs: int,
    lr: float,
    seed: int,
    df_latih: pd.DataFrame,
    lipatan_list: list,
    tokenizer,
    device,
    kurva_records: list
):
    """Menjalankan 5 lipatan cross-validation untuk satu konfigurasi dan seed tertentu."""
    print(f"\n>>> MENJALANKAN {config_name} (Epoch={epochs}, LR={lr}, Seed={seed}) PADA 5 LIPATAN <<<")
    set_seed(seed)
    
    hasil_per_fold = []
    oof_data = []

    for fold_info in lipatan_list:
        fold_idx = fold_info['fold']
        val_sids = fold_info['sesi_val']

        train_df = df_latih[~df_latih['sesi_id'].isin(val_sids)]
        val_df = df_latih[df_latih['sesi_id'].isin(val_sids)]

        res = latih_satu_fold(
            fold_idx=fold_idx,
            train_df=train_df,
            val_df=val_df,
            tokenizer=tokenizer,
            device=device,
            epochs=epochs,
            lr=lr,
            seed=seed,
            config_name=config_name,
            kurva_records=kurva_records
        )
        hasil_per_fold.append(res)

        for sid, cid, lbl, p_arr, pred_cls in zip(
            res['sesi_ids'], res['chat_ids'], res['labels_str'], res['val_probs'], res['val_preds']
        ):
            oof_data.append({
                'sesi_id': sid,
                'chat_id': cid,
                'label': lbl,
                'pred_idx': pred_cls,
                'pred_str': SEMUA_KELAS[pred_cls],
                'probs': p_arr
            })

        print(f"  Fold {fold_idx}: Val Macro-F1 = {res['val_f1']:.4f} | Waktu = {res['elapsed_sec']:.1f}s | "
              f"Memori Puncak = {res['peak_mem_mb']:.1f} MB | Loss: {res['train_loss_start']:.4f} -> {res['train_loss_end']:.4f} ({res['loss_turun']})")

    f1_list = [h['val_f1'] for h in hasil_per_fold]
    mean_f1 = np.mean(f1_list)
    std_f1 = np.std(f1_list)
    total_time = sum(h['elapsed_sec'] for h in hasil_per_fold)
    max_peak_mem = max(h['peak_mem_mb'] for h in hasil_per_fold)

    print(f"  ==> Rata-rata Macro-F1: {mean_f1:.4f} (+/- {std_f1:.4f}) | Total Waktu: {total_time:.1f}s | Puncak Memori: {max_peak_mem:.1f} MB\n")

    return {
        'config_name': config_name,
        'seed': seed,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'f1_per_fold': f1_list,
        'total_time': total_time,
        'max_peak_mem': max_peak_mem,
        'oof_data': oof_data
    }


def main():
    tee = OutputTee(HASIL_DIR / "log_T6b.txt")
    sys.stdout = tee

    try:
        print("=" * 115)
        print("LAPORAN PELATIHAN DEEP LEARNING LOKAL T6b (INDOBERTWEEET)")
        print(f"{CATATAN_LABEL}")
        print("=" * 115)

        # 1. Pengecekan Memori GPU Bebas
        if not torch.cuda.is_available():
            print("[GALAT KRITIS] CUDA tidak terdeteksi! Pelatihan memerlukan GPU lokal. Berhenti.")
            return

        device = torch.device("cuda:0")
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_gb = free_bytes / (1024 ** 3)
        total_gb = total_bytes / (1024 ** 3)

        print("\n[STATUS PERANGKAT GPU SEBELUM PELATIHAN]")
        print(f"Perangkat      : {torch.cuda.get_device_name(device)}")
        print(f"Total Memori   : {total_gb:.2f} GB")
        print(f"Memori Bebas   : {free_gb:.2f} GB")

        if free_gb < 2.0:
            print(f"[PERINGATAN KRITIS] Memori GPU bebas ({free_gb:.2f} GB) kurang dari 2.0 GB!")
            print("Sesuai instruksi kerja, proses BERHENTI. Mohon tutup aplikasi lain yang menggunakan GPU.")
            return

        print(f"Pemeriksaan memori GPU lulus: {free_gb:.2f} GB bebas (>= 2.0 GB). Melanjutkan pelatihan.\n")

        # 2. Muat Tokenizer
        print(f"Memuat tokenizer dari: {NAMA_MODEL}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(NAMA_MODEL)
            print("Tokenizer berhasil dimuat.")
        except Exception as e:
            print(f"[GALAT TRANSFORMERS / UNDUHAN] Gagal memuat tokenizer '{NAMA_MODEL}': {e}")
            print("BERHENTI sesuai instruksi tanpa menurunkan versi atau mengganti model.")
            return

        # 3. Muat Data & Lipatan
        print("\nMempersiapkan data dan lipatan...")
        kamus_kampus = muat_kamus_kampus()
        df_sesi = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_SESI)
        df_pesan = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_PESAN)

        kolom_label = "topik_koreksi (isi manual)"
        df_sesi_valid = df_sesi[df_sesi[kolom_label].notna() & df_sesi[kolom_label].isin(SEMUA_KELAS)].copy()
        pesan_mhs = df_pesan[df_pesan['pengirim'] == 'Mahasiswa'].copy()

        # Baca lipatan.json
        berkas_lipatan = HASIL_DIR / "lipatan.json"
        with open(berkas_lipatan, "r", encoding="utf-8") as f:
            lipatan_json = json.load(f)

        chat_uji_akhir = lipatan_json['chat_uji_akhir']

        # Siapkan data latih per sesi
        sesi_list = []
        for _, row in df_sesi_valid.iterrows():
            sid = row['sesi_id']
            cid = row['chat_id']
            lbl = row[kolom_label]

            p_sesi = pesan_mhs[pesan_mhs['sesi_id'] == sid].sort_values('no')
            p_text = " ".join([str(p) for p in p_sesi['teks_anonim'].head(ROUTER_MAX_PESAN).dropna().tolist()])
            p_clean = bersihkan_teks(p_text, kamus_kampus)
            p_penutup = is_penutup_murni(p_text)

            sesi_list.append({
                'sesi_id': sid,
                'chat_id': cid,
                'label': lbl,
                'teks': p_clean,
                'teks_mentah': p_text,
                'is_penutup': p_penutup
            })

        df_all = pd.DataFrame(sesi_list)
        df_latih = df_all[~df_all['chat_id'].isin(chat_uji_akhir)].copy().reset_index(drop=True)

        print(f"Total Sesi Data Latih Lipatan : {len(df_latih)} sesi (dari {df_latih['chat_id'].nunique()} chat)")
        print(f"Chat Uji Akhir Dikecualikan  : {chat_uji_akhir} (TIDAK disentuh)")
        print(f"Normalisasi yang Digunakan   : Kamus Kampus Saja ({len(kamus_kampus)} entri), tanpa Kamus Alay (sesuai Keputusan K-019)")

        kurva_records = []

        # ------------------------------------------------------------------
        # TAHAP 1: PERBANDINGAN DUA KONFIGURASI PADA SEED 42
        # ------------------------------------------------------------------
        print("\n" + "=" * 115)
        print("BAGIAN 1: PERBANDINGAN KONFIGURASI HIPERPARAMETER PADA SEED 42")
        print("  - Konfigurasi (A): Epoch 8, LR 3e-5")
        print("  - Konfigurasi (B): Epoch 15, LR 5e-5")
        print("=" * 115)

        # Konfigurasi A
        res_A_seed42 = evaluasi_5_lipatan(
            config_name="Konfigurasi A (Epoch 8, LR 3e-5)",
            epochs=8,
            lr=3e-5,
            seed=42,
            df_latih=df_latih,
            lipatan_list=lipatan_json['lipatan'],
            tokenizer=tokenizer,
            device=device,
            kurva_records=kurva_records
        )

        # Konfigurasi B
        res_B_seed42 = evaluasi_5_lipatan(
            config_name="Konfigurasi B (Epoch 15, LR 5e-5)",
            epochs=15,
            lr=5e-5,
            seed=42,
            df_latih=df_latih,
            lipatan_list=lipatan_json['lipatan'],
            tokenizer=tokenizer,
            device=device,
            kurva_records=kurva_records
        )

        # Bandingkan validasi
        print("-" * 115)
        print(f"HASIL PERBANDINGAN SEED 42:")
        print(f"  Konfigurasi A (Epoch 8,  LR 3e-5): Macro-F1 = {res_A_seed42['mean_f1']:.4f} (+/- {res_A_seed42['std_f1']:.4f}) | Waktu = {res_A_seed42['total_time']:.1f}s")
        print(f"  Konfigurasi B (Epoch 15, LR 5e-5): Macro-F1 = {res_B_seed42['mean_f1']:.4f} (+/- {res_B_seed42['std_f1']:.4f}) | Waktu = {res_B_seed42['total_time']:.1f}s")

        if res_A_seed42['mean_f1'] >= res_B_seed42['mean_f1']:
            best_config_name = "Konfigurasi A (Epoch 8, LR 3e-5)"
            best_epochs = 8
            best_lr = 3e-5
            best_res_seed42 = res_A_seed42
            print(f"==> Konfigurasi Terbaik Terpilih (Validasi Tertinggi): Konfigurasi A (Epoch 8, LR 3e-5)\n")
        else:
            best_config_name = "Konfigurasi B (Epoch 15, LR 5e-5)"
            best_epochs = 15
            best_lr = 5e-5
            best_res_seed42 = res_B_seed42
            print(f"==> Konfigurasi Terbaik Terpilih (Validasi Tertinggi): Konfigurasi B (Epoch 15, LR 5e-5)\n")

        # ------------------------------------------------------------------
        # TAHAP 2: PENGULANGAN KONFIGURASI TERBAIK PADA SEED 43 DAN 44
        # ------------------------------------------------------------------
        print("=" * 115)
        print(f"BAGIAN 2: PENGULANGAN {best_config_name} PADA SEED 43 DAN SEED 44")
        print("=" * 115)

        res_seed43 = evaluasi_5_lipatan(
            config_name=best_config_name,
            epochs=best_epochs,
            lr=best_lr,
            seed=43,
            df_latih=df_latih,
            lipatan_list=lipatan_json['lipatan'],
            tokenizer=tokenizer,
            device=device,
            kurva_records=kurva_records
        )

        res_seed44 = evaluasi_5_lipatan(
            config_name=best_config_name,
            epochs=best_epochs,
            lr=best_lr,
            seed=44,
            df_latih=df_latih,
            lipatan_list=lipatan_json['lipatan'],
            tokenizer=tokenizer,
            device=device,
            kurva_records=kurva_records
        )

        # Simpan kurva belajar ke kurva_belajar.csv
        kurva_df = pd.DataFrame(kurva_records)
        kurva_csv_path = HASIL_DIR / "kurva_belajar.csv"
        kurva_df.to_csv(kurva_csv_path, index=False)
        print(f"Kurva belajar per epoch berhasil disimpan ke: {kurva_csv_path}")

        # Agregasi performa 3 seed
        all_seeds_results = [best_res_seed42, res_seed43, res_seed44]
        all_fold_f1s = [f1 for r in all_seeds_results for f1 in r['f1_per_fold']]
        overall_mean_f1 = np.mean(all_fold_f1s)
        overall_std_f1 = np.std(all_fold_f1s)

        print("\n" + "=" * 115)
        print(f"RANGKUMAN MULTI-SEED ({best_config_name})")
        print("=" * 115)
        print(f"Seed 42 : Macro-F1 = {best_res_seed42['mean_f1']:.4f} (+/- {best_res_seed42['std_f1']:.4f})")
        print(f"Seed 43 : Macro-F1 = {res_seed43['mean_f1']:.4f} (+/- {res_seed43['std_f1']:.4f})")
        print(f"Seed 44 : Macro-F1 = {res_seed44['mean_f1']:.4f} (+/- {res_seed44['std_f1']:.4f})")
        print(f"GABUNGAN 3 SEED (15 Lipatan) : Macro-F1 = {overall_mean_f1:.4f} (+/- {overall_std_f1:.4f})")
        print("=" * 115)

        # ------------------------------------------------------------------
        # TAHAP 3: PENYIMPANAN PROBABILITAS OOF KE oof_indobertweet.csv
        # ------------------------------------------------------------------
        oof_best = best_res_seed42['oof_data']
        # Susun DataFrame OOF dengan 6 probabilitas kelas
        oof_rows = []
        for item in oof_best:
            row_dict = {
                'sesi_id': item['sesi_id'],
                'chat_id': item['chat_id'],
                'label': item['label'],
                'pred_label': item['pred_str']
            }
            for idx_c, c_name in enumerate(SEMUA_KELAS):
                row_dict[f"prob_{c_name}"] = round(float(item['probs'][idx_c]), 6)
            oof_rows.append(row_dict)

        df_oof = pd.DataFrame(oof_rows)
        oof_csv_path = HASIL_DIR / "oof_indobertweet.csv"
        df_oof.to_csv(oof_csv_path, index=False)
        print(f"\nProbabilitas OOF berhasil disimpan ke: {oof_csv_path}")

        # ------------------------------------------------------------------
        # TAHAP 4: METRIK PER KELAS & MATRIKS KEBINGUNGAN AGREGAT
        # ------------------------------------------------------------------
        y_true_all = df_oof['label'].tolist()
        y_pred_all = df_oof['pred_label'].tolist()

        prec_per_class = precision_score(y_true_all, y_pred_all, labels=SEMUA_KELAS, average=None, zero_division=0)
        rec_per_class = recall_score(y_true_all, y_pred_all, labels=SEMUA_KELAS, average=None, zero_division=0)
        f1_per_class = f1_score(y_true_all, y_pred_all, labels=SEMUA_KELAS, average=None, zero_division=0)
        cm_agregat = confusion_matrix(y_true_all, y_pred_all, labels=SEMUA_KELAS)

        print("\n" + "=" * 115)
        print("METRIK PER KELAS — INDOBERTWEET (OOF SEED 42)")
        print("=" * 115)
        print(f"{'Kelas':<25} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Dukungan (Support)'}")
        print("-" * 75)
        for idx_c, c_name in enumerate(SEMUA_KELAS):
            supp = sum(1 for yt in y_true_all if yt == c_name)
            print(f"{c_name:<25} | {prec_per_class[idx_c]:<10.4f} | {rec_per_class[idx_c]:<10.4f} | {f1_per_class[idx_c]:<10.4f} | {supp}")
        print("-" * 75)

        print("\nMATRIKS KEBINGUNGAN AGREGAT (OOF SEED 42):")
        print(f"Baris = Aktual (True), Kolom = Prediksi")
        print(f"{'':<24} | " + " | ".join([f"{c[:6]:<6}" for c in SEMUA_KELAS]))
        print("-" * 75)
        for idx_r, c_row in enumerate(SEMUA_KELAS):
            row_str = " | ".join([f"{cm_agregat[idx_r][idx_col]:<6}" for idx_col in range(len(SEMUA_KELAS))])
            print(f"{c_row:<24} | {row_str}")
        print("-" * 75)

        # ------------------------------------------------------------------
        # TAHAP 5: ANALISIS AMBANG PELUANG ESKALASI (SAMA PERSIS DENGAN T5)
        # ------------------------------------------------------------------
        # P(eskalasi) = P(beasiswa_pendaftaran) + P(lainnya)
        idx_beasiswa = SEMUA_KELAS.index('beasiswa_pendaftaran')
        idx_lainnya = SEMUA_KELAS.index('lainnya')

        sesi_dict = {row['sesi_id']: row for _, row in df_all.iterrows()}

        oof_analisis = []
        for item in oof_best:
            sid = item['sesi_id']
            p_esk = float(item['probs'][idx_beasiswa] + item['probs'][idx_lainnya])
            is_penutup = sesi_dict[sid]['is_penutup']

            oof_analisis.append({
                'sesi_id': sid,
                'chat_id': item['chat_id'],
                'y_true': item['label'],
                'p_eskalasi': p_esk,
                'is_penutup': is_penutup
            })

        oof_substantif = [d for d in oof_analisis if not d['is_penutup']]

        # AUROC
        y_bin_all = [1 if d['y_true'] in KELAS_ESKALASI else 0 for d in oof_analisis]
        scores_all = [d['p_eskalasi'] for d in oof_analisis]
        auroc_all = roc_auc_score(y_bin_all, scores_all)

        y_bin_sub = [1 if d['y_true'] in KELAS_ESKALASI else 0 for d in oof_substantif]
        scores_sub = [d['p_eskalasi'] for d in oof_substantif]
        auroc_sub = roc_auc_score(y_bin_sub, scores_sub)

        print("\n" + "=" * 115)
        print("HASIL AUROC PEMISAHAN KELAS ESKALASI VS KELAS JAWAB (INDOBERTWEET)")
        print("=" * 115)
        print(f"AUROC Versi (a) - Seluruh 166 sesi latih OOF : {auroc_all:.4f}")
        print(f"AUROC Versi (b) - Tanpa sesi penutup murni   : {auroc_sub:.4f}")

        # Tabel Ambang 0.20 - 0.80
        ambang_range = np.arange(0.20, 0.81, 0.05)
        total_sesi = len(oof_analisis)
        total_esk_a = sum(1 for d in oof_analisis if d['y_true'] in KELAS_ESKALASI)
        total_esk_b = sum(1 for d in oof_substantif if d['y_true'] in KELAS_ESKALASI)
        total_jawab = sum(1 for d in oof_analisis if d['y_true'] in KELAS_BOT)

        print("\n" + "=" * 125)
        print("TABEL ANALISIS AMBANG PELUANG ESKALASI INDOBERTWEET (0.20 - 0.80)")
        print("Aturan Keputusan: Jika Peluang Eskalasi >= T -> Teruskan; Jika < T -> Dijawab Otomatis")
        print(f"{CATATAN_LABEL}")
        print("=" * 125)
        hdr = (
            f"{'Ambang T':<9} | {'% Jawab':<9} | "
            f"{'Kesalahan Berbahaya (a) [Semua]':<32} | {'Wilson 95% (a)':<16} | "
            f"{'Kesalahan Berbahaya (b) [Substantif]':<35} | {'Wilson 95% (b)':<16} | "
            f"{'Eskalasi Berlebih'}"
        )
        print(hdr)
        print("-" * 125)

        t6b_t_030 = {}

        for T_val in ambang_range:
            T = round(T_val, 2)
            n_jawab = sum(1 for d in oof_analisis if d['p_eskalasi'] < T)
            pct_jawab = (n_jawab / total_sesi) * 100

            k_a = sum(1 for d in oof_analisis if d['y_true'] in KELAS_ESKALASI and d['p_eskalasi'] < T)
            pct_kb_a = (k_a / total_esk_a) * 100
            w_low_a, w_high_a = wilson_score_interval(k_a, total_esk_a)

            k_b = sum(1 for d in oof_substantif if d['y_true'] in KELAS_ESKALASI and d['p_eskalasi'] < T)
            pct_kb_b = (k_b / total_esk_b) * 100
            w_low_b, w_high_b = wilson_score_interval(k_b, total_esk_b)

            k_eb = sum(1 for d in oof_analisis if d['y_true'] in KELAS_BOT and d['p_eskalasi'] >= T)
            pct_eb = (k_eb / total_jawab) * 100

            baris = (
                f"{T:<9.2f} | {pct_jawab:<8.1f}% | "
                f"{pct_kb_a:<5.1f}% ({k_a:>2}/{total_esk_a}){'':<18} | "
                f"[{w_low_a:>5.2f}%, {w_high_a:>5.2f}%] | "
                f"{pct_kb_b:<5.1f}% ({k_b:>2}/{total_esk_b}){'':<21} | "
                f"[{w_low_b:>5.2f}%, {w_high_b:>5.2f}%] | "
                f"{pct_eb:<5.1f}% ({k_eb:>2}/{total_jawab})"
            )
            print(baris)

            if abs(T - 0.30) < 1e-4:
                t6b_t_030 = {
                    'pct_jawab': pct_jawab,
                    'pct_kb_a': pct_kb_a,
                    'k_a': k_a,
                    'w_a': (w_low_a, w_high_a),
                    'pct_kb_b': pct_kb_b,
                    'k_b': k_b,
                    'w_b': (w_low_b, w_high_b),
                    'pct_eb': pct_eb,
                    'k_eb': k_eb
                }

        print("=" * 125)

        # ------------------------------------------------------------------
        # TAHAP 6: TABEL BERDAMPINGAN ROUTER ATURAN VS KLASIK VS INDOBERTWEET
        # ------------------------------------------------------------------
        print("\n" + "=" * 115)
        print("TABEL PERBANDINGAN BERDAMPINGAN: ROUTER ATURAN VS KLASIK (T5) VS INDOBERTWEET (T6b)")
        print(f"{CATATAN_LABEL}")
        print("=" * 115)
        header_cmp = (
            f"{'Model Router':<32} | {'Macro-F1 (std)':<20} | "
            f"{'KB Versi (a)':<15} | {'KB Versi (b)':<15} | {'Eskalasi Berlebih':<18} | {'AUROC'}"
        )
        print(header_cmp)
        print("-" * 115)

        # Ambil metrik router aturan & klasik dari catatan T5
        print(f"{'Baseline Router Aturan':<32} | {'0.4997 (+/- 0.0475)':<20} | {'27.60%':<15} | {'-':<15} | {'21.46%':<18} | {'-'}")
        print(f"{'Router Klasik TFIDF + LogReg':<32} | {'0.5112 (+/- 0.0295)':<20} | {'44.51%':<15} | {'-':<15} | {'15.85%':<18} | {'-'}")
        print(f"{'Router Klasik (T=0.30, Eskalasi)':<32} | {'0.7826 (AUROC)':<20} | {'7.50%':<15} | {'8.90%':<15} | {'50.40%':<18} | {'0.7826 / 0.7469'}")

        bert_str = f"{overall_mean_f1:.4f} (+/- {overall_std_f1:.4f})"
        kb_a_str = f"{t6b_t_030.get('pct_kb_a', 0.0):.2f}%"
        kb_b_str = f"{t6b_t_030.get('pct_kb_b', 0.0):.2f}%"
        eb_str = f"{t6b_t_030.get('pct_eb', 0.0):.2f}%"
        auroc_str = f"{auroc_all:.4f} / {auroc_sub:.4f}"

        print(f"{'IndoBERTweet (T6b, Multi-Seed)':<32} | {bert_str:<20} | {'-':<15} | {'-':<15} | {'-':<18} | {'-'}")
        print(f"{'IndoBERTweet (T=0.30, Eskalasi)':<32} | {f'{auroc_all:.4f} (AUROC)':<20} | {kb_a_str:<15} | {kb_b_str:<15} | {eb_str:<18} | {auroc_str}")
        print("=" * 115)

        # Tambahkan ke hasil_router.csv
        csv_router_path = HASIL_DIR / "hasil_router.csv"
        df_router = pd.read_csv(csv_router_path) if csv_router_path.exists() else pd.DataFrame()
        if not df_router.empty and 'model' in df_router.columns:
            df_router = df_router[~df_router['model'].isin([
                "IndoBERTweet Base Uncased (T6b)",
                "IndoBERTweet (Peluang Eskalasi T=0.30, Versi a)",
                "IndoBERTweet (Peluang Eskalasi T=0.30, Versi b - Substantif)"
            ])]

        baris_baru_bert = pd.DataFrame([
            {
                'model': 'IndoBERTweet Base Uncased (T6b)',
                'macro_f1_mean': round(overall_mean_f1, 4),
                'macro_f1_std': round(overall_std_f1, 4),
                'kesalahan_berbahaya_persen': 0.0,
                'eskalasi_berlebih_persen': 0.0,
                'catatan': f"3 Seed (42, 43, 44), 15 Lipatan; {best_config_name}; {CATATAN_LABEL}"
            },
            {
                'model': 'IndoBERTweet (Peluang Eskalasi T=0.30, Versi a)',
                'macro_f1_mean': round(auroc_all, 4),
                'macro_f1_std': 0.0,
                'kesalahan_berbahaya_persen': round(t6b_t_030.get('pct_kb_a', 0.0), 2),
                'eskalasi_berlebih_persen': round(t6b_t_030.get('pct_eb', 0.0), 2),
                'catatan': f"AUROC={auroc_all:.4f}; Wilson 95%=[{t6b_t_030.get('w_a', (0,0))[0]}%, {t6b_t_030.get('w_a', (0,0))[1]}%]; {CATATAN_LABEL}"
            },
            {
                'model': 'IndoBERTweet (Peluang Eskalasi T=0.30, Versi b - Substantif)',
                'macro_f1_mean': round(auroc_sub, 4),
                'macro_f1_std': 0.0,
                'kesalahan_berbahaya_persen': round(t6b_t_030.get('pct_kb_b', 0.0), 2),
                'eskalasi_berlebih_persen': round(t6b_t_030.get('pct_eb', 0.0), 2),
                'catatan': f"AUROC={auroc_sub:.4f}; Wilson 95%=[{t6b_t_030.get('w_b', (0,0))[0]}%, {t6b_t_030.get('w_b', (0,0))[1]}%]; {CATATAN_LABEL}"
            }
        ])
        df_router_updated = pd.concat([df_router, baris_baru_bert], ignore_index=True)
        df_router_updated.to_csv(csv_router_path, index=False)
        print(f"Hasil evaluasi T6b berhasil diperbarui ke: {csv_router_path}")

        print("\n" + "=" * 115)
        print("PELATIHAN T6b SELESAI SELURUHNYA!")
        print("=" * 115)

    finally:
        sys.stdout = tee.terminal
        tee.close()


if __name__ == "__main__":
    main()

