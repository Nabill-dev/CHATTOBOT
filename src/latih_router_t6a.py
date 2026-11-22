# ==============================================================
# latih_router_t6a.py — Uji Alur Pelatihan Deep Learning (T6a)
#
# Tahap T6a Proyek Tugas Akhir CHATTOBOT:
#   - Model: indolem/indobertweet-base-uncased (BERT lokal)
#   - Uji alur: 1 epoch, 2 lipatan (Fold 0 dan Fold 1 dari lipatan.json), seed 42
#   - Pengecekan memori GPU bebas (> 2 GB)
#   - Loop pelatihan PyTorch standar (tanpa Trainer / tanpa accelerate)
#   - Konfigurasi: batch 8, max_length 96, fp32, lr 3e-5, weight decay 0.01,
#     warmup 10%, gradient clipping 1.0, loss CrossEntropy berbobot kelas
#   - Pencatatan: waktu per epoch, memori puncak, tren penurunan loss, macro-F1
#   - Keluaran mentah disimpan ke hasil\log_T6a.txt
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
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

# Output terminal UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
HASIL_DIR = ROOT / "hasil"

sys.path.insert(0, str(SRC_DIR))
from config import (
    EXCEL_PATH, SHEET_SESI, SHEET_PESAN, SHEET_KAMUS,
    SEMUA_KELAS, SEED, ROUTER_MAX_PESAN, ROUTER_MAX_KARAKTER
)

CATATAN_LABEL = "Catatan: label dibuat AI, TIDAK diverifikasi manusia (hasil bersifat indikatif)"
NAMA_MODEL = "indolem/indobertweet-base-uncased"
MAX_LENGTH = 96
BATCH_SIZE = 8
EPOCHS = 1
LR = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
GRAD_CLIP = 1.0


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


def muat_kamus_lokal() -> dict:
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
    """Dataset PyTorch untuk sesi obrolan mahasiswa."""
    def __init__(self, teks_list, label_list, tokenizer, max_len=96):
        self.teks_list = teks_list
        self.label_list = label_list
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.teks_list)

    def __getitem__(self, idx):
        teks = str(self.teks_list[idx])
        label = self.label_list[idx]
        encoding = self.tokenizer(
            teks,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def main():
    log_file_path = HASIL_DIR / "log_T6a.txt"
    tee = OutputTee(log_file_path)
    sys.stdout = tee

    try:
        print("=" * 85)
        print("UJI ALUR PELATIHAN DEEP LEARNING ROUTER (T6a)")
        print(f"Model       : {NAMA_MODEL}")
        print(f"Pengaturan  : 1 Epoch, 2 Lipatan (Fold 0 & Fold 1), Seed {SEED}")
        print(f"Perangkat   : GPU Lokal GTX 1650 (PyTorch murni, FP32, Batch {BATCH_SIZE})")
        print(f"{CATATAN_LABEL}")
        print("=" * 85)

        set_seed(SEED)

        # 1. Pengecekan Memori GPU Bebas
        if not torch.cuda.is_available():
            print("[GALAT KRITIS] CUDA tidak terdeteksi! Pelatihan memerlukan GPU lokal. Berhenti.")
            return

        device = torch.device("cuda:0")
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_gb = free_bytes / (1024 ** 3)
        total_gb = total_bytes / (1024 ** 3)

        print(f"\n[STATUS PERANGKAT GPU]")
        print(f"Device Name : {torch.cuda.get_device_name(device)}")
        print(f"Total Memori: {total_gb:.2f} GB")
        print(f"Memori Bebas: {free_gb:.2f} GB")

        if free_gb < 2.0:
            print(f"[PERINGATAN KRITIS] Memori GPU bebas ({free_gb:.2f} GB) kurang dari 2.0 GB!")
            print("Sesuai aturan, pelatihan dihentikan. Mohon tutup aplikasi lain yang menggunakan GPU.")
            return
        else:
            print(f"Memori GPU mencukupi ({free_gb:.2f} GB >= 2.0 GB). Melanjutkan pelatihan.\n")

        # 2. Muat Tokenizer
        print(f"Memuat tokenizer dari: {NAMA_MODEL}...")
        tokenizer = AutoTokenizer.from_pretrained(NAMA_MODEL)
        print("Tokenizer berhasil dimuat.")

        # 3. Muat Data & Lipatan
        kamus = muat_kamus_lokal()
        df_sesi = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_SESI)
        df_pesan = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_PESAN)

        berkas_lipatan = HASIL_DIR / "lipatan.json"
        with open(berkas_lipatan, "r", encoding="utf-8") as f:
            lipatan_data = json.load(f)

        pesan_mhs = df_pesan[df_pesan['pengirim'] == 'Mahasiswa']
        df_sesi_valid = df_sesi[df_sesi['topik_koreksi (isi manual)'].isin(SEMUA_KELAS)].copy()

        sesi_dict = {}
        for _, row in df_sesi_valid.iterrows():
            sid = row['sesi_id']
            lbl = row['topik_koreksi (isi manual)']
            msgs = pesan_mhs[pesan_mhs['sesi_id'] == sid].sort_values('no')['teks_anonim'].head(ROUTER_MAX_PESAN).dropna().tolist()
            teks = bersihkan_teks(" ".join(str(m) for m in msgs), kamus)
            sesi_dict[sid] = {'label': lbl, 'teks': teks}

        label_to_id = {k: i for i, k in enumerate(SEMUA_KELAS)}
        id_to_label = {i: k for i, k in enumerate(SEMUA_KELAS)}

        # 4. Loop Pelatihan 2 Lipatan (Fold 0 dan Fold 1)
        hasil_t6a = []

        for fold_info in lipatan_data['lipatan'][:2]:
            fold_idx = fold_info['fold']
            val_sids = fold_info['sesi_val']
            train_sids = [sid for f_other in lipatan_data['lipatan'] if f_other['fold'] != fold_idx for sid in f_other['sesi_val']]

            print("\n" + "-" * 75)
            print(f"MEMULAI LIPATAN {fold_idx} (Train: {len(train_sids)} sesi, Val: {len(val_sids)} sesi)")
            print("-" * 75)

            # Siapkan Dataset & DataLoader
            train_teks = [sesi_dict[s]['teks'] for s in train_sids]
            train_labels = [label_to_id[sesi_dict[s]['label']] for s in train_sids]
            val_teks = [sesi_dict[s]['teks'] for s in val_sids]
            val_labels = [label_to_id[sesi_dict[s]['label']] for s in val_sids]

            # Bobot kelas untuk Loss berbobot
            classes_arr = np.array(range(len(SEMUA_KELAS)))
            weights = compute_class_weight('balanced', classes=classes_arr, y=np.array(train_labels))
            class_weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)

            train_ds = TeksSesiDataset(train_teks, train_labels, tokenizer, max_len=MAX_LENGTH)
            val_ds = TeksSesiDataset(val_teks, val_labels, tokenizer, max_len=MAX_LENGTH)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

            # Inisialisasi Model IndoBERTweet
            print(f"Menginisialisasi model {NAMA_MODEL} untuk lipatan {fold_idx}...")
            model = AutoModelForSequenceClassification.from_pretrained(
                NAMA_MODEL,
                num_labels=len(SEMUA_KELAS),
                id2label=id_to_label,
                label2id=label_to_id
            )
            model.to(device)

            # Optimizer & Scheduler
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            total_steps = len(train_loader) * EPOCHS
            warmup_steps = int(total_steps * WARMUP_RATIO)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

            # Reset statistik memori GPU
            torch.cuda.reset_peak_memory_stats(device)
            waktu_mulai = time.perf_counter()

            # --- TRAINING LOOP (1 Epoch) ---
            model.train()
            train_losses = []
            print(f"Memulai pelatihan Epoch 1 ({len(train_loader)} batch)...")

            for step, batch in enumerate(train_loader, 1):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                loss = criterion(logits, labels)
                loss.backward()

                # Gradient Clipping 1.0
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                optimizer.step()
                scheduler.step()

                loss_val = loss.item()
                train_losses.append(loss_val)

                if step == 1 or step == len(train_loader) // 2 or step == len(train_loader):
                    print(f"  Batch [{step:>2}/{len(train_loader)}] — Loss: {loss_val:.4f}")

            waktu_epoch = time.perf_counter() - waktu_mulai
            memori_puncak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

            loss_awal = train_losses[0]
            loss_akhir = train_losses[-1]
            loss_turun = loss_akhir < loss_awal
            loss_rata_train = np.mean(train_losses)

            print(f"\nHasil Pelatihan Lipatan {fold_idx}:")
            print(f"  Waktu Epoch     : {waktu_epoch:.2f} detik")
            print(f"  Memori Puncak   : {memori_puncak_mb:.1f} MB ({memori_puncak_mb/1024:.2f} GB)")
            print(f"  Loss Awal Batch : {loss_awal:.4f}")
            print(f"  Loss Akhir Batch: {loss_akhir:.4f}")
            print(f"  Apakah Loss Turun: {'YA, loss turun' if loss_turun else 'TIDAK (fluktuasi/naik)'}")
            print(f"  Rata-rata Loss  : {loss_rata_train:.4f}")

            # --- VALIDATION LOOP ---
            model.eval()
            val_losses = []
            semua_preds = []
            semua_targets = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = criterion(logits, labels)
                    val_losses.append(loss.item())

                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    semua_preds.extend(preds)
                    semua_targets.extend(labels.cpu().numpy())

            val_loss_rata = np.mean(val_losses)
            val_macro_f1 = f1_score(semua_targets, semua_preds, average='macro', zero_division=0)
            val_f1_per_kelas = f1_score(semua_targets, semua_preds, labels=range(len(SEMUA_KELAS)), average=None, zero_division=0)

            print(f"\nHasil Evaluasi Validasi Lipatan {fold_idx}:")
            print(f"  Val Loss : {val_loss_rata:.4f}")
            print(f"  Macro-F1 : {val_macro_f1:.4f}")
            print("  F1 per kelas:")
            for idx_k, nama_k in enumerate(SEMUA_KELAS):
                print(f"    - {nama_k:<22}: {val_f1_per_kelas[idx_k]:.4f}")

            hasil_t6a.append({
                'fold': fold_idx,
                'waktu_detik': waktu_epoch,
                'memori_puncak_mb': memori_puncak_mb,
                'loss_awal': loss_awal,
                'loss_akhir': loss_akhir,
                'loss_turun': loss_turun,
                'train_loss_mean': loss_rata_train,
                'val_loss_mean': val_loss_rata,
                'macro_f1': val_macro_f1,
            })

            # Kosongkan memori GPU antar lipatan
            del model, optimizer, scheduler, criterion
            torch.cuda.empty_cache()
            print(f"Memori cache GPU lipatan {fold_idx} telah dibersihkan.")

        # 5. Rangkuman Uji Alur T6a
        print("\n" + "=" * 85)
        print("RANGKUMAN UJI ALUR PELATIHAN T6a")
        print(f"{CATATAN_LABEL}")
        print("=" * 85)
        print(f"{'Lipatan':<10} | {'Waktu (detik)':<15} | {'Memori Puncak':<18} | {'Loss Turun?':<14} | {'Val Macro-F1'}")
        print("-" * 85)
        for h in hasil_t6a:
            print(
                f"Fold {h['fold']:<5} | {h['waktu_detik']:<15.2f} | "
                f"{h['memori_puncak_mb']:.1f} MB ({h['memori_puncak_mb']/1024:.2f} GB) | "
                f"{'Ya' if h['loss_turun'] else 'Tidak':<14} | {h['macro_f1']:.4f}"
            )
        print("=" * 85)
        print("Status Uji Alur T6a: SUKSES TANPA GALAT (Alur loop PyTorch terbukti berjalan lancar di GPU lokal).")
        print("BERHENTI di sini sesuai instruksi (menunggu persetujuan sebelum melangkah ke T6b).")
        print("=" * 85)

    finally:
        sys.stdout = tee.terminal
        tee.close()


if __name__ == "__main__":
    main()
