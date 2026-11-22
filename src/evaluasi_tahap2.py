"""
src/evaluasi_tahap2.py - Evaluasi Tahap 2: Data Sekunder dari Internet (Hanya Baca)

Tugas:
1. Normalisasi Kamus Alay (filter ambigu, lindungi istilah kampus, prioritas kamus kampus).
2. Perbandingan Router Klasik (TF-IDF + LogReg) pada lipatan.json: Tanpa vs Dengan Kamus Alay.
3. Uji cakupan normalisasi pada 2000 kalimat informal STIF-Indonesia.
4. Uji penolakan di luar domain pada 50 pertanyaan unanswerable IDK-MRC lewat Sistem B.
5. Inventarisasi repositori MASSIVE.
"""

import sys
import os
import io
import re
import csv
import json
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Setup UTF-8 encoding untuk Windows terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Setup path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from config import (
    EXCEL_PATH, SHEET_SESI, SHEET_PESAN, SHEET_KAMUS,
    HASIL_DIR, ROUTER_MAX_KARAKTER, ROUTER_MAX_PESAN,
    SEMUA_KELAS, KELAS_ESKALASI, SEED
)
import bot

# Pastikan reproduktifitas
random.seed(SEED)
np.random.seed(SEED)

CATATAN_LABEL = "Catatan: label dibuat AI, TIDAK diverifikasi manusia (hasil bersifat indikatif)"

# ==============================================================
# 1. PENGELOLAAN KAMUS NORMALISASI
# ==============================================================

ISTILAH_KAMPUS_DILINDUNGI = {
    'ektm', 'e-ktm', 'pmb', 'her', 'uas', 'uts', 'va', 'bsi', 'bsi.id', 'bsi id',
    'mybest', 'mystudent', 'angsuran', 'ormik', 'ormawa', 'serkom', 'juklak', 'po',
    'bju', 'kip', 'sem', 'semester', 'spp', 'krs', 'kprs', 'khs', 'transkrip',
    'ijazah', 'wisuda', 'skripsi', 'ta', 'cuti', 'nim', 'prodi', 'dosen', 'rekening',
    'virtual account', 'admin', 'teller', 'atm', 'm-banking', 'mobile banking'
}

def muat_kamus_kampus():
    """Memuat kamus kampus dari Excel dan fallback bawaan."""
    fallback = {
        'nnti': 'nanti', 'bsk': 'besok', 'skrng': 'sekarang', 'knpa': 'kenapa',
        'udh': 'sudah', 'sdh': 'sudah', 'blm': 'belum', 'gk': 'tidak',
        'gak': 'tidak', 'ga': 'tidak', 'gabisa': 'tidak bisa', 'gkpp': 'tidak apa-apa',
        'makasih': 'terima kasih', 'mksh': 'terima kasih', 'makasi': 'terima kasih',
        'klo': 'kalau', 'kalo': 'kalau', 'kl': 'kalau', 'bwt': 'buat',
        'utk': 'untuk', 'dgn': 'dengan', 'krn': 'karena', 'krna': 'karena',
        'bgt': 'banget', 'bgtu': 'begitu', 'bgmn': 'bagaimana', 'bbrp': 'beberapa',
        'ad': 'ada', 'tdk': 'tidak', 'jg': 'juga', 'jga': 'juga', 'tp': 'tapi',
        'tpi': 'tapi', 'sy': 'saya', 'ak': 'aku', 'km': 'kamu', 'mrk': 'mereka',
        'bkn': 'bukan', 'dr': 'dari', 'dri': 'dari', 'pd': 'pada', 'pda': 'pada',
        'trs': 'terus', 'lg': 'lagi', 'lgi': 'lagi', 'bs': 'bisa', 'hrs': 'harus',
        'hrus': 'harus', 'sdg': 'sedang', 'cuma': 'hanya', 'hny': 'hanya',
        'info': 'informasi', 'adm': 'admin', 'klh': 'kuliah', 'jurusan': 'program studi',
        'pswd': 'kata sandi', 'pasword': 'kata sandi', 'ortu': 'orang tua',
        'maba': 'mahasiswa baru', 'byr': 'bayar', 'gmn': 'bagaimana',
    }
    kamus = dict(fallback)
    if EXCEL_PATH.exists():
        df_k = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_KAMUS)
        for _, row in df_k.iterrows():
            chat = str(row.get('bentuk di chat', '')).strip().lower()
            baku = str(row.get('bentuk baku / arti', '')).strip().lower()
            if chat and baku and chat not in ISTILAH_KAMPUS_DILINDUNGI:
                if len(baku.split()) <= 3:
                    kamus[chat] = baku
    return kamus

def muat_kamus_gabungan(kamus_kampus):
    """
    Menggabungkan Kamus Alay ke kamus kampus:
    - Kamus kampus berprioritas lebih tinggi.
    - Menolak istilah kampus yang dilindungi.
    - Menolak pemetaan ambigu (>1 arti baku).
    """
    lexicon_path = ROOT_DIR / "data" / "sekunder" / "kamus_alay" / "colloquial-indonesian-lexicon.csv"
    slang_to_targets = defaultdict(set)
    with open(lexicon_path, 'r', encoding='utf-8', errors='replace') as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            s = str(r.get('slang', '')).strip().lower()
            f = str(r.get('formal', '')).strip().lower()
            if s and f and s != f:
                slang_to_targets[s].add(f)

    entri_ditambahkan = {}
    ambigu_count = 0
    terlindungi_count = 0
    kampus_override_count = 0

    for s, targets in slang_to_targets.items():
        if len(targets) > 1:
            ambigu_count += 1
            continue
        target_baku = list(targets)[0]
        if s in ISTILAH_KAMPUS_DILINDUNGI:
            terlindungi_count += 1
            continue
        if s in kamus_kampus:
            kampus_override_count += 1
            continue
        entri_ditambahkan[s] = target_baku

    kamus_gabung = dict(kamus_kampus)
    kamus_gabung.update(entri_ditambahkan)
    # Pastikan kamus kampus tidak tertimpa
    for k, v in kamus_kampus.items():
        kamus_gabung[k] = v

    info = {
        'total_alay_raw': len(slang_to_targets),
        'ambigu_dibuang': ambigu_count,
        'istilah_kampus_dilindungi': terlindungi_count,
        'dilewati_ada_di_kampus': kampus_override_count,
        'entri_baru_masuk': len(entri_ditambahkan),
        'total_gabungan': len(kamus_gabung),
        'kamus_alay_entries': entri_ditambahkan
    }
    return kamus_gabung, info

def bersihkan_teks_dengan_kamus(teks: str, kamus: dict) -> str:
    """Membersihkan dan menormalisasi teks menggunakan kamus tertentu."""
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

    # Tokenisasi sederhana per kata
    kata_kata = re.findall(r'\b\w+\b', teks_l)
    kata_normal = [kamus.get(k, k) for k in kata_kata]
    hasil = " ".join(kata_normal)
    return hasil[:ROUTER_MAX_KARAKTER]

def hitung_keamanan(y_true, y_pred):
    total_eskalasi = sum(1 for yt in y_true if yt in KELAS_ESKALASI)
    total_jawab = sum(1 for yt in y_true if yt not in KELAS_ESKALASI)
    kb = sum(1 for yt, yp in zip(y_true, y_pred) if yt in KELAS_ESKALASI and yp not in KELAS_ESKALASI)
    eb = sum(1 for yt, yp in zip(y_true, y_pred) if yt not in KELAS_ESKALASI and yp in KELAS_ESKALASI)
    pct_kb = (kb / total_eskalasi * 100) if total_eskalasi > 0 else 0.0
    pct_eb = (eb / total_jawab * 100) if total_jawab > 0 else 0.0
    return pct_kb, pct_eb

# ==============================================================
# MAIN TAHAP 2 RUNNER
# ==============================================================

def jalankan_tahap2():
    print("=" * 80)
    print("MEMULAI EKSEKUSI TAHAP 2: DATA SEKUNDER DARI INTERNET (HANYA BACA)")
    print("=" * 80)

    # ----------------------------------------------------------
    # 2b. Kamus Alay -> Normalisasi
    # ----------------------------------------------------------
    print("\n[Tahap 2b] Memproses dan Menggabungkan Kamus Alay...")
    kamus_kampus = muat_kamus_kampus()
    kamus_gabung, info_ka = muat_kamus_gabungan(kamus_kampus)

    print(f"  Total Kamus Kampus Awal       : {len(kamus_kampus)}")
    print(f"  Total Entri Unik Kamus Alay   : {info_ka['total_alay_raw']}")
    print(f"  Entri Ambigu Dibuang          : {info_ka['ambigu_dibuang']}")
    print(f"  Istilah Kampus Dilindungi     : {info_ka['istilah_kampus_dilindungi']}")
    print(f"  Dilewati (Sudah di Kampus)    : {info_ka['dilewati_ada_di_kampus']}")
    print(f"  Entri Baru Alay Dimasukkan    : {info_ka['entri_baru_masuk']}")
    print(f"  Total Kamus Gabungan          : {info_ka['total_gabungan']}")

    # 20 Contoh Acak Entri Alay Masuk
    sample_alay_keys = sorted(random.sample(sorted(info_ka['kamus_alay_entries'].keys()), 20))
    contoh_20_teks = []
    for idx_s, k in enumerate(sample_alay_keys, 1):
        contoh_20_teks.append(f"{idx_s:2d}. '{k}' -> '{info_ka['kamus_alay_entries'][k]}'")

    # ----------------------------------------------------------
    # 2c. Dampak Normalisasi pada Router (Cross-Validation)
    # ----------------------------------------------------------
    print("\n[Tahap 2c] Mengevaluasi Dampak Normalisasi pada Router Klasik (5 Lipatan)...")
    
    # Baca data latih
    df_sesi = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_SESI)
    df_pesan = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_PESAN)
    kolom_label = "topik_koreksi (isi manual)"
    df_sesi_valid = df_sesi[df_sesi[kolom_label].notna() & df_sesi[kolom_label].isin(SEMUA_KELAS)].copy()
    pesan_mhs = df_pesan[df_pesan['pengirim'] == 'Mahasiswa'].copy()

    # Kumpulkan teks mentah per sesi
    sesi_raw = []
    for _, row in df_sesi_valid.iterrows():
        sid = row['sesi_id']
        cid = row['chat_id']
        lbl = row[kolom_label]
        p_sesi = pesan_mhs[pesan_mhs['sesi_id'] == sid].sort_values('no')
        p_text = " ".join([str(p) for p in p_sesi['teks_anonim'].head(ROUTER_MAX_PESAN).dropna().tolist()])
        sesi_raw.append({'sesi_id': sid, 'chat_id': cid, 'label': lbl, 'raw_text': p_text})
    df_raw = pd.DataFrame(sesi_raw)

    # Muat lipatan.json
    berkas_lipatan = HASIL_DIR / "lipatan.json"
    with open(berkas_lipatan, "r", encoding="utf-8") as f:
        lipatan_json = json.load(f)

    # Filter data latih sesuai lipatan
    chat_uji_akhir = lipatan_json['chat_uji_akhir']
    df_latih_raw = df_raw[~df_raw['chat_id'].isin(chat_uji_akhir)].copy()

    # Siapkan dua versi teks
    df_latih_raw['teks_tanpa_alay'] = df_latih_raw['raw_text'].apply(lambda t: bersihkan_teks_dengan_kamus(t, kamus_kampus))
    df_latih_raw['teks_dengan_alay'] = df_latih_raw['raw_text'].apply(lambda t: bersihkan_teks_dengan_kamus(t, kamus_gabung))

    hasil_tanpa = []
    hasil_dengan = []

    for fold_info in lipatan_json['lipatan']:
        f_idx = fold_info['fold']
        val_sids = fold_info['sesi_val']
        
        train_df = df_latih_raw[~df_latih_raw['sesi_id'].isin(val_sids)]
        val_df = df_latih_raw[df_latih_raw['sesi_id'].isin(val_sids)]

        y_tr = train_df['label'].values
        y_va = val_df['label'].values

        # 1. Tanpa Kamus Alay
        pipe_tanpa = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
            ('clf', LogisticRegression(class_weight='balanced', random_state=SEED, max_iter=1000, C=1.0))
        ])
        pipe_tanpa.fit(train_df['teks_tanpa_alay'], y_tr)
        pred_tanpa = pipe_tanpa.predict(val_df['teks_tanpa_alay'])
        f1_t = f1_score(y_va, pred_tanpa, average='macro', zero_division=0)
        kb_t, eb_t = hitung_keamanan(y_va, pred_tanpa)
        hasil_tanpa.append({'fold': f_idx, 'macro_f1': f1_t, 'kb': kb_t, 'eb': eb_t})

        # 2. Dengan Kamus Alay
        pipe_dengan = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
            ('clf', LogisticRegression(class_weight='balanced', random_state=SEED, max_iter=1000, C=1.0))
        ])
        pipe_dengan.fit(train_df['teks_dengan_alay'], y_tr)
        pred_dengan = pipe_dengan.predict(val_df['teks_dengan_alay'])
        f1_d = f1_score(y_va, pred_dengan, average='macro', zero_division=0)
        kb_d, eb_d = hitung_keamanan(y_va, pred_dengan)
        hasil_dengan.append({'fold': f_idx, 'macro_f1': f1_d, 'kb': kb_d, 'eb': eb_d})

    f1_t_arr = [h['macro_f1'] for h in hasil_tanpa]
    f1_d_arr = [h['macro_f1'] for h in hasil_dengan]
    mean_t, std_t = np.mean(f1_t_arr), np.std(f1_t_arr)
    mean_d, std_d = np.mean(f1_d_arr), np.std(f1_d_arr)
    kb_t_mean, eb_t_mean = np.mean([h['kb'] for h in hasil_tanpa]), np.mean([h['eb'] for h in hasil_tanpa])
    kb_d_mean, eb_d_mean = np.mean([h['kb'] for h in hasil_dengan]), np.mean([h['eb'] for h in hasil_dengan])

    print(f"  Versi Tanpa Kamus Alay : Macro-F1 = {mean_t:.4f} (+/- {std_t:.4f}), KB = {kb_t_mean:.2f}%, EB = {eb_t_mean:.2f}%")
    print(f"  Versi Dengan Kamus Alay: Macro-F1 = {mean_d:.4f} (+/- {std_d:.4f}), KB = {kb_d_mean:.2f}%, EB = {eb_d_mean:.2f}%")

    # Tulis ke hasil/log_normalisasi.txt
    log_norm_path = HASIL_DIR / "log_normalisasi.txt"
    with open(log_norm_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("LOG EVALUASI TAHAP 2: NORMALISASI KAMUS ALAY & STIF-INDONESIA\n")
        f.write(f"Catatan: {CATATAN_LABEL}\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. INVENTARISASI & PENGGABUNGAN KAMUS ALAY\n")
        f.write(f"  - Total entri unik Kamus Alay      : {info_ka['total_alay_raw']}\n")
        f.write(f"  - Entri ambigu dibuang (>1 arti)   : {info_ka['ambigu_dibuang']}\n")
        f.write(f"  - Istilah kampus dilindungi        : {info_ka['istilah_kampus_dilindungi']}\n")
        f.write(f"  - Entri dilewati (ada di kampus)   : {info_ka['dilewati_ada_di_kampus']}\n")
        f.write(f"  - Entri BARU dari Kamus Alay       : {info_ka['entri_baru_masuk']}\n")
        f.write(f"  - Total entri kamus gabungan       : {info_ka['total_gabungan']}\n\n")
        f.write("20 CONTOH ACAK ENTRI KAMUS ALAY YANG DIMASUKKAN:\n")
        for c in contoh_20_teks:
            f.write(f"  {c}\n")
        f.write("\n" + "-" * 80 + "\n")
        f.write("2. DAMPAK NORMALISASI PADA ROUTER KLASIK (5 LIPATAN VALIDASI)\n")
        f.write(f"  Fold 0..4 Tanpa Alay  : {[round(x, 4) for x in f1_t_arr]}\n")
        f.write(f"  Fold 0..4 Dengan Alay : {[round(x, 4) for x in f1_d_arr]}\n\n")
        f.write(f"  Rata-rata Macro-F1 (Tanpa Alay)  : {mean_t:.4f} (+/- {std_t:.4f})\n")
        f.write(f"  Rata-rata Macro-F1 (Dengan Alay) : {mean_d:.4f} (+/- {std_d:.4f})\n")
        f.write(f"  Selisih Macro-F1                 : {mean_d - mean_t:+.4f}\n")
        f.write(f"  Kesalahan Berbahaya (Tanpa/Dengan): {kb_t_mean:.2f}% -> {kb_d_mean:.2f}%\n")
        f.write(f"  Eskalasi Berlebih (Tanpa/Dengan)  : {eb_t_mean:.2f}% -> {eb_d_mean:.2f}%\n")
        f.write("\n" + "=" * 80 + "\n")

    print(f"  Log normalisasi berhasil ditulis ke: {log_norm_path}")

    # Tambahkan baris baru ke hasil/hasil_router.csv jika belum ada
    csv_router_path = HASIL_DIR / "hasil_router.csv"
    df_router = pd.read_csv(csv_router_path) if csv_router_path.exists() else pd.DataFrame()
    
    # Hapus baris jika sudah ada nama model yang sama (re-run safety)
    if not df_router.empty and 'model' in df_router.columns:
        df_router = df_router[~df_router['model'].isin([
            "Router Klasik (Tanpa Kamus Alay)",
            "Router Klasik (Dengan Kamus Alay)"
        ])]

    baris_baru = pd.DataFrame([
        {
            'model': 'Router Klasik (Tanpa Kamus Alay)',
            'macro_f1_mean': round(mean_t, 4),
            'macro_f1_std': round(std_t, 4),
            'kesalahan_berbahaya_persen': round(kb_t_mean, 2),
            'eskalasi_berlebih_persen': round(eb_t_mean, 2),
            'catatan': CATATAN_LABEL
        },
        {
            'model': 'Router Klasik (Dengan Kamus Alay)',
            'macro_f1_mean': round(mean_d, 4),
            'macro_f1_std': round(std_d, 4),
            'kesalahan_berbahaya_persen': round(kb_d_mean, 2),
            'eskalasi_berlebih_persen': round(eb_d_mean, 2),
            'catatan': f"Selisih F1={mean_d - mean_t:+.4f}; {CATATAN_LABEL}"
        }
    ])
    df_router_updated = pd.concat([df_router, baris_baru], ignore_index=True)
    df_router_updated.to_csv(csv_router_path, index=False)
    print(f"  Baris baru berhasil ditambahkan ke: {csv_router_path}")

    # ----------------------------------------------------------
    # 2d. STIF-Indonesia (Ukur Cakupan Normalisasi)
    # ----------------------------------------------------------
    print("\n[Tahap 2d] Menguji Cakupan Normalisasi pada 2000 Kalimat STIF-Indonesia...")
    stif_dir = ROOT_DIR / "data" / "sekunder" / "stif_indonesia"
    inf_files = list(stif_dir.rglob("*.inf"))
    all_inf_sentences = []
    for fp in inf_files:
        lines = fp.read_text(encoding='utf-8', errors='replace').splitlines()
        for ln in lines:
            ln_s = ln.strip()
            if ln_s:
                all_inf_sentences.append(ln_s)

    print(f"  Total kalimat informal STIF ditemukan: {len(all_inf_sentences)}")
    sample_2000 = random.sample(all_inf_sentences, min(2000, len(all_inf_sentences)))

    total_words = 0
    words_changed = 0
    sentences_changed = 0
    contoh_pairs = []

    for s_orig in sample_2000:
        words_orig = re.findall(r'\b\w+\b', s_orig.lower())
        total_words += len(words_orig)

        # Normalisasi kata demi kata menggunakan kamus gabungan
        s_norm_words = [kamus_gabung.get(w, w) for w in words_orig]
        diff_w = sum(1 for wo, wn in zip(words_orig, s_norm_words) if wo != wn)
        words_changed += diff_w

        s_norm_str = " ".join(s_norm_words)
        s_orig_str = " ".join(words_orig)
        if s_orig_str != s_norm_str:
            sentences_changed += 1
            if len(contoh_pairs) < 30:
                contoh_pairs.append((s_orig, s_norm_str))

    pct_words_changed = (words_changed / total_words * 100) if total_words > 0 else 0.0
    pct_sentences_changed = (sentences_changed / len(sample_2000) * 100) if sample_2000 else 0.0

    print(f"  Total kata diuji              : {total_words}")
    print(f"  Kata informal dinormalisasi   : {words_changed} ({pct_words_changed:.2f}%)")
    print(f"  Kalimat yang berubah bentuk   : {sentences_changed}/{len(sample_2000)} ({pct_sentences_changed:.2f}%)")

    # Tambahkan hasil STIF ke log_normalisasi.txt
    with open(log_norm_path, "a", encoding="utf-8") as f:
        f.write("\n3. CAKUPAN NORMALISASI PADA STIF-INDONESIA (2000 KALIMAT ACAK)\n")
        f.write(f"  - Total kata informal dianalisis : {total_words}\n")
        f.write(f"  - Kata yang berhasil dinormalisasi: {words_changed} ({pct_words_changed:.2f}%)\n")
        f.write(f"  - Kalimat yang mengalami perubahan: {sentences_changed}/{len(sample_2000)} ({pct_sentences_changed:.2f}%)\n\n")
        f.write("30 CONTOH KALIMAT SEBELUM & SESUDAH NORMALISASI:\n")
        for idx_p, (bef, aft) in enumerate(contoh_pairs, 1):
            f.write(f"  [{idx_p:02d}] Sebelum : {bef}\n")
            f.write(f"       Sesudah : {aft}\n")
        f.write("\n" + "=" * 80 + "\n")

    # ----------------------------------------------------------
    # 2e. IDK-MRC -> Uji Penolakan di Luar Domain (Sistem B)
    # ----------------------------------------------------------
    print("\n[Tahap 2e] Menguji Ketahanan Penolakan di Luar Domain (50 Pertanyaan IDK-MRC)...")
    idk_dir = ROOT_DIR / "data" / "sekunder" / "idk_mrc" / "dataset" / "idk_mrc"
    unans_questions = []

    for split in ["valid.json", "test.json"]:
        sp_path = idk_dir / split
        if sp_path.exists():
            with open(sp_path, "r", encoding="utf-8") as f:
                d_list = json.load(f)
                for item in d_list:
                    for qa in item.get("qas", []):
                        if qa.get("is_impossible", False):
                            q_text = qa.get("question", "").strip()
                            if q_text:
                                unans_questions.append({
                                    'id': qa.get('id', ''),
                                    'split': split,
                                    'question': q_text
                                })

    print(f"  Total pertanyaan unanswerable IDK-MRC ditemukan: {len(unans_questions)}")
    # Sample 50 pertanyaan dengan seed 42
    sample_50_idk = random.sample(unans_questions, min(50, len(unans_questions)))

    hasil_idk = []
    dialihkan_ke_admin = 0
    tidak_dialihkan = []

    for i, item in enumerate(sample_50_idk, 1):
        q = item['question']
        res = bot.jawab(q, sistem="B")
        kep = res['keputusan']
        
        is_eskalasi = (kep == "teruskan_ke_staf")
        if is_eskalasi:
            dialihkan_ke_admin += 1
        else:
            tidak_dialihkan.append({
                'no': i,
                'id': item['id'],
                'question': q,
                'keputusan': kep,
                'alasan': res['alasan'],
                'jarak': res['jarak_terbaik'],
                'overlap': res.get('tumpang_tindih_kata'),
                'jawaban': res['jawaban']
            })

        hasil_idk.append({
            'no': i,
            'id': item['id'],
            'question': q,
            'keputusan': kep,
            'alasan': res['alasan'],
            'jarak': res['jarak_terbaik'],
            'overlap': res.get('tumpang_tindih_kata'),
            'sumber': res.get('sumber'),
            'jawaban': res['jawaban']
        })

    pct_dialihkan = (dialihkan_ke_admin / len(sample_50_idk) * 100) if sample_50_idk else 0.0
    print(f"  Total Pertanyaan Uji          : {len(sample_50_idk)}")
    print(f"  Dialihkan ke Admin (Eskalasi) : {dialihkan_ke_admin} ({pct_dialihkan:.1f}%)")
    print(f"  Tidak Dialihkan (Dijawab)     : {len(tidak_dialihkan)} ({100.0 - pct_dialihkan:.1f}%)")

    # Tulis ke hasil/log_penolakan_idkmrc.txt
    log_idk_path = HASIL_DIR / "log_penolakan_idkmrc.txt"
    with open(log_idk_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("LOG EVALUASI TAHAP 2e: UJI KETAHANAN PENOLAKAN DI LUAR DOMAIN (IDK-MRC)\n")
        f.write(f"Catatan: {CATATAN_LABEL}\n")
        f.write("CATATAN PENTING METODOLOGIS:\n")
        f.write("Pertanyaan ini adalah pertanyaan unanswerable dari dataset umum IDK-MRC (EMNLP 2022),\n")
        f.write("berada di luar domain layanan administrasi kampus, sehingga secara alamiah memiliki jarak\n")
        f.write("semantik yang jauh dari basis dokumen kampus. Hasil penolakan tinggi menunjukkan bahwa\n")
        f.write("gerbang leksikal & ambang jarak berhasil menolak kueri luar domain, namun TIDAK mewakili\n")
        f.write("pertanyaan jebakan dalam domain kampus yang semantik kata-katanya mirip rujukan resmi.\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Pertanyaan Diuji     : {len(sample_50_idk)}\n")
        f.write(f"Dialihkan ke Admin (Aman)  : {dialihkan_ke_admin} ({pct_dialihkan:.1f}%)\n")
        f.write(f"Tidak Dialihkan (Terjawab) : {len(tidak_dialihkan)} ({100.0 - pct_dialihkan:.1f}%)\n\n")
        
        if tidak_dialihkan:
            f.write("DAFTAR PERTANYAAN YANG TIDAK DIALIHKAN (DIJAWAB OTOMATIS):\n")
            for td in tidak_dialihkan:
                f.write(f"  No: {td['no']} | ID: {td['id']}\n")
                f.write(f"  Pertanyaan: {td['question']}\n")
                f.write(f"  Keputusan : {td['keputusan']} | Jarak: {td['jarak']} | Overlap: {td['overlap']}\n")
                f.write(f"  Alasan    : {td['alasan']}\n")
                f.write(f"  Jawaban   : {td['jawaban'][:150]}...\n")
                f.write("-" * 60 + "\n")
        else:
            f.write("SEMUA 50 PERTANYAAN (100%) BERHASIL DIALIHKAN KE ADMIN / DITOLAK AMAN.\n")
            f.write("-" * 80 + "\n\n")

        f.write("RINCIAN 50 PERTANYAAN UJI IDK-MRC:\n")
        f.write("-" * 80 + "\n")
        for h in hasil_idk:
            j_str = f"{h['jarak']:.3f}" if h['jarak'] is not None else "-"
            o_str = f"{h['overlap']:.2f}" if h['overlap'] is not None else "-"
            f.write(f"[{h['no']:02d}] {h['question']}\n")
            f.write(f"     Keputusan: {h['keputusan']} | Jarak: {j_str} | Overlap: {o_str}\n")
            f.write(f"     Alasan   : {h['alasan']}\n\n")

    print(f"  Log penolakan IDK-MRC berhasil ditulis ke: {log_idk_path}")

    # ----------------------------------------------------------
    # 2f. MASSIVE (Inventarisasi)
    # ----------------------------------------------------------
    print("\n[Tahap 2f] Inventarisasi Repositori MASSIVE...")
    massive_dir = ROOT_DIR / "data" / "sekunder" / "massive_kode"
    has_massive_dir = massive_dir.exists()
    massive_data_files = list(massive_dir.rglob("*.jsonl")) if has_massive_dir else []
    print(f"  Folder massive_kode ada    : {has_massive_dir}")
    print(f"  Berkas data ditemukan      : {len(massive_data_files)} berkas data (data belum diunduh, hanya ada repositori kode)")
    print(f"  Dokumentasi NLU MASSIVE    : Subset Bahasa Indonesia (id-ID), 60 intent, 55 slot, 18 domain.")
    print(f"  Status                     : Hanya inventarisasi, TIDAK dipakai melatih model.")

    print("\n" + "=" * 80)
    print("TAHAP 2 SELESAI SELURUHNYA!")
    print("=" * 80)

if __name__ == "__main__":
    jalankan_tahap2()

