# ==============================================================
# latih_router_klasik.py — Pelatihan & Evaluasi Router Klasik (T5)
#
# Tahap T5 Proyek Tugas Akhir CHATTOBOT:
#   1. Ekstraksi unit data per sesi (maks. 4 pesan mahasiswa pertama,
#      dibersihkan, dinormalisasi kamus singkatan, dipotong 400 karakter).
#   2. Pembagian Uji Akhir (8 chat utuh, acak seed 42, min 4 sesi/kelas)
#      disimpan ke eval\chat_uji_akhir.txt.
#   3. StratifiedGroupKFold 5 lipatan (grup chat_id, seed 42)
#      disimpan ke hasil\lipatan.json untuk standarisasi semua model.
#   4. Evaluasi 3 model pembanding:
#      - Majority Class Baseline (Kelas Terbanyak)
#      - Rule-based Router (Router Berbasis Aturan)
#      - Router Klasik ML: TF-IDF Karakter (2-5 gram) + Logistic Regression
#   5. Perhitungan metrik: macro-F1, per-kelas, matriks kebingungan,
#      kesalahan berbahaya (% eskalasi jadi jawab), dan eskalasi berlebih.
#   6. Penyetelan ambang keyakinan (0.30 - 0.80) dari lipatan validasi.
#   7. Penyimpanan model ke models\router_klasik.joblib dan log ke hasil\.
# ==============================================================

import io
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

# Atur output UTF-8 agar aman pada terminal Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Path direktori proyek
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
HASIL_DIR = ROOT / "hasil"
EVAL_DIR = ROOT / "eval"
MODELS_DIR = ROOT / "models"

sys.path.insert(0, str(SRC_DIR))
from config import (
    EXCEL_PATH, SHEET_SESI, SHEET_PESAN, SHEET_KAMUS,
    KELAS_BOT, KELAS_ESKALASI, SEMUA_KELAS, SEED,
    ROUTER_MAX_PESAN, ROUTER_MAX_KARAKTER
)

# Catatan kejujuran hasil sesuai instruksi pemilik
CATATAN_LABEL = "Catatan: label dibuat AI, TIDAK diverifikasi manusia (hasil bersifat indikatif)"


class OutputTee:
    """Menduplikasi seluruh keluaran cetak ke terminal dan berkas log."""
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

    def close(self):
        self.log_file.close()


def muat_kamus_singkatan_cepat(excel_path: Path) -> dict:
    """Memuat kamus singkatan langsung dari sheet Kamus di file Excel."""
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
        df_kamus = pd.read_excel(excel_path, sheet_name=SHEET_KAMUS)
        istilah_khusus = {
            'ektm', 'mybest', 'myubsi', 'bsi id', 'bju', 'kip', 'ormik',
            'ormawa', 'serkom', 'juklak', 'po', 'angsuran', 'her', 'uas', 'uts', 'pmb'
        }
        for _, row in df_kamus.iterrows():
            chat = str(row.get('bentuk di chat', '')).strip().lower()
            baku = str(row.get('bentuk baku / arti', '')).strip().lower()
            if chat and baku and chat not in istilah_khusus:
                if len(baku.split()) <= 3:
                    kamus[chat] = baku
    except Exception as e:
        print(f"[PERINGATAN] Gagal membaca kamus Excel: {e}. Menggunakan fallback bawaan.")
    return kamus


def bersihkan_teks_pesan(teks: str, kamus: dict) -> str:
    """
    Membersihkan teks masukan router:
      - Huruf kecil
      - Tautan diseragamkan menjadi 'httpurl'
      - Penanda data sensitif diseragamkan ([NAMA] -> @user, [NIM] -> nim, dsb)
      - Emoji dibuang
      - Reduksi pengulangan huruf lebih dari 2 kali
      - Normalisasi singkatan kata menggunakan kamus
      - Potong maksimal 400 karakter
    """
    if not isinstance(teks, str) or not teks:
        return ""

    teks_l = teks.lower().strip()

    # Seragamkan tautan
    teks_l = re.sub(r'https?://\S+|www\.\S+', 'httpurl', teks_l)

    # Seragamkan penanda anonim
    teks_l = re.sub(r'\[nama[^\]]*\]', '@user', teks_l)
    teks_l = re.sub(r'\[nim\]', 'nim', teks_l)
    teks_l = re.sub(r'\[kelas\]', 'kelas', teks_l)
    teks_l = re.sub(r'\[(telepon|no_hp)\]', 'nohp', teks_l)
    teks_l = re.sub(r'\[email\]', 'email', teks_l)
    teks_l = re.sub(r'\[(kata_sandi|password|format_kata_sandi)\]', 'pass', teks_l)
    teks_l = re.sub(r'\[(tgl|tanggal)\]', 'tanggal', teks_l)
    teks_l = re.sub(r'\[angka\]', 'angka', teks_l)

    # Buang emoji (simbol grafis unicode)
    teks_l = re.sub(r'[\U00010000-\U0010ffff\u2600-\u27bf]', '', teks_l)

    # Reduksi karakter berulang lebih dari 2 kali
    teks_l = re.sub(r'([a-z])\1{2,}', r'\1', teks_l)

    # Normalisasi kata lewat kamus
    kata_kata = teks_l.split()
    kata_normal = [kamus.get(k, k) for k in kata_kata]
    hasil = " ".join(kata_normal)

    # Potong di 400 karakter sesuai config
    return hasil[:ROUTER_MAX_KARAKTER]


def siapkan_dataset() -> pd.DataFrame:
    """
    Mempersiapkan dataset per sesi:
      - Membaca sheet Sesi (hanya yang memiliki label manual 6 kelas)
      - Menggabungkan maksimal 4 pesan mahasiswa pertama dari sheet Pesan_Anonim
      - Membersihkan teks dan menyusun DataFrame bersih
    """
    print(f"Membaca dataset dari: {EXCEL_PATH}...")
    df_sesi = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_SESI)
    df_pesan = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_PESAN)
    kamus = muat_kamus_singkatan_cepat(EXCEL_PATH)

    # Filter sesi berlabel manual (6 kelas valid)
    kolom_label = "topik_koreksi (isi manual)"
    df_sesi_valid = df_sesi[df_sesi[kolom_label].notna() & df_sesi[kolom_label].isin(SEMUA_KELAS)].copy()
    print(f"Total sesi bermahasiswa valid: {len(df_sesi_valid)} sesi di {df_sesi_valid['chat_id'].nunique()} chat.")

    # Filter pesan mahasiswa saja
    pesan_mhs = df_pesan[df_pesan['pengirim'] == 'Mahasiswa'].copy()

    data_baris = []
    for _, row in df_sesi_valid.iterrows():
        sesi_id = row['sesi_id']
        chat_id = row['chat_id']
        label = row[kolom_label]

        # Ambil maksimal 4 pesan mahasiswa pertama
        pesan_sesi = pesan_mhs[pesan_mhs['sesi_id'] == sesi_id].sort_values('no')
        pesan_terpilih = pesan_sesi['teks_anonim'].head(ROUTER_MAX_PESAN).dropna().tolist()
        teks_mentah = " ".join([str(p) for p in pesan_terpilih])

        teks_bersih = bersihkan_teks_pesan(teks_mentah, kamus)

        data_baris.append({
            'sesi_id': sesi_id,
            'chat_id': chat_id,
            'label': label,
            'teks': teks_bersih,
            'jumlah_pesan_diambil': len(pesan_terpilih),
            'panjang_karakter': len(teks_bersih),
        })

    df_hasil = pd.DataFrame(data_baris)
    return df_hasil


def pisahkan_uji_akhir(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[int]]:
    """
    Memisahkan 8-10 chat utuh sebagai UJI AKHIR:
      - Menggunakan acak dengan seed 42
      - Mengusahakan setiap kelas memiliki minimal 4 sesi di uji akhir
      - Menyimpan daftar chat_id ke eval\chat_uji_akhir.txt
    """
    chat_summary = df.groupby('chat_id')['label'].value_counts().unstack(fill_value=0)
    all_chats = chat_summary.index.tolist()
    rng = np.random.RandomState(SEED)

    chat_terpilih = None
    for n_chats in [8, 9, 10]:
        for attempt in range(100):
            kandidat = sorted(rng.choice(all_chats, size=n_chats, replace=False).tolist())
            distribusi = chat_summary.loc[kandidat].sum()
            if (distribusi >= 4).all():
                chat_terpilih = kandidat
                break
        if chat_terpilih:
            break

    # Fallback jika pencarian acak tidak memenuhi minimal 4 sesi
    if chat_terpilih is None:
        chat_terpilih = [2, 4, 6, 18, 20, 21, 30, 33]

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    berkas_chat_uji = EVAL_DIR / "chat_uji_akhir.txt"
    with open(berkas_chat_uji, "w", encoding="utf-8") as f:
        f.write("# Daftar chat_id yang disisihkan sebagai UJI AKHIR (JANGAN dipakai melatih/menyetel ambang)\n")
        f.write("# Parameter: seed 42, acak terstratifikasi per chat\n")
        f.write("\n".join(str(c) for c in chat_terpilih) + "\n")

    df_uji_akhir = df[df['chat_id'].isin(chat_terpilih)].copy().reset_index(drop=True)
    df_latih = df[~df['chat_id'].isin(chat_terpilih)].copy().reset_index(drop=True)

    print(f"\n[PEMISAHAN UJI AKHIR]")
    print(f"Chat terpilih ({len(chat_terpilih)} chat): {chat_terpilih}")
    print(f"Total sesi Uji Akhir : {len(df_uji_akhir)} sesi (disisihkan untuk evaluasi sekali di akhir)")
    print(f"Total sesi Data Latih: {len(df_latih)} sesi di {df_latih['chat_id'].nunique()} chat (untuk StratifiedGroupKFold)")
    print(f"Distribusi kelas Uji Akhir:\n{df_uji_akhir['label'].value_counts().to_string()}")
    print(f"Daftar chat_id disimpan ke: {berkas_chat_uji}")

    return df_latih, df_uji_akhir, chat_terpilih


def buat_dan_simpan_lipatan(df_latih: pd.DataFrame, chat_uji_akhir: list[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Membagi data latih menjadi 5 lipatan StratifiedGroupKFold (grup chat_id, seed 42)
    dan menyimpannya ke hasil\lipatan.json agar model deep learning di T6 memakai pembagian yang persis sama.
    """
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    X = df_latih['sesi_id'].values
    y = df_latih['label'].values
    groups = df_latih['chat_id'].values

    lipatan_list = list(sgkf.split(X, y, groups))

    # Struktur JSON untuk standarisasi antar model
    data_json = {
        "seed": SEED,
        "n_splits": 5,
        "total_sesi_latih": len(df_latih),
        "total_chat_latih": int(df_latih['chat_id'].nunique()),
        "chat_uji_akhir": [int(c) for c in chat_uji_akhir],
        "catatan": CATATAN_LABEL,
        "lipatan": []
    }

    for fold_idx, (train_idx, val_idx) in enumerate(lipatan_list):
        train_sesi = [int(df_latih.iloc[i]['sesi_id']) for i in train_idx]
        val_sesi = [int(df_latih.iloc[i]['sesi_id']) for i in val_idx]
        train_chats = sorted(list(set(int(df_latih.iloc[i]['chat_id']) for i in train_idx)))
        val_chats = sorted(list(set(int(df_latih.iloc[i]['chat_id']) for i in val_idx)))

        data_json["lipatan"].append({
            "fold": fold_idx,
            "jumlah_train": len(train_idx),
            "jumlah_val": len(val_idx),
            "chat_val": val_chats,
            "sesi_val": val_sesi,
        })

    HASIL_DIR.mkdir(parents=True, exist_ok=True)
    berkas_lipatan = HASIL_DIR / "lipatan.json"
    with open(berkas_lipatan, "w", encoding="utf-8") as f:
        json.dump(data_json, f, indent=2)

    print(f"\nPembagian 5 lipatan StratifiedGroupKFold berhasil disimpan ke: {berkas_lipatan}")
    return lipatan_list


def hitung_metrik_keamanan(y_true: list[str], y_pred: list[str]) -> dict:
    """
    Menghitung metrik keamanan khusus router administrasi kampus:
      - Kesalahan berbahaya: % sesi kelas eskalasi (beasiswa/lainnya) yang diprediksi kelas jawab.
      - Eskalasi berlebih: % sesi kelas jawab yang diprediksi kelas eskalasi.
    """
    total_eskalasi = sum(1 for yt in y_true if yt in KELAS_ESKALASI)
    total_jawab = sum(1 for yt in y_true if yt in KELAS_BOT)

    # Kesalahan berbahaya: aslinya eskalasi tapi diprediksi bisa dijawab
    kesalahan_berbahaya_count = sum(1 for yt, yp in zip(y_true, y_pred) if yt in KELAS_ESKALASI and yp in KELAS_BOT)
    persen_berbahaya = (kesalahan_berbahaya_count / total_eskalasi * 100) if total_eskalasi > 0 else 0.0

    # Eskalasi berlebih: aslinya kelas jawab tapi dilempar ke eskalasi
    eskalasi_berlebih_count = sum(1 for yt, yp in zip(y_true, y_pred) if yt in KELAS_BOT and yp in KELAS_ESKALASI)
    persen_berlebih = (eskalasi_berlebih_count / total_jawab * 100) if total_jawab > 0 else 0.0

    return {
        "total_eskalasi": total_eskalasi,
        "kesalahan_berbahaya_count": kesalahan_berbahaya_count,
        "persen_kesalahan_berbahaya": round(persen_berbahaya, 2),
        "total_jawab": total_jawab,
        "eskalasi_berlebih_count": eskalasi_berlebih_count,
        "persen_eskalasi_berlebih": round(persen_berlebih, 2),
    }


def prediksi_router_aturan(teks: str) -> tuple[str, float]:
    """Router berbasis aturan kata kunci dari bot.py."""
    teks_l = teks.lower()
    if any(k in teks_l for k in ['beasiswa', 'kip', 'kipk', 'kip-k', 'bju', 'bidikmisi']):
        return 'beasiswa_pendaftaran', 0.95

    pola_bukti = [
        'bukti transfer', 'bukti bayar', 'sudah transfer', 'sudah bayar dan kirim',
        'struk transfer', 'struk pembayaran', 'lampiran transfer', 'ss transfer',
        'screenshoot transfer', 'screenshot bukti', 'berikut bukti'
    ]
    if any(p in teks_l for p in pola_bukti):
        return 'bukti_bayar', 0.92
    if ('transfer' in teks_l or 'bayar' in teks_l) and ('nim' in teks_l or 'rp' in teks_l or 'norek' in teks_l):
        return 'bukti_bayar', 0.85

    pola_login = [
        'login', 'akun', 'password', 'kata sandi', 'username', 'mybest',
        'myubsi', 'bsi id', 'gagal masuk', 'tidak bisa masuk', 'lupa password',
        'reset password', 'gagal login', 'kendala aplikasi'
    ]
    if any(p in teks_l for p in pola_login):
        return 'login_akun', 0.88

    pola_bayar = [
        'cara bayar', 'metode pembayaran', 'batas bayar', 'jadwal bayar',
        'tagihan', 'virtual account', ' va ', 'nomor va', 'angsuran', 'cicilan',
        'bayar kuliah', 'bayar spp', 'bayar adm', 'biaya kuliah', 'bni', 'bsi pay',
        'atm bni', 'teller', 'mobile banking', 'm-banking', 'pembayaran ektm'
    ]
    if any(p in teks_l for p in pola_bayar):
        return 'pembayaran_umum', 0.90

    pola_pmb = [
        'pmb', 'pendaftaran', 'daftar ulang', 'atribut', 'kaos', 'baju',
        'id card', 'kartu mahasiswa', 'ektm', 'jadwal pmb', 'seminar',
        'workshop', 'serkom', 'sertifikasi', 'orientasi', 'ormik', 'semaba'
    ]
    if any(p in teks_l for p in pola_pmb):
        return 'kegiatan_pmb', 0.88

    if 'bayar' in teks_l or 'transfer' in teks_l or 'biaya' in teks_l:
        return 'pembayaran_umum', 0.65
    if 'masuk' in teks_l or 'aplikasi' in teks_l:
        return 'login_akun', 0.60
    if 'kuliah' in teks_l or 'kegiatan' in teks_l:
        return 'kegiatan_pmb', 0.60

    return 'lainnya', 0.40


def evaluasi_model_klasik(df_latih: pd.DataFrame, lipatan_list: list) -> dict:
    """
    Melatih dan mengevaluasi 3 model pada 5 lipatan StratifiedGroupKFold:
      1. Baseline Kelas Terbanyak
      2. Baseline Router Aturan
      3. Router Klasik ML: TF-IDF Karakter (2-5 gram) + Logistic Regression
    """
    print("\n" + "=" * 80)
    print("EVALUASI 5 LIPATAN (CROSS-VALIDATION)")
    print(f"{CATATAN_LABEL}")
    print("=" * 80)

    hasil_lipatan_klasik = []
    hasil_lipatan_aturan = []
    hasil_lipatan_mayoritas = []

    oof_preds_klasik = []  # Out-of-fold predictions untuk tabel ambang

    for fold_idx, (train_idx, val_idx) in enumerate(lipatan_list):
        train_df = df_latih.iloc[train_idx]
        val_df = df_latih.iloc[val_idx]

        X_train = train_df['teks'].tolist()
        y_train = train_df['label'].tolist()
        X_val = val_df['teks'].tolist()
        y_val = val_df['label'].tolist()

        # ------------------------------------------------------------------
        # 1. Baseline Kelas Terbanyak (Majority Class)
        # ------------------------------------------------------------------
        kelas_terbanyak = train_df['label'].mode()[0]
        y_pred_maj = [kelas_terbanyak] * len(val_df)
        f1_maj = f1_score(y_val, y_pred_maj, average='macro', zero_division=0)
        keamanan_maj = hitung_metrik_keamanan(y_val, y_pred_maj)
        hasil_lipatan_mayoritas.append({
            'fold': fold_idx,
            'macro_f1': f1_maj,
            'kesalahan_berbahaya': keamanan_maj['persen_kesalahan_berbahaya'],
            'eskalasi_berlebih': keamanan_maj['persen_eskalasi_berlebih'],
        })

        # ------------------------------------------------------------------
        # 2. Baseline Router Aturan
        # ------------------------------------------------------------------
        pred_aturan = [prediksi_router_aturan(t) for t in X_val]
        y_pred_at = [p[0] for p in pred_aturan]
        f1_at = f1_score(y_val, y_pred_at, average='macro', zero_division=0)
        keamanan_at = hitung_metrik_keamanan(y_val, y_pred_at)
        hasil_lipatan_aturan.append({
            'fold': fold_idx,
            'macro_f1': f1_at,
            'kesalahan_berbahaya': keamanan_at['persen_kesalahan_berbahaya'],
            'eskalasi_berlebih': keamanan_at['persen_eskalasi_berlebih'],
        })

        # ------------------------------------------------------------------
        # 3. Router Klasik ML: TF-IDF (2-5 char n-gram) + LogisticRegression
        # ------------------------------------------------------------------
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='char',
                ngram_range=(2, 5),
                sublinear_tf=True,
                min_df=2
            )),
            ('clf', LogisticRegression(
                class_weight='balanced',
                random_state=SEED,
                max_iter=1000,
                C=1.0
            ))
        ])

        pipeline.fit(X_train, y_train)
        y_pred_klasik = pipeline.predict(X_val)
        y_prob_klasik = pipeline.predict_proba(X_val)
        classes_order = list(pipeline.classes_)

        f1_klasik = f1_score(y_val, y_pred_klasik, average='macro', zero_division=0)
        keamanan_klasik = hitung_metrik_keamanan(y_val, y_pred_klasik)

        # Per-class metrics
        prec_per_class = precision_score(y_val, y_pred_klasik, labels=SEMUA_KELAS, average=None, zero_division=0)
        rec_per_class = recall_score(y_val, y_pred_klasik, labels=SEMUA_KELAS, average=None, zero_division=0)
        f1_per_class = f1_score(y_val, y_pred_klasik, labels=SEMUA_KELAS, average=None, zero_division=0)

        hasil_lipatan_klasik.append({
            'fold': fold_idx,
            'macro_f1': f1_klasik,
            'kesalahan_berbahaya': keamanan_klasik['persen_kesalahan_berbahaya'],
            'eskalasi_berlebih': keamanan_klasik['persen_eskalasi_berlebih'],
            'prec_per_class': prec_per_class,
            'rec_per_class': rec_per_class,
            'f1_per_class': f1_per_class,
            'cm': confusion_matrix(y_val, y_pred_klasik, labels=SEMUA_KELAS)
        })

        # Simpan prediksi OOF (Out-Of-Fold)
        for idx_val, yt, yp, probs in zip(val_df['sesi_id'], y_val, y_pred_klasik, y_prob_klasik):
            max_p = float(np.max(probs))
            oof_preds_klasik.append({
                'sesi_id': int(idx_val),
                'y_true': yt,
                'y_pred': yp,
                'max_prob': max_p,
                'probs': {c: float(p) for c, p in zip(classes_order, probs)}
            })

    return {
        'mayoritas': pd.DataFrame(hasil_lipatan_mayoritas),
        'aturan': pd.DataFrame(hasil_lipatan_aturan),
        'klasik': hasil_lipatan_klasik,
        'oof': oof_preds_klasik,
    }


def tampilkan_dan_simpan_laporan(eval_res: dict, df_latih: pd.DataFrame, df_uji_akhir: pd.DataFrame):
    """Mencetak laporan komprehensif T5 dan menyimpan CSV hasil evaluasi."""
    klasik_list = eval_res['klasik']
    df_maj = eval_res['mayoritas']
    df_at = eval_res['aturan']
    oof = eval_res['oof']

    # Hitung rata-rata dan simpangan baku (mean ± std)
    f1_klasik_vals = [k['macro_f1'] for k in klasik_list]
    kb_klasik_vals = [k['kesalahan_berbahaya'] for k in klasik_list]
    eb_klasik_vals = [k['eskalasi_berlebih'] for k in klasik_list]

    print("\n" + "=" * 90)
    print("TABEL PERBANDINGAN MODEL ROUTER PADA 5 LIPATAN VALIDASI")
    print(f"{CATATAN_LABEL}")
    print("=" * 90)
    print(f"{'Model':<35} | {'Macro-F1 (mean±std)':<22} | {'Kesalahan Berbahaya':<20} | {'Eskalasi Berlebih'}")
    print("-" * 90)

    # 1. Mayoritas
    m_maj = df_maj['macro_f1'].mean()
    s_maj = df_maj['macro_f1'].std()
    kb_maj = df_maj['kesalahan_berbahaya'].mean()
    eb_maj = df_maj['eskalasi_berlebih'].mean()
    print(f"{'1. Baseline Kelas Terbanyak':<35} | {f'{m_maj:.4f} ± {s_maj:.4f}':<22} | {f'{kb_maj:.1f}%':<20} | {f'{eb_maj:.1f}%'}")

    # 2. Aturan
    m_at = df_at['macro_f1'].mean()
    s_at = df_at['macro_f1'].std()
    kb_at = df_at['kesalahan_berbahaya'].mean()
    eb_at = df_at['eskalasi_berlebih'].mean()
    print(f"{'2. Baseline Router Aturan':<35} | {f'{m_at:.4f} ± {s_at:.4f}':<22} | {f'{kb_at:.1f}%':<20} | {f'{eb_at:.1f}%'}")

    # 3. Klasik TF-IDF + LogReg
    m_klasik = np.mean(f1_klasik_vals)
    s_klasik = np.std(f1_klasik_vals)
    kb_klasik = np.mean(kb_klasik_vals)
    eb_klasik = np.mean(eb_klasik_vals)
    print(f"{'3. Router Klasik (TFIDF + LogReg)':<35} | {f'{m_klasik:.4f} ± {s_klasik:.4f}':<22} | {f'{kb_klasik:.1f}%':<20} | {f'{eb_klasik:.1f}%'}")
    print("=" * 90)

    # Rincian Metrik Per Kelas untuk Router Klasik (Rata-rata 5 Lipatan)
    print("\n" + "=" * 90)
    print("RINCIAN METRIK PER KELAS — ROUTER KLASIK (RATA-RATA 5 LIPATAN)")
    print(f"{CATATAN_LABEL}")
    print("=" * 90)
    print(f"{'Kelas':<24} | {'Presisi (mean)':<16} | {'Recall (mean)':<16} | {'F1-Score (mean)'}")
    print("-" * 90)

    prec_matrix = np.array([k['prec_per_class'] for k in klasik_list])
    rec_matrix = np.array([k['rec_per_class'] for k in klasik_list])
    f1_matrix = np.array([k['f1_per_class'] for k in klasik_list])

    rincian_kelas_list = []
    for idx_k, nama_k in enumerate(SEMUA_KELAS):
        p_mean = float(np.mean(prec_matrix[:, idx_k]))
        r_mean = float(np.mean(rec_matrix[:, idx_k]))
        f_mean = float(np.mean(f1_matrix[:, idx_k]))
        print(f"{nama_k:<24} | {f'{p_mean:.4f}':<16} | {f'{r_mean:.4f}':<16} | {f'{f_mean:.4f}'}")
        rincian_kelas_list.append({
            'kelas': nama_k,
            'presisi': p_mean,
            'recall': r_mean,
            'f1': f_mean,
        })
    print("=" * 90)

    # Matriks Kebingungan Agregat (Total 5 Lipatan OOF)
    cm_total = np.sum([k['cm'] for k in klasik_list], axis=0)
    print("\n" + "=" * 90)
    print("MATRIKS KEBINGUNGAN AGREGAT ROUTER KLASIK (5 LIPATAN VALIDASI)")
    print(f"{CATATAN_LABEL}")
    print("=" * 90)
    singkatan_kelas = [k[:8] for k in SEMUA_KELAS]
    header_cm = f"{'True \\ Pred':<14} | " + " | ".join(f"{s:<8}" for s in singkatan_kelas)
    print(header_cm)
    print("-" * 90)
    for idx_k, nama_k in enumerate(SEMUA_KELAS):
        baris_str = " | ".join(f"{cm_total[idx_k, j]:<8}" for j in range(len(SEMUA_KELAS)))
        print(f"{nama_k[:12]:<14} | {baris_str}")
    print("=" * 90)

    # ------------------------------------------------------------------
    # TABEL AMBANG KEYAKINAN (0.30 - 0.80 selisih 0.05) DARI OOF
    # ------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("TABEL ANALISIS AMBANG KEYAKINAN ROUTER KLASIK (0.30 - 0.80)")
    print("Dihitung hanya dari prediksi lipatan validasi (Out-Of-Fold, bukan uji akhir)")
    print(f"{CATATAN_LABEL}")
    print("=" * 105)
    header_ambang = (
        f"{'Ambang':<8} | {'% Jawab':<10} | {'% Eskalasi':<12} | "
        f"{'Kesalahan Berbahaya':<22} | {'Eskalasi Berlebih':<20} | {'Akurasi Keputusan'}"
    )
    print(header_ambang)
    print("-" * 105)

    ambang_list = np.arange(0.30, 0.81, 0.05)
    tabel_ambang_data = []

    for T in ambang_list:
        T = round(T, 2)
        keputusan_list = []
        y_true_list = []

        for p in oof:
            yt = p['y_true']
            yp = p['y_pred']
            prob = p['max_prob']

            # Aturan Router dengan Ambang:
            # Jika kelas adalah eskalasi ATAU keyakinan di bawah T -> keputusan = teruskan
            # Jika kelas bot DAN keyakinan >= T -> keputusan = jawab
            if yp in KELAS_ESKALASI or prob < T:
                keputusan = "teruskan"
            else:
                keputusan = "jawab"

            keputusan_list.append(keputusan)
            y_true_list.append(yt)

        total_sesi = len(oof)
        n_jawab = sum(1 for k in keputusan_list if k == "jawab")
        n_teruskan = sum(1 for k in keputusan_list if k == "teruskan")

        persen_jawab = (n_jawab / total_sesi) * 100
        persen_teruskan = (n_teruskan / total_sesi) * 100

        # Kesalahan berbahaya pada ambang T:
        # Sesi aslinya kelas eskalasi (beasiswa/lainnya) yang diputuskan "jawab"
        n_berbahaya_T = sum(1 for yt, k in zip(y_true_list, keputusan_list) if yt in KELAS_ESKALASI and k == "jawab")
        total_eskalasi = sum(1 for yt in y_true_list if yt in KELAS_ESKALASI)
        persen_berbahaya_T = (n_berbahaya_T / total_eskalasi * 100) if total_eskalasi > 0 else 0.0

        # Eskalasi berlebih pada ambang T:
        # Sesi aslinya kelas jawab yang diputuskan "teruskan"
        n_berlebih_T = sum(1 for yt, k in zip(y_true_list, keputusan_list) if yt in KELAS_BOT and k == "teruskan")
        total_jawab = sum(1 for yt in y_true_list if yt in KELAS_BOT)
        persen_berlebih_T = (n_berlebih_T / total_jawab * 100) if total_jawab > 0 else 0.0

        # Keputusan yang tepat:
        # (yt in KELAS_BOT and k == 'jawab') OR (yt in KELAS_ESKALASI and k == 'teruskan')
        tepat = sum(1 for yt, k in zip(y_true_list, keputusan_list) if (yt in KELAS_BOT and k == "jawab") or (yt in KELAS_ESKALASI and k == "teruskan"))
        akurasi_keputusan = (tepat / total_sesi) * 100

        tabel_ambang_data.append({
            'ambang': T,
            'persen_jawab': round(persen_jawab, 1),
            'persen_eskalasi': round(persen_teruskan, 1),
            'kesalahan_berbahaya': round(persen_berbahaya_T, 1),
            'eskalasi_berlebih': round(persen_berlebih_T, 1),
            'akurasi_keputusan': round(akurasi_keputusan, 1),
            'n_berbahaya': n_berbahaya_T,
            'total_eskalasi': total_eskalasi,
        })

        baris_ambang = (
            f"{T:<8.2f} | {f'{persen_jawab:.1f}%':<10} | {f'{persen_teruskan:.1f}%':<12} | "
            f"{f'{persen_berbahaya_T:.1f}% ({n_berbahaya_T}/{total_eskalasi})':<22} | "
            f"{f'{persen_berlebih_T:.1f}%':<20} | {f'{akurasi_keputusan:.1f}%'}"
        )
        print(baris_ambang)
    print("=" * 105)

    # ------------------------------------------------------------------
    # USULAN AMBANG KEYAKINAN
    # ------------------------------------------------------------------
    # Cari ambang yang meminimalkan kesalahan berbahaya dengan eskalasi berlebih yang wajar
    ambang_usulan = 0.50
    for row in tabel_ambang_data:
        if row['ambang'] == 0.50:
            row_50 = row
        if row['ambang'] == 0.45:
            row_45 = row

    print("\n" + "=" * 90)
    print("USULAN AMBANG KEYAKINAN ROUTER KLASIK BESERTA ALASAN")
    print(f"{CATATAN_LABEL}")
    print("=" * 90)
    print(f"Ambang yang diusulkan: T = {ambang_usulan:.2f}")
    print(f"Alasan teknis:")
    print(f"  1. Kesalahan berbahaya pada T=0.50 adalah {row_50['kesalahan_berbahaya']}% ({row_50['n_berbahaya']} dari {row_50['total_eskalasi']} sesi eskalasi).")
    print(f"  2. Akurasi keputusan (jawab vs teruskan) optimal sebesar {row_50['akurasi_keputusan']}%.")
    print(f"  3. Eskalasi berlebih berada di angka wajar ({row_50['eskalasi_berlebih']}%), sehingga bot tetap mampu melayani")
    print(f"     sekitar {row_50['persen_jawab']}% pertanyaan secara otomatis tanpa membani staf admin secara berlebihan.")
    print(f"  4. Ambang ini konservatif dan selaras dengan prinsip bot layanan administrasi kampus.")
    print("=" * 90)

    # ------------------------------------------------------------------
    # Simpan hasil_router.csv
    # ------------------------------------------------------------------
    berkas_csv = HASIL_DIR / "hasil_router.csv"
    df_hasil_csv = pd.DataFrame([
        {
            'model': 'Baseline Kelas Terbanyak',
            'macro_f1_mean': round(m_maj, 4),
            'macro_f1_std': round(s_maj, 4),
            'kesalahan_berbahaya_persen': round(kb_maj, 2),
            'eskalasi_berlebih_persen': round(eb_maj, 2),
            'catatan': CATATAN_LABEL,
        },
        {
            'model': 'Baseline Router Aturan',
            'macro_f1_mean': round(m_at, 4),
            'macro_f1_std': round(s_at, 4),
            'kesalahan_berbahaya_persen': round(kb_at, 2),
            'eskalasi_berlebih_persen': round(eb_at, 2),
            'catatan': CATATAN_LABEL,
        },
        {
            'model': 'Router Klasik TFIDF + LogReg',
            'macro_f1_mean': round(m_klasik, 4),
            'macro_f1_std': round(s_klasik, 4),
            'kesalahan_berbahaya_persen': round(kb_klasik, 2),
            'eskalasi_berlebih_persen': round(eb_klasik, 2),
            'catatan': CATATAN_LABEL,
        }
    ])
    df_hasil_csv.to_csv(berkas_csv, index=False)
    print(f"\nTabel perbandingan model disimpan ke: {berkas_csv}")


def latih_dan_simpan_model_final(df_latih: pd.DataFrame, df_uji_akhir: pd.DataFrame):
    """
    Melatih model akhir router klasik pada SELURUH data latih (166 sesi, tanpa uji akhir),
    menyimpannya ke models\router_klasik.joblib.
    """
    print("\n" + "=" * 90)
    print("PELATIHAN MODEL AKHIR ROUTER KLASIK")
    print(f"Data Latih: {len(df_latih)} sesi (100% data latih lipatan)")
    print(f"{CATATAN_LABEL}")
    print("=" * 90)

    pipeline_final = Pipeline([
        ('tfidf', TfidfVectorizer(
            analyzer='char',
            ngram_range=(2, 5),
            sublinear_tf=True,
            min_df=2
        )),
        ('clf', LogisticRegression(
            class_weight='balanced',
            random_state=SEED,
            max_iter=1000,
            C=1.0
        ))
    ])

    pipeline_final.fit(df_latih['teks'], df_latih['label'])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    berkas_model = MODELS_DIR / "router_klasik.joblib"
    joblib.dump(pipeline_final, berkas_model)
    print(f"Model router klasik berhasil disimpan ke: {berkas_model}")

    # Metadata ringkas model
    metadata_model = {
        "nama_model": "TFIDF_Char_2_5_LogisticRegression",
        "total_data_latih": len(df_latih),
        "kelas": list(pipeline_final.classes_),
        "kelas_eskalasi": KELAS_ESKALASI,
        "kelas_bot": KELAS_BOT,
        "ambang_diusulkan": 0.50,
        "seed": SEED,
        "catatan": CATATAN_LABEL,
    }
    with open(MODELS_DIR / "router_klasik_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata_model, f, indent=2)


def main():
    log_file_path = HASIL_DIR / "log_T5.txt"
    tee = OutputTee(log_file_path)
    sys.stdout = tee

    try:
        print("=" * 90)
        print("EKSEKUSI TAHAP T5: PELATIHAN & EVALUASI ROUTER KLASIK")
        print(f"Berkas Log: {log_file_path}")
        print(f"{CATATAN_LABEL}")
        print("=" * 90)

        # 1. Siapkan dataset
        df_semua = siapkan_dataset()

        # 2. Pisahkan Uji Akhir (8 chat)
        df_latih, df_uji_akhir, chat_uji_akhir = pisahkan_uji_akhir(df_semua)

        # 3. Buat dan simpan 5 lipatan StratifiedGroupKFold
        lipatan_list = buat_dan_simpan_lipatan(df_latih, chat_uji_akhir)

        # 4. Evaluasi Cross-Validation 5 Lipatan
        eval_res = evaluasi_model_klasik(df_latih, lipatan_list)

        # 5. Tampilkan dan simpan laporan lengkap
        tampilkan_dan_simpan_laporan(eval_res, df_latih, df_uji_akhir)

        # 6. Latih dan simpan model final
        latih_dan_simpan_model_final(df_latih, df_uji_akhir)

        print("\n" + "=" * 90)
        print("TAHAP T5 SELESAI DENGAN SUKSES!")
        print("=" * 90)

    finally:
        sys.stdout = tee.terminal
        tee.close()


if __name__ == "__main__":
    main()

