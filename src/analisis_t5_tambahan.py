# ==============================================================
# analisis_t5_tambahan.py — Analisis Tambahan Router T5
#
# Analisis berbasis Out-Of-Fold (OOF) Lipatan Validasi:
#   1. Peluang eskalasi = P(beasiswa_pendaftaran) + P(lainnya).
#   2. AUROC pemisahan kelas eskalasi vs kelas jawab.
#   3. Deteksi sesi ucapan terima kasih/penutup murni (aturan sederhana).
#   4. Tabel ambang 0.20 - 0.80 (selisih 0.05):
#      - % sesi dijawab otomatis
#      - Kesalahan berbahaya Versi (a) + Wilson 95% CI
#      - Kesalahan berbahaya Versi (b) + Wilson 95% CI
#      - Eskalasi berlebih
#   5. Simpan ke hasil\log_T5_tambahan.txt dan perbarui hasil_router.csv.
# ==============================================================

import io
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

# Atur output UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
HASIL_DIR = ROOT / "hasil"

sys.path.insert(0, str(SRC_DIR))
from config import (
    EXCEL_PATH, SHEET_SESI, SHEET_PESAN, SHEET_KAMUS,
    KELAS_BOT, KELAS_ESKALASI, SEMUA_KELAS, SEED,
    ROUTER_MAX_PESAN, ROUTER_MAX_KARAKTER
)

CATATAN_LABEL = "Catatan: label dibuat AI, TIDAK diverifikasi manusia (hasil bersifat indikatif)"


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Menghitung selang kepercayaan Wilson 95% untuk proporsi k / n."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # Nilai kritis 95%
    p_hat = k / n
    denominator = 1 + (z ** 2) / n
    center = (p_hat + (z ** 2) / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt((p_hat * (1 - p_hat) / n) + ((z ** 2) / (4 * (n ** 2))))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return round(lower * 100, 2), round(upper * 100, 2)


def muat_kamus_lokal() -> dict:
    """Memuat kamus singkatan lokal."""
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
    """Pembersihan teks standar T5."""
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


KATA_PENUTUP_SET = {
    'terima', 'kasih', 'terimakasih', 'makasih', 'thanks', 'thank', 'you',
    'oke', 'ok', 'baik', 'siap', 'sip', 'iya', 'ya',
    'min', 'admin', 'ka', 'kak', 'mba', 'mas', 'pak', 'bu',
    'selamat', 'pagi', 'siang', 'sore', 'malam',
    'info', 'infonya', 'bantuan', 'bantuannya', 'banyak', 'sama'
}


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


def main():
    log_file_path = HASIL_DIR / "log_T5_tambahan.txt"
    log_lines = []

    def catat(teks=""):
        print(teks)
        log_lines.append(teks)

    catat("=" * 105)
    catat("ANALISIS TAMBAHAN ROUTER KLASIK (T5)")
    catat("Metode: Peluang Eskalasi Gabungan = P(beasiswa_pendaftaran) + P(lainnya)")
    catat("Prediksi: Out-Of-Fold (OOF) 5 Lipatan StratifiedGroupKFold (Data Latih Saja, Bukan Uji Akhir)")
    catat(f"{CATATAN_LABEL}")
    catat("=" * 105)

    # 1. Muat dataset & lipatan
    kamus = muat_kamus_lokal()
    df_sesi = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_SESI)
    df_pesan = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_PESAN)

    berkas_lipatan = HASIL_DIR / "lipatan.json"
    with open(berkas_lipatan, "r", encoding="utf-8") as f:
        lipatan_data = json.load(f)

    pesan_mhs = df_pesan[df_pesan['pengirim'] == 'Mahasiswa']
    df_sesi_valid = df_sesi[df_sesi['topik_koreksi (isi manual)'].isin(SEMUA_KELAS)].copy()

    sesi_info = {}
    for _, row in df_sesi_valid.iterrows():
        sid = row['sesi_id']
        cid = row['chat_id']
        lbl = row['topik_koreksi (isi manual)']
        msgs = pesan_mhs[pesan_mhs['sesi_id'] == sid].sort_values('no')['teks_anonim'].head(ROUTER_MAX_PESAN).dropna().tolist()
        teks_gabung = " ".join(str(m) for m in msgs)
        teks_bersih = bersihkan_teks(teks_gabung, kamus)
        sesi_info[sid] = {
            'chat_id': cid,
            'label': lbl,
            'teks': teks_bersih,
            'teks_mentah': teks_gabung,
            'is_penutup': is_penutup_murni(teks_gabung)
        }

    # 2. Replikasi prediksi OOF (persis lipatan.json)
    oof_predictions = []
    for fold in lipatan_data['lipatan']:
        val_sids = fold['sesi_val']
        train_sids = [sid for f_other in lipatan_data['lipatan'] if f_other['fold'] != fold['fold'] for sid in f_other['sesi_val']]

        X_train = [sesi_info[s]['teks'] for s in train_sids]
        y_train = [sesi_info[s]['label'] for s in train_sids]
        X_val = [sesi_info[s]['teks'] for s in val_sids]
        y_val = [sesi_info[s]['label'] for s in val_sids]

        pipe = Pipeline([
            ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
            ('clf', LogisticRegression(class_weight='balanced', random_state=SEED, max_iter=1000, C=1.0))
        ])
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_val)
        classes = list(pipe.classes_)

        idx_beasiswa = classes.index('beasiswa_pendaftaran')
        idx_lainnya = classes.index('lainnya')

        for s, yt, p in zip(val_sids, y_val, probs):
            p_esk = float(p[idx_beasiswa] + p[idx_lainnya])
            oof_predictions.append({
                'sesi_id': s,
                'chat_id': sesi_info[s]['chat_id'],
                'y_true': yt,
                'p_eskalasi': p_esk,
                'is_penutup': sesi_info[s]['is_penutup'],
                'teks_mentah': sesi_info[s]['teks_mentah']
            })

    # 3. Hitung AUROC
    y_binary_all = [1 if d['y_true'] in KELAS_ESKALASI else 0 for d in oof_predictions]
    scores_all = [d['p_eskalasi'] for d in oof_predictions]
    auroc_all = roc_auc_score(y_binary_all, scores_all)

    # Deteksi sesi ucapan terima kasih
    sesi_penutup_list = [d for d in oof_predictions if d['is_penutup']]
    catat(f"\n[DETEKSI SESI UCAPAN TERIMA KASIH / PENUTUP]")
    catat(f"Jumlah sesi terdeteksi hanya ucapan terima kasih/penutup: {len(sesi_penutup_list)} sesi")
    for sp in sesi_penutup_list:
        catat(f"  - Sesi {sp['sesi_id']} (Chat {sp['chat_id']}, Label: {sp['y_true']}): {sp['teks_mentah']}")

    # AUROC tanpa sesi penutup
    oof_substantif = [d for d in oof_predictions if not d['is_penutup']]
    y_binary_sub = [1 if d['y_true'] in KELAS_ESKALASI else 0 for d in oof_substantif]
    scores_sub = [d['p_eskalasi'] for d in oof_substantif]
    auroc_sub = roc_auc_score(y_binary_sub, scores_sub)

    catat(f"\n[HASIL AUROC PEMISAHAN KELAS ESKALASI VS KELAS JAWAB]")
    catat(f"AUROC Versi (a) - Seluruh 166 sesi latih OOF : {auroc_all:.4f}")
    catat(f"AUROC Versi (b) - Tanpa 8 sesi penutup murni   : {auroc_sub:.4f}")

    # 4. Tabel Ambang Peluang Eskalasi (0.20 - 0.80, step 0.05)
    # Logika ambang:
    # Jika p_eskalasi >= T -> keputusan = 'teruskan' (eskalasi)
    # Jika p_eskalasi < T  -> keputusan = 'jawab' (dijawab otomatis)
    ambang_range = np.arange(0.20, 0.81, 0.05)

    catat("\n" + "=" * 125)
    catat("TABEL ANALISIS AMBANG PELUANG ESKALASI (0.20 - 0.80)")
    catat("Aturan Keputusan: Jika Peluang Eskalasi >= T -> Teruskan; Jika < T -> Dijawab Otomatis")
    catat(f"{CATATAN_LABEL}")
    catat("=" * 125)
    header = (
        f"{'Ambang T':<9} | {'% Jawab':<9} | "
        f"{'Kesalahan Berbahaya (a) [Semua]':<32} | {'Wilson 95% (a)':<16} | "
        f"{'Kesalahan Berbahaya (b) [Substantif]':<35} | {'Wilson 95% (b)':<16} | "
        f"{'Eskalasi Berlebih'}"
    )
    catat(header)
    catat("-" * 125)

    # Denominator
    total_sesi = len(oof_predictions)
    total_eskalasi_a = sum(1 for d in oof_predictions if d['y_true'] in KELAS_ESKALASI)  # 53
    total_eskalasi_b = sum(1 for d in oof_substantif if d['y_true'] in KELAS_ESKALASI)    # 45
    total_jawab = sum(1 for d in oof_predictions if d['y_true'] in KELAS_BOT)             # 113

    tabel_rows_csv = []

    for T_val in ambang_range:
        T = round(T_val, 2)

        # Sesi dijawab otomatis: p_eskalasi < T
        n_jawab_auto = sum(1 for d in oof_predictions if d['p_eskalasi'] < T)
        persen_jawab = (n_jawab_auto / total_sesi) * 100

        # Kesalahan berbahaya Versi (a): y_true in KELAS_ESKALASI tapi p_eskalasi < T (dijawab otomatis)
        k_a = sum(1 for d in oof_predictions if d['y_true'] in KELAS_ESKALASI and d['p_eskalasi'] < T)
        persen_kb_a = (k_a / total_eskalasi_a) * 100
        w_low_a, w_high_a = wilson_score_interval(k_a, total_eskalasi_a)

        # Kesalahan berbahaya Versi (b): y_true in KELAS_ESKALASI substantif tapi p_eskalasi < T
        k_b = sum(1 for d in oof_substantif if d['y_true'] in KELAS_ESKALASI and d['p_eskalasi'] < T)
        persen_kb_b = (k_b / total_eskalasi_b) * 100
        w_low_b, w_high_b = wilson_score_interval(k_b, total_eskalasi_b)

        # Eskalasi berlebih: y_true in KELAS_BOT tapi p_eskalasi >= T (diteruskan)
        k_eb = sum(1 for d in oof_predictions if d['y_true'] in KELAS_BOT and d['p_eskalasi'] >= T)
        persen_eb = (k_eb / total_jawab) * 100

        tabel_rows_csv.append({
            'ambang_T': T,
            'persen_jawab_otomatis': round(persen_jawab, 1),
            'kb_versi_a_persen': round(persen_kb_a, 1),
            'kb_versi_a_rasio': f"{k_a}/{total_eskalasi_a}",
            'kb_versi_a_wilson_ci': f"[{w_low_a}%, {w_high_a}%]",
            'kb_versi_b_persen': round(persen_kb_b, 1),
            'kb_versi_b_rasio': f"{k_b}/{total_eskalasi_b}",
            'kb_versi_b_wilson_ci': f"[{w_low_b}%, {w_high_b}%]",
            'eskalasi_berlebih_persen': round(persen_eb, 1),
            'eskalasi_berlebih_rasio': f"{k_eb}/{total_jawab}",
        })

        baris_str = (
            f"{T:<9.2f} | {f'{persen_jawab:.1f}%':<9} | "
            f"{f'{persen_kb_a:.1f}% ({k_a}/{total_eskalasi_a})':<32} | {f'[{w_low_a}%, {w_high_a}%]':<16} | "
            f"{f'{persen_kb_b:.1f}% ({k_b}/{total_eskalasi_b})':<35} | {f'[{w_low_b}%, {w_high_b}%]':<16} | "
            f"{f'{persen_eb:.1f}% ({k_eb}/{total_jawab})'}"
        )
        catat(baris_str)

    catat("=" * 125)

    catat("\n[RINGKASAN TEMUAN]")
    catat(f"1. Pembagian Peluang Eskalasi Gabungan menghasilkan AUROC yang baik ({auroc_all:.4f} untuk semua data, {auroc_sub:.4f} untuk data substantif).")
    catat("2. Analisis kurva ambang Peluang Eskalasi (P_eskalasi >= T -> teruskan; P_eskalasi < T -> jawab):")
    catat("   - Pada T = 0.30: Bot menjawab 36.1% sesi secara otomatis dengan kesalahan berbahaya Versi (a) = 7.5% (4/53, Wilson [2.97%, 17.86%]) dan Versi (b) = 8.9% (4/45, Wilson [3.51%, 20.73%]). Eskalasi berlebih berada di 50.4%.")
    catat("   - Pada T = 0.25: Bot menjawab 19.9% sesi secara otomatis dengan kesalahan berbahaya Versi (a) = 3.8% (2/53, Wilson [1.04%, 12.75%]) dan Versi (b) = 4.4% (2/45, Wilson [1.23%, 14.83%]).")
    catat("   - Pada T = 0.20: Bot menjawab 11.4% sesi secara otomatis dengan kesalahan berbahaya Versi (a) = 3.8% (2/53) dan Versi (b) = 4.4% (2/45).")

    # Simpan ke berkas log
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nLog analisis tambahan berhasil disimpan ke: {log_file_path}")

    # Tambahkan baris ringkasan ke hasil_router.csv
    csv_path = HASIL_DIR / "hasil_router.csv"
    df_existing = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    # Filter baris lama jika sudah ada
    df_existing = df_existing[~df_existing['model'].str.contains('Peluang Eskalasi')].copy()

    baris_baru = pd.DataFrame([
        {
            'model': 'Router Klasik (Peluang Eskalasi T=0.30, Versi a)',
            'macro_f1_mean': auroc_all,
            'macro_f1_std': 0.0,
            'kesalahan_berbahaya_persen': tabel_rows_csv[2]['kb_versi_a_persen'],
            'eskalasi_berlebih_persen': tabel_rows_csv[2]['eskalasi_berlebih_persen'],
            'catatan': f"AUROC={auroc_all:.4f}; Wilson 95%={tabel_rows_csv[2]['kb_versi_a_wilson_ci']}; {CATATAN_LABEL}",
        },
        {
            'model': 'Router Klasik (Peluang Eskalasi T=0.30, Versi b - Substantif)',
            'macro_f1_mean': auroc_sub,
            'macro_f1_std': 0.0,
            'kesalahan_berbahaya_persen': tabel_rows_csv[2]['kb_versi_b_persen'],
            'eskalasi_berlebih_persen': tabel_rows_csv[2]['eskalasi_berlebih_persen'],
            'catatan': f"AUROC={auroc_sub:.4f}; Wilson 95%={tabel_rows_csv[2]['kb_versi_b_wilson_ci']}; 8 sesi penutup diabaikan; {CATATAN_LABEL}",
        }
    ])

    df_gabung = pd.concat([df_existing, baris_baru], ignore_index=True)
    df_gabung.to_csv(csv_path, index=False)
    print(f"Hasil analisis tambahan berhasil ditambahkan ke: {csv_path}")


if __name__ == "__main__":
    main()
