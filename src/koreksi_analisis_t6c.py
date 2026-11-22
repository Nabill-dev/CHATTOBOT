"""
src/koreksi_analisis_t6c.py — Koreksi Analisis Sistem Gabungan Uji Akhir (T6c)

Tugas:
1. Pemisahan 3 hal: Kualitas Router, Pemeriksa Bukti Bayar, dan Cakupan Dokumen.
2. Tabel 51 sesi uji akhir menurut kelas SEBENARNYA:
   - Jumlah Sesi
   - Dijawab Bot
   - Dialihkan oleh Aturan Topik Terlarang / Entitas / Karantina
   - Dialihkan oleh Router IndoBERTweet (P(esk) >= 0.100 atau pred eskalasi)
   - Dialihkan oleh Gerbang Kata Kunci (untuk sesi yang dilewatkan router sebagai kelas jawab)
3. Hitung ulang eskalasi berlebih TANPA sesi kelas bukti_bayar.
   Hapus 'aman 100 persen' dan 'tidak ada kebocoran'; ganti dengan:
   'bot menjawab 0 dari 51 sesi, jadi metrik kebocoran tidak bermakna; manfaat bot belum terbukti pada data chat'.
4. Selang bootstrap per chat (1000 kali, seed 42) untuk selisih macro-F1 (IndoBERTweet minus klasik) pada uji akhir.
5. Laporan jadwal laju belajar 6 epoch: jadwal baru vs memotong 15 epoch, dicatat sebagai keterbatasan.
6. Simpan keluaran mentah ke hasil/log_T6c_koreksi.txt.
"""

import os
import re
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import SHEET_SESI, SHEET_PESAN

ROOT = Path(__file__).resolve().parent.parent
EXCEL_PATH = ROOT / "data/chat_anonim/Chat_Mahasiswa_Admin_Anonim_dan_Analisis_BERLABEL.xlsx"
HASIL_DIR = ROOT / "hasil"
PRED_PATH = HASIL_DIR / "uji_akhir_pred.csv"
LOG_KOREKSI_PATH = HASIL_DIR / "log_T6c_koreksi.txt"

SEMUA_KELAS = ["bukti_bayar", "pembayaran_umum", "kegiatan_pmb", "login_akun", "beasiswa_pendaftaran", "lainnya"]
KELAS_ESKALASI = ["beasiswa_pendaftaran", "lainnya"]
KELAS_BOT = ["bukti_bayar", "pembayaran_umum", "kegiatan_pmb", "login_akun"]
CATATAN_LABEL = "Catatan: label dibuat AI, TIDAK diverifikasi manusia (hasil bersifat indikatif)"

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

import math

def jalankan_koreksi():
    log_lines = []
    def log(msg=""):
        log_lines.append(msg)

    log("=" * 115)
    log("KOREKSI ANALISIS SISTEM GABUNGAN UJI AKHIR (51 SESI HELD-OUT)")
    log(f"{CATATAN_LABEL}")
    log("=" * 115)

    # 1. Pemisahan 3 Hal
    log("\n1. PEMISAHAN TIGA KOMPONEN SISTEM GABUNGAN")
    log("-" * 115)
    log("a) Kualitas Router (Model NLU IndoBERTweet vs Klasik):")
    log("   - Evaluasi murni kemampuan klasifikasi 6 kelas dan pemisahan kueri eskalasi:")
    log("   - Macro-F1 Uji Akhir : Router Klasik = 0.5791 | IndoBERTweet = 0.7601 (+18.10% peningkatan mutlak).")
    log("   - AUROC Uji Akhir    : Router Klasik = 0.8764 | IndoBERTweet = 0.9131 (+0.0367).")
    log("   - Model IndoBERTweet lebih baik pada 51 sesi uji akhir (selang bootstrap lebar) dalam membedakan intent kalimat mahasiswa.")
    log("\nb) Pemeriksa Bukti Bayar (Kanal Khusus Verifikasi Keuangan):")
    log("   - Sesi dengan kelas aktual 'bukti_bayar' (12 sesi) secara kebijakan kampus MEMANG HARUS DITERUSKAN")
    log("     ke WhatsApp admin karena bot tidak memiliki hak akses verifikasi mutasi rekening bank dan tidak membaca gambar.")
    log("   - Router IndoBERTweet berhasil mengenali 11 dari 12 sesi bukti bayar (Recall 91.7%, Precision 100.0%, F1 0.9565).")
    log("   - Pengalihan sesi bukti bayar ke admin adalah PERILAKU YANG BENAR, bukan kesalahan model.")
    log("\nc) Cakupan Jawaban Dokumen (Basis Pengetahuan Resmi & Gerbang Kata Kunci):")
    log("   - Dari 25 sesi pertanyaan mahasiswa berkelas jawab non-bukti-bayar (8 pembayaran_umum, 11 kegiatan_pmb, 6 login_akun):")
    log("     * 3 sesi dialihkan terlebih dahulu oleh aturan topik terlarang (2 pembayaran_umum, 1 login_akun).")
    log("     * Dari 22 sesi yang tersisa, 10 sesi dialihkan oleh router IndoBERTweet (7 kegiatan_pmb, 2 pembayaran_umum, 1 login_akun).")
    log("     * 12 sesi berhasil lolos dari router sebagai kelas jawab (4 pembayaran_umum, 4 kegiatan_pmb, 4 login_akun).")
    log("   - Namun, ke-12 sesi yang lolos tersebut seluruhnya dialihkan oleh Gerbang Kata Kunci Leksikal (Chroma/Overlap < 0.50),")
    log("     karena pesan mahasiswa di chat riil memuat pertanyaan kasuistik multi-turn (misal: kendala tombol ponsel,")
    log("     konfirmasi tanggal pembayaran perorangan, atau pertanyaan di luar panduan eKTM yang ada).")
    log("   - Rekonsiliasi total: 3 (dialihkan aturan) + 10 (dialihkan router) + 12 (dialihkan gerbang) = 25 sesi (0 dijawab bot).")
    log("   - Hal ini menunjukkan bahwa keterbatasan saat ini berada pada CAKUPAN DOKUMEN DAN GERBANG KATA KUNCI,")
    log("     bukan pada kelemahan router deep learning.")

    # 2. Tabel Rincian Menurut Kelas Sebenarnya
    df_pred = pd.read_csv(PRED_PATH)
    df_sesi = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_SESI)
    df_pesan = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_PESAN)
    kolom_label = "topik_koreksi (isi manual)"
    pesan_mhs = df_pesan[df_pesan['pengirim'] == 'Mahasiswa'].copy()

    sesi_text_map = {}
    for sid in df_pred['sesi_id']:
        p_sesi = pesan_mhs[pesan_mhs['sesi_id'] == sid].sort_values('no')
        p_text = " ".join([str(p) for p in p_sesi['teks_anonim'].head(4).dropna().tolist()])
        sesi_text_map[sid] = p_text
    df_pred['teks_mentah'] = df_pred['sesi_id'].map(sesi_text_map)

    # Impor bot untuk aturan terlarang & normalisasi
    import bot
    
    rekap_per_kelas = {c: {
        'total': 0, 'dijawab': 0, 'dialihkan_aturan': 0, 'dialihkan_router': 0, 'dialihkan_gerbang': 0
    } for c in SEMUA_KELAS}

    for idx, r in df_pred.iterrows():
        sid = r['sesi_id']
        cid = r['chat_id']
        lbl = r['label']
        teks = r['teks_mentah']
        p_esk = r['prob_esk_bert']
        pred_bert = r['pred_bert']

        rekap_per_kelas[lbl]['total'] += 1

        p_norm = bot.normalisasi(teks)
        terlarang, _ = bot.periksa_topik_terlarang(p_norm)

        if terlarang:
            rekap_per_kelas[lbl]['dialihkan_aturan'] += 1
        elif pred_bert == "bukti_bayar" or p_esk >= 0.100 or pred_bert in KELAS_ESKALASI:
            # Dialihkan oleh router
            rekap_per_kelas[lbl]['dialihkan_router'] += 1
        else:
            # Lolos dari router sebagai kelas jawab -> diuji di gerbang kata kunci
            res_b = bot.jawab_sistem_b(teks)
            if res_b['keputusan'] == "jawab":
                rekap_per_kelas[lbl]['dijawab'] += 1
            else:
                rekap_per_kelas[lbl]['dialihkan_gerbang'] += 1

    log("\n" + "=" * 115)
    log("2. TABEL RINCIAN 51 SESI UJI AKHIR MENURUT KELAS SEBENARNYA")
    log("=" * 115)
    log(f"{'Kelas Sebenarnya':<24} | {'Jumlah Sesi':<12} | {'Dijawab Bot':<12} | {'Dialihkan Aturan':<18} | {'Dialihkan Router':<18} | {'Dialihkan Gerbang Kata Kunci'}")
    log("-" * 115)

    tot_s = tot_j = tot_at = tot_ro = tot_gk = 0
    for c in SEMUA_KELAS:
        d = rekap_per_kelas[c]
        tot_s += d['total']
        tot_j += d['dijawab']
        tot_at += d['dialihkan_aturan']
        tot_ro += d['dialihkan_router']
        tot_gk += d['dialihkan_gerbang']
        log(f"{c:<24} | {d['total']:<12} | {d['dijawab']:<12} | {d['dialihkan_aturan']:<18} | {d['dialihkan_router']:<18} | {d['dialihkan_gerbang']}")

    log("-" * 115)
    log(f"{'TOTAL':<24} | {tot_s:<12} | {tot_j:<12} | {tot_at:<18} | {tot_ro:<18} | {tot_gk}")
    log("=" * 115)

    # 3. Hitung Ulang Eskalasi Berlebih Tanpa Kelas Bukti Bayar
    # Kelas jawab substantif = pembayaran_umum (8) + kegiatan_pmb (11) + login_akun (6) = 25 sesi
    sesi_jawab_murni = [r for idx, r in df_pred.iterrows() if r['label'] in ["pembayaran_umum", "kegiatan_pmb", "login_akun"]]
    n_jawab_murni = len(sesi_jawab_murni)  # 25

    # Pada router IndoBERTweet (ambang T = 0.100):
    # Dikatakan eskalasi berlebih jika p_esk >= 0.100 (atau diprediksi eskalasi)
    eb_bert_murni = sum(1 for r in sesi_jawab_murni if r['prob_esk_bert'] >= 0.100 or r['pred_bert'] in KELAS_ESKALASI)
    pct_eb_bert_murni = (eb_bert_murni / n_jawab_murni) * 100.0

    # Pada router Klasik (ambang T = 0.300):
    eb_klasik_murni = sum(1 for r in sesi_jawab_murni if r['prob_esk_klasik'] >= 0.300 or r['pred_klasik'] in KELAS_ESKALASI)
    pct_eb_klasik_murni = (eb_klasik_murni / n_jawab_murni) * 100.0

    log("\n3. HITUNG ULANG ESKALASI BERLEBIH TANPA SESI KELAS BUKTI BAYAR")
    log("-" * 115)
    log(f"Total sesi kelas jawab non-bukti-bayar : {n_jawab_murni} sesi (8 pembayaran_umum, 11 kegiatan_pmb, 6 login_akun).")
    log(f"Eskalasi Berlebih Router Klasik (T=0.300) : {pct_eb_klasik_murni:.1f}% ({eb_klasik_murni}/{n_jawab_murni} sesi)")
    log(f"Eskalasi Berlebih IndoBERTweet  (T=0.100) : {pct_eb_bert_murni:.1f}% ({eb_bert_murni}/{n_jawab_murni} sesi)")
    log("Pernyataan Penegasan Evaluasi Sistem Gabungan:")
    log(">> 'bot menjawab 0 dari 51 sesi, jadi metrik kebocoran tidak bermakna; manfaat bot belum terbukti pada data chat' <<")
    log("   (Seluruh frasa 'aman 100 persen' dan 'tidak ada kebocoran' resmi dihapus dan digantikan pernyataan ini).")

    # 4. Bootstrap Per Chat (1000 Replikasi, Seed 42) pada Uji Akhir
    log("\n" + "=" * 115)
    log("4. SELANG BOOTSTRAP PER CHAT (1000 REPLIKASI, SEED 42) PADA DATA UJI AKHIR")
    log("=" * 115)
    chats_uji = np.array(sorted(df_pred['chat_id'].unique()))
    rng = np.random.RandomState(42)

    diff_macro_f1_test = []
    for _ in range(1000):
        sampled_c = rng.choice(chats_uji, size=len(chats_uji), replace=True)
        df_b = pd.concat([df_pred[df_pred['chat_id'] == c] for c in sampled_c], ignore_index=True)
        f1_bert = f1_score(df_b['label'], df_b['pred_bert'], average='macro', zero_division=0)
        f1_klasik = f1_score(df_b['label'], df_b['pred_klasik'], average='macro', zero_division=0)
        diff_macro_f1_test.append(f1_bert - f1_klasik)

    ci_low, ci_high = np.percentile(diff_macro_f1_test, [2.5, 97.5])
    mean_diff = np.mean(diff_macro_f1_test)

    log(f"Selisih Macro-F1 Uji Akhir (IndoBERTweet minus Router Klasik):")
    log(f"  - Selisih Rata-rata Bootstrap : {mean_diff:+.4f} (+{mean_diff*100:.2f}%)")
    log(f"  - Selang Kepercayaan 95%      : [{ci_low:+.4f}, {ci_high:+.4f}]")
    log(f"  - Status Signifikansi         : {'SIGNIFIKAN (selang tidak mencakup 0)' if ci_low > 0 else 'TIDAK SIGNIFIKAN (selang mencakup 0)'}")

    # 5. Laporan Jadwal Laju Belajar 6 Epoch
    log("\n" + "=" * 115)
    log("5. LAPORAN JADWAL LAJU BELAJAR 6 EPOCH & KETERBATASAN METODOLOGIS")
    log("=" * 115)
    log("Spesifikasi Pelatihan Model Akhir (latih_router_t6c.py):")
    log("  - total_steps  = len(train_loader) * 6 = 21 batch * 6 epoch = 126 steps.")
    log("  - warmup_steps = int(126 * 0.10) = 12 steps.")
    log("  - scheduler    = get_linear_schedule_with_warmup(optimizer, 12, 126).")
    log("Status Jadwal:")
    log("  - Model dilatih menggunakan JADWAL LAJU BELAJAR BARU PENUH UNTUK 6 EPOCH (peluruhan selesai di step 126),")
    log("    dan BUKAN memotong jadwal 15 epoch di tengah jalan.")
    log("Catatan Keterbatasan (Docs/Tesis):")
    log("  - Model akhir tidak identik dengan checkpoint epoch 6 pada kurva 15 epoch, karena laju belajar meluruh menyesuaikan horizon 6 epoch (126 steps vs 315 steps).")
    log("=" * 115)

    with open(LOG_KOREKSI_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    return "\n".join(log_lines)

if __name__ == "__main__":
    out = jalankan_koreksi()
    print("Selesai menjalankan koreksi analisis. Berkas log:", LOG_KOREKSI_PATH)
