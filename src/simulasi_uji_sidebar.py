# ==============================================================
# simulasi_uji_sidebar.py — Pengujian 8 Soal Sidebar pada 3 Router
# ==============================================================

import sys
import io
from pathlib import Path
from datetime import datetime
import csv

# Set stdout utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Tambahkan src ke sys.path
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import ROOT, HASIL_DIR
import bot
from app import LOG_FILE_PATH, LOG_COLUMNS, samarkan_data_pribadi

CONTOH_PERTANYAAN = [
    "Halo min, gmn cara byr ektm lwt atm bni?",
    "kalo mau bayar kuliah lewat teller bni prosedurnya gimana ya kak?",
    "ka kalo status di ektm tertulis pembayaran ditutup itu artinya apa?",
    "cara cek riwayat pembayaran",
    "cara bayar kuliah lewat BRI",
    "apa itu E-KTM",
    "pembayaran seminar SWE masih dibuka?",
    "apakah pembayaran kegiatan masih bisa?",
]

ROUTER_LIST = ["indobertweet", "klasik", "aturan"]

def simpan_log(pertanyaan, router, keputusan, sumber, catatan=""):
    HASIL_DIR.mkdir(parents=True, exist_ok=True)
    tulis_header = not LOG_FILE_PATH.exists() or LOG_FILE_PATH.stat().st_size == 0
    waktu_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if tulis_header:
            writer.writerow(LOG_COLUMNS)
        writer.writerow([
            waktu_str,
            samarkan_data_pribadi(pertanyaan),
            "B",
            router,
            keputusan,
            sumber,
            "Uji Otomatis Sidebar",
            catatan
        ])

def jalankan_simulasi():
    out_file = HASIL_DIR / "log_uji_sidebar.txt"
    with open(out_file, "w", encoding="utf-8") as f_out:
        def cetak(teks=""):
            print(teks)
            f_out.write(teks + "\n")

        cetak("=" * 115)
        cetak("HASIL PENGUJIAN 8 CONTOH PERTANYAAN SIDEBAR UNTUK TIAP PILIHAN ROUTER")
        cetak("Waktu Pengujian : " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        cetak("Sistem          : Sistem B (Utama)")
        cetak("Berkas Log CSV  : " + str(LOG_FILE_PATH))
        cetak("=" * 115)

        for router in ROUTER_LIST:
            cetak(f"\n" + "#" * 115)
            cetak(f"PILIHAN ROUTER: {router.upper()}")
            cetak("#" * 115)
            header = f"{'No':<3} | {'Pertanyaan':<35} | {'Kelas Router':<20} | {'Keputusan':<18} | {'Sumber Dokumen'}"
            cetak(header)
            cetak("-" * 115)

            for idx, soal in enumerate(CONTOH_PERTANYAAN, 1):
                res = bot.jawab(soal, sistem="B", router_mode=router)
                kelas = res.get('kelas_router', '-')
                keputusan = res.get('keputusan', '-')
                sumber = res.get('sumber', '') if res.get('sumber') else "-"
                
                # Simpan ke CSV
                catatan = f"Kelas: {kelas}, P(esk): {res.get('peluang_eskalasi', 0):.4f}, Alasan: {res.get('alasan', '-')}"
                simpan_log(soal, router, keputusan, sumber, catatan)

                # Format baris
                soal_ringkas = (soal[:32] + "...") if len(soal) > 35 else soal
                sumber_ringkas = (sumber[:32] + "...") if len(sumber) > 35 else sumber
                baris = f"{idx:<3} | {soal_ringkas:<35} | {kelas:<20} | {keputusan:<18} | {sumber_ringkas}"
                cetak(baris)

            cetak("-" * 115)

            # Rincian detail per pertanyaan untuk router ini
            cetak(f"\n[Rincian Detail Router {router.upper()}]:")
            for idx, soal in enumerate(CONTOH_PERTANYAAN, 1):
                res = bot.jawab(soal, sistem="B", router_mode=router)
                cetak(f"  {idx}. Pertanyaan : {soal}")
                cetak(f"     Kelas      : {res.get('kelas_router', '-')}")
                cetak(f"     Keputusan  : {res.get('keputusan', '-')}")
                cetak(f"     Sumber     : {res.get('sumber', '-') if res.get('sumber') else '-'}")
                cetak(f"     Alasan     : {res.get('alasan', '-')}")
                cetak(f"     P(esk)     : {res.get('peluang_eskalasi', 0):.4f} | Keyakinan: {res.get('keyakinan', 0):.4f}")
                cetak("")

        cetak("=" * 115)
        cetak("PENGUJIAN SELESAI")
        cetak("=" * 115)

if __name__ == "__main__":
    jalankan_simulasi()

