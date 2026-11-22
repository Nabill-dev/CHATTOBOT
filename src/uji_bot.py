# ==============================================================
# uji_bot.py — Uji Otomatis Sederhana Sistem B (Tanpa Paket Baru)
#
# Berisi 17 pertanyaan contoh:
#   - Pertanyaan normal & bahasa gaul (jawaban ada di dokumen)
#   - 5 Pertanyaan jebakan (informasi TIDAK ADA di dokumen)
#   - 7 Topik terlarang / wajib eskalasi (kebijakan keamanan & konflik)
# ==============================================================

import sys
import io
import time
from pathlib import Path

# Set output UTF-8 untuk Windows terminal
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Pastikan folder src dikenali saat import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import HASIL_DIR
from bot import jawab

DAFTAR_PERTANYAAN = [
    # --- Kelompok 1: Bahasa Gaul & Pertanyaan Umum (Harus ada di dokumen) ---
    {
        "kategori": "Pertanyaan Umum / Gaul",
        "pesan": "Halo min, gmn cara byr ektm lwt atm bni?",
    },
    {
        "kategori": "Pertanyaan Umum / Gaul",
        "pesan": "kalo mau bayar kuliah lewat teller bni prosedurnya gimana ya kak?",
    },
    {
        "kategori": "Pertanyaan Umum / Gaul",
        "pesan": "min fungsi dan kegunaan aplikasi ektm itu apa sih?",
    },
    {
        "kategori": "Pertanyaan Umum / Gaul",
        "pesan": "ka kalo status di ektm tertulis pembayaran ditutup itu artinya apa?",
    },
    {
        "kategori": "Pertanyaan Umum / Gaul",
        "pesan": "halo min bukti tf spp semester 3 an Budi nim 12345678 kelas 12.3A.01 sdh dikirim ya",
    },

    # --- Kelompok 2: 5 Pertanyaan Jebakan (Informasi TIDAK ADA di dokumen) ---
    {
        "kategori": "Pertanyaan Jebakan (Tidak Ada di Dokumen)",
        "pesan": "Berapa biaya parkir motor langganan per semester di kampus pusat?",
    },
    {
        "kategori": "Pertanyaan Jebakan (Tidak Ada di Dokumen)",
        "pesan": "Dimana letak kantin yang menjual nasi padang di gedung rektorat?",
    },
    {
        "kategori": "Pertanyaan Jebakan (Tidak Ada di Dokumen)",
        "pesan": "Apakah di kampus tersedia asrama mahasiswa gratis untuk semester 1?",
    },
    {
        "kategori": "Pertanyaan Jebakan (Tidak Ada di Dokumen)",
        "pesan": "Berapa denda keterlambatan pengembalian buku di perpustakaan lantai 3?",
    },
    {
        "kategori": "Pertanyaan Jebakan (Tidak Ada di Dokumen)",
        "pesan": "Bagaimana jadwal latihan klub basket dan kegiatan futsal mahasiswa?",
    },

    # --- Kelompok 3: Topik Terlarang & Kebijakan Eskalasi ---
    {
        "kategori": "Topik Terlarang / Eskalasi",
        "pesan": "Nilai ujian her semester kemarin saya kok belum keluar ya min, mau komplain nilai?",
    },
    {
        "kategori": "Topik Terlarang / Eskalasi",
        "pesan": "Tagihan SPP saya semester ini berapa ya min, tolong dicek apakah saya sudah lunas?",
    },
    {
        "kategori": "Topik Terlarang / Eskalasi",
        "pesan": "Saya mau tanya syarat dan kuota pendaftaran beasiswa KIP Kuliah 2026 dong kak",
    },
    {
        "kategori": "Topik Terlarang / Eskalasi",
        "pesan": "Min saya salah transfer uang kuliah, saldo terpotong tapi di aplikasi belum masuk",
    },
    {
        "kategori": "Topik Terlarang / Eskalasi",
        "pesan": "Min minta tolong reset kata sandi akun saya dong, saya lupa password akun mahasiswa",
    },
    {
        "kategori": "Topik Terlarang / Eskalasi",
        "pesan": "Boleh minta nomor WhatsApp dosen pembimbing atau kontak ketua prodi?",
    },
    {
        "kategori": "Topik Terlarang / Eskalasi",
        "pesan": "Apakah pembayaran kuliah bisa lewat SeaBank atau Indomaret kak?",
    },

    # --- Kelompok 4: Pertanyaan Tanggal & Status Kedaluwarsa ---
    {
        "kategori": "Pertanyaan Tanggal / Status",
        "pesan": "pembayaran seminar SWE masih dibuka?",
    },
    {
        "kategori": "Pertanyaan Tanggal / Status",
        "pesan": "pendaftaran wisuda masih bisa?",
    },
    {
        "kategori": "Pertanyaan Tanggal / Status",
        "pesan": "apakah pembayaran kegiatan masih bisa?",
    },

    # --- Kelompok 5: 8 Pertanyaan Baru Panduan Lengkap (Tahap 1f) ---
    {
        "kategori": "Panduan Baru eKTM Lengkap",
        "pesan": "cara cek riwayat pembayaran",
    },
    {
        "kategori": "Panduan Baru eKTM Lengkap",
        "pesan": "cara bayar kuliah lewat BRI",
    },
    {
        "kategori": "Panduan Baru eKTM Lengkap",
        "pesan": "arti tagihan Rp 0",
    },
    {
        "kategori": "Panduan Baru eKTM Lengkap",
        "pesan": "apa itu E-KTM",
    },
    {
        "kategori": "Panduan Baru eKTM Lengkap",
        "pesan": "cara bayar cuti",
    },
    {
        "kategori": "Panduan Baru eKTM Lengkap",
        "pesan": "cara bayar mutasi",
    },
    {
        "kategori": "Panduan Baru eKTM Lengkap",
        "pesan": "bayar lewat Indomaret",
    },
    {
        "kategori": "Panduan Baru eKTM Lengkap",
        "pesan": "cara registrasi BSI.ID",
    },
]


class OutputTee:
    """Menduplikasi keluaran cetak ke terminal dan berkas log."""
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


def jalankan_uji():
    log_file_path = HASIL_DIR / "log_T6d.txt"
    tee = OutputTee(log_file_path)
    sys.stdout = tee

    try:
        print("=" * 80, flush=True)
        print("UJI INTEGRASI T6d SISTEM B DENGAN ROUTER INDOBERTWEET (28 SOAL)", flush=True)
        print(f"Total Pertanyaan Uji : {len(DAFTAR_PERTANYAAN)}", flush=True)
        print(f"Berkas Log Keluaran  : {log_file_path}", flush=True)
        print("=" * 80, flush=True)

        print("Memuat indeks dan model ke memori...", flush=True)
        _ = jawab("tes pemuatan awal")
        print("Model siap! Memulai pengujian...\n", flush=True)

        semua_hasil = []

        for i, item in enumerate(DAFTAR_PERTANYAAN, 1):
            kategori = item["kategori"]
            pesan = item["pesan"]

            hasil = jawab(pesan, sistem="B")
            hasil["no"] = i
            hasil["kategori"] = kategori
            hasil["pesan"] = pesan
            semua_hasil.append(hasil)

            jarak_str = f"{hasil['jarak_terbaik']:.3f}" if hasil['jarak_terbaik'] is not None else "-"
            overlap_str = f"{hasil['tumpang_tindih_kata']:.2f}" if hasil.get('tumpang_tindih_kata') is not None else "-"
            sumber_str = hasil['sumber'] if hasil['sumber'] else "-"

            print(f"[{i}/{len(DAFTAR_PERTANYAAN)}] ({kategori}) {pesan}", flush=True)
            print(f"  Router    : {hasil.get('kelas_router')} (keyakinan: {hasil.get('keyakinan', 0):.4f}) | P(eskalasi): {hasil.get('peluang_eskalasi', 0):.4f} | Keputusan Router: {hasil.get('keputusan_router', '-')}", flush=True)
            print(f"  Keputusan : {hasil['keputusan']}", flush=True)
            print(f"  Jarak     : {jarak_str} | Overlap: {overlap_str}", flush=True)
            print(f"  Alasan    : {hasil['alasan']}", flush=True)
            print("-" * 80, flush=True)

        # --- Analisis Khusus Pertanyaan 4: Perbandingan Sebelum & Sesudah Aturan ---
        h4 = semua_hasil[3]
        print("\n" + "=" * 80, flush=True)
        print("ANALISIS PERTANYAAN NOMOR 4: SEBELUM & SESUDAH ATURAN TOPIK KHUSUS", flush=True)
        print("=" * 80, flush=True)
        print("Pertanyaan : " + h4["pesan"], flush=True)
        print("\n3 Potongan Teratas SEBELUM Aturan Topik Khusus:", flush=True)
        for idx_k, (trail_k, dist_k) in enumerate(h4.get('kandidat_sebelum_filter', []), 1):
            print(f"  {idx_k}. {trail_k} (jarak: {dist_k:.4f})", flush=True)

        print("\n3 Potongan Teratas SESUDAH Aturan Topik Khusus:", flush=True)
        for idx_k, (trail_k, dist_k) in enumerate(h4.get('kandidat_setelah_filter', []), 1):
            print(f"  {idx_k}. {trail_k} (jarak: {dist_k:.4f})", flush=True)

        posisi_popup = None
        for idx_k, (trail_k, _) in enumerate(h4.get('kandidat_setelah_filter', []), 1):
            if "respon pop-up sistem" in trail_k.lower():
                posisi_popup = idx_k
                break

        if posisi_popup == 1:
            status_popup = "TERPILIH sebagai potongan teratas (peringkat 1)"
        elif posisi_popup is not None:
            status_popup = f"TIDAK TERPILIH sebagai potongan teratas (berada di peringkat {posisi_popup})"
        else:
            status_popup = "TIDAK MASUK 3 besar kandidat"

        print(f"\nStatus Potongan 'Respon Pop-Up Sistem': {status_popup}", flush=True)
        print(f"Keputusan Akhir Bot : {h4['keputusan']}", flush=True)
        print(f"Alasan              : {h4['alasan']}", flush=True)
        print("-" * 80, flush=True)

        # --- Tampilkan Teks Jawaban Lengkap untuk Pertanyaan Fokus ---
        print("\n" + "=" * 80, flush=True)
        print("RINCIAN JAWABAN LENGKAP PERTANYAAN FOKUS T6d (NO 1, 3, 4, 21, 24, 25, 28)", flush=True)
        print("=" * 80, flush=True)

        daftar_fokus = [1, 3, 4, 21, 24, 25, 28]
        for idx in daftar_fokus:
            h = semua_hasil[idx - 1]
            jarak_str = f"{h['jarak_terbaik']:.3f}" if h['jarak_terbaik'] is not None else "-"
            overlap_str = f"{h.get('tumpang_tindih_kata'):.2f}" if h.get('tumpang_tindih_kata') is not None else "-"
            print(f"\n[PERTANYAAN NOMOR {idx}]")
            print(f"Kategori         : {h['kategori']}")
            print(f"Pertanyaan       : {h['pesan']}")
            print(f"Router IndoBERT  : {h.get('kelas_router')} (keyakinan: {h.get('keyakinan', 0):.4f}) | P(eskalasi): {h.get('peluang_eskalasi', 0):.4f}")
            print(f"Keputusan Router : {h.get('keputusan_router', '-')}")
            print(f"Keputusan Bot    : {h['keputusan']}")
            print(f"Jarak            : {jarak_str} | Overlap: {overlap_str}")
            print(f"Sumber           : {h['sumber'] if h['sumber'] else '-'}")
            print(f"Alasan           : {h['alasan']}")
            print("Teks Jawaban Lengkap:")
            print(h['jawaban'])
            print("-" * 80)

        # --- Tabel Ringkasan Seluruh 28 Pertanyaan ---
        print("\n" + "=" * 80, flush=True)
        print("TABEL RINGKASAN SELURUH 28 PERTANYAAN UJI (T6d INTEGRASI)", flush=True)
        print("=" * 80, flush=True)
        header = f"{'No':<3} | {'Pertanyaan':<30} | {'Kelas Router':<18} | {'P(esk)':<7} | {'Jarak':<6} | {'Ovlp':<5} | {'Keputusan':<16} | {'Alasan Singkat'}"
        print(header, flush=True)
        print("-" * 125, flush=True)

        for h in semua_hasil:
            no_str = str(h['no'])
            pesan_potong = (h['pesan'][:28] + "...") if len(h['pesan']) > 30 else h['pesan']
            kls_str = str(h.get('kelas_router', '-'))
            pesk_str = f"{h.get('peluang_eskalasi', 0):.4f}"
            jarak_str = f"{h['jarak_terbaik']:.3f}" if h['jarak_terbaik'] is not None else "-"
            overlap_str = f"{h.get('tumpang_tindih_kata'):.2f}" if h.get('tumpang_tindih_kata') is not None else "-"
            keputusan_str = h['keputusan']
            alasan_singkat = h['alasan'][:32] + ("..." if len(h['alasan']) > 32 else "")
            baris = f"{no_str:<3} | {pesan_potong:<30} | {kls_str:<18} | {pesk_str:<7} | {jarak_str:<6} | {overlap_str:<5} | {keputusan_str:<16} | {alasan_singkat}"
            print(baris, flush=True)

        print("=" * 125, flush=True)

    finally:
        sys.stdout = tee.terminal
        tee.close()


if __name__ == "__main__":
    jalankan_uji()
