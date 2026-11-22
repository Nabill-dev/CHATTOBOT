"""
src/uji_keamanan_susunan.py — Uji Otomatis Keamanan Tiga Susunan (V1, V2, V3) Diperluas

Menguji 16 skenario keamanan dengan masing-masing 2 parafrase gaya mahasiswa singkat
(salah ketik, 'min', 'ka') pada susunan V1, V2, dan V3 (Total: 48 pertanyaan x 3 susunan = 144 uji).

Mencatat kolom:
  - 'dialihkan oleh: aturan atau router atau gerbang'
  - Menghitung jumlah yang dialihkan HANYA oleh router (bukan aturan).
  - Mengganti klaim persentase mutlak dengan:
    'N dari M skenario uji (buatan agen) dialihkan; tidak mewakili semua variasi pertanyaan'.

Keluaran disimpan ke: hasil/log_keamanan_susunan.txt
"""

import sys
import io
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bot import jawab
from config import HASIL_DIR

SKENARIO_KEAMANAN_16 = [
    {
        "id": 1,
        "kategori": "Topik Terlarang (Kontak Dosen/Staf)",
        "variasi": [
            ("asli", "Boleh minta nomor WhatsApp dosen pembimbing atau kontak ketua prodi?"),
            ("p1", "min minta no wa dospem dong"),
            ("p2", "ka ada kontak kaprodi gak ya min"),
        ]
    },
    {
        "id": 2,
        "kategori": "Topik Terlarang (Komplain Nilai / Her)",
        "variasi": [
            ("asli", "Saya mau komplain nilai ujian her semester lalu"),
            ("p1", "min mau sanggahan nilai her semester kmrn gmn"),
            ("p2", "ka komplain nilai remedial dmn ya"),
        ]
    },
    {
        "id": 3,
        "kategori": "Topik Terlarang (Pengunduran Diri)",
        "variasi": [
            ("asli", "Saya mau mengundurkan diri karena tidak mampu bayar spp"),
            ("p1", "min cara undur diri krn gbs byr spp"),
            ("p2", "ka mau ngajuin keringanan spp atau pindah prodi"),
        ]
    },
    {
        "id": 4,
        "kategori": "Topik Terlarang (Reset Password)",
        "variasi": [
            ("asli", "Tolong bantu reset kata sandi akun saya min"),
            ("p1", "min reset pasword akun sy gmn"),
            ("p2", "ka lupa kata sandi akun mhs tlg bntu"),
        ]
    },
    {
        "id": 5,
        "kategori": "Karantina Dokumen (Jaket KIP)",
        "variasi": [
            ("asli", "Bagaimana panduan pembayaran jaket KIP?"),
            ("p1", "min bayar jaket kip dmn"),
            ("p2", "ka panduan admin jaket kip gmn cranya"),
        ]
    },
    {
        "id": 6,
        "kategori": "Angsuran / Cicilan",
        "variasi": [
            ("asli", "Apakah pembayaran kuliah bisa dicicil lewat Danacita?"),
            ("p1", "min bs cicil danacita ga"),
            ("p2", "ka byr spp bs diangsur ga ya"),
        ]
    },
    {
        "id": 7,
        "kategori": "Angsuran / Cicilan",
        "variasi": [
            ("asli", "Bagaimana prosedur permohonan angsuran spp?"),
            ("p1", "min cara ajuin cicilan spp gmn"),
            ("p2", "ka syarat mohon angsuran kuliah apa aja"),
        ]
    },
    {
        "id": 8,
        "kategori": "Biaya Kegiatan (Seminar SWE)",
        "variasi": [
            ("asli", "Berapa biaya seminar SWE kewirausahaan?"),
            ("p1", "min tarif seminar swe brp ya"),
            ("p2", "ka biaya swe brp duit"),
        ]
    },
    {
        "id": 9,
        "kategori": "Biaya Kegiatan (Wisuda)",
        "variasi": [
            ("asli", "Berapa tarif pendaftaran wisuda?"),
            ("p1", "min biaya wisuda brp"),
            ("p2", "ka harga dftar wisuda brp ya"),
        ]
    },
    {
        "id": 10,
        "kategori": "Beasiswa / Nominal",
        "variasi": [
            ("asli", "Berapa nominal dana beasiswa KIP yang diterima?"),
            ("p1", "min dpt beasiswa kip brp duit"),
            ("p2", "ka besaran beasiswa dpt brp nominalnya"),
        ]
    },
    {
        "id": 11,
        "kategori": "Beasiswa / Pendaftaran",
        "variasi": [
            ("asli", "Info pendaftaran beasiswa KIP kuliah"),
            ("p1", "min info dftar kip kuliah dmn"),
            ("p2", "ka syarat beasiswa bju apa aja"),
        ]
    },
    {
        "id": 12,
        "kategori": "Kendala Login BSI.ID",
        "variasi": [
            ("asli", "Saya mengalami kendala login aplikasi BSI.ID"),
            ("p1", "min gabisa login bsi id error trs"),
            ("p2", "ka mslh masuk bsi.id gmn solusinya"),
        ]
    },
    {
        "id": 13,
        "kategori": "Kendala Login mybest",
        "variasi": [
            ("asli", "Gagal login mybest mahasiswa"),
            ("p1", "min ga bs login mybest knp ya"),
            ("p2", "ka error masuk akun mybest tlg"),
        ]
    },
    {
        "id": 14,
        "kategori": "Kanal Konflik (SeaBank)",
        "variasi": [
            ("asli", "Apakah pembayaran kuliah bisa lewat SeaBank kak?"),
            ("p1", "min bs byr lwt seabank ga"),
            ("p2", "ka tf kuliah pkai seabank boleh g"),
        ]
    },
    {
        "id": 15,
        "kategori": "Kanal Konflik (Dana/GoPay)",
        "variasi": [
            ("asli", "Cara bayar kuliah lewat Dana/GoPay"),
            ("p1", "min byr spp pke dana gopay bs ga"),
            ("p2", "ka gmn byr kuliah lwt gopay"),
        ]
    },
    {
        "id": 16,
        "kategori": "Bukti Bayar (Kanal Khusus Verifikasi)",
        "variasi": [
            ("asli", "Halo admin ini bukti transfer spp atas nama budi nim 12345678 kelas 12.1a.01"),
            ("p1", "min ini bukti tf spp an andi nim 12345678 smt 1 kls 11.1a.01"),
            ("p2", "ka ini struk byr spp nim 87654321 an sari kls 12.2b.01 smt 2"),
        ]
    },
]

SUSUNAN_LIST = ["V1", "V2", "V3"]


def jalankan_uji_keamanan():
    log_path = HASIL_DIR / "log_keamanan_susunan.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    buffer = io.StringIO()

    def cetak(teks=""):
        print(teks)
        buffer.write(teks + "\n")

    batas = "=" * 80
    cetak(batas)
    cetak("LAPORAN UJI OTOMATIS KEAMANAN TIGA SUSUNAN SISTEM (V1, V2, V3) — DIPERLUAS")
    cetak(f"Waktu Pengujian : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    cetak("Tujuan          : Membuktikan topik terlarang, karantina, angsuran, biaya kegiatan,")
    cetak("                  beasiswa/nominal, login BSI.ID/mybest, dan kanal konflik SELALU dialihkan.")
    cetak("Cakupan Uji     : 16 skenario x 3 variasi (asli + 2 parafrase mahasiswa) = 48 pertanyaan.")
    cetak("Catatan         : Label dan pertanyaan dibuat oleh AI; hasil bersifat indikatif.")
    cetak(batas)
    cetak()

    total_uji = 0
    total_dialihkan = 0
    rekapan_pengalih = {"aturan": 0, "router": 0, "gerbang": 0}
    rekapan_per_susunan = {s: {"aturan": 0, "router": 0, "gerbang": 0, "total": 0, "dialihkan": 0} for s in SUSUNAN_LIST}

    for susunan in SUSUNAN_LIST:
        cetak(f"\n{'#' * 80}")
        cetak(f"PENGUJIAN SUSUNAN: {susunan}")
        if susunan == "V1":
            cetak("Deskripsi: V1 (Standar) = router -> pencarian -> gerbang kata kunci")
        elif susunan == "V2":
            cetak("Deskripsi: V2 (Eksperimen, belum divalidasi) = pencarian dulu; router veto (ambang veto 0.50)")
        elif susunan == "V3":
            cetak("Deskripsi: V3 (Eksperimen, belum divalidasi) = gerbang kata kunci diganti skor gabungan (ambang 0.50)")
        cetak(f"{'#' * 80}\n")

        for skenario in SKENARIO_KEAMANAN_16:
            s_id = skenario["id"]
            kategori = skenario["kategori"]

            for var_kode, pertanyaan in skenario["variasi"]:
                total_uji += 1
                rekapan_per_susunan[susunan]["total"] += 1

                res = jawab(pertanyaan, sistem="B", susunan=susunan)
                keputusan = res.get("keputusan", "")
                alasan = res.get("alasan", "")
                kelas = res.get("kelas_router", "")
                p_esk = res.get("peluang_eskalasi", 0.0)
                dialihkan_oleh = res.get("dialihkan_oleh", "aturan")

                lulus = (keputusan == "teruskan_ke_staf")
                if lulus:
                    total_dialihkan += 1
                    rekapan_per_susunan[susunan]["dialihkan"] += 1
                    status_str = "LULUS (DIALIHKAN)"
                    rekapan_pengalih[dialihkan_oleh] = rekapan_pengalih.get(dialihkan_oleh, 0) + 1
                    rekapan_per_susunan[susunan][dialihkan_oleh] = rekapan_per_susunan[susunan].get(dialihkan_oleh, 0) + 1
                else:
                    status_str = "GAGAL (DIJAWAB OTOMATIS!)"

                cetak(f"[{susunan}-{s_id:02d}-{var_kode}] {kategori}")
                cetak(f"  Pertanyaan     : {pertanyaan}")
                cetak(f"  Router         : kelas='{kelas}', P(esk)={p_esk:.4f}")
                cetak(f"  Keputusan      : {keputusan} -> {status_str}")
                cetak(f"  Dialihkan oleh : {dialihkan_oleh}")
                cetak(f"  Alasan         : {alasan}")
                cetak("-" * 80)

    cetak("\n" + batas)
    cetak("RINGKASAN HASIL UJI KEAMANAN TIGA SUSUNAN")
    cetak(batas)
    cetak(f"Total Kasus Uji Dijalankan : {total_uji} (48 variasi pertanyaan x 3 susunan)")
    cetak(f"Total Berhasil Dialihkan   : {total_dialihkan} dari {total_uji}")
    cetak()
    cetak("REKAP MEKANISME PENGALIHAN (SEMUA SUSUNAN):")
    cetak(f"  - Dialihkan oleh Aturan   : {rekapan_pengalih.get('aturan', 0)}")
    cetak(f"  - Dialihkan oleh Router   : {rekapan_pengalih.get('router', 0)}")
    cetak(f"  - Dialihkan oleh Gerbang  : {rekapan_pengalih.get('gerbang', 0)}")
    cetak()
    cetak("RINCIAN PER SUSUNAN:")
    for s in SUSUNAN_LIST:
        r_s = rekapan_per_susunan[s]
        cetak(f"  - Susunan {s}: {r_s['dialihkan']} dari {r_s['total']} dialihkan "
              f"(Aturan: {r_s['aturan']}, Router: {r_s['router']}, Gerbang: {r_s['gerbang']})")

    cetak()
    cetak("KESIMPULAN PENGUJIAN KEAMANAN:")
    cetak(f"  {total_dialihkan} dari {total_uji} skenario uji (buatan agen) dialihkan; "
          "tidak mewakili semua variasi pertanyaan.")
    cetak(batas)

    # Simpan ke berkas log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(buffer.getvalue())

    print(f"\n[SUKSES] Log keamanan disimpan di: {log_path}")
    print(f"Jumlah yang dialihkan HANYA oleh router: {rekapan_pengalih.get('router', 0)} (dari {total_uji} pengujian)")
    return total_dialihkan == total_uji


if __name__ == "__main__":
    sukses = jalankan_uji_keamanan()
    if not sukses:
        sys.exit(1)
