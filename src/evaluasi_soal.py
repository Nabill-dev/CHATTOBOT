"""
src/evaluasi_soal.py — Skrip Evaluasi Soal Uji Siap Pakai (Tahap Penutupan)

Membaca berkas soal uji (eval/soal_uji.csv) dengan format:
  set,pertanyaan,jenis,bagian_panduan,jawaban_diharapkan
  - jenis: "ada" | "tidak_ada" | "eskalasi"
  - keputusan_diharapkan: "ada" -> "jawab"; "tidak_ada" / "eskalasi" -> "teruskan_ke_staf"

Fitur Utama:
1. Menjalankan susunan V1, V2, dan/atau V3 serta router yang dipilih.
2. Menghitung ringkasan:
   - Persen jawab benar pada jenis "ada"
   - Kecocokan bagian panduan untuk jenis "ada"
   - Persen salah dijawab pada jenis "tidak_ada" dan "eskalasi" (Kesalahan Berbahaya
     dengan Wilson 95% interval dan pembilang/penyebut)
   - Breakdown per susunan dan per set.
3. PENGAMAN KETAT SET 'UJI':
   Skrip WAJIB menolak berjalan pada set 'uji' kecuali dipanggil dengan bendera '--akhir',
   disertai peringatan bahwa set 'uji' hanya boleh dijalankan SATU KALI setelah penyetelan selesai.
"""

import sys
import os
import csv
import math
import argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bot import jawab
from config import HASIL_DIR, EVAL_DIR, ROUTER_AKTIF, SUSUNAN_AKTIF


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Menghitung Wilson Score Interval 95% untuk proporsi k/n."""
    if n == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = k / n
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + ((z ** 2) / (4 * (n ** 2))))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower * 100, upper * 100


def periksa_kecocokan_bagian(bagian_panduan: str, bagian_terpilih: str) -> bool:
    """Mengecek apakah bagian dokumen yang dipilih cocok dengan bagian panduan yang diharapkan."""
    if not bagian_panduan or not bagian_terpilih:
        return False
    bp = bagian_panduan.strip().lower()
    bt = bagian_terpilih.strip().lower()
    return bp in bt or bt in bp


def evaluasi_berkas(
    path_csv: Path,
    susunan_list: list[str],
    router_mode: str,
    izinkan_set_uji: bool = False,
    output_csv_path: Path = None,
    output_log_path: Path = None,
    simpan_hasil: bool = True,
) -> dict:
    if not path_csv.exists():
        raise FileNotFoundError(f"Berkas soal evaluasi tidak ditemukan: {path_csv}")

    # 1. Baca berkas CSV (gunakan utf-8-sig untuk menangani BOM Windows jika ada)
    baris_soal = []
    with open(path_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            baris_soal.append({k.strip().replace("\ufeff", ""): v.strip() for k, v in r.items() if k})

    if not baris_soal:
        raise ValueError(f"Berkas {path_csv} kosong atau tidak memiliki baris data.")

    # 2. PENGAMAN KETAT SET 'UJI'
    ada_set_uji = any(b.get("set", "").strip().lower() == "uji" for b in baris_soal)
    if ada_set_uji and not izinkan_set_uji:
        batas = "=" * 80
        pesan_peringatan = (
            f"\n{batas}\n"
            "PERINGATAN KETAT METODOLOGI: Berkas soal memuat set 'uji'!\n"
            "Sesuai aturan penelitian, set 'uji' HANYA boleh dijalankan SATU KALI (SEKALI)\n"
            "setelah seluruh proses penyetelan dan pemilihan susunan selesai secara definitif.\n"
            "Skrip menolak berjalan pada set 'uji' tanpa bendera '--akhir'.\n\n"
            "Jika Anda memang bermaksud menjalankan evaluasi akhir yang sesungguhnya, jalankan:\n"
            "  python src/evaluasi_soal.py --akhir\n"
            f"{batas}\n"
        )
        print(pesan_peringatan, file=sys.stderr)
        sys.exit(1)

    # 3. Jalankan evaluasi per susunan
    semua_hasil = []
    ringkasan_per_susunan = {}

    for susunan in susunan_list:
        hasil_susunan = []
        for idx, item in enumerate(baris_soal, 1):
            pertanyaan = item.get("pertanyaan", "")
            set_nama = item.get("set", "umum")
            jenis = item.get("jenis", "").strip().lower()  # ada | tidak_ada | eskalasi
            bagian_diharapkan = item.get("bagian_panduan", "")
            jawaban_diharapkan = item.get("jawaban_diharapkan", "")

            # Keputusan diharapkan: 'ada' -> jawab, lainnya -> teruskan
            if jenis == "ada":
                keputusan_diharapkan = "jawab"
            else:
                keputusan_diharapkan = "teruskan_ke_staf"

            # Jalankan bot
            res = jawab(pertanyaan, sistem="B", router_mode=router_mode, susunan=susunan)
            keputusan_bot = res.get("keputusan", "teruskan_ke_staf")
            alasan_bot = res.get("alasan", "")
            sumber_bot = res.get("sumber", "")
            kelas_router = res.get("kelas_router", "")
            p_esk = res.get("peluang_eskalasi", 0.0)
            jarak = res.get("jarak_terbaik")
            overlap = res.get("tumpang_tindih_kata")
            skor_gab = res.get("skor_gabungan")

            # Ekstrak bagian terpilih dari metadata potongan terpakai jika ada
            bagian_terpilih = ""
            if res.get("potongan_terpakai") and len(res["potongan_terpakai"]) > 0:
                bagian_terpilih = res["potongan_terpakai"][0].get("metadata", {}).get("judul_bagian", "")
            if not bagian_terpilih and "Bagian:" in sumber_bot:
                try:
                    bagian_terpilih = sumber_bot.split("Bagian:")[1].split(",")[0].strip()
                except Exception:
                    bagian_terpilih = sumber_bot

            # Evaluasi kecocokan bagian untuk jenis 'ada'
            if jenis == "ada":
                if keputusan_bot == "jawab":
                    kecocokan_bagian = periksa_kecocokan_bagian(bagian_diharapkan, bagian_terpilih)
                else:
                    kecocokan_bagian = False
            else:
                kecocokan_bagian = "-"

            # Klasifikasi benar/salah
            tepat_keputusan = (keputusan_bot == keputusan_diharapkan)
            kesalahan_berbahaya = (jenis in ["tidak_ada", "eskalasi"] and keputusan_bot == "jawab")
            eskalasi_berlebih = (jenis == "ada" and keputusan_bot == "teruskan_ke_staf")

            baris_res = {
                "susunan": susunan,
                "router": router_mode,
                "set": set_nama,
                "id": idx,
                "pertanyaan": pertanyaan,
                "jenis": jenis,
                "bagian_diharapkan": bagian_diharapkan,
                "bagian_terpilih": bagian_terpilih,
                "kecocokan_bagian": kecocokan_bagian,
                "keputusan_diharapkan": keputusan_diharapkan,
                "keputusan_bot": keputusan_bot,
                "tepat_keputusan": tepat_keputusan,
                "kesalahan_berbahaya": kesalahan_berbahaya,
                "eskalasi_berlebih": eskalasi_berlebih,
                "kelas_router": kelas_router,
                "peluang_eskalasi": p_esk,
                "jarak_kosinus": jarak if jarak is not None else "",
                "overlap_kata": overlap if overlap is not None else "",
                "skor_gabungan": skor_gab if skor_gab is not None else "",
                "alasan": alasan_bot,
            }
            hasil_susunan.append(baris_res)
            semua_hasil.append(baris_res)

        ringkasan_per_susunan[susunan] = hasil_susunan

    # 4. Susun ringkasan statistik
    log_text = []
    log_text.append("=" * 85)
    log_text.append("LAPORAN EVALUASI SOAL UJI (src/evaluasi_soal.py)")
    log_text.append(f"Waktu Evaluasi     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_text.append(f"Berkas Soal Uji    : {path_csv}")
    log_text.append(f"Router Aktif       : {router_mode}")
    log_text.append(f"Susunan Dievaluasi : {', '.join(susunan_list)}")
    log_text.append(f"Ada Set 'Uji'      : {'Ya (Divalidasi dengan --akhir)' if ada_set_uji else 'Tidak (set non-uji)'}")
    log_text.append("=" * 85 + "\n")

    for susunan, hasil_list in ringkasan_per_susunan.items():
        log_text.append(f"\n{'#' * 85}")
        log_text.append(f"RINGKASAN METRIK EVALUASI — SUSUNAN: {susunan}")
        log_text.append(f"{'#' * 85}")

        # Kelompokkan per set
        sets_unik = sorted(list(set(h["set"] for h in hasil_list)))
        for s_nama in sets_unik:
            sub = [h for h in hasil_list if h["set"] == s_nama]
            total = len(sub)

            # Sub-kelompok berdasarkan jenis
            sub_ada = [h for h in sub if h["jenis"] == "ada"]
            sub_non_ada = [h for h in sub if h["jenis"] in ["tidak_ada", "eskalasi"]]

            # 1. Metrik Jenis 'ADA'
            n_ada = len(sub_ada)
            ada_jawab_benar = sum(1 for h in sub_ada if h["keputusan_bot"] == "jawab")
            ada_bagian_cocok = sum(1 for h in sub_ada if h["kecocokan_bagian"] is True)
            pct_ada_jawab = (ada_jawab_benar / n_ada * 100) if n_ada > 0 else 0.0
            pct_ada_cocok = (ada_bagian_cocok / n_ada * 100) if n_ada > 0 else 0.0

            # 2. Kesalahan Berbahaya (jenis tidak_ada / eskalasi tetapi dijawab 'jawab')
            n_non_ada = len(sub_non_ada)
            kb_count = sum(1 for h in sub_non_ada if h["kesalahan_berbahaya"])
            kb_pct = (kb_count / n_non_ada * 100) if n_non_ada > 0 else 0.0
            kb_low, kb_high = wilson_score_interval(kb_count, n_non_ada)

            # 3. Eskalasi Berlebih (jenis ada tetapi dialihkan)
            esk_berlebih_count = sum(1 for h in sub_ada if h["eskalasi_berlebih"])
            esk_berlebih_pct = (esk_berlebih_count / n_ada * 100) if n_ada > 0 else 0.0

            log_text.append(f"\n--- SET: [{s_nama}] (Total {total} Soal) ---")
            log_text.append(f"  A. Jenis 'ADA' (Dokumen Memuat Jawaban) [N = {n_ada}]:")
            log_text.append(f"     - Dijawab Benar Otomatis   : {pct_ada_jawab:.2f}% ({ada_jawab_benar}/{n_ada})")
            log_text.append(f"     - Bagian Dokumen Cocok     : {pct_ada_cocok:.2f}% ({ada_bagian_cocok}/{n_ada})")
            log_text.append(f"     - Eskalasi Berlebih        : {esk_berlebih_pct:.2f}% ({esk_berlebih_count}/{n_ada})")
            log_text.append(f"  B. Jenis 'TIDAK_ADA' & 'ESKALASI' (Harus Dialihkan) [N = {n_non_ada}]:")
            log_text.append(f"     - Kesalahan Berbahaya (KB) : {kb_pct:.2f}% ({kb_count}/{n_non_ada})")
            log_text.append(f"     - Wilson 95% Confidence    : [{kb_low:.2f}%, {kb_high:.2f}%]")

    log_str = "\n".join(log_text)
    print(log_str)

    # 5. Tulis berkas jika diminta
    if simpan_hasil:
        if output_csv_path is None:
            output_csv_path = HASIL_DIR / "evaluasi.csv"
        if output_log_path is None:
            output_log_path = HASIL_DIR / "log_evaluasi.txt"

        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        output_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Simpan CSV
        fieldnames = [
            "susunan", "router", "set", "id", "pertanyaan", "jenis",
            "bagian_diharapkan", "bagian_terpilih", "kecocokan_bagian",
            "keputusan_diharapkan", "keputusan_bot", "tepat_keputusan",
            "kesalahan_berbahaya", "eskalasi_berlebih", "kelas_router",
            "peluang_eskalasi", "jarak_kosinus", "overlap_kata", "skor_gabungan", "alasan"
        ]
        with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(semua_hasil)

        # Simpan LOG
        with open(output_log_path, "w", encoding="utf-8") as f:
            f.write(log_str)

        print(f"\n[SUKSES] Hasil evaluasi disimpan ke:")
        print(f"  - CSV : {output_csv_path}")
        print(f"  - Log : {output_log_path}")

    return {
        "semua_hasil": semua_hasil,
        "ringkasan": ringkasan_per_susunan,
        "log_text": log_str
    }


def main():
    parser = argparse.ArgumentParser(description="Skrip Evaluasi Soal Uji CHATTOBOT")
    parser.add_argument("--berkas", type=str, default=str(EVAL_DIR / "soal_uji.csv"),
                        help="Path ke berkas CSV soal uji (default: eval/soal_uji.csv)")
    parser.add_argument("--susunan", type=str, default="semua", choices=["V1", "V2", "V3", "semua"],
                        help="Susunan sistem yang diuji ('V1', 'V2', 'V3', atau 'semua')")
    parser.add_argument("--router", type=str, default=ROUTER_AKTIF, choices=["indobertweet", "klasik", "aturan"],
                        help="Router yang aktif untuk evaluasi")
    parser.add_argument("--akhir", action="store_true",
                        help="Bendera konfirmasi bahwa ini adalah uji akhir definitif (membuka pengaman set 'uji')")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Path kustom untuk keluaran evaluasi.csv")
    parser.add_argument("--output_log", type=str, default=None,
                        help="Path kustom untuk keluaran log_evaluasi.txt")

    args = parser.parse_args()

    path_csv = Path(args.berkas)
    if args.susunan.lower() == "semua":
        susunan_list = ["V1", "V2", "V3"]
    else:
        susunan_list = [args.susunan.upper()]

    output_csv = Path(args.output_csv) if args.output_csv else None
    output_log = Path(args.output_log) if args.output_log else None

    evaluasi_berkas(
        path_csv=path_csv,
        susunan_list=susunan_list,
        router_mode=args.router,
        izinkan_set_uji=args.akhir,
        output_csv_path=output_csv,
        output_log_path=output_log,
        simpan_hasil=True,
    )


if __name__ == "__main__":
    main()
