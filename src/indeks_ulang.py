# ==============================================================
# indeks_ulang.py — Membangun ulang indeks Chroma dari nol
#
# Cara pakai:
#   1. Taruh dokumen baru di data\dokumen\resmi\
#   2. Tambah satu baris di data\dokumen\katalog_dokumen.csv
#   3. Jalankan: python src\indeks_ulang.py
#
# Skrip ini akan:
#   - Membaca semua dokumen berstatus 'aktif'
#   - Memecah jadi potongan teks hierarkis
#   - Menghitung embedding (intfloat/multilingual-e5-small)
#   - Membangun ulang indeks Chroma dari nol
#   - Menulis laporan lengkap ke hasil\log_indeks.txt
# ==============================================================

import sys
import io
import csv
import random
import re
from datetime import date, datetime
from pathlib import Path

# Atur encoding output agar aman di Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Tambah path src agar bisa impor modul lain
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DOKUMEN_RESMI_DIR, DOKUMEN_WAWANCARA, FAQ_DIR, KATALOG_CSV,
    CHROMA_DIR, EMBEDDING_MODEL, EMBEDDING_PREFIX_PASSAGE,
    TANGGAL_HARI_INI, HASIL_DIR, SEED,
)
from pemroses_dokumen import proses_dokumen


class OutputTee:
    """Mencatat keluaran stdout ke terminal sekaligus ke berkas log."""
    def __init__(self, file_path: Path):
        self.terminal = sys.stdout
        self.log_file = open(file_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return getattr(self.terminal, "isatty", lambda: False)()

    def close(self):
        self.log_file.close()


def baca_katalog(path_csv: Path) -> list[dict]:
    """Membaca katalog_dokumen.csv dan mengembalikan list of dict."""
    if not path_csv.exists():
        print(f"[GALAT] Katalog tidak ditemukan: {path_csv}")
        return []

    with open(path_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


def cari_path_dokumen(nama_berkas: str, jenis: str) -> Path | None:
    """Mencari path file berdasarkan nama dan jenis."""
    if jenis == 'resmi':
        path = DOKUMEN_RESMI_DIR / nama_berkas
    elif jenis == 'wawancara':
        path = DOKUMEN_WAWANCARA / nama_berkas
    elif jenis == 'faq':
        path = FAQ_DIR / nama_berkas
    else:
        path = DOKUMEN_RESMI_DIR / nama_berkas

    return path if path.exists() else None


def cek_kedaluwarsa(tgl_str: str) -> bool:
    """Mengecek apakah tanggal kedaluwarsa sudah lewat."""
    if not tgl_str or tgl_str.strip() == '':
        return False

    try:
        tgl = datetime.strptime(tgl_str.strip(), '%Y-%m-%d').date()
        return tgl < TANGGAL_HARI_INI
    except ValueError:
        print(f"  [PERINGATAN] Format tanggal tidak dikenali: '{tgl_str}'")
        return False


def cek_kesesuaian_judul_isi(potongan_list: list[dict]) -> list[str]:
    """
    Mengecek apakah ada anomali judul tidak cocok dengan isi teks potongan.
    Contoh: Judul 'Admin Jaket KIP' tetapi isinya tentang 'Keperawatan'.
    """
    peringatan = []
    kamus_cek = {
        'jaket kip': (['kip', 'jaket', 'almamater'], ['keperawatan', 'kursus']),
        'kursus bahasa': (['kursus', 'bahasa', 'inggris'], ['keperawatan', 'jaket']),
        'keperawatan': (['keperawatan', 'perawat'], ['jaket', 'kursus bahasa']),
        'wisuda': (['wisuda', 'kelulusan'], ['seminar swe', 'ujian her']),
        'seminar swe': (['seminar', 'swe', 'entrepreneur'], ['wisuda']),
        'cuti': (['cuti'], ['mutasi']),
        'mutasi': (['mutasi'], ['cuti']),
        'ujian her': (['her', 'perbaikan'], ['wisuda']),
        'danacita': (['danacita', 'cicilan'], []),
        'indomaret': (['indomaret'], ['briva']),
        'bri': (['bri', 'briva'], ['indomaret']),
        'angsuran': (['angsuran', 'cicil'], []),
    }

    for p in potongan_list:
        meta = p['metadata']
        judul = meta.get('judul_bagian', '').lower()
        teks = p['teks'].lower()
        chunk_id = p['id']

        for kata_kunci_judul, (kata_harus_ada, kata_kontradiksi) in kamus_cek.items():
            if kata_kunci_judul in judul:
                ada = any(k in teks for k in kata_harus_ada)
                kontradiksi = [k for k in kata_kontradiksi if k in teks]
                if not ada:
                    peringatan.append(
                        f"Potongan {chunk_id} (Judul: '{meta.get('judul_bagian')}') "
                        f"TIDAK memuat kata kunci yang diharapkan ({kata_harus_ada})."
                    )
                if kontradiksi and not ada:
                    peringatan.append(
                        f"Potongan {chunk_id} (Judul: '{meta.get('judul_bagian')}') "
                        f"memuat topik kontradiktif: {kontradiksi} padahal kata kunci utama tidak ditemukan."
                    )
    return peringatan


def bangun_indeks():
    """Fungsi utama: baca katalog, proses dokumen, bangun indeks Chroma, dan catat log."""
    log_path = HASIL_DIR / "log_indeks.txt"
    tee = OutputTee(log_path)
    old_stdout = sys.stdout
    sys.stdout = tee

    try:
        print("=" * 70)
        print("MEMBANGUN ULANG INDEKS CHROMA")
        print(f"Tanggal: {TANGGAL_HARI_INI}")
        print("=" * 70)

        # 1. Baca katalog
        katalog = baca_katalog(KATALOG_CSV)
        if not katalog:
            print("[GALAT] Katalog kosong atau tidak ditemukan. Berhenti.")
            return

        print(f"\nDokumen di katalog: {len(katalog)}")

        semua_potongan = []
        laporan_skip = []

        for dok in katalog:
            doc_id = dok['id']
            nama = dok['nama_berkas']
            status = dok.get('status', '').strip().lower()
            verifikasi = dok.get('diverifikasi_admin', '').strip().lower()
            jenis = dok.get('jenis', '').strip().lower()
            tgl_kadaluarsa = dok.get('tanggal_kedaluwarsa', '').strip()

            print(f"\n--- {doc_id}: {nama} ---")

            if status != 'aktif':
                alasan = f"Status bukan aktif ({status})"
                print(f"  [SKIP] {alasan}")
                laporan_skip.append((doc_id, nama, alasan))
                continue

            # Dokumen resmi aktif boleh diindeks meskipun diverifikasi_admin='tidak' (DOC005)
            # Dokumen non-resmi (wawancara/faq) wajib diverifikasi_admin='ya'
            if jenis != 'resmi' and verifikasi != 'ya':
                alasan = f"Bukan dokumen resmi dan belum diverifikasi admin ({verifikasi})"
                print(f"  [SKIP] {alasan}")
                laporan_skip.append((doc_id, nama, alasan))
                continue

            if cek_kedaluwarsa(tgl_kadaluarsa):
                alasan = f"Sudah kedaluwarsa ({tgl_kadaluarsa})"
                print(f"  [SKIP] {alasan}")
                laporan_skip.append((doc_id, nama, alasan))
                continue

            path_file = cari_path_dokumen(nama, jenis)
            if path_file is None:
                alasan = "File tidak ditemukan di folder"
                print(f"  [SKIP] {alasan}")
                laporan_skip.append((doc_id, nama, alasan))
                continue

            try:
                potongan = proses_dokumen(path_file, doc_id, dok, ukuran_potongan=1800)
                print(f"  [OK] {len(potongan)} potongan dihasilkan dari {nama}")
                semua_potongan.extend(potongan)
            except Exception as e:
                alasan = f"Galat saat memproses: {e}"
                print(f"  [GALAT] {alasan}")
                laporan_skip.append((doc_id, nama, alasan))

        print(f"\n{'=' * 70}")
        print(f"Total potongan yang akan diindeks: {len(semua_potongan)}")

        if not semua_potongan:
            print("[PERINGATAN] Tidak ada potongan untuk diindeks. Berhenti.")
            return

        # 2. Cek kesesuaian judul dengan isi
        print("\nMemeriksa kesesuaian judul dengan isi potongan...")
        peringatan_mismatch = cek_kesesuaian_judul_isi(semua_potongan)
        if peringatan_mismatch:
            print(f"[PERINGATAN] Ditemukan {len(peringatan_mismatch)} potensi ketidakcocokan judul-isi:")
            for p_warn in peringatan_mismatch:
                print(f"  ⚠️ {p_warn}")
        else:
            print("  [OK] Tidak ada ketidakcocokan judul dan isi yang terdeteksi.")

        # 3. Hitung embedding
        print("\nMemuat model embedding...")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"Model dimuat: {EMBEDDING_MODEL}")

        teks_list = [EMBEDDING_PREFIX_PASSAGE + p['teks'] for p in semua_potongan]

        print(f"Menghitung embedding untuk {len(teks_list)} potongan...")
        embeddings = model.encode(
            teks_list,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,
        )
        print(f"Embedding selesai. Dimensi: {embeddings.shape}")

        # 4. Bangun indeks Chroma
        print("\nMembangun indeks Chroma...")
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        try:
            client.delete_collection("dokumen_kampus")
            print("  Koleksi 'dokumen_kampus' lama berhasil dibersihkan.")
        except Exception:
            pass

        collection = client.create_collection(
            name="dokumen_kampus",
            metadata={"hnsw:space": "cosine"},
        )

        ids = [p['id'] for p in semua_potongan]
        documents = [p['teks'] for p in semua_potongan]
        metadatas = []
        for p in semua_potongan:
            meta_clean = {}
            for k, v in p['metadata'].items():
                if v is None:
                    meta_clean[k] = ''
                else:
                    meta_clean[k] = str(v)
            metadatas.append(meta_clean)

        embeddings_list = embeddings.tolist()

        BATCH = 100
        for i in range(0, len(ids), BATCH):
            j = min(i + BATCH, len(ids))
            collection.add(
                ids=ids[i:j],
                embeddings=embeddings_list[i:j],
                documents=documents[i:j],
                metadatas=metadatas[i:j],
            )

        print(f"  {collection.count()} potongan berhasil dimasukkan ke indeks Chroma.")
        print(f"  Indeks tersimpan di: {CHROMA_DIR}")

        # 5. Daftar Judul Bagian
        print(f"\n{'=' * 70}")
        print("DAFTAR JUDUL BAGIAN DOKUMEN AKTIF")
        print(f"{'=' * 70}")
        for idx_p, p in enumerate(semua_potongan):
            meta = p['metadata']
            print(f"{idx_p+1:02d}. [{meta['id']}] {meta['jejak_judul']} (Biaya Rp: {meta['memuat_biaya']}, 3rd: {meta['pihak_ketiga'] or '-'})")

        # 6. 8 Potongan Acak Utuh
        print(f"\n{'=' * 70}")
        print("8 POTONGAN ACAK UTUH (REPRODUCIBLE SEED=42)")
        print(f"{'=' * 70}")
        rng = random.Random(SEED)
        sampel_8 = rng.sample(semua_potongan, min(8, len(semua_potongan)))

        for idx_s, s in enumerate(sampel_8, 1):
            meta = s['metadata']
            print(f"\n--- POTONGAN ACAK #{idx_s} ---")
            print(f"ID           : {meta['id']}")
            print(f"Jejak Judul  : {meta['jejak_judul']}")
            print(f"Pihak Ketiga : {meta['pihak_ketiga'] or '-'}")
            print(f"Memuat Biaya : {meta['memuat_biaya']}")
            print("Teks Utuh:")
            print(s['teks'])
            print("-" * 50)

        # 7. Ringkasan
        print(f"\n{'=' * 70}")
        print("RINGKASAN INDEKSASI")
        print(f"{'=' * 70}")
        print(f"Dokumen di katalog : {len(katalog)}")
        print(f"Dokumen diskip     : {len(laporan_skip)}")
        print(f"Potongan diindeks  : {collection.count()}")
        if laporan_skip:
            print("\nRincian dokumen diskip:")
            for doc_id, nama, alasan in laporan_skip:
                print(f"  - {doc_id} ({nama}): {alasan}")
        print("\nSelesai dengan sukses!")

    finally:
        sys.stdout = old_stdout
        tee.close()


if __name__ == "__main__":
    bangun_indeks()
