# ==============================================================
# pemroses_dokumen.py — Membaca dokumen & memecah jadi potongan
#
# Strategi pemotongan:
#   1. Memecah dokumen per bagian / judul panduan hierarkis (jejak judul induk).
#   2. Menjaga teks utuh di batas kalimat (tidak terpotong di tengah kalimat).
#   3. Membawa jejak judul induk di metadata dan teks yang di-embed.
#   4. Menandai metadata pihak ketiga dan informasi biaya (Rp).
#   5. Tidak ada potongan yang hanya berisi judul tanpa isi.
# ==============================================================

import re
from pathlib import Path
from docx import Document


def ekstrak_tanggal_dari_teks(teks: str) -> dict:
    """
    Mencari tanggal buka dan tutup di teks bagian.
    Format yang dicari: "Tanggal Buka: DD-MM-YYYY" dan "Tanggal Tutup: DD-MM-YYYY".
    Mengembalikan dict {'tanggal_buka': str, 'tanggal_tutup': str, 'status': str}.
    """
    hasil = {'tanggal_buka': '', 'tanggal_tutup': '', 'status': ''}

    m_buka = re.search(r'Tanggal Buka:\s*(\d{2}-\d{2}-\d{4})', teks)
    if m_buka:
        hasil['tanggal_buka'] = m_buka.group(1)

    m_tutup = re.search(r'Tanggal Tutup:\s*(\d{2}-\d{2}-\d{4})', teks)
    if m_tutup:
        hasil['tanggal_tutup'] = m_tutup.group(1)

    m_status = re.search(r'Status:\s*(TERBUKA|DITUTUP|PEMBAYARAN DITUTUP)', teks, re.IGNORECASE)
    if m_status:
        hasil['status'] = m_status.group(1).upper()
    elif 'PEMBAYARAN DITUTUP' in teks.upper():
        hasil['status'] = 'PEMBAYARAN DITUTUP'
    elif 'TERBUKA' in teks.upper() and 'status' in teks.lower():
        hasil['status'] = 'TERBUKA'

    return hasil


def cek_pihak_ketiga(jejak_judul: str, teks: str) -> str:
    """
    Menandai metadata pihak ketiga untuk:
    E-Wallet/BSI PAY, isi saldo, biaya transfer, FAQ, kontak MotionPay, Cicilan Danacita.
    """
    gabungan = (jejak_judul + " " + teks).lower()
    kata_kunci = [
        'e-wallet', 'bsi pay', 'bsi-pay', 'motionpay', 'isi saldo',
        'top up', 'biaya transfer', 'faq', 'pusat bantuan',
        'cs@motionpay.id', 'danacita'
    ]
    if any(k in gabungan for k in kata_kunci):
        # Pastikan relevan dengan layanan pihak ketiga
        if any(k in jejak_judul.lower() for k in ['e-wallet', 'bsi pay', 'faq', 'danacita', 'isi saldo']) or ('motionpay' in gabungan) or ('danacita' in gabungan):
            return "informasi pihak ketiga, dapat berubah"
    return ""


def cek_memuat_biaya(teks: str) -> str:
    """Menandai apakah potongan memuat angka biaya (Rp)."""
    if re.search(r'Rp\s*[\d\.]+', teks, re.IGNORECASE):
        return "ya"
    return "tidak"


def baca_docx_hierarkis(path_file: str | Path) -> list[dict]:
    """
    Membaca dokumen Word (.docx) secara hierarkis (paragraf & tabel).
    Memecah per bagian/judul panduan dengan jejak judul induk.
    Menghilangkan bagian yang hanya berupa judul tanpa isi.
    """
    doc = Document(str(path_file))
    
    # Pola judul utama level 1 (misal "1. ...", "7. ...", hindari langkah bernomor seperti "7. Validasi...")
    pola_level1 = re.compile(
        r'^[1-7]\.\s+(Autentikasi|Beranda|Menu\s+|Pusat Bantuan|Sidebar\s+|Tutorial\s+).*',
        re.IGNORECASE
    )
    
    # Pola judul seksi level 2
    pola_level2 = re.compile(
        r'^(Petunjuk Pembayaran|Panduan & Informasi|Tata Cara Pembayaran|Tata Cara Bayar|'
        r'Info Terkini|Terkait Menu|Petunjuk Histori|Rincian Kategori|Tab Pengumuman|'
        r'Respon Pop-Up|Menu XPLORE|Panduan Isi Saldo|Pusat Bantuan|Navigasi Bawah|'
        r'Kategori NAVIGATION|Kategori TERKINI|Kategori INFO APP)',
        re.IGNORECASE
    )
    
    # Pola sub-kanal pembayaran level 3
    pola_level3 = re.compile(
        r'^(Melalui |Pembayaran dari )',
        re.IGNORECASE
    )

    seksi_list = []
    curr_major = ""
    curr_sub = ""
    curr_title = "Pendahuluan"
    curr_trail = "Pendahuluan"
    curr_body = []

    def simpan_blok():
        nonlocal curr_body
        body_text = "\n".join(curr_body).strip()
        # Jangan simpan potongan yang hanya judul atau kosong (< 30 karakter)
        if body_text and len(body_text) >= 30:
            seksi_list.append({
                'judul_induk': curr_major if curr_major else curr_title,
                'judul_bagian': curr_title,
                'jejak_judul': curr_trail,
                'teks_isi': body_text
            })
        curr_body = []

    for child in doc.element.body:
        tag = child.tag.split('}')[-1]
        if tag == 'p':
            p = Document.Paragraph(child, doc) if hasattr(Document, 'Paragraph') else doc.paragraphs[0].__class__(child, doc)
            teks = p.text.strip()
            if not teks or teks.startswith('―――'):
                continue

            # Standardisasi bullet
            for ch in ['\uf0b7', '\uf0a7', '\u2022', '\u25cf']:
                teks = teks.replace(ch, '- ')

            # Paragraf bullet / rincian tidak boleh dianggap judul seksi
            is_bullet = bool(p.style and 'bullet' in p.style.name.lower())
            if is_bullet:
                curr_body.append(teks)
                continue

            m_lvl1 = pola_level1.match(teks)
            m_lvl3 = pola_level3.match(teks)
            m_lvl2 = pola_level2.match(teks)

            if m_lvl1:
                simpan_blok()
                curr_major = teks
                curr_sub = ""
                curr_title = teks
                curr_trail = teks
            elif m_lvl3:
                simpan_blok()
                curr_title = teks
                parent_ctx = curr_sub if curr_sub else curr_major
                if curr_major and curr_sub and curr_sub != curr_major:
                    curr_trail = f"{curr_major} > {curr_sub} > {teks}"
                elif curr_major:
                    curr_trail = f"{curr_major} > {teks}"
                else:
                    curr_trail = teks
            elif m_lvl2:
                simpan_blok()
                curr_sub = teks
                curr_title = teks
                if curr_major:
                    curr_trail = f"{curr_major} > {teks}"
                else:
                    curr_trail = teks
            else:
                curr_body.append(teks)

        elif tag == 'tbl':
            tbl = Document.Table(child, doc) if hasattr(Document, 'Table') else doc.tables[0].__class__(child, doc)
            tbl_lines = []
            for r in tbl.rows:
                row_txt = ' | '.join(c.text.strip().replace('\n', ' ') for c in r.cells)
                tbl_lines.append(row_txt)
            curr_body.append('\n'.join(tbl_lines))

    simpan_blok()
    return seksi_list


def pecah_teks_di_batas_kalimat(teks: str, batas_maks: int = 1400) -> list[str]:
    """
    Memastikan teks potongan utuh di batas kalimat (titik/baris baru).
    Jika panjang teks <= batas_maks, dipertahankan utuh.
    """
    teks = teks.strip()
    if not teks:
        return []

    if len(teks) <= batas_maks:
        return [teks]

    paragraf_list = [p.strip() for p in teks.split('\n') if p.strip()]
    potongan = []
    buffer_sekarang = []
    panjang_sekarang = 0

    for par in paragraf_list:
        if panjang_sekarang + len(par) + 1 <= batas_maks or not buffer_sekarang:
            buffer_sekarang.append(par)
            panjang_sekarang += len(par) + 1
        else:
            potongan.append("\n".join(buffer_sekarang))
            buffer_sekarang = [par]
            panjang_sekarang = len(par)

    if buffer_sekarang:
        potongan.append("\n".join(buffer_sekarang))

    return potongan


def proses_dokumen(
    path_file: str | Path,
    doc_id: str,
    metadata_katalog: dict,
    ukuran_potongan: int = 1400,
    tumpang_tindih: int = 0,
) -> list[dict]:
    """
    Pipeline lengkap pemrosesan dokumen:
      - Baca docx secara hierarkis per bagian panduan
      - Membawa jejak judul induk di metadata dan teks yang di-embed
      - Tidak ada potongan yang hanya berisi judul
      - Menandai metadata pihak ketiga dan memuat_biaya (Rp)
      - Ekstrak tanggal buka/tutup dan status kegiatan
    """
    path_file = Path(path_file)
    if not path_file.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path_file}")

    seksi_list = baca_docx_hierarkis(path_file)
    semua_potongan = []
    nomor_global = 0

    for idx_bagian, seksi in enumerate(seksi_list):
        judul_induk = seksi['judul_induk']
        judul_bagian = seksi['judul_bagian']
        jejak_judul = seksi['jejak_judul']
        teks_isi = seksi['teks_isi']

        # Pecah jika melebihi ukuran batas, selalu di batas kalimat/baris
        sub_potongan = pecah_teks_di_batas_kalimat(teks_isi, batas_maks=ukuran_potongan)

        for idx_pot, pot_teks in enumerate(sub_potongan):
            nomor_global += 1
            chunk_id = f"{doc_id}_b{idx_bagian:02d}_p{idx_pot:02d}"

            info_tanggal = ekstrak_tanggal_dari_teks(pot_teks)
            pihak_ketiga = cek_pihak_ketiga(jejak_judul, pot_teks)
            memuat_biaya = cek_memuat_biaya(pot_teks)

            # Teks yang di-embed membawa konteks jejak judul induk di awal
            teks_lengkap = f"[Bagian: {jejak_judul}]\n{pot_teks}"

            metadata = {
                'id': chunk_id,
                'doc_id': doc_id,
                'nama_berkas': path_file.name,
                'judul_induk': judul_induk,
                'judul_bagian': judul_bagian,
                'jejak_judul': jejak_judul,
                'bagian_ke': idx_bagian,
                'potongan_ke': idx_pot,
                'nomor_global': nomor_global,
                'pihak_ketiga': pihak_ketiga,
                'memuat_biaya': memuat_biaya,
                # Dari katalog
                'jenis': metadata_katalog.get('jenis', ''),
                'sumber': metadata_katalog.get('sumber', ''),
                'tanggal_berlaku': metadata_katalog.get('tanggal_berlaku', ''),
                'tanggal_kedaluwarsa': metadata_katalog.get('tanggal_kedaluwarsa', ''),
                'status': metadata_katalog.get('status', ''),
                'diverifikasi_admin': metadata_katalog.get('diverifikasi_admin', ''),
                'berlaku_untuk': metadata_katalog.get('berlaku_untuk', ''),
                # Dari teks
                'tanggal_buka': info_tanggal.get('tanggal_buka', ''),
                'tanggal_tutup': info_tanggal.get('tanggal_tutup', ''),
                'status_kegiatan': info_tanggal.get('status', ''),
            }

            semua_potongan.append({
                'id': chunk_id,
                'teks': teks_lengkap,
                'metadata': metadata,
            })

    return semua_potongan


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    from config import DOKUMEN_RESMI_DIR

    path = DOKUMEN_RESMI_DIR / "Penjelasan_Fitur_EKTM_BSI_ID_Lengkap.docx"
    meta = {
        'jenis': 'resmi',
        'sumber': 'admin kampus',
        'tanggal_berlaku': '2026-11-04',
        'tanggal_kedaluwarsa': '',
        'status': 'aktif',
        'diverifikasi_admin': 'tidak',
        'berlaku_untuk': 'semua angkatan',
    }

    potongan = proses_dokumen(path, "DOC005", meta)
    print(f"Total potongan yang dihasilkan: {len(potongan)}")
    print()
    for p in potongan[:5]:
        print(f"ID       : {p['id']}")
        print(f"Jejak    : {p['metadata']['jejak_judul']}")
        print(f"3rd Party: {p['metadata']['pihak_ketiga']}")
        print(f"Biaya Rp : {p['metadata']['memuat_biaya']}")
        print(f"Awal     : {p['teks'][:110]}...")
        print("---")
