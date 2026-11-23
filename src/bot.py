# ==============================================================
# bot.py — Inti Sistem Chatbot Administrasi Kampus (Sistem B)
#
# Modul ini menyediakan fungsi utama `jawab(pesan, sistem="B")`.
# Alur Sistem B:
#   1. Normalisasi pesan (kamus singkatan dari data Excel)
#   2. Pengecekan aturan penolakan awal (topik terlarang / konflik)
#   3. Router (klasifikasi topik dan estimasi keyakinan)
#   4. Pengecekan bukti pembayaran (kelengkapan teks & teruskan ke staf)
#   5. Pencarian potongan dokumen di Chroma DB (E5 embedding)
#   6. Validasi jarak kemiripan dan tanggal batas/tutup kegiatan
#   7. Penyusunan respons tanpa mengarang teks
# ==============================================================

import re
import time
from datetime import date, datetime
from pathlib import Path
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

from config import (
    ROOT, CHROMA_DIR, EXCEL_PATH, SHEET_KAMUS,
    EMBEDDING_MODEL, EMBEDDING_PREFIX_QUERY,
    ROUTER_AKTIF, ROUTER_CONFIDENCE_THRESHOLD,
    SEARCH_DISTANCE_THRESHOLD, TOP_K_RESULTS,
    KELAS_BOT, KELAS_ESKALASI, TOPIK_TERLARANG,
    KONFLIK_METODE_BAYAR, FIELD_BUKTI_BAYAR,
    TANGGAL_HARI_INI, JAM_LAYANAN_TEKS,
    MIN_OVERLAP_KATA, KANAL_PEMBAYARAN, KATA_UMUM_STOPWORDS,
    KARANTINA_BAGIAN, BAGIAN_TOPIK_KHUSUS,
    SUSUNAN_AKTIF, AMBANG_VETO_V2, AMBANG_V3,
)

# --- Cache internal untuk model dan database (mempercepat eksekusi) ---
_CHROMA_CLIENT = None
_CHROMA_COLLECTION = None
_EMBEDDING_MODEL_OBJ = None
_KAMUS_SINGKATAN = None


def muat_kamus_singkatan() -> dict[str, str]:
    """
    Memuat kamus singkatan dari sheet 'Kamus' di file Excel analisis.
    Hanya mengambil kata-kata obrolan umum (bukan istilah sistem/aplikasi).
    Menyediakan fallback jika file Excel tidak dapat diakses.
    """
    global _KAMUS_SINGKATAN
    if _KAMUS_SINGKATAN is not None:
        return _KAMUS_SINGKATAN

    kamus = {}
    try:
        if EXCEL_PATH.exists():
            df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_KAMUS)
            # Filter baris yang bukan istilah aplikasi/akademik khusus
            istilah_khusus = {
                'ektm', 'mybest', 'myubsi', 'bsi id', 'bju', 'kip', 'ormik',
                'ormawa', 'serkom', 'juklak', 'po', 'angsuran', 'her', 'uas', 'uts', 'pmb'
            }
            for _, row in df.iterrows():
                bentuk_chat = str(row.get('bentuk di chat', '')).strip().lower()
                bentuk_baku = str(row.get('bentuk baku / arti', '')).strip().lower()
                if bentuk_chat and bentuk_baku and bentuk_chat not in istilah_khusus:
                    # Ambil arti sederhana jika tidak mengandung penjelasan panjang
                    if len(bentuk_baku.split()) <= 3:
                        kamus[bentuk_chat] = bentuk_baku
    except Exception as e:
        print(f"[PERINGATAN] Gagal membaca kamus Excel: {e}. Menggunakan kamus bawaan.")

    # Kamus bawaan esensial sebagai cadangan
    fallback = {
        'nnti': 'nanti', 'bsk': 'besok', 'skrng': 'sekarang', 'knpa': 'kenapa',
        'udh': 'sudah', 'sdh': 'sudah', 'blm': 'belum', 'gk': 'tidak',
        'gak': 'tidak', 'ga': 'tidak', 'gabisa': 'tidak bisa', 'gkpp': 'tidak apa-apa',
        'klo': 'kalau', 'kalo': 'kalau', 'brpa': 'berapa', 'tgl': 'tanggal',
        'jgn': 'jangan', 'yg': 'yang', 'bkn': 'bukan', 'lgi': 'lagi',
        'dlu': 'dulu', 'lngsung': 'langsung', 'tf': 'transfer', 'smt': 'semester',
        'pswd': 'kata sandi', 'pasword': 'kata sandi', 'ortu': 'orang tua',
        'maba': 'mahasiswa baru', 'byr': 'bayar', 'gmn': 'bagaimana', 'sy': 'saya',
    }
    fallback.update(kamus)
    _KAMUS_SINGKATAN = fallback
    return _KAMUS_SINGKATAN


def normalisasi(pesan: str) -> str:
    """
    Menormalisasi pesan chat:
      - Ubah ke huruf kecil
      - Reduksi karakter berulang berlebihan (misal: 'tolooong' -> 'tolong')
      - Ganti singkatan dengan kata baku berdasarkan kamus
    """
    if not pesan:
        return ""

    teks = pesan.lower().strip()
    # Reduksi karakter berulang lebih dari 2 kali menjadi 1 karakter
    teks = re.sub(r'([a-z])\1{2,}', r'\1', teks)

    kamus = muat_kamus_singkatan()
    # Tokenisasi berbasis kata dan pemetaan kata
    kata_kata = teks.split()
    kata_normal = [kamus.get(k, k) for k in kata_kata]
    return " ".join(kata_normal)


def ekstrak_kata_isi(teks: str) -> list[str]:
    """
    Mengekstrak kata-kata bermakna (kata isi) dari teks yang telah dinormalisasi.
    Membuang kata umum, kata tanya, partikel, dan kata sapaan (KATA_UMUM_STOPWORDS).
    Hanya menyaring token alfanumerik dengan panjang minimal 2 karakter.
    """
    token_list = re.findall(r'\b[a-z0-9_]+\b', teks.lower())
    return [t for t in token_list if t not in KATA_UMUM_STOPWORDS and len(t) > 1]


def get_embedding_model() -> SentenceTransformer:
    """Memuat model SentenceTransformer sekali saja (singleton). Default inferensi di CPU."""
    global _EMBEDDING_MODEL_OBJ
    if _EMBEDDING_MODEL_OBJ is None:
        _EMBEDDING_MODEL_OBJ = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return _EMBEDDING_MODEL_OBJ


def get_chroma_collection():
    """Memuat koleksi Chroma DB sekali saja."""
    global _CHROMA_CLIENT, _CHROMA_COLLECTION
    if _CHROMA_COLLECTION is None:
        _CHROMA_CLIENT = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _CHROMA_COLLECTION = _CHROMA_CLIENT.get_collection("dokumen_kampus")
    return _CHROMA_COLLECTION


# ==============================================================
# ROUTER
# ==============================================================

def periksa_whitelist_dokumen_resmi(teks: str) -> tuple[bool, str]:
    """
    Mendeteksi apakah pertanyaan menanyakan:
    1. Definisi, fungsi, atau navigasi fitur resmi aplikasi BSI.ID/EKTM.
    2. Tata cara bayar resmi via kanal Virtual Account / Merchant yang tercantum di panduan resmi:
       Alfamart, Indomaret, BCA, Mandiri (Livin), BNI, BRI, Permata, BSI.
    Mengembalikan: (is_whitelist, kelas_target)
    """
    t = teks.lower()

    # Jangan pernah whitelist topik terlarang atau eskalasi wajib
    for k in ['beasiswa', 'kip', 'kipk', 'bju', 'bidikmisi', 'reset password', 'lupa password', 'gagal login']:
        if k in t:
            return False, ''

    # 1. Definisi / Pengertian / Apa itu BSI.ID, MotionPay, EKTM, BSI PAY
    pola_tanya_definisi = r'\b(apa (definisi|arti|pengertian|fungsi|maksud)|bagaimana definisi|jelaskan (definisi|arti|fungsi))\b'
    pola_subjek_definisi = r'\b(bsi\.id|bsi id|bsi pay|motionpay|ektm|e-ktm|student id)\b'
    if re.search(pola_tanya_definisi, t) and re.search(pola_subjek_definisi, t):
        return True, 'kegiatan_pmb'

    frasa_definisi = [
        'definisi bsi', 'arti bsi', 'fungsi bsi', 'apa itu bsi.id', 'apa itu bsi id',
        'apa itu motionpay', 'fungsi motionpay', 'definisi motionpay', 'apa itu bsi pay',
        'definisi ektm', 'arti ektm', 'apa itu ektm', 'fungsi ektm',
        'apa itu e-ktm', 'arti e-ktm', 'fungsi e-ktm', 'definisi e-ktm',
        'kegunaan aplikasi ektm', 'kegunaan ektm', 'kegunaan bsi.id', 'kegunaan bsi id'
    ]
    if any(p in t for p in frasa_definisi):
        return True, 'kegiatan_pmb'

    # 2. Cara melihat / cek / menampilkan kartu EKTM mahasiswa
    pola_lihat = r'\b(cara|bagaimana|gimana|bisa)\b.*?\b(lihat|melihat|cek|mengecek|menampilkan|buka)\b.*?\b(ektm|e-ktm|kartu mahasiswa|student id)\b'
    if re.search(pola_lihat, t):
        return True, 'kegiatan_pmb'
    if any(p in t for p in ['lihat ektm', 'melihat ektm', 'cek ektm', 'mengecek ektm', 'tampilan ektm', 'kartu mahasiswa digital']):
        return True, 'kegiatan_pmb'

    # 3. Fitur / Menu resmi BSI.ID / MotionPay
    pola_fitur = r'\b(fitur|menu|tampilan|navigasi|panduan)\b.*?\b(bsi\.id|bsi id|ektm|e-ktm|motionpay|bsi pay|info bayar|e-wallet|xplore)\b'
    if re.search(pola_fitur, t):
        return True, 'kegiatan_pmb'
    if any(p in t for p in ['cara daftar motionpay', 'registrasi bsi.id', 'aktivasi bsi pay', 'menu info bayar', 'menu e-wallet', 'bottom bar']):
        return True, 'kegiatan_pmb'

    # 4. Tata Cara Bayar Resmi pada Kanal & Jenis Tagihan yang Tersedia Langkah Rinci di Dokumen
    pola_kanal_resmi = r'\b(alfamart|indomaret|bca|klikbca|livin|mandiri|bni|bri|briva|permata|bsi|cuti|mutasi)\b'
    pola_aksi_bayar = r'(bayar|transfer|langkah|panduan|tata cara|petunjuk|cara|tutorial|proses)'
    if re.search(pola_kanal_resmi, t) and (re.search(pola_aksi_bayar, t) or re.search(r'\b(atm|teller|m-banking|mbanking|va|virtual account)\b', t)):
        return True, 'pembayaran_umum'

    return False, ''


def router_aturan(teks: str) -> tuple[str, float]:
    """
    Router berbasis aturan kata kunci sementara (sebelum router ML dilatih di T4-T6).
    Mengembalikan: (kelas_router, keyakinan)
    6 Kelas:
      - bukti_bayar
      - pembayaran_umum
      - kegiatan_pmb
      - login_akun
      - beasiswa_pendaftaran
      - lainnya
    """
    teks_l = teks.lower()

    # 1. Beasiswa (selalu prioritas deteksi)
    if any(k in teks_l for k in ['beasiswa', 'kip', 'kipk', 'kip-k', 'bju', 'bidikmisi']):
        return 'beasiswa_pendaftaran', 0.95

    # 2. Bukti bayar
    pola_bukti = [
        'bukti transfer', 'bukti bayar', 'sudah transfer', 'sudah bayar dan kirim',
        'struk transfer', 'struk pembayaran', 'lampiran transfer', 'ss transfer',
        'screenshoot transfer', 'screenshot bukti', 'berikut bukti'
    ]
    if any(p in teks_l for p in pola_bukti):
        return 'bukti_bayar', 0.92

    # Jika ada penyebutan transfer/bayar yang disertai NIM/nominal/rekening
    if ('transfer' in teks_l or 'bayar' in teks_l) and ('nim' in teks_l or 'rp' in teks_l or 'norek' in teks_l):
        return 'bukti_bayar', 0.85

    # 2.5. Whitelist Dokumen Resmi (fitur/definisi/tata cara bayar resmi dialirkan ke pencarian)
    is_wl, target_kls = periksa_whitelist_dokumen_resmi(teks)
    if is_wl:
        return target_kls, 0.90

    # 3. Login & akun
    pola_login = [
        'login', 'akun', 'password', 'kata sandi', 'username', 'mybest',
        'myubsi', 'bsi id', 'gagal masuk', 'tidak bisa masuk', 'lupa password',
        'reset password', 'gagal login', 'kendala aplikasi'
    ]
    if any(p in teks_l for p in pola_login):
        return 'login_akun', 0.88

    # 4. Pembayaran umum
    pola_bayar = [
        'cara bayar', 'metode pembayaran', 'batas bayar', 'jadwal bayar',
        'tagihan', 'virtual account', ' va ', 'nomor va', 'angsuran', 'cicilan',
        'bayar kuliah', 'bayar spp', 'bayar adm', 'biaya kuliah', 'bni', 'bsi pay',
        'atm bni', 'teller', 'mobile banking', 'm-banking', 'pembayaran ektm'
    ]
    if any(p in teks_l for p in pola_bayar):
        return 'pembayaran_umum', 0.90

    # 5. Kegiatan PMB & Akademik
    pola_pmb = [
        'pmb', 'pendaftaran', 'daftar ulang', 'atribut', 'kaos', 'baju',
        'id card', 'kartu mahasiswa', 'ektm', 'jadwal pmb', 'seminar',
        'workshop', 'serkom', 'sertifikasi', 'orientasi', 'ormik', 'semaba'
    ]
    if any(p in teks_l for p in pola_pmb):
        return 'kegiatan_pmb', 0.88

    # Deteksi sekunder kata umum
    if 'bayar' in teks_l or 'transfer' in teks_l or 'biaya' in teks_l:
        return 'pembayaran_umum', 0.65
    if 'masuk' in teks_l or 'aplikasi' in teks_l:
        return 'login_akun', 0.60
    if 'kuliah' in teks_l or 'kegiatan' in teks_l:
        return 'kegiatan_pmb', 0.60

    # Default eskalasi
    return 'lainnya', 0.40


_MODEL_KLASIK = None


def klasifikasi_router_detail(teks: str, mode: str = None) -> dict:
    """
    Klasifikasi teks menggunakan router aktif ("aturan" | "klasik" | "indobertweet").
    Mengembalikan dict:
      {
          'kelas': str,
          'keyakinan': float,
          'peluang_eskalasi': float,
          'keputusan': 'teruskan' | 'jawab',
          'distribusi_probabilitas': dict[str, float]
      }
    """
    global _MODEL_KLASIK
    if mode is None:
        mode = ROUTER_AKTIF.strip().lower()
    else:
        mode = mode.strip().lower()

    if mode == "indobertweet":
        try:
            from router_infer import prediksi as prediksi_indobertweet
            res = prediksi_indobertweet(teks)
            # Whitelist Dokumen Resmi: Jangan biarkan eskalasi memblokir panduan/definisi/tata cara bayar resmi
            is_wl, target_kls = periksa_whitelist_dokumen_resmi(teks)
            if is_wl:
                res['kelas'] = target_kls
                res['keyakinan'] = 0.90
                res['peluang_eskalasi'] = 0.05
                res['keputusan'] = 'jawab'
            return res
        except Exception as e:
            pass

    elif mode == "klasik":
        try:
            import joblib
            klasik_path = ROOT / "models" / "router_klasik.joblib"
            if _MODEL_KLASIK is None and klasik_path.exists():
                _MODEL_KLASIK = joblib.load(klasik_path)
            if _MODEL_KLASIK is not None:
                probs = _MODEL_KLASIK.predict_proba([teks])[0]
                classes = list(_MODEL_KLASIK.classes_)
                idx_max = int(probs.argmax())
                kls = classes[idx_max]
                conf = float(probs[idx_max])

                # Peluang eskalasi = P(beasiswa_pendaftaran) + P(lainnya)
                p_esk = 0.0
                if "beasiswa_pendaftaran" in classes:
                    p_esk += float(probs[classes.index("beasiswa_pendaftaran")])
                if "lainnya" in classes:
                    p_esk += float(probs[classes.index("lainnya")])

                ambang_klasik = 0.300  # Ambang FPR(a) <= 10% dari T6b lanjutan
                if kls in KELAS_ESKALASI or p_esk >= ambang_klasik:
                    is_wl, target_kls = periksa_whitelist_dokumen_resmi(teks)
                    if is_wl:
                        kls = target_kls
                        conf = 0.90
                        p_esk = 0.05
                        keputusan = "jawab"
                    else:
                        keputusan = "teruskan"
                else:
                    keputusan = "jawab"

                dist = {c: round(float(p), 4) for c, p in zip(classes, probs)}
                return {
                    'kelas': kls,
                    'keyakinan': round(conf, 4),
                    'peluang_eskalasi': round(p_esk, 4),
                    'keputusan': keputusan,
                    'distribusi_probabilitas': dist
                }
        except Exception as e:
            pass

    # Default / Fallback: Router Aturan
    kls, conf = router_aturan(teks)
    p_esk = 1.0 if kls in KELAS_ESKALASI else 0.0
    keputusan = "teruskan" if (kls in KELAS_ESKALASI or conf < ROUTER_CONFIDENCE_THRESHOLD) else "jawab"
    return {
        'kelas': kls,
        'keyakinan': round(conf, 4),
        'peluang_eskalasi': round(p_esk, 4),
        'keputusan': keputusan,
        'distribusi_probabilitas': {kls: round(conf, 4)}
    }


def klasifikasi_router(teks: str) -> tuple[str, float]:
    """
    Fungsi perantara router. Memilih implementasi sesuai config.ROUTER_AKTIF.
    Cukup ubah config.ROUTER_AKTIF ("aturan" | "klasik" | "indobertweet")
    tanpa perlu mengubah kode bot lainnya.
    """
    res = klasifikasi_router_detail(teks)
    return res['kelas'], res['keyakinan']


# ==============================================================
# PENGECEKAN ATURAN KHUSUS & KEAMANAN
# ==============================================================

def periksa_topik_terlarang(pesan: str) -> tuple[bool, str]:
    """
    Memeriksa apakah pesan menyentuh topik terlarang yang wajib dieskalasi.
    Aturan dari prompt:
      - komplain / keputusan ujian / nilai / her
      - nominal pribadi (tagihan pribadi mahasiswa)
      - beasiswa
      - koreksi pembayaran (salah transfer, salah input, uang terpotong)
      - kata sandi (permintaan reset kata sandi akun pribadi)
      - nomor pribadi dosen / staf
      - permohonan undur diri / tidak mampu bayar
    Mengembalikan: (apakah_terlarang, alasan)
    """
    pesan_l = pesan.lower()

    # 1. Komplain / nilai / ujian
    if any(k in pesan_l for k in ['nilai', 'komplain nilai', 'sanggahan nilai', 'her', 'remedial', 'yudisium', 'kelulusan']):
        return True, "Topik terkait komplain, nilai, atau hasil ujian wajib ditangani langsung oleh staf akademik."

    # 2. Tagihan/nominal pribadi & angsuran
    pola_nominal_pribadi = [
        'tagihan saya', 'cek tagihan', 'spp saya', 'sisa pembayaran saya',
        'apakah saya sudah lunas', 'cek spp saya', 'status pembayaran saya'
    ]
    if any(p in pesan_l for p in pola_nominal_pribadi):
        return True, "Pengecekan nominal pribadi dan status tagihan mahasiswa memerlukan verifikasi sistem internal staf keuangan."

    # Angsuran / Cicilan
    if any(k in pesan_l for k in ['angsuran', 'cicilan', 'danacita']):
        return True, "Ketentuan, prosedur, atau permohonan angsuran/cicilan pembayaran kuliah dialihkan langsung ke bagian keuangan."

    # Biaya cuti
    if any(k in pesan_l for k in ['biaya cuti', 'nominal cuti', 'bayar cuti berapa']):
        return True, "Informasi rincian biaya cuti akademik dialihkan langsung ke staf administrasi."

    # Hal bertentangan: Ikon centang di riwayat pembayaran
    if any(k in pesan_l for k in ['ikon centang', 'centang merah', 'centang hijau', 'arti centang', 'tanda centang']):
        return True, "Penjelasan mengenai arti ikon atau tanda centang pada riwayat pembayaran dialihkan langsung ke staf admin."

    # Admin Jaket KIP
    if any(k in pesan_l for k in ['jaket kip', 'admin jaket']):
        return True, "Ketentuan dan pembayaran Admin Jaket KIP dialihkan langsung ke staf administrasi."

    # Nominal beasiswa
    if any(k in pesan_l for k in ['nominal beasiswa', 'jumlah beasiswa', 'dana beasiswa', 'besaran beasiswa']):
        return True, "Ketentuan nominal beasiswa dialihkan langsung ke staf administrasi."

    # Biaya dan tarif kegiatan (jangan tampilkan angka biaya kegiatan apa pun)
    pola_biaya_jadwal = [
        'biaya kegiatan', 'biaya seminar', 'biaya pmb', 'biaya ormik',
        'jadwal ormik', 'jadwal pmb', 'biaya wisuda', 'tarif kegiatan',
        'tarif seminar', 'tarif wisuda', 'biaya serkom', 'biaya workshop',
        'biaya jaket', 'tarif serkom', 'biaya ta', 'biaya skripsi'
    ]
    if any(p in pesan_l for p in pola_biaya_jadwal):
        return True, "Biaya dan tarif kegiatan kampus dialihkan ke staf admin."

    # 3. Koreksi pembayaran / kendala transaksi perorangan
    pola_koreksi = [
        'salah transfer', 'salah bayar', 'salah input', 'salah nim',
        'uang terpotong', 'sudah bayar tapi belum masuk', 'belum lunas padahal sudah bayar',
        'gerai tidak masuk', 'uang belum masuk'
    ]
    if any(p in pesan_l for p in pola_koreksi):
        return True, "Koreksi kesalahan pembayaran atau transaksi yang belum tercatat wajib ditindaklanjuti oleh staf keuangan."

    # 4. Kata sandi / Reset kata sandi pribadi (jangan menyebut kata sandi ke publik)
    if any(k in pesan_l for k in ['reset kata sandi', 'reset password', 'lupa kata sandi', 'lupa password', 'ganti password saya']):
        return True, "Reset akses akun pribadi mahasiswa wajib melalui prosedur verifikasi identitas oleh staf admin."

    # Masalah login BSI.ID dan mybest (jangan menyebut kata sandi)
    # BSI.ID dan mybest adalah aplikasi resmi kampus; kendala akses/login dialihkan ke admin
    is_app_bsi_mybest = any(app in pesan_l for app in ['bsi.id', 'bsi id', 'mybest', 'myubsi'])
    is_masalah_akses = any(k in pesan_l for k in [
        'login', 'masuk', 'gagal', 'kendala', 'masalah', 'error', 'tidak bisa',
        'gabisa', 'gak bisa', 'reset', 'password', 'kata sandi', 'lupa', 'akun', 'aplikasi'
    ])
    if is_app_bsi_mybest and is_masalah_akses:
        return True, "Kendala akses dan login pada aplikasi BSI.ID atau mybest dialihkan langsung ke staf administrasi."
    if any(k in pesan_l for k in ['login bsi.id', 'login mybest', 'gagal login bsi.id', 'gagal login mybest', 'masalah bsi.id', 'masalah mybest', 'kendala bsi.id', 'kendala mybest', 'login bsi']):
        return True, "Kendala akses dan login pada aplikasi BSI.ID atau mybest dialihkan langsung ke staf administrasi."

    # 5. Aturan kata kunci keras: nomor atau kontak + dosen atau staf atau ketua prodi
    kata_kontak = ['nomor', 'kontak', 'wa', 'no hp', 'nohp', 'telepon', 'telp', 'no']
    kata_personel = ['dosen', 'staf', 'staff', 'ketua prodi', 'kaprodi', 'prodi', 'pembimbing', 'rektor', 'dekan']
    if any(c in pesan_l for c in kata_kontak) and any(p in pesan_l for p in kata_personel):
        return True, "Aturan kata kunci: Permintaan nomor atau kontak dosen, staf, atau ketua prodi dialihkan langsung ke staf administrasi."
    pola_kontak = [
        'nomor dosen', 'kontak dosen', 'wa dosen', 'nomor pak', 'nomor bu',
        'no hp dosen', 'kontak staf', 'nomor pribadi'
    ]
    if any(p in pesan_l for p in pola_kontak):
        return True, "Chatbot tidak berwenang memberikan kontak pribadi dosen atau staf. Silakan hubungi bagian administrasi kampus."

    # 5b. Aturan kata kunci keras: beasiswa atau KIP atau BJU
    if any(k in pesan_l for k in ['beasiswa', 'kip', 'kipk', 'kip-k', 'bju', 'bidikmisi']):
        return True, "Aturan kata kunci: Informasi dan ketentuan beasiswa, KIP, atau BJU dialihkan langsung ke staf administrasi."

    # 5c. Aturan kata kunci keras: tarif/biaya/harga + kegiatan atau wisuda atau seminar atau workshop atau serkom atau jaket
    kata_biaya = ['tarif', 'biaya', 'harga', 'nominal']
    kata_kegiatan = ['kegiatan', 'wisuda', 'seminar', 'workshop', 'serkom', 'jaket', 'ormik', 'semaba', 'pmb', 'ta', 'skripsi']
    # Gunakan batasan kata utuh agar kata seperti 'mutasi' tidak cocok dengan 'ta'
    ada_kata_biaya = any(re.search(r'\b' + re.escape(b) + r'\b', pesan_l) for b in kata_biaya)
    ada_kata_kegiatan = any(re.search(r'\b' + re.escape(k) + r'\b', pesan_l) for k in kata_kegiatan)
    if ada_kata_biaya and ada_kata_kegiatan:
        return True, "Aturan kata kunci: Informasi tarif/biaya/harga kegiatan, wisuda, seminar, workshop, serkom, atau jaket dialihkan langsung ke staf administrasi."

    # Pertanyaan cara/petunjuk bayar mutasi bukan menanyakan nominal, diperiksa terpisah jika menanyakan nominal rincian mutasi
    if 'mutasi' in pesan_l and any(re.search(r'\b' + re.escape(b) + r'\b', pesan_l) for b in ['berapa', 'nominal', 'tarif', 'harga']):
        return True, "Informasi rincian nominal biaya mutasi akademik dialihkan langsung ke staf administrasi."

    # 6. Pengunduran diri / ketidakmampuan bayar
    if any(k in pesan_l for k in ['undur diri', 'mengundurkan diri', 'tidak mampu bayar', 'keringanan spp', 'pindah prodi']):
        return True, "Pengajuan pengunduran diri, keringanan biaya, atau mutasi prodi wajib diajukan resmi ke pimpinan/staf kampus."

    # 7. Konflik metode pembayaran (K-008: SeaBank, Indomaret, Dana/GoPay, unggah atribut)
    for k in KONFLIK_METODE_BAYAR:
        if k.lower() in pesan_l:
            return True, f"Pertanyaan mengenai metode/ketentuan '{k}' saat ini berstatus konflik kebijakan dan dialihkan ke admin."

    return False, ""


def periksa_kelengkapan_bukti_bayar(pesan: str) -> dict:
    """
    Menganalisis kelengkapan keterangan teks saat mahasiswa mengirimkan bukti bayar.
    Kriteria: nama, NIM, kelas, jenis pembayaran, semester atau angsuran ke-, jenis beasiswa.
    Bot TIDAK membaca gambar bukti bayar secara mandiri, hanya mengecek keterangan teks.
    """
    pesan_l = pesan.lower()
    kelengkapan = {}

    # Indikasi Nama
    kelengkapan['nama'] = bool(re.search(r'(nama|an\.|a\.n\.|atas nama)', pesan_l) or '@user' in pesan_l or re.search(r'\ban\b', pesan_l))

    # Indikasi NIM (deretan angka 6-12 digit atau kata 'nim')
    kelengkapan['NIM'] = bool(re.search(r'\b\d{6,12}\b', pesan_l) or 'nim' in pesan_l)

    # Indikasi Kelas / Prodi
    kelengkapan['kelas'] = bool(any(k in pesan_l for k in ['kelas', 'prodi', 'jurusan', 'pagi', 'malam', 'karyawan']) or re.search(r'\b\d{2}\.\d[a-z]\.\d{2}\b', pesan_l))

    # Indikasi Jenis Pembayaran
    kelengkapan['jenis pembayaran'] = bool(any(k in pesan_l for k in ['spp', 'adm', 'pmb', 'formulir', 'daftar ulang', 'kegiatan', 'serkom', 'seminar', 'bayar', 'tf', 'transfer', 'kuliah']))

    # Indikasi Semester/Angsuran
    kelengkapan['semester atau angsuran ke-'] = bool(any(k in pesan_l for k in ['semester', 'smt', 'angsuran', 'cicilan', 'tahap']))

    # Field wajib utama
    field_wajib = ['nama', 'NIM', 'kelas', 'jenis pembayaran', 'semester atau angsuran ke-']

    # Jika beasiswa disebut, pastikan jenis beasiswa tercantum
    if 'beasiswa' in pesan_l:
        field_wajib.append('jenis beasiswa')
        kelengkapan['jenis beasiswa'] = bool(any(k in pesan_l for k in ['kip', 'kipk', 'kip-k', 'bju', 'bidikmisi', 'prestasi', 'dhuafa']))

    field_hilang = [f for f in field_wajib if not kelengkapan.get(f, False)]

    return {
        'status_lengkap': len(field_hilang) == 0,
        'rincian': kelengkapan,
        'field_hilang': field_hilang,
    }


def periksa_tanggal_kedaluwarsa_teks(metadata: dict) -> tuple[bool, str, str]:
    """
    Memeriksa tanggal tutup/kegiatan yang tercantum di metadata potongan.
    Dibandingkan dengan tanggal hari ini dari sistem.
    Mengembalikan: (sudah_tutup: bool, alasan: str, tgl_tutup_str: str)
    """
    tgl_tutup_str = metadata.get('tanggal_tutup', '').strip()
    if not tgl_tutup_str:
        return False, "", ""

    try:
        tgl_tutup = datetime.strptime(tgl_tutup_str, "%d-%m-%Y").date()
        if tgl_tutup < TANGGAL_HARI_INI:
            return True, f"Batas waktu kegiatan ({tgl_tutup_str}) telah terlewati per tanggal hari ini ({TANGGAL_HARI_INI}).", tgl_tutup_str
    except Exception:
        pass

    return False, "", tgl_tutup_str


def bersihkan_klaim_status_saat_ini(teks: str) -> str:
    """
    Kalimat berpola 'Saat ini ...' atau 'Pada saat ...' yang menyatakan status (buka, tutup, aktif)
    jangan ditampilkan sebagai fakta terkini. Hilangkan kalimat itu dari jawaban dan tambahkan:
    'Status menu berubah sesuai jadwal dan diumumkan lewat Story WhatsApp admin.' (Perbaikan 2)
    """
    pola = re.compile(
        r'(?:Status Menu\s*/\s*Jadwal:\s*)?(?:[^\.\n]*?(?:saat ini|pada saat)[^\.\n]*?(?:ditutup|dibuka|aktif|terbuka)[^\.\n]*?\.\s*)',
        re.IGNORECASE
    )
    if pola.search(teks):
        teks = pola.sub('', teks).strip()
        teks = re.sub(r'(?:Status Menu\s*/\s*Jadwal:\s*)', '', teks, flags=re.IGNORECASE).strip()
        teks = teks + "\n\nStatus menu berubah sesuai jadwal dan diumumkan lewat Story WhatsApp admin."
    return teks


def format_jawaban_manusiawi(teks_raw: str, metadata: dict, pesan_pengguna: str) -> str:
    """
    Mengemas teks jawaban dokumen resmi menjadi respons percakapan yang ramah, sopan,
    dan alami seperti staf admin manusia, tanpa mengubah fakta teknis dokumen resmi.
    """
    judul_bagian = metadata.get('judul_bagian', '')
    jejak_judul = metadata.get('jejak_judul', '')
    t_tanya = pesan_pengguna.lower()

    # 1. Bersihkan prefix teknis [Bagian: ...] dari teks dokumen
    teks_bersih = re.sub(r'^\[Bagian:[^\]]+\]\s*', '', teks_raw.strip(), flags=re.MULTILINE).strip()

    # 2. Pisahkan baris non-kosong
    baris_list = [b.strip() for b in teks_bersih.split('\n') if b.strip()]

    # 3. Tentukan salam pembuka yang ramah dan kontekstual
    if any(k in judul_bagian for k in ['Melalui ', 'Pembayaran dari ']):
        nama_kanal = judul_bagian.replace('Melalui ', '').replace('Pembayaran dari ', '').strip()
        pembuka = f"Halo! Untuk melakukan pembayaran kuliah melalui **{nama_kanal}**, ini dia langkah-langkahnya ya:\n\n"
        isi_terformat = "\n".join(baris_list)
        penutup = "\n\nSemoga membantu! Pastikan selalu menyimpan bukti transaksi Anda ya. Jika ada langkah yang kurang jelas, silakan tanyakan lagi."

    elif 'Petunjuk Pembayaran Biaya Cuti' in judul_bagian:
        pembuka = "Halo! Untuk panduan pembayaran **Biaya Cuti Akademik**, berikut langkah-langkahnya:\n\n"
        isi_terformat = "\n".join(baris_list)
        penutup = "\n\nSemoga membantu! Pastikan pendaftaran cuti sudah dilakukan terlebih dahulu di aplikasi MyStudents ya."

    elif 'Petunjuk Pembayaran Biaya Mutasi' in judul_bagian:
        pembuka = "Halo! Untuk alur pembayaran **Biaya Mutasi Akademik**, berikut panduan lengkapnya:\n\n"
        isi_terformat = "\n".join(baris_list)
        penutup = "\n\nSemoga membantu! Pastikan proses mutasi sudah diajukan secara resmi ya."

    elif 'Petunjuk Pembayaran Kuliah' in judul_bagian:
        pembuka = "Halo! Untuk panduan dan langkah pembayaran **Biaya Kuliah**, berikut informasinya ya:\n\n"
        isi_terformat = "\n".join(baris_list)
        penutup = "\n\nSemoga membantu! Pastikan selalu mengecek status transaksi di aplikasi EKTM Anda ya."

    elif 'Histori Pembayaran' in judul_bagian:
        pembuka = "Halo! Untuk melihat **Riwayat / Histori Pembayaran Akademik**, berikut langkah-langkahnya:\n\n"
        isi_terformat = "\n".join(baris_list)
        penutup = "\n\nSemoga informasinya jelas! Anda dapat melihat seluruh riwayat transaksi sebelumnya pada menu tersebut."

    elif 'Beranda Utama' in jejak_judul or 'Kartu Mahasiswa' in teks_bersih:
        pembuka = "Halo! Terkait tampilan dan cara melihat **Kartu Mahasiswa Digital (E-KTM)**, berikut penjelasannya:\n\n"
        isi_terformat = "\n".join(baris_list)
        penutup = "\n\nSemoga informasinya jelas! Ada hal lain seputar fitur aplikasi BSI.ID yang ingin Anda ketahui?"

    elif any(k in t_tanya for k in ['definisi', 'arti', 'pengertian', 'apa itu']):
        # Jika pertanyaan spesifik definisi, tampilkan baris definisi yang cocok atau seluruh rincian
        baris_cocok = []
        for b in baris_list:
            if any(k in b.lower() for k in ['definisi', 'tentang', 'profil', 'adalah']):
                baris_cocok.append(b)

        pembuka = "Halo! Mengenai informasi yang Anda tanyakan, berikut penjelasan dari panduan resmi aplikasi BSI.ID:\n\n"
        if baris_cocok:
            isi_terformat = "\n\n".join(baris_cocok)
        else:
            isi_terformat = "\n".join(baris_list)
        penutup = "\n\nSemoga membantu! Jika ada fitur lain yang ingin diketahui, silakan tanyakan ya."

    elif 'Registrasi' in judul_bagian or 'Autentikasi' in jejak_judul:
        pembuka = "Halo! Berikut panduan registrasi dan aktivasi akun aplikasi BSI.ID:\n\n"
        isi_terformat = "\n".join(baris_list)
        penutup = "\n\nSemoga proses registrasi Anda lancar! Jika mengalami kendala teknis aktivasi, silakan tanyakan ya."

    else:
        pembuka = "Halo! Berikut informasi resmi yang tercantum pada dokumen panduan kampus:\n\n"
        isi_terformat = "\n".join(baris_list)
        penutup = "\n\nSemoga informasinya bermanfaat! Jika ada hal yang kurang jelas, silakan tanyakan lagi ya."

    # 4. Catatan pihak ketiga (diletakkan rapi di paling akhir sebagai catatan kaki)
    catatan_kaki = ""
    if metadata.get('pihak_ketiga'):
        catatan_kaki = "\n\n📌 *Catatan: Fitur ini terhubung dengan layanan mitra (MotionPay/pihak ketiga), rincian berkala dapat dicek langsung pada aplikasi.*"

    return pembuka + isi_terformat + penutup + catatan_kaki


# ==============================================================
# ROUTER CASCADE & ATURAN SEGMEN PANDUAN BSI.ID
# ==============================================================

def inferensi_klasik_lazy(teks: str) -> dict:
    """
    Memuat router klasik TF-IDF + Regresi Logistik secara lazy (on-demand),
    menghitung probabilitas inferensi, dan langsung membersihkan memori.
    Ambang P(eskalasi) = 0.300.
    """
    import joblib, gc
    klasik_path = ROOT / "models" / "router_klasik.joblib"
    if not klasik_path.exists():
        return None
    model_obj = joblib.load(klasik_path)
    probs = model_obj.predict_proba([teks])[0]
    classes = list(model_obj.classes_)
    idx_max = int(probs.argmax())
    kls = classes[idx_max]
    conf = float(probs[idx_max])

    p_esk = 0.0
    if "beasiswa_pendaftaran" in classes:
        p_esk += float(probs[classes.index("beasiswa_pendaftaran")])
    if "lainnya" in classes:
        p_esk += float(probs[classes.index("lainnya")])

    ambang_klasik = 0.300
    keputusan = "teruskan" if (kls in KELAS_ESKALASI or p_esk >= ambang_klasik) else "jawab"
    dist = {c: round(float(p), 4) for c, p in zip(classes, probs)}
    del model_obj
    gc.collect()
    return {
        'kelas': kls,
        'keyakinan': round(conf, 4),
        'peluang_eskalasi': round(p_esk, 4),
        'keputusan': keputusan,
        'distribusi_probabilitas': dist
    }


SEGMEN_ATURAN_LIST = [
    {
        'id': 'K1',
        'nama': 'Cara membayar Biaya Kuliah & Metode Pembayaran',
        'target_judul': 'Petunjuk Pembayaran Kuliah',
        'pola': [
            r'langkah.*bayar.*kuliah',
            r'gimana.*langkah.*bayar.*kuliah',
            r'cara.*bayar.*biaya.*kuliah',
            r'alur.*bayar.*kuliah',
            r'prosedur.*bayar.*kuliah',
            r'tata.*cara.*bayar.*kuliah',
            r'bayar.*kuliah.*bisa.*dengan.*apa.*saja',
            r'bayar.*kuliah.*bisa.*apa.*saja',
            r'bayar.*kuliah.*pakai.*apa.*saja',
            r'metode.*bayar.*kuliah',
            r'metode.*pembayaran.*kuliah',
            r'pembayaran.*kuliah.*bisa.*lewat.*apa',
            r'opsi.*bayar.*kuliah',
        ]
    },
    {
        'id': 'K4',
        'nama': 'Histori Pembayaran Akademik',
        'target_judul': 'Petunjuk Histori Pembayaran Akademik',
        'pola': [
            r'cara.*lihat.*histori.*pembayaran',
            r'cara.*cek.*riwayat.*pembayaran',
            r'lihat.*riwayat.*bayar',
            r'cek.*histori.*pembayaran',
            r'melihat.*riwayat.*pembayaran',
            r'cek.*riwayat.*pembayaran',
            r'lihat.*histori.*pembayaran',
            r'histori.*pembayaran.*akademik',
            r'riwayat.*pembayaran.*akademik',
            r'melihat.*histori.*transaksi',
        ]
    },
    {
        'id': 'K3',
        'nama': 'Petunjuk Pembayaran Biaya Mutasi',
        'target_judul': 'Petunjuk Pembayaran Biaya Mutasi',
        'pola': [
            r'petunjuk.*pembayaran.*biaya.*mutasi',
            r'petunjuk.*bayar.*mutasi',
            r'cara.*bayar.*mutasi',
            r'cara.*membayar.*mutasi',
            r'panduan.*bayar.*biaya.*mutasi',
            r'panduan.*bayar.*mutasi',
            r'langkah.*bayar.*mutasi',
            r'prosedur.*bayar.*mutasi',
        ]
    },
    {
        'id': 'K2',
        'nama': 'Petunjuk Pembayaran Biaya Cuti',
        'target_judul': 'Petunjuk Pembayaran Biaya Cuti',
        'pola': [
            r'petunjuk.*pembayaran.*biaya.*cuti',
            r'petunjuk.*bayar.*cuti',
            r'cara.*bayar.*cuti',
            r'cara.*membayar.*cuti',
            r'panduan.*bayar.*biaya.*cuti',
            r'panduan.*bayar.*cuti',
            r'langkah.*bayar.*cuti',
            r'prosedur.*bayar.*cuti',
        ]
    }
]


def is_kueri_riwayat_pembayaran(teks: str) -> bool:
    """Mendeteksi apakah pertanyaan menanyakan histori/riwayat, bukan mengirimkan bukti bayar."""
    t = teks.lower()
    return bool(re.search(r'\b(histori|history|riwayat)\b', t) and any(k in t for k in ['bayar', 'pembayaran', 'transaksi', 'kuliah', 'spp', 'tagihan']))


def cocokkan_segmen_aturan(pesan_normal: str) -> dict | None:
    """Mencocokkan frasa pertanyaan dengan segmen spesifik Dokumen Panduan BSI.ID."""
    matches = []
    for r in SEGMEN_ATURAN_LIST:
        for p in r['pola']:
            if re.search(p, pesan_normal):
                matches.append(r)
                break
    if len(matches) == 1:
        return matches[0]
    return None


def eksekusi_jawaban_segmen_aturan(pesan_normal: str, rule_segmen: dict, susunan: str = "V1", mulai_waktu: float = None) -> tuple[bool, dict]:
    """Mengambil dokumen resmi langsung dari basis data untuk segmen aturan dan memvalidasi leksikal."""
    if mulai_waktu is None:
        mulai_waktu = time.perf_counter()

    col = get_chroma_collection()
    target_judul = rule_segmen['target_judul']

    res_db = col.get(where={"judul_bagian": target_judul})
    if not res_db or not res_db['documents']:
        return False, {'alasan': f"Dokumen chunk untuk '{target_judul}' tidak ditemukan di basis data."}

    doc_text = res_db['documents'][0]
    meta = res_db['metadatas'][0]
    nama_sumber = meta.get('nama_berkas', 'Penjelasan_Fitur_EKTM_BSI_ID_Lengkap update.docx')
    tgl_berlaku = meta.get('tanggal_berlaku', '2026-11-09')
    judul_bagian = meta.get('judul_bagian', target_judul)
    sumber_str = f"{nama_sumber} (Bagian: {judul_bagian}, dimasukkan ke indeks: {tgl_berlaku})"

    kata_isi = ekstrak_kata_isi(pesan_normal)
    doc_cek = doc_text.lower().replace('e-ktm', 'ektm').replace('-', ' ')
    if kata_isi:
        kata_cocok = [k for k in kata_isi if re.search(r'\b' + re.escape(k) + r'\b', doc_cek)]
        overlap = round(len(kata_cocok) / len(kata_isi), 3)
    else:
        overlap = 1.0

    if overlap < MIN_OVERLAP_KATA:
        return False, {
            'alasan': f"Frasa cocok dengan segmen {rule_segmen['id']}, namun overlap kata isi ({overlap:.2f}) < {MIN_OVERLAP_KATA:.2f}",
            'tumpang_tindih_kata': overlap
        }

    teks_jawaban = format_jawaban_manusiawi(doc_text.strip(), meta, pesan_normal)
    waktu_ms = round((time.perf_counter() - mulai_waktu) * 1000, 2)

    potongan_list = [{
        'teks': doc_text,
        'jarak': 0.0,
        'metadata': meta,
    }]

    return True, {
        'jawaban': teks_jawaban,
        'sumber': sumber_str,
        'kelas_router': 'aturan_segmen',
        'keyakinan': 1.0,
        'peluang_eskalasi': 0.0,
        'keputusan_router': 'jawab',
        'jarak_terbaik': 0.0,
        'tumpang_tindih_kata': overlap,
        'skor_gabungan': 1.0,
        'susunan': susunan,
        'kata_isi': kata_isi,
        'keputusan': "jawab",
        'alasan': f"Aturan segmen {rule_segmen['id']} ({target_judul}) cocok jelas dan overlap leksikal ({overlap:.2f}) >= {MIN_OVERLAP_KATA:.2f}.",
        'dialihkan_oleh': "-",
        'potongan_terpakai': potongan_list,
        'waktu_ms': waktu_ms,
    }


KANAL_TERDAFTAR_VA = {"motionbank", "indomaret", "alfamart", "bca", "permata", "mandiri", "bni", "bri", "bsi"}
KANAL_DENGAN_LANGKAH_RINCI = {"bri", "indomaret", "alfamart", "bca", "mandiri", "bni", "permata", "bsi"}

def eksekusi_pencarian_dan_gerbang(pesan_normal: str, susunan: str, kls_router: str, conf_router: float, p_esk: float, mulai_waktu: float) -> tuple[bool, dict]:
    """
    Menjalankan alur pencarian Chroma DB dan verifikasi gerbang leksikal/keamanan untuk Sistem B.
    Mengembalikan (sukses: bool, data_hasil: dict).
    """
    try:
        col = get_chroma_collection()
        embedder = get_embedding_model()

        query_teks = EMBEDDING_PREFIX_QUERY + pesan_normal
        query_emb = embedder.encode([query_teks], normalize_embeddings=True)

        hasil_cari = col.query(
            query_embeddings=query_emb.tolist(),
            n_results=15,
        )
    except Exception as e:
        waktu_ms = round((time.perf_counter() - mulai_waktu) * 1000, 2)
        return False, {
            'jawaban': f"Terjadi kendala teknis dalam pencarian basis pengetahuan. Silakan hubungi admin lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}.",
            'sumber': "",
            'kelas_router': kls_router,
            'keyakinan': conf_router,
            'peluang_eskalasi': p_esk,
            'keputusan_router': "teruskan",
            'jarak_terbaik': None,
            'tumpang_tindih_kata': None,
            'skor_gabungan': None,
            'susunan': susunan,
            'kata_isi': [],
            'keputusan': "teruskan_ke_staf",
            'alasan': f"Galat basis data Chroma: {e}",
            'potongan_terpakai': [],
            'waktu_ms': waktu_ms,
        }

    docs_raw = hasil_cari.get('documents', [[]])[0]
    distances_raw = hasil_cari.get('distances', [[]])[0]
    metadatas_raw = hasil_cari.get('metadatas', [[]])[0]

    kandidat_sebelum_filter = [
        (metadatas_raw[i].get('jejak_judul', ''), float(distances_raw[i]))
        for i in range(min(3, len(distances_raw)))
    ]

    docs = []
    distances = []
    metadatas = []
    for d, dist, meta in zip(docs_raw, distances_raw, metadatas_raw):
        jejak_l = meta.get('jejak_judul', '').lower()
        judul_l = meta.get('judul_bagian', '').lower()
        gabung = f"{jejak_l} {judul_l}"

        perlu_kw = False
        kws_wajib = []
        for nama_topik, kws in BAGIAN_TOPIK_KHUSUS.items():
            if nama_topik in gabung:
                perlu_kw = True
                kws_wajib.extend(kws)

        if perlu_kw:
            ada_kw = any(re.search(r'\b' + re.escape(kw) + r'\b', pesan_normal) for kw in kws_wajib)
            if not ada_kw:
                continue

        docs.append(d)
        distances.append(dist)
        metadatas.append(meta)

    kanal_disebut = []
    for k in KANAL_PEMBAYARAN:
        if k == "bsi":
            if re.search(r'\b(bank bsi|va bsi|transfer bsi|atm bsi|teller bsi|mbanking bsi|m-banking bsi|rekening bsi|setor bsi)\b', pesan_normal):
                kanal_disebut.append("bsi")
        else:
            if re.search(r'\b' + re.escape(k) + r'\b', pesan_normal):
                kanal_disebut.append(k)

    if kanal_disebut:
        kandidat_prioritas = []
        kandidat_lain = []
        for d, dist, meta in zip(docs, distances, metadatas):
            j_bag = (meta.get('judul_bagian', '') + ' ' + meta.get('jejak_judul', '')).lower()
            if any(k in j_bag for k in kanal_disebut):
                kandidat_prioritas.append((d, dist, meta))
            else:
                kandidat_lain.append((d, dist, meta))
        if kandidat_prioritas:
            gabung = kandidat_prioritas + kandidat_lain
            docs = [k[0] for k in gabung]
            distances = [k[1] for k in gabung]
            metadatas = [k[2] for k in gabung]

    if re.search(r'\b(definisi|pengertian|arti)\b', pesan_normal):
        kandidat_def = []
        kandidat_non_def = []
        for d, dist, meta in zip(docs, distances, metadatas):
            if 'definisi ' in d.lower():
                kandidat_def.append((d, dist, meta))
            else:
                kandidat_non_def.append((d, dist, meta))
        if kandidat_def:
            gabung = kandidat_def + kandidat_non_def
            docs = [k[0] for k in gabung]
            distances = [k[1] for k in gabung]
            metadatas = [k[2] for k in gabung]

    docs = docs[:TOP_K_RESULTS]
    distances = distances[:TOP_K_RESULTS]
    metadatas = metadatas[:TOP_K_RESULTS]

    kandidat_setelah_filter = [
        (metadatas[i].get('jejak_judul', ''), float(distances[i]))
        for i in range(min(3, len(distances)))
    ]

    if not docs or not distances:
        waktu_ms = round((time.perf_counter() - mulai_waktu) * 1000, 2)
        return False, {
            'jawaban': f"Informasi yang Anda cari tidak ditemukan dalam dokumen resmi kampus. Silakan hubungi admin lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}.",
            'sumber': "",
            'kelas_router': kls_router,
            'keyakinan': conf_router,
            'peluang_eskalasi': p_esk,
            'keputusan_router': "teruskan",
            'jarak_terbaik': None,
            'tumpang_tindih_kata': None,
            'skor_gabungan': None,
            'susunan': susunan,
            'kata_isi': [],
            'keputusan': "teruskan_ke_staf",
            'alasan': "Tidak ada dokumen yang relevan setelah penyaringan topik khusus.",
            'potongan_terpakai': [],
            'kandidat_sebelum_filter': kandidat_sebelum_filter,
            'kandidat_setelah_filter': kandidat_setelah_filter,
            'waktu_ms': waktu_ms,
        }

    jarak_terbaik = float(distances[0])
    potongan_list = []
    for d, dist, meta in zip(docs, distances, metadatas):
        potongan_list.append({
            'teks': d,
            'jarak': float(dist),
            'metadata': meta,
        })

    if susunan == "V2" and p_esk >= AMBANG_VETO_V2:
        waktu_ms = round((time.perf_counter() - mulai_waktu) * 1000, 2)
        return False, {
            'jawaban': (
                f"Pertanyaan Anda berada di luar cakupan informasi otomatis bot "
                f"atau memerlukan penanganan khusus. "
                f"Silakan hubungi admin lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}."
            ),
            'sumber': "",
            'kelas_router': kls_router,
            'keyakinan': conf_router,
            'peluang_eskalasi': p_esk,
            'keputusan_router': "teruskan",
            'jarak_terbaik': jarak_terbaik,
            'tumpang_tindih_kata': None,
            'skor_gabungan': None,
            'susunan': susunan,
            'kata_isi': [],
            'keputusan': "teruskan_ke_staf",
            'alasan': f"Veto Router V2: peluang eskalasi ({p_esk:.4f}) mencapai ambang veto ({AMBANG_VETO_V2}).",
            'dialihkan_oleh': "router",
            'potongan_terpakai': potongan_list,
            'kandidat_sebelum_filter': kandidat_sebelum_filter,
            'kandidat_setelah_filter': kandidat_setelah_filter,
            'waktu_ms': waktu_ms,
        }

    if susunan in ["V1", "V2"] and jarak_terbaik > SEARCH_DISTANCE_THRESHOLD:
        waktu_ms = round((time.perf_counter() - mulai_waktu) * 1000, 2)
        return False, {
            'jawaban': (
                f"Informasi mengenai hal tersebut belum tercantum dengan jelas pada dokumen resmi kampus. "
                f"Silakan hubungi admin lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}."
            ),
            'sumber': "",
            'kelas_router': kls_router,
            'keyakinan': conf_router,
            'peluang_eskalasi': p_esk,
            'keputusan_router': "teruskan",
            'jarak_terbaik': jarak_terbaik,
            'tumpang_tindih_kata': None,
            'skor_gabungan': None,
            'susunan': susunan,
            'kata_isi': [],
            'keputusan': "teruskan_ke_staf",
            'alasan': f"Jarak kosinus terbaik ({jarak_terbaik:.3f}) melebihi ambang batas ({SEARCH_DISTANCE_THRESHOLD}).",
            'dialihkan_oleh': "gerbang",
            'potongan_terpakai': potongan_list,
            'kandidat_sebelum_filter': kandidat_sebelum_filter,
            'kandidat_setelah_filter': kandidat_setelah_filter,
            'waktu_ms': waktu_ms,
        }

    teks_teratas = docs[0]
    teks_teratas_l = teks_teratas.lower()
    meta_top = metadatas[0]
    nama_sumber = meta_top.get('nama_berkas', 'Penjelasan_Fitur_EKTM_BSI_ID_Lengkap update.docx')
    tgl_berlaku = meta_top.get('tanggal_berlaku', '2026-11-09')

    kata_isi = ekstrak_kata_isi(pesan_normal)
    teks_teratas_cek = teks_teratas_l.replace('e-ktm', 'ektm').replace('-', ' ')
    if kata_isi:
        kata_cocok = [k for k in kata_isi if re.search(r'\b' + re.escape(k) + r'\b', teks_teratas_cek)]
        tumpang_tindih = round(len(kata_cocok) / len(kata_isi), 3)
    else:
        kata_cocok = []
        tumpang_tindih = 1.0

    if "mandiri" in kanal_disebut and re.search(r'\b(atm mandiri|transfer atm mandiri)\b', pesan_normal):
        waktu_ms = round((time.perf_counter() - mulai_waktu) * 1000, 2)
        return False, {
            'jawaban': (
                f"Panduan langkah pembayaran melalui Transfer ATM Mandiri saat ini belum tersedia di sistem "
                f"(sesuai catatan dokumen resmi, silakan hubungi admin atau gunakan aplikasi Livin by Mandiri). "
                f"Silakan hubungi admin lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}."
            ),
            'sumber': f"{nama_sumber} (dimasukkan ke indeks: {tgl_berlaku})",
            'kelas_router': kls_router,
            'keyakinan': conf_router,
            'peluang_eskalasi': p_esk,
            'keputusan_router': "teruskan",
            'jarak_terbaik': jarak_terbaik,
            'tumpang_tindih_kata': tumpang_tindih,
            'skor_gabungan': None,
            'susunan': susunan,
            'kata_isi': kata_isi,
            'keputusan': "teruskan_ke_staf",
            'alasan': "Panduan langkah pembayaran melalui Transfer ATM Mandiri tidak tersedia di dokumen resmi (catatan dokumen: tanyakan ke admin).",
            'dialihkan_oleh': "aturan",
            'potongan_terpakai': potongan_list,
            'kandidat_sebelum_filter': kandidat_sebelum_filter,
            'kandidat_setelah_filter': kandidat_setelah_filter,
            'waktu_ms': waktu_ms,
        }

    kanal_tanpa_langkah = [k for k in kanal_disebut if k not in KANAL_DENGAN_LANGKAH_RINCI]
    if kanal_tanpa_langkah:
        waktu_ms = round((time.perf_counter() - mulai_waktu) * 1000, 2)
        daftar_kanal = ", ".join([k.upper() for k in kanal_tanpa_langkah])
        if any(k in KANAL_TERDAFTAR_VA for k in kanal_tanpa_langkah):
            teks_kanal = (
                f"{daftar_kanal} tersedia sebagai pilihan Virtual Account "
                f"(sumber {nama_sumber}, dimasukkan ke indeks: {tgl_berlaku}). "
                f"Panduan langkah pembayaran lewat {daftar_kanal} belum tersedia. "
                f"Silakan hubungi admin lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}."
            )
            alasan_kanal = f"Kanal '{daftar_kanal}' tercantum di daftar pilihan VA namun rincian langkah belum tersedia di dokumen rujukan."
        else:
            teks_kanal = (
                f"Panduan langkah pembayaran lewat {daftar_kanal} belum tersedia pada dokumen rujukan resmi. "
                f"Silakan hubungi admin lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}."
            )
            alasan_kanal = f"Kanal '{daftar_kanal}' tidak tercantum dalam dokumen rujukan resmi."

        return False, {
            'jawaban': teks_kanal,
            'sumber': f"{nama_sumber} (dimasukkan ke indeks: {tgl_berlaku})",
            'kelas_router': kls_router,
            'keyakinan': conf_router,
            'peluang_eskalasi': p_esk,
            'keputusan_router': "teruskan",
            'jarak_terbaik': jarak_terbaik,
            'tumpang_tindih_kata': tumpang_tindih,
            'skor_gabungan': None,
            'susunan': susunan,
            'kata_isi': kata_isi,
            'keputusan': "teruskan_ke_staf",
            'alasan': alasan_kanal,
            'dialihkan_oleh': "aturan",
            'potongan_terpakai': potongan_list,
            'kandidat_sebelum_filter': kandidat_sebelum_filter,
            'kandidat_setelah_filter': kandidat_setelah_filter,
            'waktu_ms': waktu_ms,
        }

    skor_gabungan = None
    if susunan in ["V1", "V2"]:
        if tumpang_tindih < MIN_OVERLAP_KATA:
            waktu_ms = round((time.perf_counter() - mulai_waktu) * 1000, 2)
            return False, {
                'jawaban': (
                    f"Informasi mengenai hal tersebut belum tercantum dengan jelas pada dokumen resmi kampus. "
                    f"Silakan hubungi admin lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}."
                ),
                'sumber': "",
                'kelas_router': kls_router,
                'keyakinan': conf_router,
                'peluang_eskalasi': p_esk,
                'keputusan_router': "teruskan",
                'jarak_terbaik': jarak_terbaik,
                'tumpang_tindih_kata': tumpang_tindih,
                'skor_gabungan': None,
                'susunan': susunan,
                'kata_isi': kata_isi,
                'keputusan': "teruskan_ke_staf",
                'alasan': f"Tumpang tindih kata isi ({tumpang_tindih:.2f}) di bawah batas minimum ({MIN_OVERLAP_KATA}), di luar cakupan dokumen.",
                'dialihkan_oleh': "gerbang",
                'potongan_terpakai': potongan_list,
                'kandidat_sebelum_filter': kandidat_sebelum_filter,
                'kandidat_setelah_filter': kandidat_setelah_filter,
                'waktu_ms': waktu_ms,
            }
    elif susunan == "V3":
        sim_kosinus = max(0.0, 1.0 - jarak_terbaik)
        skor_gabungan = round(0.5 * sim_kosinus + 0.5 * tumpang_tindih, 4)
        if skor_gabungan < AMBANG_V3:
            waktu_ms = round((time.perf_counter() - mulai_waktu) * 1000, 2)
            return False, {
                'jawaban': (
                    f"Informasi mengenai hal tersebut belum tercantum dengan jelas pada dokumen resmi kampus. "
                    f"Silakan hubungi admin lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}."
                ),
                'sumber': "",
                'kelas_router': kls_router,
                'keyakinan': conf_router,
                'peluang_eskalasi': p_esk,
                'keputusan_router': "teruskan",
                'jarak_terbaik': jarak_terbaik,
                'tumpang_tindih_kata': tumpang_tindih,
                'skor_gabungan': skor_gabungan,
                'susunan': susunan,
                'kata_isi': kata_isi,
                'keputusan': "teruskan_ke_staf",
                'alasan': f"Skor gabungan V3 ({skor_gabungan:.3f}) di bawah ambang batas ({AMBANG_V3}).",
                'dialihkan_oleh': "gerbang",
                'potongan_terpakai': potongan_list,
                'kandidat_sebelum_filter': kandidat_sebelum_filter,
                'kandidat_setelah_filter': kandidat_setelah_filter,
                'waktu_ms': waktu_ms,
            }

    judul_bagian = meta_top.get('judul_bagian', '')
    jejak_judul = meta_top.get('jejak_judul', '')
    sumber_str = f"{nama_sumber} (Bagian: {judul_bagian}, dimasukkan ke indeks: {tgl_berlaku})"

    if any(k.lower() in judul_bagian.lower() or k.lower() in jejak_judul.lower() for k in KARANTINA_BAGIAN):
        waktu_ms = round((time.perf_counter() - mulai_waktu) * 1000, 2)
        return False, {
            'jawaban': f"Informasi terkait bagian ini sedang dalam karantina kebijakan. Silakan hubungi admin lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}.",
            'sumber': sumber_str,
            'kelas_router': kls_router,
            'keyakinan': conf_router,
            'peluang_eskalasi': p_esk,
            'keputusan_router': "teruskan",
            'jarak_terbaik': jarak_terbaik,
            'tumpang_tindih_kata': tumpang_tindih,
            'skor_gabungan': skor_gabungan,
            'susunan': susunan,
            'kata_isi': kata_isi,
            'keputusan': "teruskan_ke_staf",
            'alasan': f"Bagian '{judul_bagian}' berada dalam daftar KARANTINA_BAGIAN.",
            'potongan_terpakai': potongan_list,
            'waktu_ms': waktu_ms,
        }

    sudah_tutup, alasan_tgl, tgl_tutup_str = periksa_tanggal_kedaluwarsa_teks(meta_top)
    teks_jawaban = docs[0].strip()

    tanya_status_buka = any(k in pesan_normal for k in [
        'masih buka', 'masih dibuka', 'masih bisa', 'apakah bisa', 'bisa daftar',
        'kapan tutup', 'sudah tutup', 'jadwal buka', 'jadwal pendaftaran'
    ])

    if tanya_status_buka:
        if sudah_tutup:
            waktu_ms = round((time.perf_counter() - mulai_waktu) * 1000, 2)
            jawaban_tegas = (
                f"Pemberitahuan: Pembayaran/pendaftaran untuk {judul_bagian} telah resmi DITUTUP pada tanggal {tgl_tutup_str}. "
                f"Status 'TERBUKA' yang sebelumnya tercantum pada dokumen panduan aplikasi sudah tidak berlaku per tanggal hari ini ({TANGGAL_HARI_INI}). "
                f"Silakan hubungi admin lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}."
            )
            return False, {
                'jawaban': jawaban_tegas,
                'sumber': sumber_str,
                'kelas_router': kls_router,
                'keyakinan': conf_router,
                'peluang_eskalasi': p_esk,
                'keputusan_router': "teruskan",
                'jarak_terbaik': jarak_terbaik,
                'tumpang_tindih_kata': tumpang_tindih,
                'skor_gabungan': skor_gabungan,
                'susunan': susunan,
                'kata_isi': kata_isi,
                'keputusan': "teruskan_ke_staf",
                'alasan': f"Kegiatan sudah ditutup pada {tgl_tutup_str} (melewati tanggal sistem {TANGGAL_HARI_INI}).",
                'dialihkan_oleh': "aturan",
                'potongan_terpakai': potongan_list,
                'waktu_ms': waktu_ms,
            }
        elif not tgl_tutup_str:
            waktu_ms = round((time.perf_counter() - mulai_waktu) * 1000, 2)
            return False, {
                'jawaban': f"Status pembayaran kegiatan berubah sesuai jadwal dan diumumkan lewat Story WhatsApp admin. Silakan hubungi admin lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}.",
                'sumber': sumber_str,
                'kelas_router': kls_router,
                'keyakinan': conf_router,
                'peluang_eskalasi': p_esk,
                'keputusan_router': "teruskan",
                'jarak_terbaik': jarak_terbaik,
                'tumpang_tindih_kata': tumpang_tindih,
                'skor_gabungan': skor_gabungan,
                'susunan': susunan,
                'kata_isi': kata_isi,
                'keputusan': "teruskan_ke_staf",
                'alasan': "Dokumen aktif tidak memuat tanggal tutup; status diumumkan lewat Story WhatsApp admin.",
                'dialihkan_oleh': "aturan",
                'potongan_terpakai': potongan_list,
                'waktu_ms': waktu_ms,
            }

    is_kegiatan = any(k in jejak_judul.lower() for k in [
        'kegiatan', 'seminar', 'wisuda', 'serkom', 'workshop', 'jaket', 'rincian kategori pembayaran akademik'
    ])
    tanya_biaya = any(k in pesan_normal for k in ['biaya', 'tarif', 'harga', 'bayar berapa', 'berapa biaya', 'nominal'])
    if (meta_top.get('memuat_biaya') == 'ya' and is_kegiatan and tanya_biaya) or (is_kegiatan and tanya_biaya):
        waktu_ms = round((time.perf_counter() - mulai_waktu) * 1000, 2)
        return False, {
            'jawaban': f"Informasi mengenai angka atau rincian biaya kegiatan tidak ditampilkan secara otomatis oleh bot. Silakan hubungi admin lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}.",
            'sumber': sumber_str,
            'kelas_router': kls_router,
            'keyakinan': conf_router,
            'peluang_eskalasi': p_esk,
            'keputusan_router': "teruskan",
            'jarak_terbaik': jarak_terbaik,
            'tumpang_tindih_kata': tumpang_tindih,
            'skor_gabungan': skor_gabungan,
            'susunan': susunan,
            'kata_isi': kata_isi,
            'keputusan': "teruskan_ke_staf",
            'alasan': "Pertanyaan biaya kegiatan atau potongan memuat angka biaya (Rp), dialihkan sesuai kebijakan K-017.",
            'dialihkan_oleh': "aturan",
            'potongan_terpakai': potongan_list,
            'waktu_ms': waktu_ms,
        }

    teks_jawaban = bersihkan_klaim_status_saat_ini(teks_jawaban)
    teks_jawaban = format_jawaban_manusiawi(teks_jawaban, meta_top, pesan_normal)

    waktu_ms = round((time.perf_counter() - mulai_waktu) * 1000, 2)
    return True, {
        'jawaban': teks_jawaban,
        'sumber': sumber_str,
        'kelas_router': kls_router,
        'keyakinan': conf_router,
        'peluang_eskalasi': p_esk,
        'keputusan_router': "jawab",
        'jarak_terbaik': jarak_terbaik,
        'tumpang_tindih_kata': tumpang_tindih,
        'skor_gabungan': skor_gabungan,
        'susunan': susunan,
        'kata_isi': kata_isi,
        'keputusan': "jawab",
        'alasan': f"Informasi ditemukan relevan pada '{nama_sumber}' (jarak kosinus: {jarak_terbaik:.3f}, overlap leksikal: {tumpang_tindih:.2f}).",
        'dialihkan_oleh': "-",
        'potongan_terpakai': potongan_list,
        'kandidat_sebelum_filter': kandidat_sebelum_filter,
        'kandidat_setelah_filter': kandidat_setelah_filter,
        'waktu_ms': waktu_ms,
    }


# ==============================================================
# FUNGSI UTAMA BOT (SISTEM B CASCADE)
# ==============================================================

def jawab_sistem_b(pesan: str, router_mode: str = None, susunan: str = None) -> dict:
    """
    Implementasi penalaran Sistem B dengan CASCADE OTOMATIS:
    Langkah 0: Gerbang Keras Keamanan (Topik Terlarang, Bukti Bayar, Beasiswa) -> Teruskan Admin (Bypass Cascade)
    Langkah 1: Router IndoBERTweet (Ambang T = 0.100) -> Jawab jika lolos gerbang leksikal, eskalasi jika gagal
    Langkah 2: Router Klasik TF-IDF (Ambang T = 0.300, lazy loaded) -> Jawab jika lolos gerbang leksikal, eskalasi jika gagal
    Langkah 3: Router Aturan Segmen Panduan BSI.ID -> Jawab jika cocok segmen dan overlap kata isi >= 50%
    Fallback : Jika ketiganya meneruskan, barulah chatbot memberikan jawaban penolakan ramah dan meneruskan ke admin.
    """
    mulai_waktu = time.perf_counter()

    if susunan is None:
        susunan = SUSUNAN_AKTIF.strip().upper()
    else:
        susunan = susunan.strip().upper()

    pesan_normal = normalisasi(pesan)

    # ------------------------------------------------------------------
    # LANGKAH 0: GERBANG KERAS (KEAMANAN PERTAMA — TIDAK BOLEH DI-CASCADE)
    # ------------------------------------------------------------------
    terlarang, alasan_terlarang = periksa_topik_terlarang(pesan_normal)
    if terlarang:
        waktu_ms = round((time.perf_counter() - mulai_waktu) * 1000, 2)
        return {
            'jawaban': (
                f"Mohon maaf, pertanyaan Anda tidak dapat dijawab secara otomatis oleh sistem. "
                f"{alasan_terlarang} "
                f"Silakan hubungi admin lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}."
            ),
            'sumber': "",
            'kelas_router': "topik_terlarang",
            'keyakinan': 1.0,
            'peluang_eskalasi': 1.0,
            'keputusan_router': "teruskan",
            'router_penjawab': "tidak_ada",
            'telemetri_cascade': {
                'gerbang_keras': f"Topik terlarang / keamanan aktif: {alasan_terlarang}",
                'indobertweet': {'status': 'dilewati', 'alasan': 'Gerbang keras keamanan aktif di Langkah 0'},
                'klasik': {'status': 'dilewati', 'alasan': 'Gerbang keras keamanan aktif di Langkah 0'},
                'aturan': {'status': 'dilewati', 'alasan': 'Gerbang keras keamanan aktif di Langkah 0'},
                'router_penjawab': 'tidak_ada'
            },
            'jarak_terbaik': None,
            'tumpang_tindih_kata': None,
            'skor_gabungan': None,
            'susunan': susunan,
            'kata_isi': [],
            'keputusan': "teruskan_ke_staf",
            'alasan': f"Topik terlarang / kebijakan eskalasi: {alasan_terlarang}",
            'dialihkan_oleh': "aturan",
            'potongan_terpakai': [],
            'waktu_ms': waktu_ms,
        }

    is_tanya_histori = is_kueri_riwayat_pembayaran(pesan_normal)
    pola_bukti_keras = [
        'bukti transfer', 'bukti bayar', 'sudah transfer', 'sudah bayar dan kirim',
        'struk transfer', 'struk pembayaran', 'lampiran transfer', 'ss transfer',
        'screenshoot transfer', 'screenshot bukti', 'berikut bukti'
    ]
    is_bukti_keras = any(p in pesan_normal for p in pola_bukti_keras) or (
        ('transfer' in pesan_normal or 'bayar' in pesan_normal) and 
        ('nim' in pesan_normal or 'rp' in pesan_normal or 'norek' in pesan_normal)
    )
    if is_bukti_keras and not is_tanya_histori:
        cek_bukti = periksa_kelengkapan_bukti_bayar(pesan_normal)
        waktu_ms = round((time.perf_counter() - mulai_waktu) * 1000, 2)
        if cek_bukti['status_lengkap']:
            teks_status = (
                f"Silakan kirim bukti pembayaran beserta keterangan tersebut ke admin "
                f"lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}."
            )
        else:
            kurang = ", ".join(cek_bukti['field_hilang'])
            teks_status = (
                f"Keterangan bukti pembayaran Anda belum lengkap. Keterangan yang belum tercantum: {kurang}. "
                f"Silakan kirim bukti pembayaran beserta keterangan tersebut ke admin "
                f"lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}."
            )
        return {
            'jawaban': teks_status,
            'sumber': "",
            'kelas_router': "bukti_bayar",
            'keyakinan': 0.95,
            'peluang_eskalasi': 0.0,
            'keputusan_router': "teruskan",
            'router_penjawab': "tidak_ada",
            'telemetri_cascade': {
                'gerbang_keras': "Bukti bayar aktif di Langkah 0",
                'indobertweet': {'status': 'dilewati', 'alasan': 'Kanal bukti bayar wajib dialihkan ke staf'},
                'klasik': {'status': 'dilewati', 'alasan': 'Kanal bukti bayar wajib dialihkan ke staf'},
                'aturan': {'status': 'dilewati', 'alasan': 'Kanal bukti bayar wajib dialihkan ke staf'},
                'router_penjawab': 'tidak_ada'
            },
            'jarak_terbaik': None,
            'tumpang_tindih_kata': None,
            'skor_gabungan': None,
            'susunan': susunan,
            'kata_isi': [],
            'keputusan': "teruskan_ke_staf",
            'alasan': "Bukti bayar tidak dibaca otomatis oleh bot; diperiksa kelengkapan teks dan diarahkan ke admin.",
            'dialihkan_oleh': "aturan",
            'potongan_terpakai': [],
            'waktu_ms': waktu_ms,
        }

    # Jika pengguna memilih satu router spesifik secara manual (bukan cascade)
    mode_str = (router_mode or "").strip().lower()
    if mode_str in ["indobertweet", "klasik", "aturan"]:
        # Eksekusi mode non-cascade manual
        if mode_str == "aturan":
            rule_segmen = cocokkan_segmen_aturan(pesan_normal)
            if rule_segmen:
                sukses, res = eksekusi_jawaban_segmen_aturan(pesan_normal, rule_segmen, susunan=susunan, mulai_waktu=mulai_waktu)
                if sukses:
                    res['router_penjawab'] = "aturan"
                    return res
            waktu_ms = round((time.perf_counter() - mulai_waktu) * 1000, 2)
            return {
                'jawaban': f"Pertanyaan Anda berada di luar cakupan informasi otomatis bot. Silakan hubungi admin lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}.",
                'sumber': "",
                'kelas_router': "lainnya",
                'keyakinan': 0.0,
                'peluang_eskalasi': 1.0,
                'keputusan_router': "teruskan",
                'router_penjawab': "tidak_ada",
                'jarak_terbaik': None,
                'tumpang_tindih_kata': None,
                'skor_gabungan': None,
                'susunan': susunan,
                'kata_isi': [],
                'keputusan': "teruskan_ke_staf",
                'alasan': "Tidak cocok dengan aturan kata kunci segmen panduan.",
                'dialihkan_oleh': "router",
                'potongan_terpakai': [],
                'waktu_ms': waktu_ms,
            }

        # indobertweet atau klasik tunggal
        info_r = klasifikasi_router_detail(pesan_normal, mode=mode_str)
        kls_r = info_r['kelas']
        conf_r = info_r['keyakinan']
        p_esk_r = info_r.get('peluang_eskalasi', 0.0)
        ambang_r = 0.100 if mode_str == "indobertweet" else 0.300

        if kls_r == "bukti_bayar" and not is_tanya_histori:
            cek_bukti = periksa_kelengkapan_bukti_bayar(pesan_normal)
            waktu_ms = round((time.perf_counter() - mulai_waktu) * 1000, 2)
            kurang = ", ".join(cek_bukti['field_hilang']) if not cek_bukti['status_lengkap'] else ""
            teks_status = (
                f"Keterangan bukti pembayaran Anda belum lengkap. Keterangan yang belum tercantum: {kurang}. "
                f"Silakan kirim bukti pembayaran beserta keterangan tersebut ke admin lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}."
            ) if kurang else (
                f"Silakan kirim bukti pembayaran beserta keterangan tersebut ke admin lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}."
            )
            return {
                'jawaban': teks_status,
                'sumber': "",
                'kelas_router': kls_r,
                'keyakinan': conf_r,
                'peluang_eskalasi': p_esk_r,
                'keputusan_router': "teruskan",
                'router_penjawab': "tidak_ada",
                'jarak_terbaik': None,
                'tumpang_tindih_kata': None,
                'skor_gabungan': None,
                'susunan': susunan,
                'kata_isi': [],
                'keputusan': "teruskan_ke_staf",
                'alasan': "Bukti bayar tidak dibaca otomatis oleh bot; diperiksa kelengkapan teks dan diarahkan ke admin.",
                'dialihkan_oleh': "aturan",
                'potongan_terpakai': [],
                'waktu_ms': waktu_ms,
            }

        if kls_r in KELAS_ESKALASI or p_esk_r >= ambang_r or (kls_r == "bukti_bayar" and is_tanya_histori):
            waktu_ms = round((time.perf_counter() - mulai_waktu) * 1000, 2)
            return {
                'jawaban': f"Pertanyaan Anda berada di luar cakupan informasi otomatis bot. Silakan hubungi admin lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}.",
                'sumber': "",
                'kelas_router': kls_r,
                'keyakinan': conf_r,
                'peluang_eskalasi': p_esk_r,
                'keputusan_router': "teruskan",
                'router_penjawab': "tidak_ada",
                'jarak_terbaik': None,
                'tumpang_tindih_kata': None,
                'skor_gabungan': None,
                'susunan': susunan,
                'kata_isi': [],
                'keputusan': "teruskan_ke_staf",
                'alasan': f"Router {mode_str}: kelas '{kls_r}' (keyakinan: {conf_r:.2f}) atau peluang eskalasi ({p_esk_r:.4f}) >= {ambang_r}.",
                'dialihkan_oleh': "router",
                'potongan_terpakai': [],
                'waktu_ms': waktu_ms,
            }

        sukses, res = eksekusi_pencarian_dan_gerbang(pesan_normal, susunan=susunan, kls_router=kls_r, conf_router=conf_r, p_esk=p_esk_r, mulai_waktu=mulai_waktu)
        res['router_penjawab'] = mode_str if sukses else "tidak_ada"
        return res

    # ------------------------------------------------------------------
    # CASCADE ROUTER OTOMATIS: 1) IndoBERTweet -> 2) TF-IDF Klasik -> 3) Aturan
    # ------------------------------------------------------------------
    telemetri_cascade = {
        'indobertweet': {'status': '-', 'alasan': '-'},
        'klasik': {'status': '-', 'alasan': '-'},
        'aturan': {'status': '-', 'alasan': '-'},
        'router_penjawab': 'tidak_ada'
    }

    # --- LANGKAH 1: ROUTER INDOBERTWEET (UTAMA) ---
    info_bert = klasifikasi_router_detail(pesan_normal, mode="indobertweet")
    kls_bert = info_bert['kelas']
    conf_bert = info_bert['keyakinan']
    p_esk_bert = info_bert.get('peluang_eskalasi', 0.0)

    if kls_bert == "bukti_bayar" and not is_tanya_histori:
        cek_bukti = periksa_kelengkapan_bukti_bayar(pesan_normal)
        waktu_ms = round((time.perf_counter() - mulai_waktu) * 1000, 2)
        kurang = ", ".join(cek_bukti['field_hilang']) if not cek_bukti['status_lengkap'] else ""
        teks_status = (
            f"Keterangan bukti pembayaran Anda belum lengkap. Keterangan yang belum tercantum: {kurang}. "
            f"Silakan kirim bukti pembayaran beserta keterangan tersebut ke admin "
            f"lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}."
        ) if kurang else (
            f"Silakan kirim bukti pembayaran beserta keterangan tersebut ke admin "
            f"lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}."
        )
        return {
            'jawaban': teks_status,
            'sumber': "",
            'kelas_router': "bukti_bayar",
            'keyakinan': conf_bert,
            'peluang_eskalasi': p_esk_bert,
            'keputusan_router': "teruskan",
            'router_penjawab': "tidak_ada",
            'telemetri_cascade': {
                'indobertweet': {'status': 'eskalasi_bukti_bayar', 'alasan': 'Prediksi bukti_bayar diarahkan ke staf'},
                'klasik': {'status': 'dilewati', 'alasan': 'Kanal bukti bayar dialihkan di router utama'},
                'aturan': {'status': 'dilewati', 'alasan': 'Kanal bukti bayar dialihkan di router utama'},
                'router_penjawab': 'tidak_ada'
            },
            'jarak_terbaik': None,
            'tumpang_tindih_kata': None,
            'skor_gabungan': None,
            'susunan': susunan,
            'kata_isi': [],
            'keputusan': "teruskan_ke_staf",
            'alasan': "Bukti bayar tidak dibaca otomatis oleh bot; diperiksa kelengkapan teks dan diarahkan ke admin.",
            'dialihkan_oleh': "aturan",
            'potongan_terpakai': [],
            'waktu_ms': waktu_ms,
        }

    if kls_bert in KELAS_ESKALASI or p_esk_bert >= 0.100 or (kls_bert == "bukti_bayar" and is_tanya_histori):
        alasan_esk_bert = f"P(eskalasi) {p_esk_bert:.4f} >= 0.100 (kelas '{kls_bert}', keyakinan {conf_bert:.2f})"
        telemetri_cascade['indobertweet'] = {'status': 'eskalasi', 'alasan': alasan_esk_bert}
    else:
        sukses, res_cari = eksekusi_pencarian_dan_gerbang(pesan_normal, susunan=susunan, kls_router=kls_bert, conf_router=conf_bert, p_esk=p_esk_bert, mulai_waktu=mulai_waktu)
        if sukses:
            res_cari['router_penjawab'] = "indobertweet"
            telemetri_cascade['indobertweet'] = {'status': 'jawab', 'alasan': f"Lolos router (P_esk={p_esk_bert:.4f}) dan gerbang leksikal (jarak={res_cari['jarak_terbaik']:.3f}, overlap={res_cari['tumpang_tindih_kata']:.2f})"}
            telemetri_cascade['klasik'] = {'status': 'dilewati', 'alasan': 'Sudah terjawab oleh IndoBERTweet'}
            telemetri_cascade['aturan'] = {'status': 'dilewati', 'alasan': 'Sudah terjawab oleh IndoBERTweet'}
            telemetri_cascade['router_penjawab'] = "indobertweet"
            res_cari['telemetri_cascade'] = telemetri_cascade
            return res_cari
        else:
            alasan_esk_bert = f"Router lolos, namun {res_cari['alasan']}"
            telemetri_cascade['indobertweet'] = {'status': 'eskalasi', 'alasan': alasan_esk_bert}

    # --- LANGKAH 2: ROUTER TF-IDF KLASIK (PEMBANDING, LAZY LOADED) ---
    info_klasik = inferensi_klasik_lazy(pesan_normal)
    if info_klasik is not None:
        kls_klasik = info_klasik['kelas']
        conf_klasik = info_klasik['keyakinan']
        p_esk_klasik = info_klasik['peluang_eskalasi']

        if kls_klasik in KELAS_ESKALASI or p_esk_klasik >= 0.300 or (kls_klasik == "bukti_bayar" and is_tanya_histori):
            alasan_esk_klasik = f"P(eskalasi) {p_esk_klasik:.4f} >= 0.300 (kelas '{kls_klasik}', keyakinan {conf_klasik:.2f})"
            telemetri_cascade['klasik'] = {'status': 'eskalasi', 'alasan': alasan_esk_klasik}
        else:
            sukses_k, res_cari_k = eksekusi_pencarian_dan_gerbang(pesan_normal, susunan=susunan, kls_router=kls_klasik, conf_router=conf_klasik, p_esk=p_esk_klasik, mulai_waktu=mulai_waktu)
            if sukses_k:
                res_cari_k['router_penjawab'] = "klasik"
                telemetri_cascade['klasik'] = {'status': 'jawab', 'alasan': f"Lolos router klasik (P_esk={p_esk_klasik:.4f}) dan gerbang leksikal (jarak={res_cari_k['jarak_terbaik']:.3f}, overlap={res_cari_k['tumpang_tindih_kata']:.2f})"}
                telemetri_cascade['aturan'] = {'status': 'dilewati', 'alasan': 'Sudah terjawab oleh router klasik'}
                telemetri_cascade['router_penjawab'] = "klasik"
                res_cari_k['telemetri_cascade'] = telemetri_cascade
                return res_cari_k
            else:
                alasan_esk_klasik = f"Router klasik lolos, namun {res_cari_k['alasan']}"
                telemetri_cascade['klasik'] = {'status': 'eskalasi', 'alasan': alasan_esk_klasik}
    else:
        telemetri_cascade['klasik'] = {'status': 'eskalasi', 'alasan': 'Model klasik tidak tersedia'}

    # --- LANGKAH 3: ROUTER ATURAN (RULE-BASED SEGMEN MATCHING) ---
    rule_segmen = cocokkan_segmen_aturan(pesan_normal)
    if rule_segmen is not None:
        sukses_r, res_r = eksekusi_jawaban_segmen_aturan(pesan_normal, rule_segmen, susunan=susunan, mulai_waktu=mulai_waktu)
        if sukses_r:
            res_r['router_penjawab'] = "aturan"
            telemetri_cascade['aturan'] = {'status': 'jawab', 'alasan': f"Cocok segmen {rule_segmen['id']} ({rule_segmen['target_judul']}) dan overlap leksikal ({res_r['tumpang_tindih_kata']:.2f}) >= 50%"}
            telemetri_cascade['router_penjawab'] = "aturan"
            res_r['telemetri_cascade'] = telemetri_cascade
            return res_r
        else:
            telemetri_cascade['aturan'] = {'status': 'eskalasi', 'alasan': res_r['alasan']}
    else:
        telemetri_cascade['aturan'] = {'status': 'eskalasi', 'alasan': 'Tidak ada segmen aturan yang cocok secara spesifik / ambigu'}

    # --- JIKA KETIGA ROUTER MENERUSKAN (FINAL FALLBACK) ---
    waktu_ms = round((time.perf_counter() - mulai_waktu) * 1000, 2)
    alasan_lengkap = (
        f"Semua router meneruskan ke admin: "
        f"[IndoBERTweet: {telemetri_cascade['indobertweet']['alasan']}] -> "
        f"[Klasik: {telemetri_cascade['klasik']['alasan']}] -> "
        f"[Aturan: {telemetri_cascade['aturan']['alasan']}]"
    )
    return {
        'jawaban': (
            f"Pertanyaan Anda berada di luar cakupan informasi otomatis bot "
            f"atau memerlukan penanganan khusus. "
            f"Silakan hubungi admin lewat WhatsApp resmi kampus pada jam layanan {JAM_LAYANAN_TEKS}."
        ),
        'sumber': "",
        'kelas_router': kls_bert,
        'keyakinan': conf_bert,
        'peluang_eskalasi': p_esk_bert,
        'keputusan_router': "teruskan",
        'router_penjawab': "tidak_ada",
        'telemetri_cascade': telemetri_cascade,
        'jarak_terbaik': None,
        'tumpang_tindih_kata': None,
        'skor_gabungan': None,
        'susunan': susunan,
        'kata_isi': [],
        'keputusan': "teruskan_ke_staf",
        'alasan': alasan_lengkap,
        'dialihkan_oleh': "router",
        'potongan_terpakai': [],
        'waktu_ms': waktu_ms,
    }


def jawab_gemini(pesan: str) -> dict:
    """Sistem A: Pembanding menggunakan LLM Gemini (dikerjakan di T8)."""
    raise NotImplementedError("Sistem A (Gemini) belum diaktifkan. Akan diimplementasikan pada tahap T8.")


def jawab_bm25(pesan: str) -> dict:
    """Sistem Pembanding: Pencarian kata kunci BM25 (dikerjakan di T8)."""
    raise NotImplementedError("Sistem BM25 belum diaktifkan. Akan diimplementasikan pada tahap T8.")


def jawab(pesan: str, sistem: str = "B", router_mode: str = None, susunan: str = None) -> dict:
    """
    Fungsi utama antarmuka bot.
    Parameter:
      - pesan: teks masukan pertanyaan
      - sistem: pilihan sistem ("B" | "A" | "BM25")
      - router_mode: pilihan router ("cascade" | "indobertweet" | "klasik" | "aturan")
      - susunan: pilihan susunan sistem B ("V1" | "V2" | "V3")
    Mengembalikan dict hasil lengkap.
    """
    sistem_bersih = sistem.strip().upper()
    if sistem_bersih == "B" or "B " in sistem_bersih:
        return jawab_sistem_b(pesan, router_mode=router_mode, susunan=susunan)
    elif sistem_bersih == "A":
        return jawab_gemini(pesan)
    elif sistem_bersih == "BM25":
        return jawab_bm25(pesan)
    else:
        raise ValueError(f"Sistem tidak dikenali: '{sistem}'. Pilihan yang tersedia: 'B', 'A', 'BM25'.")


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    print("Menguji fungsi bot.jawab() dengan router cascade...")
    tes_pesan = "Halo min, bagaimana cara bayar ektm lewat atm bni?"
    hasil = jawab(tes_pesan)
    print("\nPertanyaan     :", tes_pesan)
    print("Keputusan      :", hasil['keputusan'])
    print("Router Penjawab:", hasil.get('router_penjawab'))
    print("Alasan         :", hasil['alasan'])
    print("Kelas          :", hasil['kelas_router'])
    print("Jarak          :", hasil['jarak_terbaik'])
    print("Sumber         :", hasil['sumber'])
    print("Jawaban        :\n", hasil['jawaban'][:200], "...")
    print(f"Waktu          : {hasil['waktu_ms']} ms")

