# ==============================================================
# config.py — Pengaturan terpusat untuk proyek CHATTOBOT
# Semua path, nama model, ambang, dan konstanta ada di sini.
# Tanggal hari ini dibaca dari sistem saat dijalankan.
# ==============================================================

import os
from pathlib import Path
from datetime import date

# --- Path proyek ---
# ROOT otomatis mengarah ke G:\CHATTOBOT (satu tingkat di atas src\)
ROOT = Path(__file__).resolve().parent.parent

# Folder data (JANGAN diubah isinya lewat kode)
DATA_DIR          = ROOT / "data"
CHAT_ANONIM_DIR   = DATA_DIR / "chat_anonim"
EXCEL_PATH        = CHAT_ANONIM_DIR / "Chat_Mahasiswa_Admin_Anonim_dan_Analisis_BERLABEL.xlsx"
DOKUMEN_DIR       = DATA_DIR / "dokumen"
DOKUMEN_RESMI_DIR = DOKUMEN_DIR / "resmi"
DOKUMEN_WAWANCARA = DOKUMEN_DIR / "wawancara"
FAQ_DIR           = DATA_DIR / "faq"
KATALOG_CSV       = DOKUMEN_DIR / "katalog_dokumen.csv"

# Folder hasil olahan (boleh ditulis)
SRC_DIR    = ROOT / "src"
MODELS_DIR = ROOT / "models"
HASIL_DIR  = ROOT / "hasil"
EVAL_DIR   = ROOT / "eval"
DOCS_DIR   = ROOT / "docs"

# Folder indeks Chroma
CHROMA_DIR = ROOT / "chroma_db"

# --- Tanggal hari ini (dibaca dari sistem, JANGAN ditulis tetap) ---
TANGGAL_HARI_INI = date.today()

# --- Jam layanan admin (belum diverifikasi admin) ---
JAM_LAYANAN_TEKS = "08.00-22.00 WIB"

# --- Seed untuk reproduksibilitas ---
SEED = 42

# --- Nama sheet di Excel ---
SHEET_PESAN    = "Pesan_Anonim"
SHEET_SESI     = "Sesi"
SHEET_FAQ      = "Draf_FAQ"
SHEET_ESKALASI = "Eskalasi"
SHEET_KAMUS    = "Kamus"

# --- Model embedding untuk pencarian ---
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
# Awalan wajib untuk model E5
EMBEDDING_PREFIX_QUERY   = "query: "
EMBEDDING_PREFIX_PASSAGE = "passage: "

# --- Model router (IndoBERTweet) ---
ROUTER_MODEL_NAME = "indolem/indobertweet-base-uncased"
ROUTER_MAX_LENGTH = 96          # panjang token maksimal untuk fine-tuning
ROUTER_BATCH_SIZE = 8           # ukuran batch (disesuaikan GPU 4 GB)
ROUTER_EPOCHS     = 10          # epoch pelatihan (bisa disetel ulang)
ROUTER_LR         = 2e-5        # learning rate
ROUTER_FP16       = False       # JANGAN ubah ke True tanpa izin

# Folder simpan model router
ROUTER_SAVE_DIR     = MODELS_DIR / "router_indobertweet"
ROUTER_TFIDF_DIR    = MODELS_DIR / "router_tfidf"
ROUTER_CONFIG_FILE  = "router_config.json"

# --- Router yang sedang aktif ---
# Pilihan: "aturan" (berbasis kata kunci) | "klasik" (TF-IDF + SVM/LogReg) | "indobertweet" (fine-tuned transformer)
# Cukup ubah nilai ini untuk beralih jenis router tanpa mengubah kode lain
ROUTER_AKTIF = "indobertweet"

# --- Tiga Susunan Sistem (Tahap Penutupan) ---
# Pilihan:
# "V1" (sekarang): router -> pencarian -> gerbang kata kunci (default saat ini, jangan diubah)
# "V2": pencarian dulu; router hanya menjadi veto jika P(eskalasi) >= AMBANG_VETO_V2 (eksperimen, belum divalidasi)
# "V3": gerbang kata kunci diganti skor gabungan = jarak kosinus dan overlap (eksperimen, belum divalidasi)
SUSUNAN_AKTIF = "V1"
AMBANG_VETO_V2 = 0.50  # Nilai awal, belum disetel
AMBANG_V3 = 0.50       # Nilai awal, belum disetel (skor gabungan = 0.5*(1 - jarak_terbaik) + 0.5*overlap)

# --- 6 Kelas router ---
# Kelas yang bisa dijawab bot (dengan syarat info tersedia)
KELAS_BOT = [
    "bukti_bayar",           # bot kumpulkan info, lalu teruskan ke admin
    "pembayaran_umum",       # bot jawab jika info ada di dokumen
    "kegiatan_pmb",          # bot jawab jika info ada di dokumen
    "login_akun",            # bot jawab langkah umum
]
# Kelas yang SELALU diteruskan ke staf
KELAS_ESKALASI = [
    "beasiswa_pendaftaran",  # selalu eskalasi
    "lainnya",               # selalu eskalasi
]
SEMUA_KELAS = KELAS_BOT + KELAS_ESKALASI

# --- Pemetaan topik otomatis → 6 kelas router ---
# Dipakai saat label manual (topik_koreksi) kosong
PEMETAAN_TOPIK = {
    "Kirim bukti bayar":                             "bukti_bayar",
    "Cara bayar dan nominal":                        "pembayaran_umum",
    "Batas atau perpanjangan pembayaran":             "pembayaran_umum",
    "Login dan akun":                                "login_akun",
    "Atribut PMB (kaos, baju, id card)":             "kegiatan_pmb",
    "Acara PMB dan aturan kegiatan":                 "kegiatan_pmb",
    "Pendaftaran, daftar ulang, prodi":              "kegiatan_pmb",
    "Kegiatan akademik (seminar, workshop, serkom)": "kegiatan_pmb",
    "Beasiswa":                                      "beasiswa_pendaftaran",
    # Semua di bawah ini → lainnya (eskalasi)
    "Lain-lain":                                     "lainnya",
    "Penutup/lain-lain":                             "lainnya",
    "Surat dan izin":                                "lainnya",
    "Minta kontak dosen/staf":                       "lainnya",
    "Nilai, jadwal, dan perkuliahan":                "lainnya",
}
# Topik yang dikeluarkan dari data latih (bukan percakapan Q&A)
TOPIK_DIKELUARKAN = [
    "Broadcast/info dari admin",
]

# --- Ambang keyakinan router ---
# Jika probabilitas tertinggi < ambang, teruskan ke staf
# Nilai awal; akan disetel dari data uji di T5/T6
ROUTER_CONFIDENCE_THRESHOLD = 0.5

# --- Ambang kemiripan pencarian ---
# Jarak kosinus maksimal untuk menerima hasil pencarian
# Nilai awal 0.6 tetap dipertahankan; ambang final disetel di T8 memakai set validasi terpisah dari set uji
SEARCH_DISTANCE_THRESHOLD = 0.6

# --- Gerbang kedua (leksikal): proporsi minimal kata isi yang harus muncul di potongan teratas ---
MIN_OVERLAP_KATA = 0.5

# Daftar nama bank atau kanal pembayaran untuk aturan entitas khusus
KANAL_PEMBAYARAN = [
    "bni", "bca", "mandiri", "bri", "bsi", "permata", "alfamart",
    "indomaret", "seabank", "dana", "gopay", "ovo", "shopeepay", "linkaja", "qris"
]

# Kata-kata umum, kata tanya, dan kata sapaan yang dibuang saat mengekstrak kata isi pertanyaan
KATA_UMUM_STOPWORDS = {
    # Sapaan
    "min", "admin", "kak", "kakak", "ka", "mba", "mas", "pak", "bu", "halo", "hai", "assalamualaikum",
    # Kata tanya
    "apa", "apakah", "bagaimana", "gimana", "dimana", "kemana", "kapan", "kenapa", "mengapa", "siapa", "berapa",
    # Partikel / penegas
    "ya", "sih", "dong", "deh", "kan", "tuh", "nih", "lah", "kah", "pun",
    # Kata hubung & preposisi
    "dan", "atau", "ke", "di", "dari", "pada", "untuk", "buat", "oleh", "dengan", "tentang", "terkait", "soal", "mengenai",
    # Kata penunjuk & kopula
    "ini", "itu", "yang", "yg", "ada", "adalah", "yaitu",
    # Kata kerja/bantu umum
    "bisa", "dapat", "tolong", "mohon", "mau", "ingin", "tanya", "info", "informasi",
    "sudah", "belum", "tidak", "bukan",
    # Kata ganti orang
    "saya", "aku", "kami", "kita", "anda", "kamu", "sy",
}

# --- Pencarian: jumlah potongan teratas ---
TOP_K_RESULTS = 5

# --- Gemini (sistem pembanding A) ---
GEMINI_MODEL = "gemini-3.8-flash"
# Batas percobaan ulang untuk galat 429/503
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_DELAY = 5  # detik antar percobaan

# --- Pemecahan dokumen (chunking) ---
CHUNK_SIZE     = 500   # karakter per potongan
CHUNK_OVERLAP  = 100   # tumpang tindih antar potongan

# --- Panjang teks input router ---
# Gabungan maksimal 4 pesan mahasiswa pertama, potong di 400 karakter
ROUTER_MAX_PESAN     = 4
ROUTER_MAX_KARAKTER  = 400

# --- Penanda anonimisasi yang diseragamkan ---
PETA_ANONIM = {
    "[NAMA]":       "@user",
    "[NIM]":        "nim",
    "[NO_HP]":      "nohp",
    "[ALAMAT]":     "alamat",
    "[EMAIL]":      "email",
    "[PASSWORD]":   "pass",
    "[TANGGAL]":    "tanggal",
}

# --- Daftar topik yang TIDAK boleh dijawab bot ---
TOPIK_TERLARANG = [
    "komplain atau konfirmasi nilai",
    "pindah program studi (mahasiswa aktif)",
    "status pembayaran belum lunas atau kurang bayar",
    "pembayaran lewat gerai tidak masuk",
    "pengunduran diri atau tidak mampu membayar",
    "reset kata sandi",            # bot hanya beri langkah umum
    "nomor pribadi dosen/staf",
    "beasiswa",                    # selalu eskalasi
    "nominal pribadi",
    "koreksi pembayaran",
]

# --- Info yang harus dikumpulkan bot sebelum eskalasi bukti bayar ---
FIELD_BUKTI_BAYAR = [
    "nama",
    "NIM",
    "kelas",
    "jenis pembayaran/kegiatan",
    "semester atau angsuran ke-",
    "jenis beasiswa (jika ada)",
]

# --- Metode pembayaran yang KONFLIK (belum diputuskan admin) ---
# Bot TIDAK boleh menjawab pilih yang mana; teruskan ke admin
KONFLIK_METODE_BAYAR = [
    "SeaBank",
    "Dana/GoPay",
    "cara unggah bukti pembelian atribut",
]

# --- Bagian dokumen yang DIKARANTINA (kebijakan K-017) ---
KARANTINA_BAGIAN = [
    "Panduan & Informasi Pembayaran Admin Jaket KIP",
]

# --- Kata kunci wajib untuk bagian dokumen khusus topik / prodi / kegiatan ---
# Potongan dengan nama topik khusus berikut HANYA boleh dipakai jika pertanyaan memuat minimal satu kata kuncinya
BAGIAN_TOPIK_KHUSUS = {
    "keperawatan": ["keperawatan", "perawat"],
    "kursus bahasa": ["kursus", "bahasa", "kursus bahasa"],
    "jaket kip": ["jaket", "kip", "almamater", "jaket kip"],
    "skripsi / tugas akhir (ta)": ["skripsi", "ta", "tugas akhir"],
    "skripsi/ta perbaikan": ["skripsi", "ta", "tugas akhir", "perbaikan skripsi", "revisi skripsi"],
    "wisuda": ["wisuda", "kelulusan"],
    "seminar swe": ["swe", "seminar", "entrepreneur"],
    "ujian her": ["her", "ujian her", "remedial"],
}
