# CHATTOBOT — Chatbot Administrasi Kampus (Sistem B)

Aplikasi Chatbot Administrasi Kampus berbasis Streamlit dengan arsitektur Retrieval-Augmented Generation (RAG) Dokumen Panduan Resmi Kampus dan Router IndoBERTweet Fine-Tuned.

## Konfigurasi Sistem
- **Sistem Utama:** Sistem B (Streamlit)
- **Susunan Default:** V1 (Keamanan → Router → Pencarian Chroma DB → Gerbang Leksikal)
- **Model Router:** IndoBERTweet Fine-Tuned (6 Kelas Intent)
- **Model Embedding:** `intfloat/multilingual-e5-small` (Prefix passage/query)
- **Basis Data Vektor:** Chroma DB (`chroma_db/`, Dokumen Panduan BSI.ID)
- **Ambang Jarak Kosinus:** Maksimum 0,60 (5 chunk teratas)
- **Ambang Leksikal:** Tumpang-tindih kata isi minimal 50%
- **Platform:** Streamlit Community Cloud

## Antarmuka Web
- Tombol **New Chat** (membersihkan sesi obrolan baru)
- Penyimpanan riwayat obrolan dalam format JSON
- 8 Tombol contoh pertanyaan cepat
- Panel telemetri teknis (waktu eksekusi, router confidence, sumber chunk, jarak kosinus, overlap)
- Format respons ramah mahasiswa
