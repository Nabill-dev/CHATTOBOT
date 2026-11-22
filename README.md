---
title: CHATTOBOT
emoji: 🎓
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.66.1"
app_file: app.py
pinned: false
---

# CHATTOBOT — Chatbot Administrasi Kampus (Sistem B)

Aplikasi Chatbot Administrasi Kampus berbasis Streamlit dengan arsitektur RAG Dokumen Panduan Resmi Kampus dan Router IndoBERTweet fine-tuned.

- **Sistem:** Sistem B (Utama)
- **Susunan Default:** V1 (Keamanan -> Router -> Chroma DB -> Gerbang Leksikal)
- **Model Router:** IndoBERTweet Fine-Tuned (6 Kelas Intent)
- **Model Embedding:** `intfloat/multilingual-e5-small`
- **Basis Data Vektor:** Chroma DB (Dokumen Panduan Resmi BSI.ID)
