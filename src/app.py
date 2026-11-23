# ==============================================================
# app.py — Antarmuka Web Streamlit Chatbot Administrasi Kampus
#
# Dijalankan dengan: streamlit run src\app.py
# Menjalankan pengujian interaktif lokal untuk Sistem B,
# logging penilaian manual, dan pemantauan detail teknis.
# ==============================================================

import csv
import io
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# Tambahkan src ke path modul
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    ROOT, HASIL_DIR, KATALOG_CSV, TANGGAL_HARI_INI,
    ROUTER_AKTIF, ROUTER_CONFIDENCE_THRESHOLD, SEARCH_DISTANCE_THRESHOLD,
    SUSUNAN_AKTIF
)
from bot import jawab, get_embedding_model, get_chroma_collection

# --- Pengaturan Halaman Streamlit ---
st.set_page_config(
    page_title="Chatbot Administrasi Kampus — Prototipe",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# File path untuk log uji lokal
LOG_FILE_PATH = HASIL_DIR / "log_uji.csv"
LOG_COLUMNS = ["waktu", "pertanyaan", "sistem", "router", "keputusan", "sumber", "penilaian", "catatan"]


# ==============================================================
# FUNGSI UTILITAS & RESOURCE CACHING
# ==============================================================

@st.cache_resource(show_spinner="Memuat indeks Chroma DB dan model embedding E5...")
def muat_sumber_daya():
    """
    Memuat model embedding E5 dan basis data Chroma sekali saja.
    Streamlit menjalankan ulang skrip pada tiap interaksi pengguna,
    fungsi ini mencegah pemuatan ulang yang memakan waktu dan memori.
    """
    model = get_embedding_model()
    koleksi = get_chroma_collection()
    return {
        "status": "siap",
        "total_dokumen": koleksi.count(),
        "model_nama": "intfloat/multilingual-e5-small",
    }


def hitung_dokumen_katalog() -> int:
    """Menghitung jumlah dokumen aktif di katalog_dokumen.csv."""
    if not KATALOG_CSV.exists():
        return 0
    try:
        import pandas as pd
        df = pd.read_csv(KATALOG_CSV)
        aktif = df[(df['status'].str.lower() == 'aktif') & (df['diverifikasi_admin'].str.lower() == 'ya')]
        return len(aktif)
    except Exception:
        return 0


def samarkan_data_pribadi(teks: str) -> str:
    """
    Menyamarkan data pribadi dasar sebelum disimpan ke berkas log:
      - NIM (8-12 digit angka berurutan)
      - Nomor HP (pola nomor telepon Indonesia)
      - Alamat Email
    """
    if not teks:
        return ""
    teks = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[EMAIL]', teks)
    teks = re.sub(r'(\+62|62|08)[0-9]{8,12}\b', '[NO_HP]', teks)
    teks = re.sub(r'\b\d{8,12}\b', '[NIM]', teks)
    return teks


def simpan_ke_log(pertanyaan: str, sistem: str, router: str, keputusan: str, sumber: str, penilaian: str, catatan: str):
    """Menyimpan hasil penilaian manual ke berkas hasil/log_uji.csv."""
    HASIL_DIR.mkdir(parents=True, exist_ok=True)
    tulis_header = not LOG_FILE_PATH.exists() or LOG_FILE_PATH.stat().st_size == 0

    pertanyaan_samar = samarkan_data_pribadi(pertanyaan)
    catatan_samar = samarkan_data_pribadi(catatan)
    waktu_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if tulis_header:
            writer.writerow(LOG_COLUMNS)
        writer.writerow([
            waktu_str,
            pertanyaan_samar,
            sistem,
            router,
            keputusan,
            sumber,
            penilaian,
            catatan_samar,
        ])


def hapus_log_uji():
    """Menghapus isi berkas log_uji.csv."""
    if LOG_FILE_PATH.exists():
        LOG_FILE_PATH.unlink()


# ==============================================================
# PENGELOLAAN SESI & RIWAYAT CHAT PERSISTEN
# ==============================================================

RIWAYAT_DIR = HASIL_DIR / "riwayat_chat"
RIWAYAT_DIR.mkdir(parents=True, exist_ok=True)

PESAN_AWAL_DEFAULT = [
    {
        "role": "assistant",
        "content": (
            "Halo! Saya asisten virtual layanan administrasi kampus (Sistem B). "
            "Silakan ajukan pertanyaan seputar pembayaran perkuliahan, akun aplikasi, "
            "atau kegiatan mahasiswa baru. Mohon diingat untuk tidak memasukkan nomor telepon, "
            "kata sandi, atau data pribadi lainnya."
        ),
        "detail": None,
        "id": 0,
    }
]


def muat_daftar_sesi() -> list[dict]:
    """Memuat daftar ringkasan sesi percakapan dari disk JSON, diurutkan terbaru."""
    sesi_list = []
    if not RIWAYAT_DIR.exists():
        return sesi_list
    for f in RIWAYAT_DIR.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                sesi_list.append({
                    "id": data.get("id", f.stem),
                    "judul": data.get("judul", "Percakapan"),
                    "waktu": data.get("waktu", ""),
                })
        except Exception:
            continue
    sesi_list.sort(key=lambda x: x.get("waktu", ""), reverse=True)
    return sesi_list


def simpan_sesi_disk(session_id: str, judul: str, messages: list[dict], waktu: str = None):
    """Menyimpan isi sesi percakapan ke disk JSON."""
    RIWAYAT_DIR.mkdir(parents=True, exist_ok=True)
    if not waktu:
        waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_path = RIWAYAT_DIR / f"{session_id}.json"
    data = {
        "id": session_id,
        "judul": judul,
        "waktu": waktu,
        "messages": messages,
    }
    with open(file_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def baca_sesi_disk(session_id: str) -> dict:
    """Membaca data sesi percakapan dari berkas JSON."""
    file_path = RIWAYAT_DIR / f"{session_id}.json"
    if file_path.exists():
        try:
            with open(file_path, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            pass
    return None


def hapus_sesi_disk(session_id: str):
    """Menghapus satu berkas sesi dari disk."""
    file_path = RIWAYAT_DIR / f"{session_id}.json"
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception:
            pass


def hapus_semua_sesi_disk():
    """Menghapus semua berkas sesi riwayat percakapan."""
    if RIWAYAT_DIR.exists():
        for f in RIWAYAT_DIR.glob("*.json"):
            try:
                f.unlink()
            except Exception:
                pass


# Inisialisasi model dan database
info_res = muat_sumber_daya()

# Inisialisasi Session State dasar sebelum sidebar dirender
if "current_session_id" not in st.session_state:
    new_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.current_session_id = new_id
    st.session_state.current_session_title = "Percakapan Baru"
    st.session_state.messages = list(PESAN_AWAL_DEFAULT)

if "messages" not in st.session_state:
    st.session_state.messages = list(PESAN_AWAL_DEFAULT)

if "penilaian_status" not in st.session_state:
    st.session_state.penilaian_status = {}

# ==============================================================
# SIDEBAR
# ==============================================================

with st.sidebar:
    st.title("🎓 Chatbot UBSI")

    # 1. Tombol Chat Baru (New Chat)
    if st.button("➕ Chat Baru", key="btn_new_chat", use_container_width=True, type="primary"):
        new_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.current_session_id = new_id
        st.session_state.current_session_title = "Percakapan Baru"
        st.session_state.messages = list(PESAN_AWAL_DEFAULT)
        st.session_state.penilaian_status = {}
        st.rerun()

    # 2. Riwayat Percakapan
    st.markdown("---")
    st.subheader("📜 Riwayat Chat")
    daftar_sesi = muat_daftar_sesi()
    if not daftar_sesi:
        st.caption("Belum ada riwayat percakapan tersimpan.")
    else:
        for sesi in daftar_sesi[:8]:  # Tampilkan 8 sesi terbaru
            sid = sesi["id"]
            judul_label = sesi["judul"]
            is_active = (sid == st.session_state.get("current_session_id"))

            col_s1, col_s2 = st.columns([5, 1])
            with col_s1:
                prefix = "👉 " if is_active else "💬 "
                if st.button(f"{prefix}{judul_label}", key=f"sesi_{sid}", use_container_width=True, disabled=is_active):
                    data_sesi = baca_sesi_disk(sid)
                    if data_sesi:
                        st.session_state.current_session_id = sid
                        st.session_state.current_session_title = data_sesi.get("judul", "Percakapan")
                        st.session_state.messages = data_sesi.get("messages", list(PESAN_AWAL_DEFAULT))
                        st.session_state.penilaian_status = {}
                        st.rerun()
            with col_s2:
                if st.button("🗑️", key=f"del_{sid}", help=f"Hapus sesi '{judul_label}'"):
                    hapus_sesi_disk(sid)
                    if sid == st.session_state.get("current_session_id"):
                        new_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.session_state.current_session_id = new_id
                        st.session_state.current_session_title = "Percakapan Baru"
                        st.session_state.messages = list(PESAN_AWAL_DEFAULT)
                        st.session_state.penilaian_status = {}
                    st.rerun()

        if st.button("🧹 Bersihkan Semua Riwayat", key="btn_clear_all_history", use_container_width=True):
            hapus_semua_sesi_disk()
            new_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state.current_session_id = new_id
            st.session_state.current_session_title = "Percakapan Baru"
            st.session_state.messages = list(PESAN_AWAL_DEFAULT)
            st.session_state.penilaian_status = {}
            st.rerun()

    # 3. Contoh Pertanyaan Cepat (8 contoh satu klik)
    st.markdown("---")
    st.subheader("💡 Contoh Pertanyaan Cepat")
    st.caption("Klik untuk langsung menguji (bebas data pribadi):")

    contoh_pertanyaan = [
        "apa definisi BSI.ID",
        "bagaimana cara melihat EKTM mahasiswa",
        "Halo min, gmn cara byr ektm lwt atm bni?",
        "kalo mau bayar kuliah lewat teller bni prosedurnya gimana ya kak?",
        "ka kalo status di ektm tertulis pembayaran ditutup itu artinya apa?",
        "cara cek riwayat pembayaran",
        "cara bayar kuliah lewat BRI",
        "pembayaran seminar SWE masih dibuka?",
    ]

    for idx_contoh, contoh_teks in enumerate(contoh_pertanyaan, 1):
        if st.button(f"{idx_contoh}. {contoh_teks}", key=f"btn_contoh_{idx_contoh}", use_container_width=True):
            st.session_state["prompt_otomatis"] = contoh_teks
            st.rerun()

    # 4. Pengaturan & Status Sistem (dalam expander agar rapi)
    st.markdown("---")
    with st.expander("⚙️ Pengaturan & Status Sistem", expanded=False):
        st.subheader("Model Perute (Router)")
        router_aktif_pilihan = st.selectbox(
            "Pilih Router Aktif:",
            options=["cascade (otomatis)", "indobertweet", "klasik", "aturan"],
            index=0,
            help="cascade (otomatis): Bertingkat IndoBERTweet -> TF-IDF Klasik -> Aturan\nindobertweet: Transformer fine-tuned (T6c)\nklasik: TF-IDF + Logistic Regression (T5)\naturan: Berbasis kata kunci (T1)"
        )

        st.subheader("Susunan Sistem (Pipeline)")
        susunan_options = [
            "V1 (Standar)",
            "V2 (Eksperimen, belum divalidasi)",
            "V3 (Eksperimen, belum divalidasi)",
        ]
        susunan_map = {
            "V1 (Standar)": "V1",
            "V2 (Eksperimen, belum divalidasi)": "V2",
            "V3 (Eksperimen, belum divalidasi)": "V3",
        }
        susunan_idx = 0 if SUSUNAN_AKTIF == "V1" else (1 if SUSUNAN_AKTIF == "V2" else 2)
        susunan_label_pilihan = st.selectbox(
            "Pilih Susunan Sistem:",
            options=susunan_options,
            index=susunan_idx,
            help=(
                "V1 (Standar): Router -> Pencarian -> Gerbang kata kunci (default stabil).\n"
                "V2 (Eksperimen, belum divalidasi): Pencarian dulu; router hanya veto jika P(eskalasi) >= 0.50.\n"
                "V3 (Eksperimen, belum divalidasi): Gerbang kata kunci diganti skor gabungan kosinus + overlap."
            )
        )
        susunan_aktif_pilihan = susunan_map[susunan_label_pilihan]

        st.subheader("Pilihan Sistem")
        sistem_terpilih = st.radio(
            "Pilih sistem backend:",
            options=["B (utama)", "BM25", "A (Gemini)"],
            index=0,
            help="Sistem B adalah sistem utama saat ini. Sistem BM25 dan Sistem A akan aktif pada tahap T8."
        )

        if sistem_terpilih in ["BM25", "A (Gemini)"]:
            st.warning(f"⚠️ Sistem **{sistem_terpilih}** saat ini nonaktif (tahap T8). Pertanyaan akan diproses menggunakan **Sistem B**.")

        st.markdown("---")
        st.subheader("📊 Status Sistem")
        st.write(f"📅 **Tanggal Hari Ini:** `{TANGGAL_HARI_INI}`")
        st.write(f"📚 **Dokumen Aktif:** `{hitung_dokumen_katalog()}` berkas")
        st.write(f"🧩 **Total Potongan Indeks:** `{info_res['total_dokumen']}` chunk")
        st.write(f"🧠 **Router Terpilih:** `{router_aktif_pilihan}`")
        st.write(f"⚙️ **Susunan Terpilih:** `{susunan_label_pilihan}`")
        st.write(f"📏 **Ambang Jarak Kosinus:** `{SEARCH_DISTANCE_THRESHOLD}`")

        st.markdown("---")
        st.subheader("🛠️ Kontrol Sesi & Log")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🧹 Kosongkan Obrolan Ini", use_container_width=True):
                st.session_state.messages = list(PESAN_AWAL_DEFAULT)
                st.session_state.penilaian_status = {}
                st.rerun()

        with col_btn2:
            if st.button("🔄 Muat Ulang Cache", use_container_width=True):
                st.cache_resource.clear()
                st.rerun()

        if st.button("🗑️ Hapus Berkas Log Uji", use_container_width=True):
            hapus_log_uji()
            st.success("Berkas log berhasil dikosongkan.")

        st.markdown("---")
        st.caption(
            "🔒 **Peringatan Privasi Log:** Log pengujian disimpan di komputer lokal (`hasil/log_uji.csv`). "
            "Data sensitif otomatis disamarkan sebelum disimpan."
        )


# ==============================================================
# AREA UTAMA (CHAT & BANNER)
# ==============================================================

# (v) Banner wajib di bagian paling atas
st.warning("⚠️ **Prototipe uji internal. Isi jawaban BELUM diverifikasi admin. Jangan memasukkan data pribadi.**")

judul_aktif = st.session_state.get("current_session_title", "Percakapan Layanan Administrasi Kampus")
st.markdown(f"### 💬 {judul_aktif}")
st.caption("Ajukan pertanyaan umum seputar biaya perkuliahan, jadwal kegiatan, akun aplikasi, atau prosedur administrasi.")

# Tampilkan Riwayat Pesan
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        konten = msg["content"]
        detail = msg.get("detail")
        msg_id = msg.get("id")

        # (i) Jawaban panjang ditampilkan 600 karakter pertama dengan tombol 'lihat selengkapnya'
        if msg["role"] == "assistant" and len(konten) > 600:
            st.markdown(konten[:600] + "...")
            with st.expander("📖 Lihat selengkapnya"):
                st.markdown(konten)
        else:
            st.markdown(konten)

        # Jika pesan dari asisten dan memiliki metadata teknis
        if detail is not None:
            # (ii) Tampilkan 'Bagian' dan 'dimasukkan ke indeks' di bawah jawaban
            meta_teratas = {}
            if detail.get("potongan_terpakai"):
                meta_teratas = detail["potongan_terpakai"][0].get("metadata", {})

            bagian_dok = meta_teratas.get("judul_bagian")
            tgl_indeks = meta_teratas.get("tanggal_berlaku", "2026-11-04")
            sumber_str = detail.get("sumber", "")

            if bagian_dok:
                st.caption(f"📌 **Bagian:** {bagian_dok} &nbsp;|&nbsp; 🗓️ **dimasukkan ke indeks:** {tgl_indeks}")
            elif "dimasukkan ke indeks:" in sumber_str:
                st.caption(f"📌 **Sumber Dokumen:** {sumber_str}")
            elif sumber_str:
                st.caption(f"📌 **Sumber:** {sumber_str} &nbsp;|&nbsp; 🗓️ **dimasukkan ke indeks:** {tgl_indeks}")

            # (iv) Expander detail teknis memuat kelas router, keyakinan, peluang eskalasi, jarak, overlap, alasan keputusan
            with st.expander("🔍 Lihat detail teknis"):
                susunan_detail = detail.get("susunan_label")
                if not susunan_detail:
                    s_kode = detail.get("susunan", "V1")
                    if s_kode == "V1":
                        susunan_detail = "V1 (Standar)"
                    elif s_kode == "V2":
                        susunan_detail = "V2 (Eksperimen, belum divalidasi)"
                    elif s_kode == "V3":
                        susunan_detail = "V3 (Eksperimen, belum divalidasi)"
                    else:
                        susunan_detail = f"{s_kode} (Eksperimen, belum divalidasi)"
                st.caption(f"⚙️ **Susunan Sistem:** `{susunan_detail}`")
                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    st.write(f"**Router Mode:** `{detail.get('router_pakai', router_aktif_pilihan)}`")
                    st.write(f"**Router Penjawab:** `{detail.get('router_penjawab', '-')}`")
                    st.write(f"**Kelas Router:** `{detail.get('kelas_router', '-')}`")
                    st.write(f"**Keyakinan Router:** `{detail.get('keyakinan', 0):.4f}`")
                    st.write(f"**Peluang Eskalasi:** `{detail.get('peluang_eskalasi', 0):.4f}`")
                with col_t2:
                    jarak_val = detail.get('jarak_terbaik')
                    ovlp_val = detail.get('tumpang_tindih_kata')
                    st.write(f"**Keputusan Bot:** `{detail.get('keputusan', '-')}`")
                    st.write(f"**Keputusan Router:** `{detail.get('keputusan_router', '-')}`")
                    st.write(f"**Jarak Cosine:** `{f'{jarak_val:.3f}' if jarak_val is not None else '-'}`")
                    st.write(f"**Overlap Kata Isi:** `{f'{ovlp_val:.2f}' if ovlp_val is not None else '-'}`")
                    st.write(f"**Waktu Respons:** `{detail.get('waktu_ms', 0)} ms`")

                st.write(f"**Alasan Keputusan:** {detail.get('alasan', '-')}")

                # Tampilkan rincian cascade jika ada
                t_cascade = detail.get("telemetri_cascade")
                if t_cascade:
                    st.markdown("**Alur Cascade Router:**")
                    c_bert = t_cascade.get('indobertweet', {})
                    c_klas = t_cascade.get('klasik', {})
                    c_atur = t_cascade.get('aturan', {})
                    st.markdown(f"- **1. IndoBERTweet:** `{c_bert.get('status', '-')}` — *{c_bert.get('alasan', '-')}*")
                    st.markdown(f"- **2. TF-IDF Klasik:** `{c_klas.get('status', '-')}` — *{c_klas.get('alasan', '-')}*")
                    st.markdown(f"- **3. Aturan Panduan:** `{c_atur.get('status', '-')}` — *{c_atur.get('alasan', '-')}*")

                # Tampilkan daftar potongan teks jika ada
                potongan = detail.get("potongan_terpakai", [])
                if potongan:
                    st.markdown("**Potongan Terkait yang Ditemukan:**")
                    for idx_p, p in enumerate(potongan[:3], 1):
                        st.markdown(f"*{idx_p}. Jarak: `{p.get('jarak', 0):.3f}`* — `{p.get('metadata', {}).get('judul_bagian', '')}`")
                        st.text(p.get("teks", "")[:200] + "...")

            # Mode Penilaian Manual
            st.markdown("---")
            sudah_dinilai = st.session_state.penilaian_status.get(msg_id)

            if sudah_dinilai:
                st.success(f"✓ Penilaian tercatat di log: **{sudah_dinilai}**")
            else:
                with st.container():
                    st.caption("📝 **Penilaian Kualitas Jawaban (Simpan ke log lokal):**")
                    col_p1, col_p2 = st.columns([3, 2])

                    with col_p1:
                        pilihan_nilai = st.radio(
                            "Kesesuaian respon:",
                            options=["Benar", "Sebagian", "Salah", "Seharusnya diteruskan ke staf"],
                            key=f"radio_{msg_id}",
                            horizontal=True,
                        )
                    with col_p2:
                        catatan_eval = st.text_input(
                            "Catatan evaluasi (opsional):",
                            key=f"catatan_{msg_id}",
                            placeholder="Contoh: potongan terlalu pendek",
                        )

                    if st.button("💾 Simpan Penilaian", key=f"simpan_{msg_id}"):
                        pertanyaan_terkait = msg.get("pertanyaan_asal", "")
                        simpan_ke_log(
                            pertanyaan=pertanyaan_terkait,
                            sistem=detail.get("sistem_nama", "B"),
                            router=detail.get("router_pakai", router_aktif_pilihan),
                            keputusan=detail.get("keputusan", ""),
                            sumber=detail.get("sumber", ""),
                            penilaian=pilihan_nilai,
                            catatan=catatan_eval,
                        )
                        st.session_state.penilaian_status[msg_id] = pilihan_nilai
                        # Simpan pembaruan status ke berkas sesi disk
                        simpan_sesi_disk(
                            st.session_state.current_session_id,
                            st.session_state.current_session_title,
                            st.session_state.messages
                        )
                        st.toast("Penilaian berhasil disimpan ke hasil/log_uji.csv!", icon="✅")
                        st.rerun()


# ==============================================================
# INPUT PENGGUNA
# ==============================================================

prompt_otomatis = st.session_state.pop("prompt_otomatis", None)
prompt_manual = st.chat_input("Ketik pertanyaan Anda di sini...")
prompt = prompt_otomatis if prompt_otomatis else prompt_manual

if prompt:
    # 1. Perbarui judul sesi jika masih berupa default 'Percakapan Baru'
    if st.session_state.get("current_session_title") == "Percakapan Baru":
        judul_bersih = prompt.replace("\n", " ").strip()
        st.session_state.current_session_title = judul_bersih[:25] + ("..." if len(judul_bersih) > 25 else "")

    # 2. Tampilkan pesan pengguna di antarmuka
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "detail": None,
        "id": len(st.session_state.messages),
    })

    # 3. Proses pertanyaan menggunakan Sistem B dengan router terpilih
    sistem_pakai = "B"
    with st.spinner("Memproses pertanyaan..."):
        try:
            hasil = jawab(prompt, sistem=sistem_pakai, router_mode=router_aktif_pilihan, susunan=susunan_aktif_pilihan)
            hasil["sistem_nama"] = sistem_terpilih
            hasil["router_pakai"] = router_aktif_pilihan
            hasil["susunan_label"] = susunan_label_pilihan
        except Exception as e:
            hasil = {
                "jawaban": f"Terjadi kesalahan saat memproses jawaban: {e}",
                "sumber": "",
                "kelas_router": "error",
                "keyakinan": 0.0,
                "peluang_eskalasi": 1.0,
                "keputusan_router": "teruskan",
                "jarak_terbaik": None,
                "tumpang_tindih_kata": None,
                "kata_isi": [],
                "keputusan": "teruskan_ke_staf",
                "alasan": str(e),
                "potongan_terpakai": [],
                "waktu_ms": 0.0,
                "sistem_nama": sistem_terpilih,
                "router_pakai": router_aktif_pilihan,
            }

    # 4. Simpan jawaban asisten ke riwayat percakapan
    st.session_state.messages.append({
        "role": "assistant",
        "content": hasil["jawaban"],
        "detail": hasil,
        "pertanyaan_asal": prompt,
        "id": len(st.session_state.messages),
    })

    # 5. Persistensikan sesi percakapan ke berkas JSON lokal
    simpan_sesi_disk(
        st.session_state.current_session_id,
        st.session_state.current_session_title,
        st.session_state.messages
    )

    # 6. Refresh tampilan agar pesan langsung tampil
    st.rerun()
