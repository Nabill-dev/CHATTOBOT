"""
src/router_infer.py — Inferensi Mandiri Router IndoBERTweet (CPU)

Tahap T6d Proyek Tugas Akhir CHATTOBOT:
  - Memuat model dan tokenizer dari models/router_indobertweet/ (inferensi CPU)
  - Membaca konfigurasi router dari router_config.json
  - Menghitung probabilitas 6 kelas dan peluang eskalasi
  - Memberikan keputusan 'teruskan' jika kelas in KELAS_ESKALASI ATAU peluang_eskalasi >= ambang
"""

import json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models" / "router_indobertweet"
CONFIG_FILE = MODELS_DIR / "router_config.json"

_MODEL = None
_TOKENIZER = None
_CONFIG = None


def muat_router():
    """Memuat model IndoBERTweet, tokenizer, dan konfigurasinya sekali saja (singleton di CPU)."""
    global _MODEL, _TOKENIZER, _CONFIG
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER, _CONFIG

    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"Direktori model router tidak ditemukan di: {MODELS_DIR}")

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        _CONFIG = json.load(f)

    # Muat tokenizer & model ke CPU
    _TOKENIZER = AutoTokenizer.from_pretrained(MODELS_DIR)
    _MODEL = AutoModelForSequenceClassification.from_pretrained(MODELS_DIR)
    _MODEL.to(torch.device("cpu"))
    _MODEL.eval()

    return _MODEL, _TOKENIZER, _CONFIG


def prediksi(teks: str) -> dict:
    """
    Melakukan inferensi klasifikasi teks pada model IndoBERTweet (CPU).

    Mengembalikan:
      {
          'kelas': str,
          'keyakinan': float,
          'peluang_eskalasi': float,
          'keputusan': 'teruskan' | 'jawab',
          'distribusi_probabilitas': dict[str, float]
      }
    """
    model, tokenizer, config = muat_router()

    semua_kelas = config.get("kelas", [
        "bukti_bayar", "pembayaran_umum", "kegiatan_pmb",
        "login_akun", "beasiswa_pendaftaran", "lainnya"
    ])
    kelas_eskalasi = set(config.get("kelas_eskalasi", ["beasiswa_pendaftaran", "lainnya"]))
    ambang_eskalasi = float(config.get("ambang_peluang_eskalasi", 0.10))
    max_len = int(config.get("max_length", 96))

    if not isinstance(teks, str) or not teks.strip():
        # Fallback jika masukan kosong
        return {
            'kelas': 'lainnya',
            'keyakinan': 1.0,
            'peluang_eskalasi': 1.0,
            'keputusan': 'teruskan',
            'distribusi_probabilitas': {c: (1.0 if c == 'lainnya' else 0.0) for c in semua_kelas}
        }

    inputs = tokenizer(
        teks,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt"
    )
    inputs = {k: v.to(torch.device("cpu")) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze(0).cpu().numpy()

    kelas_idx = int(probs.argmax())
    kelas_pred = semua_kelas[kelas_idx]
    keyakinan = float(probs[kelas_idx])

    # Hitung peluang eskalasi = P(beasiswa_pendaftaran) + P(lainnya)
    idx_beas = semua_kelas.index("beasiswa_pendaftaran")
    idx_lain = semua_kelas.index("lainnya")
    peluang_eskalasi = float(probs[idx_beas] + probs[idx_lain])

    # Logika keputusan: teruskan jika kelas termasuk eskalasi ATAU peluang_eskalasi >= ambang
    if kelas_pred in kelas_eskalasi or peluang_eskalasi >= ambang_eskalasi:
        keputusan = "teruskan"
    else:
        keputusan = "jawab"

    distribusi = {c: round(float(p), 4) for c, p in zip(semua_kelas, probs)}

    return {
        'kelas': kelas_pred,
        'keyakinan': round(keyakinan, 4),
        'peluang_eskalasi': round(peluang_eskalasi, 4),
        'keputusan': keputusan,
        'distribusi_probabilitas': distribusi
    }


if __name__ == "__main__":
    print("Menguji src/router_infer.py pada CPU...")
    tes_kalimat = [
        "bagaimana cara bayar biaya kuliah lewat bsi",
        "min saya mau tanya info pendaftaran beasiswa kip",
        "halo min bukti bayar spp an budi sudah ditransfer ya",
        "min password akun mystudent saya lupa cara gantinya gimana"
    ]
    for k in tes_kalimat:
        res = prediksi(k)
        print(f"\nKalimat   : {k}")
        print(f"Kelas     : {res['kelas']} (keyakinan: {res['keyakinan']:.4f})")
        print(f"P(eskalasi): {res['peluang_eskalasi']:.4f} -> Keputusan: {res['keputusan']}")

