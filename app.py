# ==============================================================
# app.py — Entry Point Wrapper untuk Hugging Face Spaces
# Menjalankan src/app.py dengan konteks path yang benar
# ==============================================================

import sys
import runpy
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Eksekusi antarmuka Streamlit utama
runpy.run_path(str(SRC_DIR / "app.py"), run_name="__main__")
