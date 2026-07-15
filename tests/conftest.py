import os
import sys
from pathlib import Path

# Anaconda ships its own libiomp5md.dll which clashes with torch's copy on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# make `models` and cfg/ paths importable when running pytest from anywhere
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
