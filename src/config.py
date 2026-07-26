from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"

DATABASE_URL = os.environ["DATABASE_URL"]
DOLLARS_PER_WAR = int(os.environ.get("DOLLARS_PER_WAR", 8_000_000))
DISCOUNT_RATE = float(os.environ.get("DISCOUNT_RATE", 0.08))

for d in (RAW_DIR, SNAPSHOTS_DIR):
    d.mkdir(parents=True, exist_ok=True)