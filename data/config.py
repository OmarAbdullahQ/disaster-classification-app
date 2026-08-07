from pathlib import Path

SEED       = 42
IMG_SIZE   = (160, 160)  
CLASSES    = ["Earthquake", "Fire", "Flood", "Normal"] 
SPLITS     = ["Train", "Val", "Test"]
ROOT       = Path(__file__).resolve().parents[1]
RAW_DIR    = ROOT / "data" / "raw"