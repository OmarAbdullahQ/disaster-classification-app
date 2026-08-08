import random
import numpy as np
import tensorflow as tf
from data.config import IMG_SIZE, CLASSES, SPLITS, RAW_DIR, SEED

FRAC = 0.478 # 8000 / 16723 -> ~8k total

def load_split(split, src=RAW_DIR, frac=FRAC):
    X, y = [], []
    for label, c in enumerate(CLASSES):
        files = sorted(p for p in (src / split / c).glob("*") if p.suffix.lower() == ".png")
        random.Random(SEED).shuffle(files)           # seeded -> reproducible slice
        for f in files[:int(len(files) * frac)]:     # the ratio slice
            img = tf.keras.utils.load_img(f, target_size=IMG_SIZE)
            X.append(tf.keras.utils.img_to_array(img))
            y.append(label)
    return np.asarray(X, dtype="float32") / 255.0, np.asarray(y, dtype="int64")

if __name__ == "__main__":
    X_train, y_train = load_split("Train")
    X_val,   y_val   = load_split("Val")
    X_test,  y_test  = load_split("Test")
    for name, X, yy in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        print(f"{name:5} X {X.shape}  y {yy.shape}  pixels {X.min():.2f}-{X.max():.2f}")
    print("total:", len(y_train) + len(y_val) + len(y_test))
    print("train class counts:", np.bincount(y_train))