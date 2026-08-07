# Data

Loads AIDERv2 images into normalized `X`/`y` arrays, sliced to ~8k.

## Run
```bash
uv run python -m data.pipeline
```

## Use
```python
from data.pipeline import load_split
X_train, y_train = load_split("Train")   # also "Val", "Test"
```

## Knobs
- `FRAC` in `pipeline.py` — how much data to keep (`0.478` ≈ 8k, raise toward `1.0` for more).
- Images are already `/255` normalized — don't rescale again in the model.