# Modeling Lead deliverables

- `config.py`: shared constraints and four explicit architecture configurations.
- `models.py`: all CNN builders; every model is trained from scratch.
- `train.py`: reproducible training, timing, parameter tracking, checkpointing,
  history export, validation-only winner selection, and resume/skip behavior.

Run from the repository root:

```bash
python -m modeling.train
```

The four hypotheses are stored beside their configurations and copied into the
training metadata. Use `--force` only for intentional retraining.
