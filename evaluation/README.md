# Evaluation Lead deliverables

`evaluate.py` loads the saved `.keras` models and does not train them again. It
creates:

- training/validation loss and accuracy curves;
- confusion matrices;
- classification reports and per-class F1 scores;
- a final model-comparison table;
- weakest-class and most-confused-pair tables;
- high-confidence misclassification examples.

Run from the repository root after training:

```bash
python -m evaluation.evaluate
```
