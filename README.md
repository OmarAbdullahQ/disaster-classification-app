# Disaster Classification App

CNN-based aerial disaster classification using the four AIDERv2 classes:
Earthquake, Fire, Flood, and Normal.

## Modeling and evaluation

The project compares four TensorFlow/Keras CNNs built from scratch:

1. Model A - small baseline
2. Model B - deeper CNN
3. Model C - regularized CNN with BatchNorm, Dropout, and augmentation
4. Model D - custom multi-scale CNN

Transfer learning is not used. All experiments share the same balanced dataset,
seed, optimizer, learning rate, batch size, maximum epochs, and early-stopping rule.

The recommended workflow is to train once on a Google Colab GPU and reuse the
saved artifacts locally. See [COLAB_TRAINING.md](COLAB_TRAINING.md).

```bash
python -m modeling.train
python -m evaluation.evaluate
```

Generated models, histories, comparison tables, plots, confusion matrices, and
error examples are written to `artifacts/`.
