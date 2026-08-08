# Train once on Google Colab

Training should be performed on a Colab GPU. The saved `.keras`, CSV, JSON, and PNG artifacts can then be used in VS Code without retraining.

## 1. Start a GPU runtime

In Colab, select **Runtime > Change runtime type > T4 GPU**. Verify it:

```python
!nvidia-smi
```

## 2. Clone and install

```python
!git clone https://github.com/OmarAbdullahQ/disaster-classification-app.git
%cd disaster-classification-app
!pip -q install scikit-learn seaborn requests
```

## 3. Download AIDERv2

```python
!python scripts/download_aiderv2.py --output data/raw
```

## 4. Optional: save artifacts permanently in Google Drive

Mount Drive and link the artifact directory before training:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!mkdir -p /content/drive/MyDrive/disaster_cnn_artifacts
!rm -rf artifacts
!ln -s /content/drive/MyDrive/disaster_cnn_artifacts artifacts
```

The removal above targets only the fresh Colab clone's generated `artifacts` directory. Do not run it in a local repository containing results you need.

## 5. Train all four models once

```python
!python -m modeling.train
```

The trainer saves each model immediately after it finishes. If Colab disconnects, reconnect Drive and run the same command; completed models are skipped. Use `--force` only when intentional retraining is required.

## 6. Evaluate saved models without retraining

```python
!python -m evaluation.evaluate
```

## 7. Download one results archive

```python
!zip -r disaster_cnn_artifacts.zip artifacts
from google.colab import files
files.download('disaster_cnn_artifacts.zip')
```

Extract that ZIP into the repository root in VS Code. The backend can load `artifacts/best_model.keras`, while all evaluation results remain under `artifacts/results` and `artifacts/figures`.
