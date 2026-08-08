"""Shared experiment settings and the four required CNN configurations."""

from data.config import IMG_SIZE, SEED

INPUT_SHAPE = (*IMG_SIZE, 3)
NUM_CLASSES = 4
BATCH_SIZE = 32
MAX_EPOCHS = 25
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5

MODEL_CONFIGS = {
    "model_a_baseline": {
        "filters": [32, 64],
        "kernel_size": 3,
        "batch_norm": False,
        "dropout": 0.0,
        "augmentation": False,
        "dense_units": 128,
        "custom_multiscale": False,
        "hypothesis": (
            "A small two-block CNN provides a fast reference for accuracy, "
            "generalization, parameter count, and training time."
        ),
    },
    "model_b_deeper": {
        "filters": [32, 64, 128],
        "kernel_size": 3,
        "batch_norm": False,
        "dropout": 0.0,
        "augmentation": False,
        "dense_units": 128,
        "custom_multiscale": False,
        "hypothesis": (
            "Adding a third convolution block should learn more complex "
            "disaster patterns, but may increase overfitting."
        ),
    },
    "model_c_regularized": {
        "filters": [32, 64, 128],
        "kernel_size": 3,
        "batch_norm": True,
        "dropout": 0.4,
        "augmentation": True,
        "dense_units": 128,
        "custom_multiscale": False,
        "hypothesis": (
            "Batch normalization, dropout, and training-only augmentation "
            "should reduce the generalization gap of the deeper model."
        ),
    },
    "model_d_custom": {
        "filters": [32, 64, 128],
        "kernel_size": 3,
        "batch_norm": True,
        "dropout": 0.35,
        "augmentation": True,
        "dense_units": 192,
        "custom_multiscale": True,
        "hypothesis": (
            "Parallel 3x3 and 5x5 filters should capture both local damage "
            "details and wider smoke, fire, flood, or rubble patterns."
        ),
    },
}
