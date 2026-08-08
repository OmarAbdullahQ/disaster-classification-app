"""Traditional TensorFlow/Keras CNNs built entirely from scratch."""

from tensorflow import keras
from tensorflow.keras import layers

from modeling.config import INPUT_SHAPE, NUM_CLASSES


def _augmentation() -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="training_augmentation",
    )


def _conv_block(x, filters: int, batch_norm: bool, block_number: int):
    x = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=not batch_norm,
        name=f"block_{block_number}_conv",
    )(x)
    if batch_norm:
        x = layers.BatchNormalization(name=f"block_{block_number}_bn")(x)
    x = layers.Activation("relu", name=f"block_{block_number}_relu")(x)
    return layers.MaxPooling2D(name=f"block_{block_number}_pool")(x)


def _multiscale_block(x, filters: int):
    branch_3 = layers.Conv2D(
        filters,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="custom_branch_3x3",
    )(x)
    branch_5 = layers.Conv2D(
        filters,
        kernel_size=5,
        padding="same",
        use_bias=False,
        name="custom_branch_5x5",
    )(x)
    x = layers.Concatenate(name="custom_multiscale_concat")([branch_3, branch_5])
    x = layers.BatchNormalization(name="custom_multiscale_bn")(x)
    x = layers.Activation("relu", name="custom_multiscale_relu")(x)
    return layers.MaxPooling2D(name="custom_multiscale_pool")(x)


def build_model(model_name: str, config: dict) -> keras.Model:
    inputs = keras.Input(shape=INPUT_SHAPE, name="image")
    x = inputs

    if config["augmentation"]:
        x = _augmentation()(x)

    filters = config["filters"]
    start_index = 0
    if config["custom_multiscale"]:
        x = _multiscale_block(x, filters[0])
        start_index = 1

    for index, filter_count in enumerate(filters[start_index:], start=start_index + 1):
        x = _conv_block(
            x,
            filters=filter_count,
            batch_norm=config["batch_norm"],
            block_number=index,
        )

    x = layers.GlobalAveragePooling2D(name="global_average_pool")(x)
    x = layers.Dense(
        config["dense_units"],
        activation="relu",
        name="dense_features",
    )(x)
    if config["dropout"] > 0:
        x = layers.Dropout(config["dropout"], name="dropout")(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    return keras.Model(inputs, outputs, name=model_name)
