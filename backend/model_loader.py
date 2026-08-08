import io
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "best_model.keras"
)

IMG_SIZE = (160, 160)
CLASSES = ["Earthquake", "Fire", "Flood", "Normal"]

_model = None

def get_model():
    """Load the model lazily, or raise an error if not found."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"The trained model is missing. Please place the trained model at: {MODEL_PATH}"
            )
        _model = tf.keras.models.load_model(str(MODEL_PATH))
    return _model


def get_prediction(image_bytes: bytes) -> dict:
    """
    CNN inference pipeline.
    
    Args:
        image_bytes: The bytes of the uploaded image.
        
    Returns:
        dict: The prediction response in the format expected by the frontend.
    """
    model = get_model()
    
    image = Image.open(io.BytesIO(image_bytes))
    
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    image = image.resize(IMG_SIZE)
    
    img_array = tf.keras.utils.img_to_array(image)
    img_array = img_array / 255.0
    
    img_batch = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_batch)[0]
    
    top_3_indices = np.argsort(predictions)[-3:][::-1]
    
    top_predictions = []
    for idx in top_3_indices:
        top_predictions.append({
            "class": CLASSES[idx],
            "confidence": float(predictions[idx] * 100)
        })
        
    predicted_class = top_predictions[0]["class"]
    confidence = top_predictions[0]["confidence"]
    
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_predictions": top_predictions
    }
