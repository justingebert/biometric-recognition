import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_PATH = os.path.join(ROOT, "data", "siamesemodel.keras")
VERIFICATION_IMG_DIR = os.path.join(ROOT, "application_data", "verification_images")

DETECTION_THRESHOLD = 0.5
VERIFICATION_THRESHOLD = 0.5


class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "L1Dist": L1Dist,
        "BinaryCrossentropy": tf.losses.BinaryCrossentropy,
    },
)


def preprocess_bytes(image_bytes: bytes) -> tf.Tensor:
    """Decode raw image bytes -> 100x100x3 float tensor in [0, 1]."""
    img = tf.io.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img


def preprocess_file(file_path: str) -> tf.Tensor:
    return preprocess_bytes(tf.io.read_file(file_path).numpy())


def run_verify(input_bytes: bytes) -> dict:
    input_img = tf.expand_dims(preprocess_bytes(input_bytes), axis=0)

    files = [f for f in os.listdir(VERIFICATION_IMG_DIR) if f.lower().endswith(".jpg")]
    results = []
    for name in files:
        val_img = preprocess_file(os.path.join(VERIFICATION_IMG_DIR, name))
        val_img = tf.expand_dims(val_img, axis=0)
        score = model.predict([input_img, val_img], verbose=0)
        results.append(float(score[0][0]))

    results = np.array(results)
    detection = int(np.sum(results > DETECTION_THRESHOLD))
    verification = detection / len(files) if files else 0.0
    verified = bool(verification > VERIFICATION_THRESHOLD)

    return {
        "verified": verified,
        "confidence": round(verification, 3),
        "detections": detection,
        "total": len(files),
    }
