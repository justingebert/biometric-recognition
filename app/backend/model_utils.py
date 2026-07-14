import glob
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")
MANUAL_TEST_DIR = os.path.join(DATA_DIR, "manual_test")

# Max gallery images used per verify (first N by sorted filename -> deterministic).
# Set to None to use every image in the person's verification/ folder.
GALLERY_LIMIT = 50

DETECTION_THRESHOLD = 0.5
VERIFICATION_THRESHOLD = 0.5


def _latest_model(models_dir):
    """Newest models/siamesemodel_<timestamp>.keras (lexical sort = chronological)."""
    paths = sorted(glob.glob(os.path.join(models_dir, "siamesemodel_*.keras")))
    if not paths:
        raise FileNotFoundError(
            f"No siamesemodel_*.keras in {models_dir}. Train the model or run `git lfs pull`."
        )
    return paths[-1]


MODEL_PATH = _latest_model(MODELS_DIR)
print("Loaded Model: ", MODEL_PATH)

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


def _gallery_dir(person: str) -> str:
    return os.path.join(DATA_DIR, person, "verification")


def _jpgs(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(".jpg"))


def list_persons() -> list[str]:
    """Enrolled persons = subdirs of data/ with a non-empty verification/ folder."""
    if not os.path.isdir(DATA_DIR):
        return []
    people = []
    for name in sorted(os.listdir(DATA_DIR)):
        if _jpgs(_gallery_dir(name)):
            people.append(name)
    return people


def list_test_images() -> list[str]:
    """Filenames of the staged probe images in data/manual_test/."""
    return _jpgs(MANUAL_TEST_DIR)


def preprocess_bytes(image_bytes: bytes) -> tf.Tensor:
    """Decode raw image bytes -> 100x100x3 float tensor in [0, 1].

    Center-crop to a square before resizing so the face keeps its aspect ratio
    and matches the square training crops; a plain resize would squash a
    non-square input (e.g. a 4:3 webcam frame or an arbitrary upload).
    """
    img = tf.io.decode_jpeg(image_bytes, channels=3)
    side = tf.minimum(tf.shape(img)[0], tf.shape(img)[1])
    img = tf.image.resize_with_crop_or_pad(img, side, side)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img


def preprocess_file(file_path: str) -> tf.Tensor:
    return preprocess_bytes(tf.io.read_file(file_path).numpy())


def run_verify(
    person: str,
    input_bytes: bytes,
    detection_threshold: float | None = None,
    verification_threshold: float | None = None,
) -> dict:
    """Compare the input against `person`'s verification gallery (1:1 verification).

    detection_threshold: per-image score cutoff for counting one comparison a match.
    verification_threshold: fraction of gallery matches required to accept.
    Both fall back to the module defaults when None.
    """
    det_thr = DETECTION_THRESHOLD if detection_threshold is None else detection_threshold
    ver_thr = VERIFICATION_THRESHOLD if verification_threshold is None else verification_threshold

    gallery_dir = _gallery_dir(person)
    files = _jpgs(gallery_dir)
    if not files:
        raise ValueError(f"No gallery images for person '{person}'.")
    if GALLERY_LIMIT is not None:
        files = files[:GALLERY_LIMIT]

    input_img = tf.expand_dims(preprocess_bytes(input_bytes), axis=0)

    results = []
    for name in files:
        val_img = preprocess_file(os.path.join(gallery_dir, name))
        val_img = tf.expand_dims(val_img, axis=0)
        score = model.predict([input_img, val_img], verbose=0)
        results.append(float(score[0][0]))

    results = np.array(results)
    detection = int(np.sum(results > det_thr))
    verification = detection / len(files)
    verified = bool(verification > ver_thr)

    return {
        "person": person,
        "verified": verified,
        "confidence": round(verification, 3),
        "detections": detection,
        "total": len(files),
        "detection_threshold": det_thr,
        "verification_threshold": ver_thr,
    }


def run_verify_test(
    person: str,
    name: str,
    detection_threshold: float | None = None,
    verification_threshold: float | None = None,
) -> dict:
    """Verify a staged probe from data/manual_test/ against `person`'s gallery."""
    if name not in list_test_images():
        raise ValueError(f"Unknown test image '{name}'.")
    path = os.path.join(MANUAL_TEST_DIR, name)
    return run_verify(
        person,
        tf.io.read_file(path).numpy(),
        detection_threshold,
        verification_threshold,
    )
