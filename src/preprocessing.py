import os
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

IMG_SIZE = (128, 128)
CLASSES = ['eight', 'five', 'four', 'nine', 'one', 'seven', 'six', 'three', 'two', 'zero']
CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls for cls, i in CLASS_TO_IDX.items()}


def load_image(path: str) -> np.ndarray:
    """
    Load image, convert to grayscale, invert (white ink on black bg),
    convert back to RGB, resize, then apply MobileNetV2 preprocessing (-1 to 1).
    """
    img = Image.open(path).convert('L')          # grayscale
    img = ImageOps.invert(img)                   # invert: dark ink -> bright on dark bg
    img = img.convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    # MobileNetV2 expects inputs in [-1, 1]
    arr = (arr / 127.5) - 1.0
    return arr


def load_dataset(data_dir: str):
    """
    Load all images from data_dir/class_name/image.jpg structure.
    Returns X (N, H, W, C) and y (N,) arrays.
    """
    X, y = [], []
    for cls in CLASSES:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            X.append(load_image(os.path.join(cls_dir, fname)))
            y.append(CLASS_TO_IDX[cls])
    return np.array(X), np.array(y)


def augment_image(img: np.ndarray) -> np.ndarray:
    """
    Augment a single preprocessed image with rotation, zoom, brightness,
    shear, and slight translation.
    """
    pil = Image.fromarray(((img + 1.0) * 127.5).astype(np.uint8))
    w, h = pil.size

    # random rotation ±20 degrees
    angle = np.random.uniform(-20, 20)
    pil = pil.rotate(angle, fillcolor=(0, 0, 0))

    # random zoom
    if np.random.rand() > 0.4:
        margin = int(w * np.random.uniform(0.02, 0.15))
        pil = pil.crop((margin, margin, w - margin, h - margin))
        pil = pil.resize((w, h), Image.BILINEAR)

    # random translation (shift)
    if np.random.rand() > 0.5:
        tx = int(np.random.uniform(-8, 8))
        ty = int(np.random.uniform(-8, 8))
        pil = pil.transform(pil.size, Image.AFFINE, (1, 0, tx, 0, 1, ty), fillcolor=(0, 0, 0))

    # random brightness
    factor = np.random.uniform(0.75, 1.25)
    pil = ImageEnhance.Brightness(pil).enhance(factor)

    # random contrast
    factor = np.random.uniform(0.8, 1.2)
    pil = ImageEnhance.Contrast(pil).enhance(factor)

    arr = np.array(pil, dtype=np.float32)
    return (arr / 127.5) - 1.0


def preprocess_for_prediction(path: str) -> np.ndarray:
    """Load a single image and return batch-ready array (1, H, W, C)."""
    return np.expand_dims(load_image(path), axis=0)
