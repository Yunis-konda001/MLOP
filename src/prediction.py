import numpy as np
from src.preprocessing import preprocess_for_prediction, load_image, augment_image, IDX_TO_CLASS
from src.model import load_model


def predict(image_path: str, model=None, tta_steps: int = 0) -> dict:
    """
    Predict digit class. If tta_steps > 0, uses Test-Time Augmentation.
    For production/load testing, use tta_steps=0 (default).
    """
    if model is None:
        model = load_model()

    base_img = load_image(image_path)

    if tta_steps > 0:
        # TTA: predict original + augmented versions and average
        batch = [base_img] + [augment_image(base_img) for _ in range(tta_steps)]
        batch = np.array(batch)
        all_probs = model.predict(batch, verbose=0)
        avg_probs = all_probs.mean(axis=0)
    else:
        # Fast single prediction
        batch = np.expand_dims(base_img, axis=0)
        avg_probs = model.predict(batch, verbose=0)[0]

    pred_idx = int(np.argmax(avg_probs))
    return {
        'predicted_class': IDX_TO_CLASS[pred_idx],
        'confidence': float(avg_probs[pred_idx]),
        'all_probabilities': {IDX_TO_CLASS[i]: float(p) for i, p in enumerate(avg_probs)}
    }
