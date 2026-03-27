import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import MobileNetV2
from src.preprocessing import IMG_SIZE, CLASSES, load_dataset, augment_image

MODEL_PATH = os.path.join('models', 'handwritten_digit_model.h5')
NUM_CLASSES = len(CLASSES)


def build_model() -> tf.keras.Model:
    """
    Transfer learning with MobileNetV2.
    Phase 1: train only the top layers (base frozen).
    Phase 2: unfreeze top 30 layers for fine-tuning.
    """
    base = MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False  # start fully frozen

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def augment_dataset(X: np.ndarray, y: np.ndarray, factor: int = 15):
    """Augment dataset by applying random transforms factor-1 additional times."""
    X_aug, y_aug = [X], [y]
    for _ in range(factor - 1):
        X_aug.append(np.array([augment_image(img) for img in X]))
        y_aug.append(y)
    return np.concatenate(X_aug), np.concatenate(y_aug)


def train(train_dir: str = os.path.join('data', 'train'),
          epochs: int = 50,
          save_path: str = MODEL_PATH) -> tf.keras.Model:
    """Two-phase training: frozen base first, then fine-tune top layers."""
    X_train, y_train = load_dataset(train_dir)
    X_train, y_train = augment_dataset(X_train, y_train, factor=10)
    idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[idx], y_train[idx]

    model = build_model()

    early_stop = callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=1)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    # ── Phase 1: train top layers only ───────────────────────────────────────
    print('\n=== Phase 1: Training top layers (base frozen) ===')
    model.fit(
        X_train, y_train,
        epochs=20,
        validation_split=0.15,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # ── Phase 2: unfreeze top 30 layers of base and fine-tune ────────────────
    print('\n=== Phase 2: Fine-tuning top 30 base layers ===')
    base_model = model.layers[1]  # MobileNetV2 is the second layer
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop2 = callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1)
    model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_split=0.15,
        callbacks=[early_stop2, reduce_lr],
        verbose=1
    )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    model.save(save_path)
    print(f'Model saved to {save_path}')
    return model


def load_model(path: str = MODEL_PATH) -> tf.keras.Model:
    """Load a saved model from disk."""
    return tf.keras.models.load_model(path)
