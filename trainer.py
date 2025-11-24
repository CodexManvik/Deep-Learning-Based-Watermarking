import os
import glob
import tensorflow as tf
from datetime import datetime

# project imports
from configs import *
from models.wavetf_model import WaveTFModel
from data_loaders.merged_data_loader import MergedDataLoader

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)

# ============================================================
# GPU setup
# ============================================================
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except:
            pass
    print(f"✓ GPUs found: {len(gpus)} — memory growth enabled")
else:
    print("⚠ No GPU found — running on CPU")

# ============================================================
# Dataset
# ============================================================
print(f"Loading dataset from {TRAIN_IMAGES_PATH}")

# Merged loader with manual normalization fix
train_dataset = MergedDataLoader(
    image_base_path=TRAIN_IMAGES_PATH,
    image_channels=[0],
    image_convert_type=None,       # Forces manual /255.0 normalization
    watermark_size=WATERMARK_SIZE,
    attack_min_id=ATTACK_MIN_ID,
    attack_max_id=ATTACK_MAX_ID,
    batch_size=BATCH_SIZE,
    max_images=20000
).get_data_loader()

# ============================================================
# Model Build
# ============================================================
print("Building model (from code)...")
# LL BAND STRATEGY: Use tiny delta_scale (0.1) to "whisper" into the sensitive band.
model = WaveTFModel(
    image_size=IMAGE_SIZE,
    watermark_size=WATERMARK_SIZE,
    delta_scale=delta_scale  # <--- CRITICAL: 0.1 for LL Band to prevent image destruction
).get_model()

# ============================================================
# Resume from checkpoint
# ============================================================
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

candidate = sorted(
    glob.glob(os.path.join(MODEL_OUTPUT_PATH, "*.h5")) +
    glob.glob(os.path.join(MODEL_OUTPUT_PATH, "*.keras")),
    key=os.path.getmtime
)

resume_path = candidate[-1] if candidate else None

if resume_path:
    print(f"\nResuming from checkpoint: {resume_path}")
    try:
        model.load_weights(resume_path)
        print("✓ Weights restored.")
    except Exception as e:
        print(f"✗ Failed loading checkpoint: {e}")
else:
    print("\nNo checkpoint found — training from scratch.")

# ============================================================
# Callbacks Setup
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("logs", timestamp)

best_weights_path = os.path.join(MODEL_OUTPUT_PATH, "best_weights.h5")

callbacks = [
    ModelCheckpoint(
        filepath=best_weights_path,
        monitor="output_watermark_loss",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=1
    ),
    EarlyStopping(
        monitor="output_watermark_loss",
        patience=25,
        restore_best_weights=True,
        verbose=1,
        mode="min"
    ),
    ReduceLROnPlateau(
        monitor="output_watermark_loss",
        factor=0.5,
        patience=4,        # Fast reaction to stuck loss
        min_lr=1e-7,
        verbose=1
    ),
    TensorBoard(
        log_dir=log_dir, 
        update_freq="epoch"
    )
]

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)

# ============================================================
# PHASE 1: LL BAND TRAINING (Whisper Strategy)
# ============================================================
print("\n>>> STARTING LL BAND TRAINING <<<")
print(f"Strategy: High Image Protection ({IMAGE_LOSS_WEIGHT}) + Strong Watermark Push ({WATERMARK_LOSS_WEIGHT})")

model.compile(
    optimizer=optimizer,
    loss={
        "embedded_image": "mse",
        "output_watermark": "mae"
    },
    loss_weights={
        "embedded_image": IMAGE_LOSS_WEIGHT,   # High weight to protect LL Band quality
        "output_watermark": WATERMARK_LOSS_WEIGHT  # Strong weight to force learning
    },
    metrics=["accuracy"]
)

try:
    # Reset initial_epoch to 0 so it runs for the full duration
    model.fit(
        train_dataset,
        initial_epoch=0,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Save Final
    final_path = os.path.join(MODEL_OUTPUT_PATH, f"final_weights-{timestamp}.h5")
    model.save_weights(final_path)
    print(f"\n✓ Training complete. Saved to {final_path}")

except KeyboardInterrupt:
    print("\nTraining interrupted by user.")
    model.save_weights(os.path.join(MODEL_OUTPUT_PATH, "interrupted.h5"))

except Exception as e:
    print(f"\n✗ Training crashed: {e}")