import os
import glob
import tensorflow as tf
from datetime import datetime

from configs import *
from models.wavetf_model import WaveTFModel
from data_loaders.merged_data_loader import MergedDataLoader

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)

import matplotlib.pyplot as plt

class ImageLogger(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset, save_dir, freq=1):
        super(ImageLogger, self).__init__()
        # Grab a single batch from the dataset to use as a constant benchmark
        self.val_inputs, self.val_targets = next(iter(val_dataset))
        self.save_dir = save_dir
        self.freq = freq
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.freq != 0:
            return

        # Run prediction on the fixed batch
        # Model outputs: [embedded_image, extracted_watermark]
        predictions = self.model.predict(self.val_inputs, verbose=0)
        embedded_imgs = predictions[0]
        
        # Inputs: (image, watermark, attack_id) -> We want inputs[0] (image)
        original_imgs = self.val_inputs[0]

        # Plot the first image in the batch
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Original
        axes[0].imshow(original_imgs[0, :, :, 0], cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f"Original")
        axes[0].axis('off')

        # 2. Watermarked
        axes[1].imshow(embedded_imgs[0, :, :, 0], cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f"Watermarked (Epoch {epoch+1})")
        axes[1].axis('off')

        # 3. Difference (Amplified x50 so you can see it)
        diff = tf.abs(original_imgs[0] - embedded_imgs[0])
        axes[2].imshow(diff[:, :, 0] * 50.0, cmap='inferno')
        axes[2].set_title(f"Difference (Amplified 50x)")
        axes[2].axis('off')

        path = os.path.join(self.save_dir, f"epoch_{epoch+1}_sample.png")
        plt.savefig(path)
        plt.close()
        print(f"\n[Visualizer] Sample image saved to: {path}")

# GPU Setup
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        try: tf.config.experimental.set_memory_growth(g, True)
        except: pass

# Dataset (Merged Loader with Attack IDs)
train_dataset = MergedDataLoader(
    image_base_path=TRAIN_IMAGES_PATH,
    image_channels=[0],
    image_convert_type=None,
    watermark_size=WATERMARK_SIZE,
    attack_min_id=ATTACK_MIN_ID,
    attack_max_id=ATTACK_MAX_ID,
    batch_size=BATCH_SIZE,
    max_images=20000
).get_data_loader()

# Model
print("Building model (Robust LL Strategy)...")
model = WaveTFModel(
    image_size=IMAGE_SIZE,
    watermark_size=WATERMARK_SIZE,
    delta_scale=delta_scale
).get_model()

# Resume Logic
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
candidate = sorted(glob.glob(os.path.join(MODEL_OUTPUT_PATH, "*.h5")) + 
                   glob.glob(os.path.join(MODEL_OUTPUT_PATH, "*.keras")), 
                   key=os.path.getmtime)
if candidate:
    print(f"Resuming from: {candidate[-1]}")
    try: model.load_weights(candidate[-1])
    except Exception as e: print(e)

# Callbacks
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
visualizer = ImageLogger(train_dataset, MODEL_OUTPUT_PATH)
callbacks = [
    visualizer,
    ModelCheckpoint(
        filepath=os.path.join(MODEL_OUTPUT_PATH, "best_weights.h5"),
        monitor="output_watermark_loss",
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=1
    ),
    ReduceLROnPlateau(monitor="output_watermark_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    TensorBoard(log_dir=os.path.join("logs", timestamp), update_freq="epoch")
]

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)

# ============================================================
# SINGLE PHASE TRAINING (Simultaneous Robustness & Invisibility)
# ============================================================
print("\n>>> STARTING ROBUST TRAINING (SINGLE PHASE) <<<")
print(f"Paper Configuration:")
print(f" - Image Weight: {IMAGE_LOSS_WEIGHT} (Invisibility)")
print(f" - Watermark Weight: {WATERMARK_LOSS_WEIGHT} (Robustness)")
print(f" - Attacks: ENABLED (Learning to survive noise/compression)")

model.compile(
    optimizer=optimizer,
    loss={"embedded_image": "mse", "output_watermark": "mae"},
    loss_weights={"embedded_image": IMAGE_LOSS_WEIGHT, "output_watermark": WATERMARK_LOSS_WEIGHT},
    metrics={"output_watermark": "binary_accuracy"}
)

try:
    model.fit(
        train_dataset,
        initial_epoch=0,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    model.save_weights(os.path.join(MODEL_OUTPUT_PATH, f"final_weights-{timestamp}.h5"))
    print("\n✓ Training complete.")
except KeyboardInterrupt:
    print("\nInterrupted.")
    model.save_weights(os.path.join(MODEL_OUTPUT_PATH, "interrupted.h5"))
except Exception as e:
    print(f"\nError: {e}")