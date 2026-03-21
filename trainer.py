#!/usr/bin/env python3
"""
Optimized Medical Watermarking Trainer (60K images, 100 epochs)
RTX 3050 + WSL2 VRAM fixes + Mixed Precision + XLA + Early Stopping
"""

import os
import glob
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt

# Project imports
from configs import *
from models.wavetf_model import WaveTFModel
from data_loaders.merged_data_loader import MergedDataLoader

from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)

# Visualization imports
try:
    import visualkeras
    VISUALKERAS_AVAILABLE = True
except ImportError:
    VISUALKERAS_AVAILABLE = False
    print("[WARNING] visualkeras not available - model architecture visualization disabled")

# ============================================
# PERFORMANCE BOOSTS (30% SPEEDUP)
# ============================================

# 1. XLA Compilation - DISABLED: attack simulator uses tf.switch_case with variant
# tensors inside tf.map_fn which XLA_GPU_JIT cannot compile (FakeParam op unsupported)
# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
print("[WARNING] XLA disabled (attack simulator incompatible with XLA_GPU_JIT)")

# 2. Mixed Precision - DISABLED due to WaveTF float64 kernel incompatibility
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
# print("[OK] Mixed precision enabled (50% VRAM savings)")
print("[WARNING] Mixed precision disabled (WaveTF requires float32)")

# 3. WSL2 VRAM Fix (1.7GB -> 3.5GB+)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[OK] GPU memory growth enabled (1.7GB -> 3.5GB)")
    except RuntimeError as e:
        print(f"[WARNING] GPU config error: {e}")

print(f"Available GPUs: {len(gpus)}")
if gpus:
    details = tf.config.experimental.get_device_details(gpus[0])
    print(f"GPU: {details.get('device_name', 'Unknown')}")

# ============================================
# ImageLogger Callback (Visual Progress)
# Saves original, watermarked, and difference image every epoch.
# ============================================

DIFF_OUTPUT_PATH = os.path.join(VISUALIZATION_OUTPUT_PATH, "epoch_diffs")

class ImageLogger(tf.keras.callbacks.Callback):
    """
    Saves a comparison figure (original | watermarked | amplified difference)
    for one sample image at the end of every training epoch.
    The difference is amplified by 50x so subtle watermark patterns are visible.
    """
    def __init__(self, val_dataset, save_dir: str):
        super(ImageLogger, self).__init__()
        self.val_inputs, self.val_targets = next(iter(val_dataset))
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs=None):
        predictions = self.model.predict(self.val_inputs, verbose=0)
        embedded_imgs = predictions[0]   # Clean watermarked output
        original_imgs = self.val_inputs[0]  # Original cover image

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original
        axes[0].imshow(original_imgs[0, :, :, 0], cmap='gray', vmin=0, vmax=1)
        axes[0].set_title("Original")
        axes[0].axis('off')

        # Watermarked
        axes[1].imshow(embedded_imgs[0, :, :, 0], cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f"Watermarked (Epoch {epoch + 1})")
        axes[1].axis('off')

        # Amplified difference (x50) using inferno colormap to highlight signal
        diff = np.abs(
            original_imgs[0].numpy().astype(np.float32) - embedded_imgs[0].astype(np.float32)
        )
        axes[2].imshow(diff[:, :, 0] * 50.0, cmap='inferno', vmin=0, vmax=1)
        axes[2].set_title(f"Difference x50 (Epoch {epoch + 1})")
        axes[2].axis('off')

        path = os.path.join(self.save_dir, f"epoch_{epoch + 1:04d}_diff.png")
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"[ImageLogger] Epoch {epoch + 1} diff saved -> {path}")

# ============================================
# DATASET & MODEL SETUP
# ============================================

# Load dataset (no split, no cache - minimal memory footprint)
print(f"Loading {TRAIN_IMAGES} images (no validation split for 4GB VRAM)...")
dataset = MergedDataLoader(
    image_base_path=TRAIN_IMAGES_PATH,
    image_channels=[0],
    image_convert_type=None,
    watermark_size=WATERMARK_SIZE,
    attack_min_id=ATTACK_MIN_ID,
    attack_max_id=ATTACK_MAX_ID,
    batch_size=BATCH_SIZE,
    max_images=TRAIN_IMAGES,
    attacks_disabled=ATTACKS_DISABLED
).get_data_loader()

# Simple pipeline: no shuffle, no cache, minimal memory
train_batches = TRAIN_IMAGES // BATCH_SIZE
train_dataset = dataset.repeat()

# One-batch snapshot for ImageLogger (taken before repeat() wraps the iterator)
logger_snapshot = dataset.take(1)

print(f"[OK] Training: {TRAIN_IMAGES} images ({train_batches} batches)")
print("[OK] No shuffle buffer (zero memory overhead)")
print("[OK] No validation set")
print("[OK] No disk cache (direct streaming)")

# Model (Robust LL Strategy)
print("Building Medical-Optimized Model (LL Band)...")
model = WaveTFModel(
    image_size=IMAGE_SIZE,
    watermark_size=WATERMARK_SIZE,
    delta_scale=delta_scale
).get_model()

# ============================================
# RESUME FROM ROBUST BASELINE (Critical!)
# ============================================
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
candidates = sorted(
    glob.glob(os.path.join(MODEL_OUTPUT_PATH, "*.h5")) +
    glob.glob(os.path.join(MODEL_OUTPUT_PATH, "*.keras")),
    key=os.path.getmtime
)

if candidates:
    resume_path = candidates[-1]
    print(f"[RESUME] Loading robust baseline: {os.path.basename(resume_path)}")
    try:
        model.load_weights(resume_path)
        print("[OK] Robust weights loaded successfully")
    except Exception as e:
        print(f"[WARNING] Resume failed, starting fresh: {e}")
else:
    print("[NEW] Starting from scratch (no previous weights)")

# ============================================
# Generate timestamp for this session
# ============================================
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# ============================================
# MODEL ARCHITECTURE VISUALIZATION
# ============================================
if VISUALIZE_MODEL_ARCHITECTURE and VISUALKERAS_AVAILABLE:
    try:
        arch_path = os.path.join(VISUALIZATION_OUTPUT_PATH, f"model_architecture_{timestamp}.png")
        os.makedirs(VISUALIZATION_OUTPUT_PATH, exist_ok=True)

        # Generate layered architecture diagram
        visualkeras.layered_view(
            model,
            to_file=arch_path,
            legend=True,
            scale_xy=1,
            scale_z=1,
            max_z=400
        )
        print(f"[OK] Model architecture saved: {arch_path}")

        # Also generate a graph view with layer details
        from tensorflow.keras.utils import plot_model
        graph_path = os.path.join(VISUALIZATION_OUTPUT_PATH, f"model_graph_{timestamp}.png")
        plot_model(
            model,
            to_file=graph_path,
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            dpi=150
        )
        print(f"[OK] Model graph saved: {graph_path}")
    except Exception as e:
        print(f"[WARNING] Architecture visualization failed: {e}")

# ============================================
# CALLBACKS (Auto-save Best Models)
# ============================================

# ImageLogger: saves original / watermarked / diff every epoch
os.makedirs(DIFF_OUTPUT_PATH, exist_ok=True)
image_logger = ImageLogger(logger_snapshot, DIFF_OUTPUT_PATH)

callbacks = [
    # Per-epoch visual diff logger
    image_logger,

    # BEST ROBUSTNESS (Primary - watermark extraction)
    ModelCheckpoint(
        filepath=os.path.join(MODEL_OUTPUT_PATH, "best_medical_robust.weights.h5"),
        monitor="loss",  # Training loss (no validation)
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=1
    ),

    # BEST PSNR (Secondary - image quality)
    ModelCheckpoint(
        filepath=os.path.join(MODEL_OUTPUT_PATH, "best_medical_psnr.weights.h5"),
        monitor="embedded_image_loss",  # Training image loss
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        verbose=1
    ),

    # EARLY STOPPING (Prevents overfitting - monitors training loss)
    EarlyStopping(
        monitor="loss",
        patience=20,  # Stop after 20 stagnant epochs
        restore_best_weights=True,
        verbose=1
    ),

    # LEARNING RATE SCHEDULE
    ReduceLROnPlateau(
        monitor="loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),

    # TENSORBOARD (No profiling/histograms for 4GB VRAM)
    TensorBoard(
        log_dir=os.path.join("logs", f"medical_{timestamp}"),
        histogram_freq=TENSORBOARD_HISTOGRAM_FREQ,
        write_graph=True,
        write_images=False,  # Disabled for memory
        update_freq="epoch",
        profile_batch=TENSORBOARD_PROFILE_BATCH,
        embeddings_freq=0
    )
]

# ============================================
# COMPILATION (Medical-Optimized Loss)
# ============================================
optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    clipnorm=1.0  # Gradient clipping for stability
)

print("\n[START] MEDICAL WATERMARKING TRAINING (60K Chest X-rays)")
print(f"[CONFIG] Image / Watermark loss weights: {IMAGE_LOSS_WEIGHT:.1f} / {WATERMARK_LOSS_WEIGHT:.1f}")
if ATTACKS_DISABLED:
    print("[CONFIG] Attacks: DISABLED (all batches use identity pass-through)")
else:
    print(f"[CONFIG] Attacks: ENABLED (IDs {ATTACK_MIN_ID}-{ATTACK_MAX_ID}, paper-weighted distribution)")
    print("[CONFIG] Attack distribution: 25% none | 12.5% each: salt, gaussian, jpeg, dropout, rotation, stupid")
print(f"[CONFIG] Timeline: (100 epochs @ B={BATCH_SIZE})")
print(f"[CONFIG] Diff images will be saved to: {DIFF_OUTPUT_PATH}")

model.compile(
    optimizer=optimizer,
    loss={
        "embedded_image": "mse",        # Image fidelity
        "output_watermark": "mae",      # Watermark extraction
        "attacked_image": "mse"         # Attack robustness (low weight)
    },
    loss_weights={
        "embedded_image": IMAGE_LOSS_WEIGHT,
        "output_watermark": WATERMARK_LOSS_WEIGHT,
        "attacked_image": 0.0           # Don't penalize attack distortion
    },
    metrics={"output_watermark": "binary_accuracy"},
    jit_compile=False   # Disable XLA JIT: attack simulator uses ops incompatible with XLA_GPU_JIT
)

# ============================================
# TRAINING LAUNCH
# ============================================
try:
    print("\n" + "="*70)
    print("STARTING MEDICAL TRAINING (Hit Ctrl+C to save checkpoint)")
    print("="*70)

    history = model.fit(
        train_dataset,
        steps_per_epoch=train_batches,
        initial_epoch=0,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Final save
    final_path = os.path.join(MODEL_OUTPUT_PATH, f"medical_final_{timestamp}.weights.h5")
    model.save_weights(final_path)
    print(f"\n[DONE] Training complete! Final weights: {final_path}")

except KeyboardInterrupt:
    print("\n[STOPPED] Training interrupted - saving checkpoint...")
    model.save_weights(os.path.join(MODEL_OUTPUT_PATH, "medical_interrupted.weights.h5"))

except Exception as e:
    print(f"\n[ERROR] {e}")
    model.save_weights(os.path.join(MODEL_OUTPUT_PATH, "medical_crashed.weights.h5"))
