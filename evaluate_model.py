#!/usr/bin/env python3
import os
import sys
import math
import cv2
import argparse
import numpy as np
import tensorflow as tf
from tabulate import tabulate

# Project imports
from data_loaders.merged_data_loader import MergedDataLoader
from models.wavetf_model import WaveTFModel
from configs import (
    MODEL_OUTPUT_PATH,
    IMAGE_SIZE,
    WATERMARK_SIZE,
    BATCH_SIZE,
    TEST_IMAGES_PATH
)

SAMPLE_OUTPUT_DIR = "evaluation_outputs/"
MAX_TEST_IMAGES = 2000 

# ----------------------------
# Metrics
# ----------------------------
def mse_cal(a, b):
    return np.mean((a - b) ** 2)

def psnr_cal(a, b):
    mse = mse_cal(a, b)
    if mse == 0:
        return 100
    return 10 * math.log10(1.0 / mse)

def ssim_cal(a, b):
    # Ensure inputs are float32 [0,1] tensors
    tf_a = tf.convert_to_tensor(a, dtype=tf.float32)
    tf_b = tf.convert_to_tensor(b, dtype=tf.float32)
    
    if len(tf_a.shape) == 2:
        tf_a = tf.expand_dims(tf_a, -1)
    if len(tf_b.shape) == 2:
        tf_b = tf.expand_dims(tf_b, -1)
        
    return float(tf.image.ssim(tf_a, tf_b, max_val=1.0))

def nc_cal(orig, pred):
    orig = np.array(orig).flatten()
    pred = np.array(pred).flatten()
    dot_prod = np.dot(orig, pred)
    norm_orig = np.linalg.norm(orig)
    norm_pred = np.linalg.norm(pred)
    if norm_orig == 0 or norm_pred == 0:
        return 0
    return dot_prod / (norm_orig * norm_pred)

def ber_cal(orig, pred):
    orig = np.array(orig).flatten()
    pred_bin = np.round(np.array(pred).flatten()) # Threshold at 0.5
    total = len(orig)
    correct = np.sum(orig == pred_bin)
    return 100.0 * (1 - (correct / total))

# ----------------------------
# Load model
# ----------------------------
def load_trained_model(weights_path):
    print(f"\n[INFO] Loading weights from: {weights_path}")
    model = WaveTFModel(
        image_size=IMAGE_SIZE,
        watermark_size=WATERMARK_SIZE
    ).get_model()

    try:
        model.load_weights(weights_path)
        print("✓ Weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        sys.exit(1)
    return model

# ----------------------------
# Select .h5 file
# ----------------------------
def select_model(provided_path=None):
    if provided_path:
        return provided_path

    files = sorted([f for f in os.listdir(MODEL_OUTPUT_PATH) if f.endswith(".h5") or f.endswith(".keras")], key=lambda x: os.path.getmtime(os.path.join(MODEL_OUTPUT_PATH, x)), reverse=True)
    
    if not files:
        print(f"No weight files found in {MODEL_OUTPUT_PATH}")
        return None

    print("\nAvailable weight files (Newest first):")
    for i, f in enumerate(files):
        print(f"{i}: {f}")

    try:
        selection = input("\nSelect model number (default 0): ")
        idx = int(selection) if selection else 0
        return os.path.join(MODEL_OUTPUT_PATH, files[idx])
    except:
        return None

# ----------------------------
# Evaluation Logic
# ----------------------------
def evaluate(model):
    print(f"\n[INFO] Starting Evaluation on {MAX_TEST_IMAGES} images...")
    
    psnr_vals, ssim_vals, cover_loss_vals = [], [], []
    ber_vals, nc_vals, secret_loss_vals = [], [], []
    total_imgs = 0

    # We use a custom loader loop to manually fix scaling issues
    loader = MergedDataLoader(
        image_base_path=TEST_IMAGES_PATH,
        image_channels=[0],
        image_convert_type=tf.float32, 
        watermark_size=WATERMARK_SIZE,
        attack_min_id=0,
        attack_max_id=1,
        batch_size=BATCH_SIZE
    ).get_data_loader()
    
    # Calculate how many batches needed
    limit = max(1, MAX_TEST_IMAGES // BATCH_SIZE)
    dataset = loader.take(limit)

    for (imgs, wms, attack_ids), (target_imgs, target_wms) in dataset:
        
        # --- CRITICAL FIX: SAFETY NORMALIZATION ---
        # If the data loader returns [0, 255], we force it to [0, 1]
        imgs_np = imgs.numpy()
        if imgs_np.max() > 1.5:
            imgs_np = imgs_np / 255.0
        
        target_imgs_np = target_imgs.numpy()
        if target_imgs_np.max() > 1.5:
            target_imgs_np = target_imgs_np / 255.0
        # ------------------------------------------

        preds = model.predict([imgs_np, wms, attack_ids], verbose=0)
        pred_imgs = preds[0]   # (N,H,W,1)
        pred_wms = preds[1]    # (N,WM_SIZE)

        for i in range(len(imgs)):
            if total_imgs >= MAX_TEST_IMAGES: break

            # 1. Image Metrics
            target_img = target_imgs_np[i] # Clean [0,1]
            pred_img = np.clip(pred_imgs[i], 0.0, 1.0) # Clean [0,1]

            cover_loss_vals.append(mse_cal(target_img, pred_img))
            psnr_vals.append(psnr_cal(target_img, pred_img))
            ssim_vals.append(ssim_cal(target_img, pred_img))

            # 2. Watermark Metrics
            target_wm = target_wms[i].numpy()
            pred_wm = pred_wms[i]

            secret_loss_vals.append(mse_cal(target_wm, pred_wm))
            nc_vals.append(nc_cal(target_wm, pred_wm))
            ber_vals.append(ber_cal(target_wm, pred_wm))

            total_imgs += 1

    return {
        "psnr": np.mean(psnr_vals),
        "ssim": np.mean(ssim_vals),
        "cover_loss": np.mean(cover_loss_vals),
        "secret_loss": np.mean(secret_loss_vals),
        "nc": np.mean(nc_vals),
        "ber": np.mean(ber_vals),
        "count": total_imgs
    }

def save_visual_samples(model):
    os.makedirs(SAMPLE_OUTPUT_DIR, exist_ok=True)
    print(f"\n[INFO] Saving visual samples to {SAMPLE_OUTPUT_DIR}...")

    loader = MergedDataLoader(
        image_base_path=TEST_IMAGES_PATH,
        image_channels=[0],
        image_convert_type=tf.float32,
        watermark_size=WATERMARK_SIZE,
        attack_min_id=0,
        attack_max_id=1,
        batch_size=min(5, BATCH_SIZE)
    ).get_data_loader()

    (imgs, wms, attack_ids), _ = next(iter(loader))
    
    # --- SAFETY NORMALIZATION FOR VISUALS ---
    imgs_np = imgs.numpy()
    if imgs_np.max() > 1.5:
        imgs_np = imgs_np / 255.0
    # ----------------------------------------

    preds = model.predict([imgs_np, wms, attack_ids], verbose=0)
    pred_imgs = preds[0]

    for i in range(len(imgs)):
        # Save Inputs (Convert 0-1 back to 0-255 for OpenCV)
        inp = (imgs_np[i] * 255).astype(np.uint8)
        
        # Save Predictions
        out_norm = np.clip(pred_imgs[i], 0.0, 1.0)
        out = (out_norm * 255).astype(np.uint8)

        # Difference
        diff = np.abs(inp.astype(np.float32) - out.astype(np.float32))
        diff = np.clip(diff * 50, 0, 255).astype(np.uint8)

        cv2.imwrite(f"{SAMPLE_OUTPUT_DIR}/sample_{i}_original.png", inp)
        cv2.imwrite(f"{SAMPLE_OUTPUT_DIR}/sample_{i}_watermarked.png", out)
        cv2.imwrite(f"{SAMPLE_OUTPUT_DIR}/sample_{i}_diff_x50.png", diff)

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()

    print("\nWatermarking Model Evaluation")
    if not os.path.exists(TEST_IMAGES_PATH):
        print(f"Error: '{TEST_IMAGES_PATH}' not found.")
        sys.exit(1)

    model_path = select_model(args.weights)
    if not model_path: sys.exit(0)

    model = load_trained_model(model_path)
    metrics = evaluate(model)

    print("\nEvaluation Results:")
    table = [
        ["Metric", "Value", "Target"],
        ["Images Evaluated", metrics['count'], "-"],
        ["PSNR (dB)", f"{metrics['psnr']:.2f}", "> 30"],
        ["SSIM", f"{metrics['ssim']:.4f}", "> 0.90"],
        ["Cover Pixel Loss", f"{metrics['cover_loss']:.6f}", "< 0.001"],
        ["Secret Pixel Loss", f"{metrics['secret_loss']:.6f}", "< 0.1"],
        ["NC", f"{metrics['nc']:.4f}", "> 0.95"],
        ["BER (%)", f"{metrics['ber']:.2f}", "< 1.0"],
    ]
    print(tabulate(table, headers="firstrow", tablefmt="grid"))
    
    save_visual_samples(model)
    print("\nDone.")