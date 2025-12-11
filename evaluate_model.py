#!/usr/bin/env python3
"""
Comprehensive Evaluation Script - Multiple Testing Modes + Auto Sample Saving

Automatically saves one sample per attack type for visual verification.
This helps debug if attacks are actually being applied correctly.
"""

import os
import sys
import math
import cv2
import argparse
import numpy as np
from configs import *
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
    TEST_IMAGES_PATH,
    MAX_TEST_IMAGES
)

SAMPLE_OUTPUT_DIR = "evaluation_outputs/"
MAX_TEST_IMAGES = MAX_TEST_IMAGES

# ----------------------------
# Attack Definitions
# ----------------------------

# Default: Training distribution (random parameters)
ATTACK_SUITE_RANDOM = {
    'No Attack': 0,
    'Salt & Pepper (random)': 1,
    'Gaussian Noise (random σ∈[0.05,0.15])': 2,
    'JPEG (random q∈[50,90])': 3,
    'Dropout (random)': 4,
    'Rotation (random)': 5,
}

# Paper baseline: Fixed parameters from Table 3
ATTACK_SUITE_PAPER = {
    'No Attack': 0,
    'Salt & Pepper (p=0.1)': 1,
    'Gaussian Noise (σ=0.15)': 2,
    'JPEG (q=50)': 3,
    'Dropout (p=0.3)': 4,
}

# Stratified: Multiple fixed strengths per attack
ATTACK_SUITE_STRATIFIED = {
    # No attack
    'No Attack': 0,
    
    # Salt & Pepper (not parameterizable without modifying attack code)
    'Salt & Pepper (p=0.1)': 1,
    
    # Gaussian - 3 levels + random
    'Gaussian - Light (σ=0.05)': 2,
    'Gaussian - Medium (σ=0.10)': 2,
    'Gaussian - Paper (σ=0.15)': 2,
    'Gaussian - Random': 2,
    
    # JPEG - 3 levels + random
    'JPEG - High Quality (q=90)': 3,
    'JPEG - Medium (q=70)': 3,
    'JPEG - Paper (q=50)': 3,
    'JPEG - Random': 3,
    
    # Dropout
    'Dropout (p=0.3)': 4,
}

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
    tf_a = tf.convert_to_tensor(a, dtype=tf.float32)
    tf_b = tf.convert_to_tensor(b, dtype=tf.float32)
    if len(tf_a.shape) == 2:
        tf_a = tf.expand_dims(tf_a, -1)
    if len(tf_b.shape) == 2:
        tf_b = tf.expand_dims(tf_b, -1)
    return float(tf.image.ssim(tf_a, tf_b, max_val=1.0))

def nc_cal(orig, pred):
    """Normalized Correlation"""
    orig = np.array(orig).flatten()
    pred = np.array(pred).flatten()
    dot_prod = np.dot(orig, pred)
    norm_orig = np.linalg.norm(orig)
    norm_pred = np.linalg.norm(pred)
    if norm_orig == 0 or norm_pred == 0:
        return 0
    return dot_prod / (norm_orig * norm_pred)

def ber_cal(orig, pred):
    """Bit Error Rate (%)"""
    orig = np.array(orig).flatten()
    pred_bin = np.round(np.array(pred).flatten())
    total = len(orig)
    correct = np.sum(orig == pred_bin)
    return 100.0 * (1 - (correct / total))

# ----------------------------
# Sample Saving Helper
# ----------------------------
def save_single_sample(original_img, watermarked_img, attack_name, sample_dir):
    """
    Save a single sample with attack-specific naming.
    This lets you visually verify each attack is working.
    """
    os.makedirs(sample_dir, exist_ok=True)
    
    # Sanitize attack name for filename
    safe_name = attack_name.replace(' ', '_').replace('(', '').replace(')', '').replace('σ', 'sigma').replace('±', 'pm').replace('∈', 'in')
    
    # Original
    inp = (original_img * 255).astype(np.uint8)
    
    # Watermarked
    out_norm = np.clip(watermarked_img, 0.0, 1.0)
    out = (out_norm * 255).astype(np.uint8)
    
    # Difference (amplified)
    diff = np.abs(inp.astype(np.float32) - out.astype(np.float32))
    diff_amplified = np.clip(diff * 50, 0, 255).astype(np.uint8)
    
    # Save with attack-specific names
    cv2.imwrite(f"{sample_dir}/{safe_name}_original.png", inp)
    cv2.imwrite(f"{sample_dir}/{safe_name}_watermarked.png", out)
    cv2.imwrite(f"{sample_dir}/{safe_name}_diff_x50.png", diff_amplified)
    
    return safe_name

# ----------------------------
# Load Model
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
        print(f"✗ Error loading weights: {e}")
        sys.exit(1)
    
    return model

# ----------------------------
# Model Selection
# ----------------------------
def select_model(provided_path=None):
    if provided_path:
        return provided_path
    
    files = sorted(
        [f for f in os.listdir(MODEL_OUTPUT_PATH) 
         if f.endswith(".h5") or f.endswith(".keras")],
        key=lambda x: os.path.getmtime(os.path.join(MODEL_OUTPUT_PATH, x)),
        reverse=True
    )
    
    if not files:
        print(f"No weight files found in {MODEL_OUTPUT_PATH}")
        return None
    
    print("\n=== Available Weight Files (Newest First) ===")
    for i, f in enumerate(files):
        print(f"  [{i}] {f}")
    
    try:
        selection = input("\nSelect model number (default 0): ").strip()
        idx = int(selection) if selection else 0
        return os.path.join(MODEL_OUTPUT_PATH, files[idx])
    except (ValueError, IndexError):
        print("Invalid selection.")
        return None

# ----------------------------
# Per-Attack Evaluation (WITH AUTO SAMPLE SAVING)
# ----------------------------
# In evaluate_single_attack function, update prediction handling:

def evaluate_single_attack(model, attack_name, attack_id, max_images=MAX_TEST_IMAGES, save_samples=True):
    """
    Evaluate model on a SINGLE attack type.
    Automatically saves one sample for visual verification.
    """
    print(f"\n{'='*60}")
    print(f"  Testing: {attack_name}")
    print(f"{'='*60}")
    
    psnr_vals, ssim_vals, cover_loss_vals = [], [], []
    ber_vals, nc_vals, secret_loss_vals = [], [], []
    total_imgs = 0
    sample_saved = False
    
    loader = MergedDataLoader(
        image_base_path=TEST_IMAGES_PATH,
        image_channels=[0],
        image_convert_type=None,
        watermark_size=WATERMARK_SIZE,
        attack_min_id=attack_id,
        attack_max_id=attack_id + 1,
        batch_size=BATCH_SIZE,
        use_paper_attack_distribution=False
    ).get_data_loader()
    
    limit = max(1, max_images // BATCH_SIZE)
    dataset = loader.take(limit)
    
    # UPDATED: Now unpacks 3 targets instead of 2
    for (imgs, wms, attack_ids), (target_imgs, target_wms, target_imgs_attacked) in dataset:
        # Normalize inputs
        imgs_np = imgs.numpy()
        if imgs_np.max() > 1.5:
            imgs_np = imgs_np / 255.0
        
        target_imgs_np = target_imgs.numpy()
        if target_imgs_np.max() > 1.5:
            target_imgs_np = target_imgs_np / 255.0
        
        # Force specific attack
        attack_ids = tf.fill(tf.shape(attack_ids), attack_id)
        
        # Model returns 3 outputs
        preds = model.predict([imgs_np, wms, attack_ids], verbose=0)
        pred_imgs_clean = preds[0]   # Clean watermarked (for PSNR)
        pred_wms = preds[1]           # Extracted watermarks
        pred_imgs_attacked = preds[2] # Attacked watermarked (for visualization)
        
        for i in range(len(imgs)):
            if total_imgs >= max_images:
                break
            
            # Save sample with ATTACKED image (so we can see the attack effects)
            if save_samples and not sample_saved and total_imgs == 0:
                safe_name = save_single_sample(
                    imgs_np[i],
                    pred_imgs_attacked[i],  # Use attacked version for visualization
                    attack_name,
                    SAMPLE_OUTPUT_DIR
                )
                print(f"  → Saved sample: {safe_name}_*.png")
                sample_saved = True
            
            # Image Quality Metrics (use CLEAN watermarked for PSNR)
            target_img = target_imgs_np[i]
            pred_img_clean = np.clip(pred_imgs_clean[i], 0.0, 1.0)
            
            cover_loss_vals.append(mse_cal(target_img, pred_img_clean))
            psnr_vals.append(psnr_cal(target_img, pred_img_clean))
            ssim_vals.append(ssim_cal(target_img, pred_img_clean))
            
            # Watermark Robustness Metrics
            target_wm = target_wms[i].numpy()
            pred_wm = pred_wms[i]
            
            secret_loss_vals.append(mse_cal(target_wm, pred_wm))
            nc_vals.append(nc_cal(target_wm, pred_wm))
            ber_vals.append(ber_cal(target_wm, pred_wm))
            
            total_imgs += 1
        
        if total_imgs >= max_images:
            break
    
    return {
        'attack': attack_name,
        'psnr': np.mean(psnr_vals),
        'ssim': np.mean(ssim_vals),
        'cover_loss': np.mean(cover_loss_vals),
        'secret_loss': np.mean(secret_loss_vals),
        'nc': np.mean(nc_vals),
        'ber': np.mean(ber_vals),
        'ber_std': np.std(ber_vals),
        'count': total_imgs
    }


# ----------------------------
# Full Suite Evaluation
# ----------------------------
def evaluate_full_suite(model, attack_suite, save_samples=True):
    """Run evaluation on all attack types in the specified suite."""
    all_results = []
    
    print(f"\n[INFO] Samples will be saved to: {SAMPLE_OUTPUT_DIR}")
    
    for attack_name, attack_id in attack_suite.items():
        result = evaluate_single_attack(model, attack_name, attack_id, save_samples=save_samples)
        all_results.append(result)
    
    return all_results

# ----------------------------
# Results Display
# ----------------------------
def print_results_table(results, mode="default"):
    """Print results table with context based on evaluation mode."""
    
    if mode == "paper":
        title = "EVALUATION RESULTS (Paper Table 3 - Fixed Parameters)"
        note = "Note: Uses exact paper parameters (σ=0.15, quality=50, etc.)"
    elif mode == "stratified":
        title = "EVALUATION RESULTS (Stratified - Multiple Attack Strengths)"
        note = "Note: Tests model across different attack intensities"
    else:
        title = "EVALUATION RESULTS (Training Distribution)"
        note = "Note: Random parameters matching training (σ∈[0.05,0.15], q∈[50,90])"
    
    print("\n" + "="*90)
    print(f"  {title}")
    print("="*90)
    print(f"{note}")
    print("="*90)
    
    # Summary Table
    summary_table = [
        ["Attack Type", "PSNR (dB)", "SSIM", "NC", "BER (%)", "BER Std", "Images"]
    ]
    
    for r in results:
        summary_table.append([
            r['attack'],
            f"{r['psnr']:.2f}",
            f"{r['ssim']:.4f}",
            f"{r['nc']:.4f}",
            f"{r['ber']:.2f}",
            f"±{r['ber_std']:.2f}" if r['ber_std'] > 0.5 else "-",
            r['count']
        ])
    
    print(tabulate(summary_table, headers="firstrow", tablefmt="grid"))
    
    # Paper Comparison (only for paper mode)
    if mode == "paper":
        print("\n" + "="*90)
        print("  COMPARISON WITH PAPER BASELINES (Table 3)")
        print("="*90)
        
        comparison_table = [
            ["Metric", "Your Model", "Paper Target", "Status"]
        ]
        
        # Overall PSNR
        avg_psnr = np.mean([r['psnr'] for r in results])
        comparison_table.append([
            "PSNR (dB)",
            f"{avg_psnr:.2f}",
            "> 30 (Paper: 40.1)",
            "✓" if avg_psnr > 30 else "✗"
        ])
        
        # No Attack BER
        no_attack_ber = next((r['ber'] for r in results if 'No Attack' in r['attack']), None)
        if no_attack_ber is not None:
            comparison_table.append([
                "BER - No Attack (%)",
                f"{no_attack_ber:.4f}",
                "< 0.001 (Paper: 0.0003)",
                "✓" if no_attack_ber < 0.001 else "✗"
            ])
        
        # Salt & Pepper BER
        sp_ber = next((r['ber'] for r in results if 'Salt' in r['attack']), None)
        if sp_ber is not None:
            comparison_table.append([
                "BER - Salt & Pepper (%)",
                f"{sp_ber:.2f}",
                "< 10 (Paper: 1.42)",
                "✓" if sp_ber < 10 else "✗"
            ])
        
        # Gaussian BER
        gauss_ber = next((r['ber'] for r in results if 'Gaussian' in r['attack'] and 'σ=0.15' in r['attack']), None)
        if gauss_ber is not None:
            comparison_table.append([
                "BER - Gaussian Noise (%)",
                f"{gauss_ber:.2f}",
                "< 15 (Paper: 7.54)",
                "✓" if gauss_ber < 15 else "✗"
            ])
        
        # JPEG BER
        jpeg_ber = next((r['ber'] for r in results if 'JPEG' in r['attack'] and 'q=50' in r['attack']), None)
        if jpeg_ber is not None:
            comparison_table.append([
                "BER - JPEG (q=50) (%)",
                f"{jpeg_ber:.2f}",
                "< 30 (Paper: 27.25)",
                "✓" if jpeg_ber < 30 else "✗"
            ])
        
        # Dropout BER
        dropout_ber = next((r['ber'] for r in results if 'Dropout' in r['attack']), None)
        if dropout_ber is not None:
            comparison_table.append([
                "BER - Dropout (%)",
                f"{dropout_ber:.2f}",
                "< 20 (Paper: 11.71)",
                "✓" if dropout_ber < 20 else "✗"
            ])
        
        print(tabulate(comparison_table, headers="firstrow", tablefmt="grid"))
    
    # Summary stats
    print("\n" + "="*90)
    print("  SUMMARY")
    print("="*90)
    avg_psnr = np.mean([r['psnr'] for r in results])
    avg_ber = np.mean([r['ber'] for r in results])
    avg_nc = np.mean([r['nc'] for r in results])
    
    summary_stats = [
        ["Overall Average PSNR", f"{avg_psnr:.2f} dB"],
        ["Overall Average BER", f"{avg_ber:.2f}%"],
        ["Overall Average NC", f"{avg_nc:.4f}"],
    ]
    print(tabulate(summary_stats, tablefmt="simple"))

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate watermarking model with automatic sample saving per attack"
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to .h5/.keras weights file"
    )
    parser.add_argument(
        "--mode", type=str, default="default",
        choices=["default", "paper", "stratified"],
        help="Evaluation mode: default (training dist), paper (fixed params), stratified (multiple strengths)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Test only 200 images per attack (fast iteration)"
    )
    parser.add_argument(
        "--no-samples", action="store_true",
        help="Disable automatic sample saving (faster)"
    )
    args = parser.parse_args()
    
    # Select attack suite based on mode
    if args.mode == "paper":
        attack_suite = ATTACK_SUITE_PAPER
        mode_name = "Paper Baseline (Fixed Parameters)"
    elif args.mode == "stratified":
        attack_suite = ATTACK_SUITE_STRATIFIED
        mode_name = "Stratified (Multiple Strengths)"
    else:
        attack_suite = ATTACK_SUITE_RANDOM
        mode_name = "Default (Training Distribution)"
    
    print("\n" + "="*90)
    print("  WATERMARKING MODEL EVALUATION")
    print(f"  Mode: {mode_name}")
    print("="*90)
    
    # Validate test path
    if not os.path.exists(TEST_IMAGES_PATH):
        print(f"✗ Error: Test images directory '{TEST_IMAGES_PATH}' not found.")
        sys.exit(1)
    
    # Load model
    model_path = select_model(args.weights)
    if not model_path:
        sys.exit(0)
    
    model = load_trained_model(model_path)
    
    # Adjust test size
    if args.quick:
        MAX_TEST_IMAGES = 200
        print(f"[INFO] Quick mode: Testing {MAX_TEST_IMAGES} images per attack")
    
    # Run evaluation (samples saved by default)
    save_samples = not args.no_samples
    results = evaluate_full_suite(model, attack_suite, save_samples=save_samples)
    
    # Display results
    print_results_table(results, mode=args.mode)
    
    if save_samples:
        print(f"\n✓ Visual samples saved to: {SAMPLE_OUTPUT_DIR}")
        print("  Check the *_watermarked.png files to verify attacks are being applied correctly.")
    
    print("\n✓ Evaluation complete.\n")
    print("\nTIP: Check JPEG samples especially - verify compression artifacts are visible")
