#!/usr/bin/env python3
"""
Text Watermark Evaluation Script
Tests if model trained on random bits can handle text watermarks.
"""

import os
import hashlib
import numpy as np
import tensorflow as tf
import cv2
from tabulate import tabulate

# Project imports
from models.wavetf_model import WaveTFModel
from configs import IMAGE_SIZE, WATERMARK_SIZE, MODEL_OUTPUT_PATH, TEST_IMAGES_PATH

# ============================================
# TEXT WATERMARK UTILITIES
# ============================================

def text_to_binary(text, size=256):
    """Convert text to 256-bit binary using SHA-256 hash"""
    hash_bytes = hashlib.sha256(text.encode('utf-8')).digest()
    bits = ''.join(format(b, '08b') for b in hash_bytes)
    binary = np.array([int(b) for b in bits[:size]], dtype=np.float32)
    return binary

def binary_to_text_visualization(binary, shape=(16, 16)):
    """Visualize binary watermark as image"""
    binary_int = (binary > 0.5).astype(np.uint8)
    img = binary_int.reshape(shape) * 255
    return img

def hamming_distance(bits1, bits2):
    """Calculate number of bit differences"""
    return np.sum(np.abs((bits1 > 0.5).astype(int) - (bits2 > 0.5).astype(int)))

def ber_percentage(original, extracted):
    """Calculate Bit Error Rate"""
    errors = hamming_distance(original, extracted)
    total = len(original)
    return (errors / total) * 100

def bits_to_hex(binary):
    """Convert binary array to hex string (for comparison)"""
    binary_int = (binary > 0.5).astype(int)
    hex_str = hex(int(''.join(map(str, binary_int)), 2))[2:].upper()
    return hex_str[:16] + "..."  # First 64 bits as hex

# ============================================
# TEST CONFIGURATION
# ============================================

# *** EDIT THESE TEXTS TO TEST YOUR WATERMARKS ***
TEST_TEXTS = [
    "Copyright 2025 IIT Delhi",
    "Copyright Protection",
    "Image Authentication",
    "Manvik BTech Project",
    "Hello World",
    "Property of Research Lab",
    "License MIT 2025",
    "Image ID: 12345",
    "https://example.com/img",
    "Author: John Doe",
]

# Attack configurations (same as paper)
ATTACKS = {
    'No Attack': 0,
    'Salt & Pepper': 1,
    'Gaussian Noise': 2,
    'JPEG (q=50)': 3,
    'Dropout': 4,
}

# ============================================
# MODEL LOADING WITH SELECTION
# ============================================

def list_available_weights():
    """List all available weight files"""
    files = [f for f in os.listdir(MODEL_OUTPUT_PATH) if f.endswith('.h5')]
    if not files:
        raise FileNotFoundError(f"No weights found in {MODEL_OUTPUT_PATH}")
    
    # Sort by modification time (newest first)
    files_with_time = []
    for f in files:
        full_path = os.path.join(MODEL_OUTPUT_PATH, f)
        mtime = os.path.getmtime(full_path)
        size = os.path.getsize(full_path) / (1024 * 1024)  # MB
        files_with_time.append((f, mtime, size))
    
    files_with_time.sort(key=lambda x: x[1], reverse=True)
    return files_with_time

def select_weights():
    """Interactive weight selection"""
    print("\n" + "="*70)
    print("  AVAILABLE MODEL WEIGHTS")
    print("="*70)
    
    weights_list = list_available_weights()
    
    print(f"\nFound {len(weights_list)} weight file(s) in {MODEL_OUTPUT_PATH}:\n")
    
    for idx, (filename, mtime, size) in enumerate(weights_list, 1):
        from datetime import datetime
        mod_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"  [{idx}] {filename}")
        print(f"      Modified: {mod_time} | Size: {size:.2f} MB")
    
    print("\n" + "-"*70)
    
    while True:
        try:
            choice = input(f"\nSelect model [1-{len(weights_list)}] or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("Exiting...")
                exit(0)
            
            idx = int(choice) - 1
            if 0 <= idx < len(weights_list):
                selected_file = weights_list[idx][0]
                return os.path.join(MODEL_OUTPUT_PATH, selected_file)
            else:
                print(f"⚠ Invalid choice. Please enter 1-{len(weights_list)}")
        except ValueError:
            print("⚠ Invalid input. Please enter a number or 'q'")
        except KeyboardInterrupt:
            print("\n\nExiting...")
            exit(0)

def load_model(weights_path=None):
    """Load trained watermarking model"""
    if weights_path is None:
        weights_path = select_weights()
    
    print(f"\n[INFO] Loading model from: {os.path.basename(weights_path)}")
    print(f"[INFO] Full path: {weights_path}")
    
    model = WaveTFModel(
        image_size=IMAGE_SIZE,
        watermark_size=WATERMARK_SIZE
    ).get_model()
    
    try:
        model.load_weights(weights_path)
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Error loading weights: {e}")
        exit(1)
    
    return model, weights_path

# ============================================
# TEST IMAGE LOADING
# ============================================

def load_test_image():
    """Load a random test image"""
    images = [f for f in os.listdir(TEST_IMAGES_PATH) if f.endswith(('.jpg', '.png'))]
    if not images:
        raise FileNotFoundError(f"No test images found in {TEST_IMAGES_PATH}")
    
    img_path = os.path.join(TEST_IMAGES_PATH, np.random.choice(images))
    img = cv2.imread(img_path, 0)  # Grayscale
    img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)   # Add batch dimension
    return img, os.path.basename(img_path)

# ============================================
# EVALUATION FUNCTION
# ============================================

def evaluate_text_watermark(model, text, image, attack_name, attack_id):
    """Embed and extract text watermark with specified attack"""
    
    # Convert text to binary
    watermark_bits = text_to_binary(text, size=WATERMARK_SIZE[0])
    watermark_input = watermark_bits.reshape(1, -1)  # Batch dimension
    
    # Prepare attack ID
    attack_input = np.array([[attack_id]], dtype=np.int32)
    
    # Predict: [clean_watermarked, extracted_wm, attacked_watermarked]
    predictions = model.predict([image, watermark_input, attack_input], verbose=0)
    extracted_bits = predictions[1][0]  # Remove batch dimension
    
    # Calculate metrics
    ber = ber_percentage(watermark_bits, extracted_bits)
    hamming = hamming_distance(watermark_bits, extracted_bits)
    
    return {
        'original_bits': watermark_bits,
        'extracted_bits': extracted_bits,
        'ber': ber,
        'hamming_distance': hamming,
        'total_bits': len(watermark_bits)
    }

# ============================================
# VISUALIZATION
# ============================================

def save_watermark_comparison(original_bits, extracted_bits, text, attack_name, output_dir='text_watermark_eval'):
    """Save side-by-side comparison of original vs extracted watermark"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizations
    original_img = binary_to_text_visualization(original_bits)
    extracted_img = binary_to_text_visualization(extracted_bits)
    
    # Scale up for visibility
    scale = 10
    original_scaled = cv2.resize(original_img, (16*scale, 16*scale), interpolation=cv2.INTER_NEAREST)
    extracted_scaled = cv2.resize(extracted_img, (16*scale, 16*scale), interpolation=cv2.INTER_NEAREST)
    
    # Side-by-side comparison
    comparison = np.hstack([original_scaled, extracted_scaled])
    
    # Add text labels
    comparison_rgb = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
    cv2.putText(comparison_rgb, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(comparison_rgb, "Extracted", (170, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Sanitize filename
    safe_text = text.replace(' ', '_').replace('/', '_')[:30]
    safe_attack = attack_name.replace(' ', '_').replace('(', '').replace(')', '')
    filename = f"{safe_text}_{safe_attack}.png"
    
    cv2.imwrite(os.path.join(output_dir, filename), comparison_rgb)
    return filename

# ============================================
# MAIN EVALUATION
# ============================================

def main():
    print("="*70)
    print("  TEXT WATERMARK EVALUATION")
    print("  Testing if model trained on random bits generalizes to text")
    print("="*70)
    
    # Load model (with interactive selection)
    model, weights_path = load_model()
    
    # Load test image
    test_image, img_name = load_test_image()
    print(f"[INFO] Using test image: {img_name}\n")
    
    # Store results
    all_results = []
    
    # Test each text with each attack
    for text in TEST_TEXTS:
        print(f"\n{'─'*70}")
        print(f"Testing text: \"{text}\"")
        print(f"{'─'*70}")
        
        text_results = []
        
        for attack_name, attack_id in ATTACKS.items():
            result = evaluate_text_watermark(model, text, test_image, attack_name, attack_id)
            
            # Save visualization
            save_watermark_comparison(
                result['original_bits'],
                result['extracted_bits'],
                text,
                attack_name
            )
            
            text_results.append({
                'Text': text[:30] + '...' if len(text) > 30 else text,
                'Attack': attack_name,
                'BER (%)': f"{result['ber']:.2f}",
                'Errors': f"{result['hamming_distance']}/{result['total_bits']}",
                'Original (hex)': bits_to_hex(result['original_bits']),
                'Extracted (hex)': bits_to_hex(result['extracted_bits']),
            })
            
            all_results.append({
                'text': text,
                'attack': attack_name,
                'ber': result['ber']
            })
        
        # Print results for this text
        print(tabulate(text_results, headers='keys', tablefmt='grid'))
    
    # Summary statistics
    print("\n" + "="*70)
    print("  SUMMARY STATISTICS")
    print("="*70)
    
    summary = []
    for attack_name in ATTACKS.keys():
        attack_bers = [r['ber'] for r in all_results if r['attack'] == attack_name]
        summary.append({
            'Attack Type': attack_name,
            'Avg BER (%)': f"{np.mean(attack_bers):.2f}",
            'Min BER (%)': f"{np.min(attack_bers):.2f}",
            'Max BER (%)': f"{np.max(attack_bers):.2f}",
            'Std Dev': f"{np.std(attack_bers):.2f}",
        })
    
    print(tabulate(summary, headers='keys', tablefmt='grid'))
    
    # Per-text average
    print("\n" + "="*70)
    print("  PER-TEXT AVERAGE BER")
    print("="*70)
    
    text_summary = []
    for text in TEST_TEXTS:
        text_bers = [r['ber'] for r in all_results if r['text'] == text]
        text_summary.append({
            'Text': text[:40] + '...' if len(text) > 40 else text,
            'Avg BER (%)': f"{np.mean(text_bers):.2f}",
            'No Attack BER (%)': f"{[r['ber'] for r in all_results if r['text']==text and r['attack']=='No Attack'][0]:.2f}",
            'JPEG BER (%)': f"{[r['ber'] for r in all_results if r['text']==text and r['attack']=='JPEG (q=50)'][0]:.2f}",
        })
    
    print(tabulate(text_summary, headers='keys', tablefmt='grid'))
    
    print("\n" + "="*70)
    print(f"✓ Evaluation complete!")
    print(f"✓ Model used: {os.path.basename(weights_path)}")
    print(f"✓ Visualizations saved to: text_watermark_eval/")
    print("="*70)

# ============================================
# ADVANCED: TEXT RECONSTRUCTION ATTEMPT
# ============================================

def try_text_reconstruction(extracted_bits, candidate_texts):
    """
    Try to match extracted bits to closest known text.
    (Only works if BER is very low)
    """
    min_distance = float('inf')
    best_match = None
    
    for text in candidate_texts:
        text_bits = text_to_binary(text, size=len(extracted_bits))
        distance = hamming_distance(text_bits, extracted_bits)
        
        if distance < min_distance:
            min_distance = distance
            best_match = text
    
    ber = (min_distance / len(extracted_bits)) * 100
    return best_match, ber

def advanced_evaluation():
    """
    Try to reconstruct original text from extracted bits.
    Only works if BER < 5% (rough threshold)
    """
    print("\n" + "="*70)
    print("  ADVANCED: TEXT RECONSTRUCTION TEST")
    print("="*70)
    
    model, _ = load_model()
    test_image, _ = load_test_image()
    
    reconstruction_results = []
    
    for text in TEST_TEXTS:
        # Test with no attack (should have lowest BER)
        result = evaluate_text_watermark(model, text, test_image, 'No Attack', 0)
        
        # Try to reconstruct
        best_match, match_ber = try_text_reconstruction(
            result['extracted_bits'],
            TEST_TEXTS
        )
        
        reconstruction_results.append({
            'Original Text': text,
            'Best Match': best_match,
            'Match?': '✓' if best_match == text else '✗',
            'BER to Match (%)': f"{match_ber:.2f}",
            'Extraction BER (%)': f"{result['ber']:.2f}"
        })
    
    print(tabulate(reconstruction_results, headers='keys', tablefmt='grid'))
    
    # Analysis
    exact_matches = sum(1 for r in reconstruction_results if r['Match?'] == '✓')
    print(f"\n[INFO] Exact text reconstruction: {exact_matches}/{len(TEST_TEXTS)} ({100*exact_matches/len(TEST_TEXTS):.1f}%)")
    
    if exact_matches == len(TEST_TEXTS):
        print("✓ Perfect! Model can embed and extract text with 100% accuracy (no attack)")
    elif exact_matches > len(TEST_TEXTS) * 0.8:
        print("✓ Excellent! Most texts reconstructed correctly")
    else:
        print("⚠ Note: Low exact match rate expected - extracted bits may differ by few bits")
        print("  → BER metric is more reliable than exact text matching")

if __name__ == "__main__":
    # Basic evaluation
    main()
    
    # Optional: Try text reconstruction (only works with very low BER)
    print("\n")
    user_input = input("Run text reconstruction test? (y/n): ").strip().lower()
    if user_input == 'y':
        advanced_evaluation()
