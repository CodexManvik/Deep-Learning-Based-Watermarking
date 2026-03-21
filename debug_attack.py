#!/usr/bin/env python3
"""
Debug script: Verify attacks are applied correctly
"""

import numpy as np
import tensorflow as tf
import cv2
from models.wavetf_model import WaveTFModel
from configs import IMAGE_SIZE, WATERMARK_SIZE, MODEL_OUTPUT_PATH, TEST_IMAGES_PATH
import os

def load_model():
    model = WaveTFModel(IMAGE_SIZE, WATERMARK_SIZE).get_model()
    weights = os.path.join(MODEL_OUTPUT_PATH, 'best_weights.h5')
    model.load_weights(weights)
    return model

def load_image():
    files = [f for f in os.listdir(TEST_IMAGES_PATH) if f.endswith('.jpg')]
    img_path = os.path.join(TEST_IMAGES_PATH, files[0])
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=[0, -1])
    return img

model = load_model()
image = load_image()
watermark = np.random.randint(0, 2, (1, 256)).astype(np.float32)

print("Testing if attacks are actually applied...\n")

attacks = {
    'No Attack': 0,
    'Salt & Pepper': 1,
    'Gaussian Noise': 2,
    'JPEG (q=50)': 3,
    'Dropout': 4,
}

for name, attack_id in attacks.items():
    attack_input = np.array([[attack_id]], dtype=np.int32)
    
    # Predict
    preds = model.predict([image, watermark, attack_input], verbose=0)
    clean_img = preds[0][0]
    extracted_wm = preds[1][0]
    attacked_img = preds[2][0]
    
    # Calculate image differences
    img_diff = np.mean(np.abs(clean_img - attacked_img))
    max_diff = np.max(np.abs(clean_img - attacked_img))
    
    # Calculate watermark accuracy
    original_wm = watermark[0]
    ber = np.mean(np.abs((original_wm > 0.5).astype(int) - (extracted_wm > 0.5).astype(int))) * 100
    
    print(f"{name:20} | Img Diff: {img_diff:.6f} | Max Diff: {max_diff:.3f} | BER: {ber:.2f}%")
    
    # Save attacked image
    cv2.imwrite(f'debug_{name.replace(" ", "_")}.png', (attacked_img * 255).astype(np.uint8))

print("\n✓ Check debug_*.png files to visually verify attacks")
print("\nExpected image differences:")
print("  No Attack:      ~0.0 (no change)")
print("  Salt & Pepper:  >0.05 (visible noise)")
print("  Gaussian:       >0.01 (subtle noise)")
print("  JPEG:           >0.002 (compression artifacts)")
print("  Dropout:        >0.1 (30% pixels zeroed)")
