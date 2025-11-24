#!/usr/bin/env python3
import os
import sys
import math
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
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
MAX_TEST_IMAGES = 20


# ----------------------------
# Metrics
# ----------------------------
def mse_cal(a, b):
    a = np.array(a).reshape(256, 256)
    b = np.array(b).reshape(256, 256)
    return mean_squared_error(a, b)


def psnr_cal(a, b):
    mse = mse_cal(a, b)
    if mse == 0:
        return 100
    return 10 * math.log10(1.0 / mse)


def ber_cal(orig, pred):
    orig = np.array(orig).flatten()
    pred = np.round(np.array(pred).flatten())
    total = len(orig)
    correct = np.sum(orig == pred)
    return 100 - (100 * correct / total)


# ----------------------------
# Load model (weights only)
# ----------------------------
def load_trained_model(weights_path):
    print(f"\nLoading weights from: {weights_path}")

    # Rebuild architecture exactly like trainer.py
    model = WaveTFModel(
        image_size=IMAGE_SIZE,
        watermark_size=WATERMARK_SIZE
    ).get_model()

    model.load_weights(weights_path)
    print("✓ Weights loaded successfully.")
    return model


# ----------------------------
# Select .h5 file
# ----------------------------
def select_model():
    files = [f for f in os.listdir(MODEL_OUTPUT_PATH) if f.endswith(".h5")]
    if not files:
        print("No .h5 weight files found.")
        return None

    print("\nAvailable weight files:")
    for i, f in enumerate(files):
        print(f"{i}: {f}")

    idx = int(input("\nSelect model number: "))
    return os.path.join(MODEL_OUTPUT_PATH, files[idx])


# ----------------------------
# Create dataset
# ----------------------------
def create_test_dataset():
    loader = MergedDataLoader(
        image_base_path=TEST_IMAGES_PATH,
        image_channels=[0],
        image_convert_type=tf.float32,
        watermark_size=WATERMARK_SIZE,
        attack_min_id=0,
        attack_max_id=1,
        batch_size=BATCH_SIZE
    ).get_data_loader()

    limit = MAX_TEST_IMAGES // BATCH_SIZE
    return loader.take(limit)


# ----------------------------
# Helper: normalize model image outputs to [0,1]
# ----------------------------
def normalize_pred_image(pred_img):
    """
    pred_img: numpy array, shape (H,W,1) assumed.
    Heuristics:
     - if values in [-1,1], rescale -> (pred+1)/2
     - otherwise clip to [0,1]
    """
    mn = pred_img.min()
    mx = pred_img.max()
    # debug prints for a few images (optional)
    # print(f"pred min/max = {mn:.6f}/{mx:.6f}")

    # if roughly symmetric around 0 and bounded by ~1, assume tanh
    if mn >= -1.1 and mx <= 1.1 and (mn < -0.1 or mx > 1.0):
        out = (pred_img + 1.0) / 2.0
        out = np.clip(out, 0.0, 1.0)
        return out
    # else just clamp
    return np.clip(pred_img, 0.0, 1.0)


def evaluate(model):
    psnr_vals = []
    ber_vals = []
    total = 0

    for (imgs, wms, attack_ids), (target_imgs, target_wms) in create_test_dataset():
        preds = model.predict([imgs, wms, attack_ids], verbose=0)

        # model returns list: [embedded_image, extracted_watermark]
        pred_imgs = preds[0]   # numpy array (N,H,W,1)
        pred_wms = preds[1]    # numpy array (N,WM_SIZE)

        # diagnostic: log global min/max for first batch
        if total == 0:
            print("DEBUG pred_imgs min/max:", float(pred_imgs.min()), float(pred_imgs.max()))
            print("DEBUG pred_wms min/max:", float(pred_wms.min()), float(pred_wms.max()))

        for i in range(len(imgs)):
            # imgs[i] is a tensor -> convert to numpy
            input_img = imgs[i].numpy()
            target_img = target_imgs[i].numpy()

            # pred_img is numpy already
            pimg = pred_imgs[i]

            # normalize predicted image into [0,1] using heuristics
            pimg_norm = normalize_pred_image(pimg)

            # compute PSNR using range = 1.0
            psnr_vals.append(psnr_cal(target_img, pimg_norm))

            # watermark: pred_wms is a float in [0,1] (sigmoid) OR otherwise; round at 0.5
            pred_w = pred_wms[i]
            ber_vals.append(ber_cal(target_wms[i], pred_w))

            total += 1

    return np.mean(psnr_vals), np.mean(ber_vals), total


def save_samples(model):
    os.makedirs(SAMPLE_OUTPUT_DIR, exist_ok=True)

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

    preds = model.predict([imgs, wms, attack_ids], verbose=0)
    pred_imgs = preds[0]

    for i in range(len(imgs)):
        inp = (imgs[i].numpy() * 255).astype(np.uint8)

        out_raw = pred_imgs[i]
        out_norm = normalize_pred_image(out_raw)
        out = (out_norm * 255).astype(np.uint8)

        diff = np.abs(inp - out)
        cv2.imwrite(f"{SAMPLE_OUTPUT_DIR}/input_{i}.png", inp)
        cv2.imwrite(f"{SAMPLE_OUTPUT_DIR}/watermarked_{i}.png", out)
        cv2.imwrite(f"{SAMPLE_OUTPUT_DIR}/diff_{i}.png", diff)

    print("Sample outputs saved.")

# ----------------------------
# Main
# ----------------------------
def main():
    print("\nWatermarking Model Evaluation\n")

    if not os.path.exists(TEST_IMAGES_PATH):
        print(f"Test image directory '{TEST_IMAGES_PATH}' not found.")
        return

    model_path = select_model()
    if not model_path:
        return

    model = load_trained_model(model_path)

    print("\nEvaluating...")
    psnr, ber, total = evaluate(model)

    print("\nResults:")
    table = [
        ["Images Evaluated", total],
        ["Average PSNR (dB)", f"{psnr:.2f}"],
        ["Average BER (%)", f"{ber:.2f}"],
    ]
    print(tabulate(table, tablefmt="grid"))

    save_samples(model)

    print("\nEvaluation complete.\n")


if __name__ == "__main__":
    main()
