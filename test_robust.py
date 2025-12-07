import os
import sys
import numpy as np
import tensorflow as tf
from tabulate import tabulate

from models.wavetf_model import WaveTFModel
from data_loaders.merged_data_loader import MergedDataLoader
from configs import *

# --- CONFIG ---
MAX_IMAGES = 2000  
BATCH_SIZE_TEST = 10
# --------------

def load_best_model():
    files = sorted([f for f in os.listdir(MODEL_OUTPUT_PATH) if f.endswith(".h5")], 
                   key=lambda x: os.path.getmtime(os.path.join(MODEL_OUTPUT_PATH, x)), 
                   reverse=True)
    if not files:
        print("No weights found!")
        sys.exit(1)
        
    weight_path = os.path.join(MODEL_OUTPUT_PATH, files[0])
    print(f"Loading weights: {weight_path}")
    
    model = WaveTFModel(IMAGE_SIZE, WATERMARK_SIZE).get_model()
    model.load_weights(weight_path)
    return model

def ber_cal(orig, pred):
    orig = np.array(orig).flatten()
    pred_bin = np.round(np.array(pred).flatten())
    total = len(orig)
    correct = np.sum(orig == pred_bin)
    return 100.0 * (1 - (correct / total))

def nc_cal(orig, pred):
    # Formula: DotProduct(A, B) / (Norm(A) * Norm(B))
    orig = np.array(orig).flatten()
    pred = np.array(pred).flatten()
    
    dot_prod = np.dot(orig, pred)
    norm_orig = np.linalg.norm(orig)
    norm_pred = np.linalg.norm(pred)
    
    if norm_orig == 0 or norm_pred == 0:
        return 0
    
    return dot_prod / (norm_orig * norm_pred)

def run_attack_test(model, attack_name):
    attack_map = {
        "No Attack": 0,
        "Salt & Pepper": 1,
        "Gaussian Noise": 2,
        "JPEG": 3,
        "Dropout": 4
    }
    
    if attack_name not in attack_map:
        return 0.0, 0.0

    target_id = attack_map[attack_name]

    # Initialize Loader
    loader = MergedDataLoader(
        image_base_path=TEST_IMAGES_PATH,
        image_channels=[0],
        image_convert_type=tf.float32,
        watermark_size=WATERMARK_SIZE,
        attack_min_id=0, attack_max_id=1, 
        batch_size=BATCH_SIZE_TEST
    ).get_data_loader()

    batches_needed = MAX_IMAGES // BATCH_SIZE_TEST
    dataset = loader.take(batches_needed)

    total_ber = []
    total_nc = []
    processed_count = 0

    for (imgs, wms, _), (_, target_wms) in dataset:
        
        # 1. Safety Normalization
        imgs_np = imgs.numpy()
        if imgs_np.max() > 1.5:
            imgs_np = imgs_np / 255.0

        # 2. Force Attack ID
        id_tensor = tf.fill((len(imgs), 1), target_id)
        id_tensor = tf.cast(id_tensor, tf.int32)
        
        # 3. Predict
        preds = model.predict([imgs_np, wms, id_tensor], verbose=0)
        extracted_wms = preds[1]
        
        # 4. Calculate Stats (BER and NC)
        for i in range(len(imgs)):
            # Bit Error Rate
            ber = ber_cal(target_wms[i], extracted_wms[i])
            total_ber.append(ber)
            
            # Normalized Correlation
            nc = nc_cal(target_wms[i], extracted_wms[i])
            total_nc.append(nc)
            
            processed_count += 1
            
    print(f"Tested {processed_count} images.", end=" ")
    return np.mean(total_ber), np.mean(total_nc)

def main():
    if not os.path.exists(TEST_IMAGES_PATH):
        print(f"Error: {TEST_IMAGES_PATH} not found.")
        sys.exit(1)

    model = load_best_model()
    
    print(f"\nRunning Robustness Evaluation (BER & NC) on {MAX_IMAGES} images...")
    print("---------------------------------------------------------------")
    
    results = []
    attacks = ["No Attack", "Salt & Pepper", "Gaussian Noise", "Dropout", "JPEG"]
    
    for atk in attacks:
        print(f"Testing {atk}...", end="\r")
        
        # Get both scores
        ber_score, nc_score = run_attack_test(model, atk)
        
        results.append([atk, f"{ber_score:.2f} %", f"{nc_score:.4f}"])
        print(f"Testing {atk: <20} Done. BER: {ber_score:.2f}% | NC: {nc_score:.4f}")
        
    print("\nFinal Robustness Results:")
    print(tabulate(results, headers=["Attack Type", "BER (Lower is Better)", "NC (Higher is Better)"], tablefmt="grid"))

if __name__ == "__main__":
    main()