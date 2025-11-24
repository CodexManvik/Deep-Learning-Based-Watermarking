import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from models.wavetf_model import WaveTFModel
from configs import IMAGE_SIZE, WATERMARK_SIZE, MODEL_OUTPUT_PATH

# Force CPU to avoid messing with training GPU memory
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def analyze_model_state():
    weights_path = os.path.join(MODEL_OUTPUT_PATH, "best_weights.h5")
    # Fallback to final weights if best doesn't exist
    if not os.path.exists(weights_path):
        weights_path = os.path.join(MODEL_OUTPUT_PATH, f"final_weights-{os.listdir(MODEL_OUTPUT_PATH)[0].split('-')[-1]}")
    
    print(f"\n>>> INSPECTING WEIGHTS: {weights_path}")

    # 1. Load Model
    model = WaveTFModel(IMAGE_SIZE, WATERMARK_SIZE).get_model()
    try:
        model.load_weights(weights_path)
        print("✓ Weights loaded successfully.")
    except Exception as e:
        print(f"✗ Could not load weights: {e}")
        return

    # 2. Create Synthetic Test Data
    # Use a real image if possible, otherwise random noise
    test_img_path = "test_images/test.jpg"
    if os.path.exists(test_img_path):
        print(f"Loading real image: {test_img_path}")
        img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        img_in = img.astype("float32") / 255.0
    else:
        print("⚠ No test image found, using random noise image.")
        img_in = np.random.rand(256, 256).astype("float32")

    img_batch = np.expand_dims(img_in, axis=(0, -1))

    # Create a structured watermark (Half 0s, Half 1s) to test contrast
    wm_in = np.zeros((1, WATERMARK_SIZE[0]), dtype="float32")
    wm_in[0, :WATERMARK_SIZE[0]//2] = 1.0  # First half 1, second half 0
    
    # Attack ID 0 (No attack)
    attack_id = np.zeros((1, 1), dtype="int32")

    # 3. Run Prediction
    print("\n>>> RUNNING INFERENCE...")
    embedded_img, extracted_wm = model.predict([img_batch, wm_in, attack_id], verbose=0)
    
    # Remove batch dimensions
    emb_out = embedded_img[0, :, :, 0]
    wm_out = extracted_wm[0]

    # 4. Statistical Analysis
    print("\n" + "="*40)
    print("       DIAGNOSTIC REPORT")
    print("="*40)

    # --- A. IMAGE ANALYSIS ---
    diff = np.abs(img_in - emb_out)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"\n[IMAGE STATUS]")
    print(f"Max Pixel Change:  {max_diff:.6f} (0.0 = Identity/Stuck)")
    print(f"Avg Pixel Change:  {mean_diff:.6f}")
    
    if max_diff < 1e-6:
        print("🔴 CRITICAL: The model is NOT modifying the image at all.")
    else:
        print("🟢 GOOD: The model is attempting to embed information.")

    # --- B. WATERMARK ANALYSIS ---
    print(f"\n[WATERMARK STATUS]")
    print(f"Input:  Min={np.min(wm_in):.1f}, Max={np.max(wm_in):.1f}")
    print(f"Output: Min={np.min(wm_out):.4f}, Max={np.max(wm_out):.4f}, Mean={np.mean(wm_out):.4f}")
    
    # Bit Error Rate (Threshold at 0.5)
    wm_rounded = np.round(wm_out)
    errors = np.sum(np.abs(wm_in[0] - wm_rounded))
    ber = errors / WATERMARK_SIZE[0]
    
    print(f"Bit Error Rate:    {ber*100:.2f}% (Target: < 1.0%)")
    
    if ber > 0.4:
        print("🔴 CRITICAL: Model is random guessing (BER ~50%).")
    elif ber > 0.05:
        print("🟠 WARNING: Model is learning but struggling.")
    else:
        print("🟢 EXCELLENT: Watermark is being recovered!")

    # --- C. VISUALIZATION ---
    plt.figure(figsize=(15, 5))

    # 1. Original vs Watermarked
    plt.subplot(1, 4, 1)
    plt.title("Original")
    plt.imshow(img_in, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Watermarked")
    plt.imshow(emb_out, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')

    # 2. The Hidden Signal (Difference)
    plt.subplot(1, 4, 3)
    plt.title(f"Added Noise (Max: {max_diff:.4f})")
    # Normalize difference for visibility
    if max_diff > 0:
        norm_diff = diff / max_diff
    else:
        norm_diff = diff
    plt.imshow(norm_diff, cmap='inferno')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    # 3. Watermark Recovery
    plt.subplot(1, 4, 4)
    plt.title("Watermark: Input (Blue) vs Output (Red)")
    plt.plot(wm_in[0], label="Target", color="blue", alpha=0.6, linewidth=2)
    plt.plot(wm_out, label="Predicted", color="red", alpha=0.6, linewidth=1)
    plt.legend(loc='upper right')
    plt.ylim(-0.1, 1.1)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ensure you have a dummy image or this will use noise
    analyze_model_state()