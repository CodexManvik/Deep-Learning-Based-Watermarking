import os
import cv2
import numpy as np
from tqdm import tqdm

# --- CONFIG ---
SOURCE_FOLDER = 'train_images/'
OUTPUT_FILE = 'dataset_75k.npz'
IMAGE_SIZE = (256, 256)
MAX_IMAGES = 75000

def create_npz():
    images = []
    # Support multiple extensions
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    files = [f for f in os.listdir(SOURCE_FOLDER) if f.lower().endswith(exts)]
    
    # Shuffle and slice to 75k
    np.random.shuffle(files)
    files = files[:MAX_IMAGES]
    
    print(f"Found {len(files)} images. Starting preprocessing...")
    
    for filename in tqdm(files):
        try:
            path = os.path.join(SOURCE_FOLDER, filename)
            # Read as Grayscale (matching your 256, 256, 1 config)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            img = cv2.resize(img, IMAGE_SIZE)
            # Normalize to float32 [0, 1] to save computation during training
            img = img.astype(np.float32) / 255.0
            images.append(img)
        except Exception as e:
            continue
            
    # Convert to numpy array and add channel dimension (N, H, W, 1)
    images_array = np.array(images)
    images_array = np.expand_dims(images_array, axis=-1)
    
    print(f"Final array shape: {images_array.shape}")
    print(f"Saving to {OUTPUT_FILE} (this may take a few minutes)...")
    
    # We use save() instead of savez_compressed for faster IO with mmap later
    np.savez(OUTPUT_FILE, images=images_array)
    print("Pre-processing complete!")

if __name__ == "__main__":
    create_npz()