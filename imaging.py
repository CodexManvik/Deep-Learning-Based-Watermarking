import numpy as np
from PIL import Image
import io
from pathlib import Path

# Paths (Windows-friendly)
npz_path = Path(r"./marchive/mfull_archive.npz")  # Update full path
output_dir = Path(r"./siim_extracted_images")
output_dir.mkdir(parents=True, exist_ok=True)

# Load with allow_pickle for object arrays
data = np.load(npz_path, allow_pickle=True)

print("Keys:", data.files)
for key in data.files:
    arr = data[key]
    print(f"{key}: shape {arr.shape if hasattr(arr, 'shape') else 'N/A'}, dtype {arr.dtype}")
    if len(arr) > 0:
        print(f"  Sample type: {type(arr[0])}")

# Extract from 'image' key
if 'image' in data:
    images = data['image']
    total = len(images)
    print(f"Extracting {total} images...")
    
    for i, img_obj in enumerate(images):
        try:
            if isinstance(img_obj, Image.Image):
                img = img_obj
            elif isinstance(img_obj, np.ndarray):
                img = Image.fromarray(img_obj.astype(np.uint8))
            elif isinstance(img_obj, bytes):
                img = Image.open(io.BytesIO(img_obj))
            else:
                # Fallback: treat as pickled image data
                img = Image.open(io.BytesIO(img_obj))
            
            # Save PNG
            out_path = output_dir / f"siim_{i:05d}.png"
            img.save(out_path)
            
            if (i + 1) % 500 == 0:
                print(f"Saved {i+1}/{total}")
        except Exception as e:
            print(f"Error on {i}: {e}")
    
    print(f"Done! Check {output_dir} ({len([p for p in output_dir.glob('*.png')])} PNGs extracted)")
else:
    print("No 'image' key found.")
