import os
import shutil
import random
from pathlib import Path

# Paths - adjust these to your setup
coco_images_dir = Path("./train_images_coco")  # e.g., /mnt/c/Project/watermarking/coco/train2017
train_images_fixed_dir = Path("./train_images")

# Create output directory
train_images_fixed_dir.mkdir(parents=True, exist_ok=True)

# Get all image files from COCO directory
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
all_images = []
for ext in image_extensions:
    all_images.extend(coco_images_dir.glob(f'*{ext}'))
    all_images.extend(coco_images_dir.glob(f'*{ext.upper()}'))

print(f"Found {len(all_images)} images in COCO directory")

# Randomly select 20,000 images (or all if less available)
num_to_select = min(30000, len(all_images))
selected_images = random.sample(all_images, num_to_select)

# Copy to train_images_fixed
copied_count = 0
for img_path in selected_images:
    dest_path = train_images_fixed_dir / img_path.name
    # Handle filename conflicts by adding counter
    counter = 1
    while dest_path.exists():
        name, ext = dest_path.stem, dest_path.suffix
        dest_path = train_images_fixed_dir / f"{name}_{counter}{ext}"
        counter += 1
    
    shutil.copy2(img_path, dest_path)
    copied_count += 1
    if copied_count % 1000 == 0:
        print(f"Copied {copied_count}/{num_to_select} images")

print(f"Successfully copied {copied_count} random images to {train_images_fixed_dir}")
