import os

IMAGE_SIZE = (256, 256, 1)
WATERMARK_SIZE = (256,)

# --- TRAINING HYPERPARAMETERS ---
TRAIN_IMAGES = 50000
EPOCHS = 60
BATCH_SIZE = 16
LEARNING_RATE = 0.0005

# --- LOSS WEIGHTS ---
IMAGE_LOSS_WEIGHT = 100.0
WATERMARK_LOSS_WEIGHT = 1.0

# --- SIGNAL STRENGTH ---
delta_scale = 0.5

# --- ATTACKS ---
ATTACKS_DISABLED = False

# Attack IDs: 0=None, 1=Salt, 2=Gauss, 3=JPEG, 4=Dropout, 5=Rotation, 6=Stupid
# FIXED: Was 7 (undefined), now 6 (max valid ID)
ATTACK_MIN_ID = 0
ATTACK_MAX_ID = 6  # Changed from 7

# Use paper-compliant weighted distribution (1/3 no-attack, 1/6 each for 4 attacks)
USE_PAPER_ATTACK_DISTRIBUTION = True

# --- PATHS ---
MODEL_OUTPUT_PATH = 'pure_wavelet/'
TRAIN_IMAGES_PATH = 'train_images/'
TEST_IMAGES_PATH = 'test_images/'
MAX_TEST_IMAGES = 2500
