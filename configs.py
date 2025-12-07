import os

IMAGE_SIZE = (256, 256, 1)
WATERMARK_SIZE = (256,)

EPOCHS = 60             # Paper used 60 epochs [cite: 1162]
BATCH_SIZE = 10         # Paper used 10 [cite: 1162]
LEARNING_RATE = 0.001   # Paper used 0.001 

# --- PAPER HYPERPARAMETERS  ---
IMAGE_LOSS_WEIGHT = 10.0    # Lambda 1
WATERMARK_LOSS_WEIGHT = 1.0 # Lambda 2
delta_scale = 1.0           # Standard strength

# --- ENABLE ATTACKS (Single Phase Training) ---
ATTACKS_DISABLED = False

TRAIN_IMAGES = 20000 

MODEL_OUTPUT_PATH = 'pure_wavelet/'
TRAIN_IMAGES_PATH = 'train_images/'
TEST_IMAGES_PATH = 'test_images/'

# Attack IDs: 0=Identity, 1=Salt&Pepper, 2=Gaussian, 3=JPEG, 4=Dropout
ATTACK_MIN_ID = 0
ATTACK_MAX_ID = 5