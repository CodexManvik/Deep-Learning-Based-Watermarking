import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("\n" + "="*60)
print("GPU Detection:")
print("="*60)

# List physical devices
gpus = tf.config.list_physical_devices('GPU')
print(f"\nNumber of GPUs detected: {len(gpus)}")

if gpus:
    for i, gpu in enumerate(gpus):
        print(f"\nGPU {i}: {gpu}")
        print(f"  Name: {gpu.name}")
        print(f"  Type: {gpu.device_type}")
else:
    print("\n⚠️  No GPU detected - TensorFlow will use CPU")
    print("\nPossible reasons:")
    print("  1. No NVIDIA GPU installed")
    print("  2. CUDA/cuDNN not installed")
    print("  3. TensorFlow CPU version installed instead of GPU version")

# Check if CUDA is available
print("\n" + "="*60)
print("CUDA Availability:")
print("="*60)
print(f"CUDA available: {tf.test.is_built_with_cuda()}")
print(f"GPU available: {tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)}")

# Try to create a tensor on GPU
print("\n" + "="*60)
print("GPU Test:")
print("="*60)
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print("✓ Successfully created and computed tensor on GPU")
        print(f"  Result shape: {c.shape}")
except RuntimeError as e:
    print(f"✗ Cannot use GPU: {e}")
    print("  TensorFlow will use CPU instead")
