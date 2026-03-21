"""
Setup and Test Script for Watermarking Project
This script helps verify the environment and run a basic test
"""

import sys
import subprocess

def check_imports():
    """Check if required packages can be imported"""
    required_packages = [
        'tensorflow',
        'numpy',
        'cv2',  # opencv-python
        'wavetf',
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            missing.append(package)
    
    return missing

def test_basic_pipeline():
    """Test basic watermarking pipeline"""
    try:
        import tensorflow as tf
        import numpy as np
        from models.wavetf_model import WaveTFModel
        
        print("\n=== Testing Basic Pipeline ===")
        
        # Create model
        IMAGE_SIZE = (256, 256, 1)
        WATERMARK_SIZE = (16 * 16,)
        
        print("Creating WaveTF model...")
        model = WaveTFModel(image_size=IMAGE_SIZE, watermark_size=WATERMARK_SIZE).get_model()
        print("✓ Model created successfully")
        
        # Test with dummy data
        print("Testing with dummy data...")
        dummy_image = np.random.rand(1, 256, 256, 1).astype(np.float32)
        dummy_watermark = np.random.rand(1, 256).astype(np.float32)
        dummy_attack_id = np.array([[0]], dtype=np.int32)
        
        output = model.predict([dummy_image, dummy_watermark, dummy_attack_id])
        print(f"✓ Model prediction successful")
        print(f"  - Embedded image shape: {output[0].shape}")
        print(f"  - Extracted watermark shape: {output[1].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Watermarking Project - Setup and Test")
    print("=" * 60)
    
    print("\n1. Checking Python version...")
    print(f"Python {sys.version}")
    
    print("\n2. Checking required packages...")
    missing = check_imports()
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("\nTo install missing packages, run:")
        print("  venv_watermark_win\\Scripts\\pip.exe install tensorflow opencv-python numpy")
        print("  venv_watermark_win\\Scripts\\pip.exe install tensorflow-wavelets")
        sys.exit(1)
    
    print("\n3. Testing basic pipeline...")
    if test_basic_pipeline():
        print("\n" + "=" * 60)
        print("✓ All tests passed! The pipeline is ready.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ Pipeline test failed. Please check the errors above.")
        print("=" * 60)
        sys.exit(1)
