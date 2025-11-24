#!/usr/bin/env python3
"""Check if all required packages are installed."""

import sys

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} - NOT INSTALLED")
        return False

def main():
    print("=" * 60)
    print("Checking Required Packages")
    print("=" * 60)
    
    packages = [
        ("tensorflow", "tensorflow"),
        ("numpy", "numpy"),
        ("opencv-python", "cv2"),
        ("matplotlib", "matplotlib"),
        ("scikit-learn", "sklearn"),
        ("scikit-image", "skimage"),
        ("PyWavelets", "pywt"),
        ("pandas", "pandas"),
        ("tabulate", "tabulate"),
        ("wavetf", "wavetf"),
    ]
    
    all_installed = True
    missing = []
    
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_installed = False
            missing.append(package_name)
    
    print("=" * 60)
    
    if all_installed:
        print("✓ All packages are installed!")
        
        # Check GPU
        print("\n" + "=" * 60)
        print("GPU Check")
        print("=" * 60)
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ {len(gpus)} GPU(s) detected")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        else:
            print("⚠️  No GPU detected - will use CPU")
        
        return 0
    else:
        print(f"✗ Missing {len(missing)} package(s)")
        print("\nTo install missing packages:")
        print("  pip install " + " ".join(missing))
        if "wavetf" in missing:
            print("  pip install git+https://github.com/fversaci/WaveTF.git")
        return 1

if __name__ == "__main__":
    sys.exit(main())
