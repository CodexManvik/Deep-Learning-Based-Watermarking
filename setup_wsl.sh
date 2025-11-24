#!/bin/bash
echo "============================================================"
echo "Setting up Watermarking Project in WSL"
echo "============================================================"

# Step 1: Create virtual environment
#echo ""
#echo "Step 1: Creating Python virtual environment..."
#python3 -m venv venv_watermark
#source venv_watermark/bin/activate

# Step 2: Upgrade pip
#echo ""
#echo "Step 2: Upgrading pip..."
#pip install --upgrade pip

# Step 3: Install TensorFlow GPU
#echo ""
#echo "Step 3: Installing TensorFlow with GPU support..."
#pip install "tensorflow[and-cuda]>=2.12.0,<2.16.0"

# Step 4: Install other requirements
echo ""
echo "Step 4: Installing other dependencies..."
pip install opencv-python numpy matplotlib scikit-learn scikit-image PyWavelets jupyter pandas tabulate

# Step 5: Install WaveTF
echo ""
echo "Step 5: Installing WaveTF from GitHub..."
pip install git+https://github.com/fversaci/WaveTF.git

# Step 6: Verify GPU
echo ""
echo "Step 6: Verifying GPU detection..."
python3 check_gpu.py

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "To activate the environment in future sessions:"
echo "  source venv_watermark/bin/activate"
echo ""
echo "To start training:"
echo "  python3 trainer.py"
