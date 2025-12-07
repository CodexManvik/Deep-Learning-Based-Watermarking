#!/bin/bash
echo "============================================================"
echo "Starting Training in WSL"
echo "============================================================"

# Check GPU
#echo ""
#echo "Checking GPU availability..."
#python3 -c "import tensorflow as tf; print(f'GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')"

# Start training
echo ""
echo "Starting training..."
python3 trainer.py
