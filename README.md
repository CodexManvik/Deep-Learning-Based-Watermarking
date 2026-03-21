# Deep Learning Based Invisible Watermarking

A robust, deep-learning-based invisible image watermarking system built on top of the Discrete Wavelet Transform (DWT). The system embeds a 256-bit binary watermark into a grayscale cover image in the wavelet domain and learns to recover that watermark after a range of signal-processing attacks. The architecture is designed for medical imaging (chest X-rays at 256x256 grayscale) but is general enough to work with any single-channel image dataset of the same resolution.

---

## Table of Contents

1. [Overview and Motivation](#overview-and-motivation)
2. [High-Level Architecture](#high-level-architecture)
3. [Full End-to-End Pipeline](#full-end-to-end-pipeline)
4. [Repository Structure](#repository-structure)
5. [Module Reference](#module-reference)
   - [configs.py](#configspy)
   - [models/](#models)
   - [data_loaders/](#data_loaders)
   - [attacks/](#attacks)
   - [trainer.py](#trainerpy)
   - [evaluate_model.py](#evaluate_modelpy)
   - [text_eval.py](#text_evalpy)
   - [test_robust.py](#test_robustpy)
   - [making_npz.py](#making_npzpy)
   - [Utility Scripts](#utility-scripts)
6. [Data Flow in Detail](#data-flow-in-detail)
   - [Image Loader](#image-loader)
   - [Watermark Generator](#watermark-generator)
   - [Attack ID Sampler](#attack-id-sampler)
   - [MergedDataLoader and Batch Format](#mergeddataloader-and-batch-format)
7. [Model Architecture in Detail](#model-architecture-in-detail)
   - [Inputs](#inputs)
   - [Stage 1: DWT and LL Band Extraction](#stage-1-dwt-and-ll-band-extraction)
   - [Stage 2: Watermark Preprocessing](#stage-2-watermark-preprocessing)
   - [Stage 3: Embedding CNN](#stage-3-embedding-cnn)
   - [Stage 4: IDWT Reconstruction](#stage-4-idwt-reconstruction)
   - [Stage 5: Attack Simulator](#stage-5-attack-simulator)
   - [Stage 6: Extraction CNN](#stage-6-extraction-cnn)
   - [Model Outputs](#model-outputs)
8. [Attack Simulator](#attack-simulator)
   - [Salt and Pepper Noise](#salt-and-pepper-noise)
   - [Gaussian Noise](#gaussian-noise)
   - [Differentiable JPEG Compression](#differentiable-jpeg-compression)
   - [Pixel Dropout](#pixel-dropout)
   - [Random Rotation](#random-rotation)
   - [Average Blur (Stupid Attack)](#average-blur-stupid-attack)
   - [Identity Pass-Through](#identity-pass-through)
   - [Attack Distribution and Scheduling](#attack-distribution-and-scheduling)
9. [Loss Function and Compilation](#loss-function-and-compilation)
10. [Training Strategy and Callbacks](#training-strategy-and-callbacks)
11. [Evaluation Modes](#evaluation-modes)
12. [Metrics](#metrics)
13. [Hardware Constraints and Engineering Decisions](#hardware-constraints-and-engineering-decisions)
14. [Saved Weights and Checkpointing](#saved-weights-and-checkpointing)
15. [Installation and Setup](#installation-and-setup)
16. [Running the Code](#running-the-code)

---

## Overview and Motivation

Invisible digital watermarking is the practice of embedding information into a digital medium in a way that is perceptually transparent to human observers but can be recovered algorithmically later. Applications include copyright protection, provenance tracking, and tamper detection in medical imaging pipelines.

Classical watermarking methods operate directly in the pixel domain or in frequency-domain representations such as the DCT. These methods are brittle against modern signal-processing attacks. This project replaces handcrafted embedding rules with trained convolutional networks that learn, entirely from data, how to hide a 256-bit binary payload inside the low-frequency (LL) subband of a Haar Discrete Wavelet Transform and then extract that payload reliably after attacks such as JPEG compression, Gaussian noise, pixel dropout, rotation, salt-and-pepper noise, and average blurring.

The approach follows the high-level encoder-decoder architecture described in the academic watermarking literature. The key design decisions specific to this implementation are:

- The watermark is embedded exclusively in the LL (approximation) subband of the DWT rather than across all four subbands. This concentrates the information in the most perceptually salient and most robust frequency region.
- The attack simulator is baked directly into the Keras computation graph as a non-trainable Lambda layer, meaning the extractor sees attacked images during every forward pass and gradients flow cleanly through the entire encode-attack-decode chain.
- The JPEG simulation is fully differentiable by using a straight-through estimator for the quantization rounding step, allowing gradients to propagate through the DCT quantization bottleneck during training.
- The system was developed and validated under tight GPU memory constraints (NVIDIA RTX 3050, nominally 4 GB VRAM, approximately 3.5 GB usable under WSL2), which influenced several engineering decisions described later.

---

## High-Level Architecture

The model follows a three-stage pipeline:

```
Cover Image (256x256x1)
        |
        v
  [ ENCODER ]
  DWT -> LL Band Extraction -> Watermark Preprocessing -> Embed CNN -> IDWT Reconstruction
        |
        v
  Watermarked Image (256x256x1)   <-- Output 1: used for PSNR / image fidelity loss
        |
        v
  [ ATTACK SIMULATOR ]
  One randomly selected attack applied to the entire batch
        |
        v
  Attacked Watermarked Image      <-- Output 3: stored for visualization
        |
        v
  [ DECODER ]
  DWT -> LL Band Extraction -> Extract CNN
        |
        v
  Extracted Watermark (256-bit)  <-- Output 2: used for watermark recovery loss
```

The model takes three inputs simultaneously: the cover image, the 256-bit binary watermark, and an integer attack ID that selects which attack to apply. It produces three outputs simultaneously, each associated with a different training loss.

---

## Full End-to-End Pipeline

The following sequence describes exactly what happens from raw image files on disk to a trained model checkpoint, in the order that tensors flow through the system.

**Step 1 - Image Collection:** Raw image files (PNG or JPEG) are read from `train_images/` or `test_images/` using the `ImageDataLoader`. Each file is decoded as a three-channel image, the first channel is extracted to produce a single-channel grayscale image, the image is resized to (256, 256), and pixel values are normalized from uint8 [0, 255] to float32 [0.0, 1.0].

**Step 2 - Watermark Generation:** For each image in the dataset, a completely independent, randomly generated 256-bit binary vector is produced by the `WatermarkDataLoader`. The watermark is never stored; it is generated on-the-fly from a TensorFlow Dataset pipeline. At evaluation time, the watermark can be replaced with a deterministic SHA-256 hash of an arbitrary text string.

**Step 3 - Attack ID Sampling:** For each image-watermark pair, the `AttackIdDataLoader` samples an integer ID from {0, 1, 2, 3, 4, 5, 6} according to a weighted distribution: 25% probability for no-attack (ID 0), and 12.5% probability for each of the six attack types. A single attack ID is drawn per batch (not per image), so every image in a batch sees the same attack type.

**Step 4 - Batching:** The three independent streams (images, watermarks, attack IDs) are zipped together by `MergedDataLoader` and then batched into tensors of shape `(batch_size, ...)`. The final dataset element format is `((image_batch, watermark_batch, attack_id_batch), (image_batch, watermark_batch, image_batch))` - the targets tuple repeats the inputs because the model is expected to reconstruct the original image and watermark.

**Step 5 - DWT Forward:** Inside the model, the cover image passes through a forward Haar Discrete Wavelet Transform via the WaveTF library. The transform decomposes the (256, 256, 1) image into a (128, 128, 4) representation consisting of four subbands: LL (approximation), LH (horizontal detail), HL (vertical detail), and HH (diagonal detail). The LL band, stored at channel index 0, is extracted and scaled by 0.5.

**Step 6 - Watermark Upsampling:** The raw 256-bit watermark vector is reshaped from (256,) to (16, 16, 1) and then passed through three transposed convolution blocks that progressively upsample it from 16x16 to 32x32, 64x64, and finally 128x128. Each transposed convolution is followed by batch normalization and ReLU. The final (128, 128, 1) tensor spatially matches the LL band of the DWT.

**Step 7 - Concatenation and Embedding:** The scaled LL band and the upsampled watermark are concatenated along the channel axis to produce a (128, 128, 2) tensor. This merged representation is passed through the embedding CNN, which consists of three Conv2D layers each with 64 filters and 3x3 kernels, each followed by batch normalization and ReLU. A final Conv2D with 1 filter and a tanh activation produces a delta (perturbation) tensor of shape (128, 128, 1). The delta is scaled by the `delta_scale` hyperparameter (0.55) and added to the LL band to produce the modified LL band.

**Step 8 - IDWT Reconstruction:** The modified LL band is scaled back up by a factor of 2.0 (to invert the earlier 0.5 scaling) and recombined with the unmodified LH, HL, and HH subbands. The resulting (128, 128, 4) tensor is passed through the inverse Haar Discrete Wavelet Transform to reconstruct a (256, 256, 1) image. The reconstructed image is clipped to [0.0, 1.0]. This is the clean watermarked image.

**Step 9 - Attack Application:** The clean watermarked image and the attack ID tensor are passed into the attack simulator layer. The batch-level integer ID is read from the first element of the attack ID tensor. A chain of `tf.cond` statements selects exactly one attack function to apply - only the selected branch actually executes, making this XLA-compatible and computationally efficient. The result is the attacked watermarked image, shaped identically to the input at (batch, 256, 256, 1).

**Step 10 - LL Band Re-extraction and Decoding:** The attacked watermarked image passes through a second DWT forward transform, and only the LL band is retained. This LL band is fed into the extraction CNN, which uses two Conv2D layers with strides of 2 and filter counts of 128 and 256 to spatially downsample the 128x128 feature map down to 32x32 and then 16x16. A final strided Conv2D with 1 filter followed by a sigmoid activation produces a (16, 16, 1) output that is flattened to the predicted 256-bit watermark vector.

**Step 11 - Loss and Backpropagation:** Three losses are computed simultaneously. MSE between the clean watermarked image and the original cover image, weighted by 80.0, optimizes for image fidelity. MAE between the extracted watermark and the ground-truth watermark, weighted by 2.0, optimizes for watermark recoverability. MSE between the attacked watermarked image and the original cover image, weighted by 0.0, is computed but contributes zero gradient - it exists only as a metric to monitor attack distortion severity. The Adam optimizer with learning rate 0.0001 and gradient clipping at norm 1.0 updates all trainable weights.

---

## Repository Structure

```
watermarking/
|
|-- configs.py                    # All global hyperparameters and paths
|
|-- trainer.py                    # Main training script
|-- evaluate_model.py             # Full evaluation suite with multiple modes
|-- test_robust.py                # Quick BER and NC evaluation
|-- text_eval.py                  # Text-based watermark embedding and evaluation
|-- making_npz.py                 # Offline preprocessing: images to NPZ
|
|-- models/
|   |-- base_model.py             # Abstract base class for models
|   |-- wavetf_model.py           # WaveTF-based watermarking model (primary)
|
|-- data_loaders/
|   |-- base_data_loader.py       # Abstract base class for loaders
|   |-- configs.py                # Data loader constants (PREFETCH, formats)
|   |-- merged_data_loader.py     # Combines image, watermark, and attack streams
|   |-- image_data_loaders/
|   |   |-- image_data_loader.py  # Reads images from disk via tf.data
|   |-- watermark_data_loaders/
|   |   |-- watermark_data_loader.py  # Generates random binary watermarks
|   |-- attack_id_data_loader/
|       |-- attack_id_data_loader.py  # Samples attack IDs with weighted distribution
|
|-- attacks/
|   |-- base_attack.py            # Abstract base class for attacks
|   |-- salt_pepper_attack.py     # Salt and pepper noise (p=0.1 total)
|   |-- gaussian_noise_attack.py  # Additive Gaussian noise (sigma in [0.05, 0.20])
|   |-- jpeg_attack.py            # Differentiable JPEG simulation via DCT
|   |-- drop_out_attack.py        # Random pixel zeroing (rate in [0.10, 0.50])
|   |-- rotation_attack.py        # Random rotation up to 30 degrees
|   |-- stupid_attack.py          # 3x3 average pool blur (was formerly identity)
|
|-- pure_wavelet/                 # Saved model weights (.weights.h5 files)
|-- train_images/                 # Training images directory
|-- test_images/                  # Test images directory
|-- evaluation_outputs/           # Per-attack sample images saved during evaluation
|-- visualizations/               # Model architecture diagrams and epoch diffs
|   |-- epoch_diffs/              # Per-epoch comparison images saved by ImageLogger
|-- logs/                         # TensorBoard log directories
|
|-- check_gpu.py                  # GPU detection and memory check utility
|-- check_requirements.py         # Dependency verification script
|-- setup_and_test.py             # Environment setup tester
|-- debug.py                      # Debugging utilities
|-- debug_attack.py               # Attack-specific debug script
|-- imaging.py                    # Image utility functions
|-- move_images.py                # Dataset organization helper
|-- setup_wsl.sh                  # WSL2 setup shell script
|-- train_wsl.sh                  # Quick training launch shell script
|-- requirements.txt              # Full pinned dependency list
|-- requirements_updated.txt      # Minimal modern dependency list
```

---

## Module Reference

### configs.py

This is the single source of truth for all hyperparameters. Every other module imports from it using `from configs import *`. Changing a value here affects the entire pipeline.

- `IMAGE_SIZE = (256, 256, 1)` - All images are 256 pixels square, single channel (grayscale). This is fixed throughout the architecture.
- `WATERMARK_SIZE = (256,)` - The watermark is a flat 256-bit binary vector. When reshaped to 2D it becomes (16, 16), a perfect square.
- `TRAIN_IMAGES = 75000` - Maximum number of images loaded from the training directory.
- `EPOCHS = 100` - Maximum training epochs before early stopping applies.
- `BATCH_SIZE = 24` - Chosen specifically for the RTX 3050 with approximately 3.5 GB of usable VRAM under WSL2. Larger batches cause out-of-memory errors.
- `LEARNING_RATE = 0.0001` - Conservative learning rate chosen to preserve previously trained PSNR while the model adapts to harder attack scenarios.
- `IMAGE_LOSS_WEIGHT = 80.0` - Heavily weights image fidelity in the total loss. The high weight ensures the watermark perturbation stays nearly invisible.
- `WATERMARK_LOSS_WEIGHT = 2.0` - Weight for watermark extraction loss. Lower than the image loss but still meaningful.
- `delta_scale = 0.55` - Controls how strongly the learned embedding delta is applied to the LL band. Higher values improve watermark robustness at the cost of more visible distortion.
- `ATTACKS_DISABLED = False` - Global flag. When True every batch uses attack ID 0 (identity), useful for debugging image quality in isolation.
- `ATTACK_MIN_ID = 0`, `ATTACK_MAX_ID = 6` - Range of valid attack IDs for training.
- `USE_PAPER_ATTACK_DISTRIBUTION = True` - Uses the weighted distribution (25% no-attack, 12.5% each for 6 attacks) rather than uniform random sampling.

### models/

**base_model.py** defines an abstract base class `BaseModel` that enforces a `get_model()` interface. Any new model variant must subclass this and implement `get_model()` to return a compiled Keras Model object.

**wavetf_model.py** contains the entire model construction logic in the `WaveTFModel` class. The class is a pure model factory; it does not inherit from any Keras class itself. All parameters (image size, watermark size, wavelet type, delta scale) are passed at construction time. The `get_model()` method builds and returns an uncompiled Keras functional API Model. Compilation happens in `trainer.py`.

Key architectural constants set in `__init__`:
- `embed_filters = [64, 64, 64]` - Three-layer embedding CNN with 64 filters each.
- `extract_filters = [128, 256]` - Two-layer extraction CNN with increasing filter counts and stride-2 convolutions for downsampling.
- `wm_pre_filters = [256, 128, 64]` - Watermark preprocessing transposed convolution filter counts (but the actual implementation uses a fixed three-block upsampling sequence with 512, 128, and 1 filters rather than this list, which appears to be an older design artifact).
- `wm_side = 16` - sqrt(256) = 16, the 2D side length of the reshaped watermark.

### data_loaders/

**base_data_loader.py** defines an abstract `BaseDataLoader` class with an abstract `get_data_loader()` method that must return a `tf.data.Dataset`.

**configs.py** (inside data_loaders/) defines `PREFETCH = tf.data.AUTOTUNE` and the list of accepted image formats (PNG, JPEG, JPG, BMP, WEBP).

**image_data_loader.py** discovers all image files in a given directory by globbing for supported extensions. It builds a `tf.data.Dataset` from the list of file paths and uses `tf.image.decode_image` with `channels=3`, then extracts only the first channel with slice indexing to produce a grayscale (H, W, 1) tensor. Images are resized to (256, 256) via bilinear interpolation and normalized to float32 [0, 1]. Parallel decoding is enabled with `tf.data.AUTOTUNE`.

**watermark_data_loader.py** creates an infinite stream of random binary vectors. It starts from `tf.data.Dataset.range(10**12)` (effectively infinite) and maps each index to a fresh `tf.random.uniform()` call that samples integers from [0, 2), giving 0 or 1 per element. The resulting integer tensor is cast to float32. Because this uses a pure TensorFlow op, it has no Python execution overhead and is safe to run in parallel with the image loader.

**attack_id_data_loader.py** samples attack identifiers. When `attacks_disabled=True` it returns a constant stream of zeros. When `min_value == max_value` it returns a constant stream of that fixed ID (used during evaluation to lock in a specific attack). Otherwise it uses either the weighted paper distribution or uniform sampling. The paper distribution uses `tf.random.categorical` applied to log-probabilities derived from the weight vector `[2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]`.

**merged_data_loader.py** is the orchestrator. It instantiates one of each sub-loader and uses `tf.data.Dataset.zip()` to align them sample-by-sample. The `format_batch` function packages the three streams into the Keras expected format: inputs as a tuple `(image, watermark, attack_id)` and targets as a tuple `(image, watermark, image)`. Note that the image appears twice in targets because the model has both an image reconstruction output and an attacked image output, both of which are compared against the original. The dataset is batched and prefetched but not shuffled in order to minimize memory overhead.

### attacks/

**base_attack.py** defines `BaseAttack(tf.keras.layers.Layer)`. Each attack is a Keras layer that implements the `call()` method and also exposes a module-level function `<attack>_function(x)` that instantiates the layer and calls it, for use inside the model's Lambda layer.

**salt_pepper_attack.py** generates a uniform random probability map over the entire image tensor. Pixels where the map falls below 0.05 are set to 1.0 (salt). Pixels where the map falls above 0.95 are zeroed (pepper). The net effect is approximately 5% salt pixels and 5% pepper pixels, for a total noise rate of p=0.1, matching the paper's evaluation parameter.

**gaussian_noise_attack.py** draws a single standard deviation scalar uniformly from [0.05, 0.20] (the upper bound was extended from the paper's 0.15 to provide a harder training signal). A Gaussian noise tensor of the same shape as the input is drawn from this stddev and added to the image. The result is clipped to [0, 1].

**jpeg_attack.py** is the most complex attack. It simulates JPEG compression entirely within TensorFlow:
1. The image is scaled from [0, 1] to [0, 255].
2. The spatial dimensions are padded to the nearest multiple of 8.
3. The image is reshaped into a (Batch, H/8, W/8, 1, 8, 8) tensor of 8x8 blocks.
4. A 2D DCT is applied via two sequential 1D `tf.signal.dct` calls on the last two axes with appropriate transpositions.
5. A quality-dependent luminance quantization matrix (derived from the standard JPEG table) is computed and used to quantize the DCT coefficients.
6. Rounding is performed using a straight-through estimator so gradients pass through unchanged in the backward pass.
7. The inverse DCT is applied.
8. The 8x8 blocks are merged back into the full image, padding is cropped, and values are normalized back to [0, 1].
Quality factor is sampled uniformly from [50, 90] during training.

**drop_out_attack.py** samples a drop rate from [0.10, 0.50] and generates a binary mask where each pixel is independently set to zero with that probability. The input is multiplied by the mask. This is a more aggressive version of the paper's fixed p=0.3 dropout; the wider range forces the model to generalize.

**rotation_attack.py** uses `tensorflow_addons.image.rotate` if available, applying a random rotation angle sampled uniformly from [-30 degrees, +30 degrees] independently per image in the batch (the rotation is per-image for spatial diversity). If tensorflow-addons is not installed, it falls back to 90-degree multiples using `tf.image.rot90`. Fill mode is reflective to avoid black border artifacts.

**stupid_attack.py** applies a 3x3 average pooling operation with stride 1 and SAME padding using `tf.nn.avg_pool2d`. This is equivalent to a 3x3 box blur that slightly smooths the image, removing some high-frequency watermark signal. This attack replaced an earlier version that was just the identity transform, which wasted 12.5% of the attack probability budget without providing any learning signal.

### trainer.py

This is the main entry point for training. It sets up the GPU memory growth configuration, constructs the dataset, builds the model, optionally loads the most recent checkpoint, attaches callbacks, compiles the model, and calls `model.fit()`.

The `ImageLogger` callback runs at the end of every epoch. It takes one batch of data captured before the `repeat()` call on the dataset (to avoid calling it on the infinite stream), runs a prediction, and saves a three-panel figure: the original cover image, the clean watermarked output, and the absolute difference amplified by 50x using the inferno colormap. These are saved to `visualizations/epoch_diffs/` as numbered PNGs, providing a visual record of how the watermark strength and image quality evolve over training.

The trainer looks for any `.h5` or `.keras` file in `pure_wavelet/` and loads the most recently modified one as a starting point. This allows training to be resumed after an interruption or to fine-tune from a previously trained checkpoint. If no checkpoint exists, training starts from random initialization.

A keyboard interrupt during training saves the current weights to `pure_wavelet/medical_interrupted.weights.h5`. An unhandled exception saves to `pure_wavelet/medical_crashed.weights.h5`. On normal completion, the final epoch weights are saved to a timestamped file.

### evaluate_model.py

The evaluation script supports three distinct modes, each targeting a different evaluation philosophy:

- **Default mode** (`--mode default`) uses the same randomized attack parameters that were used during training (Gaussian with sigma in [0.05, 0.15], JPEG with quality in [50, 90], dropout with rate in [0.10, 0.50]). This measures the model under its training distribution.
- **Paper mode** (`--mode paper`) uses fixed parameters matching the benchmarks in the referenced paper (sigma=0.15, JPEG quality=50, dropout p=0.3). This mode also prints a comparison table showing whether each metric passes or fails the paper's reported values.
- **Stratified mode** (`--mode stratified`) runs multiple fixed settings for each attack (e.g., three different JPEG quality levels) to profile how model performance degrades as attack severity increases.

The `--quick` flag limits evaluation to 200 images per attack for fast iteration. The `--no-samples` flag skips saving per-attack sample images.

For each attack, the script runs `MergedDataLoader` with the attack ID locked to the target attack, calls `model.predict()`, and computes image quality metrics against the clean watermarked output (not the attacked output) and watermark extraction metrics against the target watermark. Results are reported per-attack and then aggregated.

### text_eval.py

This script tests whether the model, which was trained on random binary watermarks, can generalize to watermarks derived from meaningful text.

Text is converted to a deterministic 256-bit binary string by computing the SHA-256 hash of the UTF-8 encoded text and representing each byte as 8 bits. This means different texts produce uncorrelated binary patterns, and the same text always produces the same pattern.

The script tests a predefined list of ten text strings against five attack scenarios and computes BER and Hamming distance for each combination. It also saves side-by-side visualizations of the original and extracted watermark rendered as 16x16 binary images, scaled up 10x for visibility.

An advanced mode attempts nearest-neighbor text reconstruction: for each extracted binary vector, it finds the text in the known candidate list whose SHA-256 hash has the smallest Hamming distance to the extracted bits.

### test_robust.py

A lighter alternative to `evaluate_model.py` that quickly computes BER and Normalized Correlation (NC) for five attack types on 2000 test images. It loads the most recently modified weight file automatically without any interactive menu. Useful for rapid checks during development.

### making_npz.py

An offline preprocessing utility. It reads up to 75,000 images from `train_images/`, resizes each to (256, 256), converts to grayscale, normalizes to float32 [0, 1], stacks them into a (N, 256, 256, 1) NumPy array, and saves the array to a `.npz` file using `np.savez()` (uncompressed, for faster memory-mapping during training). This script was written to support a batch-loading workflow that was later replaced by the streaming `tf.data` pipeline, and it is no longer used by the main trainer. It remains available for dataset inspection.

### Utility Scripts

- **check_gpu.py** - Detects available GPUs, prints device names, memory limits, and runs a small TensorFlow computation to verify GPU execution.
- **check_requirements.py** - Imports each major dependency and prints its version. Useful for diagnosing environment issues.
- **setup_and_test.py** - Runs a comprehensive environment check: GPU detection, package versions, a small forward pass through the model with synthetic data.
- **debug.py** - Contains debugging routines for inspecting model layer outputs, tensor shapes, and intermediate values during development.
- **debug_attack.py** - Applies each attack individually to a test image and saves the result, allowing visual verification that each attack is functioning correctly.
- **imaging.py** - Utility functions for image normalization, display, and comparison.
- **move_images.py** - Helper for reorganizing downloaded datasets into the expected directory structure.
- **setup_wsl.sh** - Shell script that installs system-level dependencies needed for WSL2 GPU access and sets TensorFlow environment variables.
- **train_wsl.sh** - One-line script that activates the virtual environment and launches `trainer.py`.

---

## Data Flow in Detail

### Image Loader

```
Disk (PNG/JPEG at any resolution)
  -> tf.io.read_file()
  -> tf.image.decode_image(channels=3)    # Force RGB decode
  -> img[:, :, 0:1]                       # Take first channel only -> (H, W, 1)
  -> tf.image.resize((256, 256))          # Bilinear resize
  -> tf.cast(img, float32) / 255.0        # Normalize to [0, 1]
  -> Dataset of (256, 256, 1) float32 tensors
```

### Watermark Generator

```
tf.data.Dataset.range(10^12)
  -> map: tf.random.uniform(shape=(256,), minval=0, maxval=2, dtype=int32)
  -> map: tf.cast(x, float32)
  -> Infinite dataset of (256,) float32 tensors with values in {0.0, 1.0}
```

### Attack ID Sampler

```
Paper distribution weights: [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  -> tf.random.categorical(log([weights]), num_samples=1)
  -> Shape (1,) int32 tensor with value in {0, 1, 2, 3, 4, 5, 6}
  -> Infinite dataset of (1,) int32 tensors
```

### MergedDataLoader and Batch Format

```
Dataset.zip((images, watermarks, attack_ids))
  -> map: format_batch(img, wm, atk)
      inputs  = (img, wm, atk)
      targets = (img, wm, img)   # img twice: once for embedded_image, once for attacked_image
      return (inputs, targets)
  -> .batch(24)
  -> .prefetch(AUTOTUNE)
  -> Final batch shape:
      inputs:   ((24,256,256,1), (24,256), (24,1))
      targets:  ((24,256,256,1), (24,256), (24,256,256,1))
```

---

## Model Architecture in Detail

### Inputs

The model accepts three named inputs created with `tf.keras.layers.Input`:

- `image_input`: shape (256, 256, 1), dtype float32 - the cover image.
- `watermark_input`: shape (256,), dtype float32 - the binary watermark to embed.
- `attack_id_input`: shape (1,), dtype int32 - the attack identifier for this batch.

### Stage 1: DWT and LL Band Extraction

```
image_input (256, 256, 1)
  -> WaveTFFactory().build("haar", dim=2)
  -> full_dwt: (128, 128, 4)   # 4 subbands: LL, LH, HL, HH
  -> target_band = full_dwt[..., 0:1] / 2.0   # LL band scaled by 0.5
     shape: (128, 128, 1)
```

The 0.5 scaling of the LL band before embedding and 2.0 rescaling before reconstruction is described in the paper (Section 4.2). It keeps the LL band values in a numerically stable range for the embedding CNN and is exactly inverted before the IDWT.

### Stage 2: Watermark Preprocessing

```
watermark_input (256,)
  -> Reshape((16, 16, 1))                          # (16, 16, 1)
  -> Conv2DTranspose(512, 3x3, stride=2, same)     # (32, 32, 512)
  -> BatchNormalization -> ReLU
  -> Conv2DTranspose(128, 3x3, stride=2, same)     # (64, 64, 128)
  -> BatchNormalization -> ReLU
  -> Conv2DTranspose(1, 3x3, stride=2, same)       # (128, 128, 1)
  -> Lambda: tf.image.resize(t, (128, 128))        # Ensures exact shape match
  -> wm_pre: (128, 128, 1)
```

The transposed convolution sequence learns how to spread the watermark bits spatially across the 128x128 LL band, distributing each bit's influence over a local neighborhood rather than at a single pixel. The Lambda resize at the end is a safety measure against subpixel rounding in the transposed convolution output dimensions.

### Stage 3: Embedding CNN

```
Concatenate([target_band, wm_pre], axis=-1)      # (128, 128, 2)
  -> Conv2D(64, 3x3, same) -> BN -> ReLU
  -> Conv2D(64, 3x3, same) -> BN -> ReLU
  -> Conv2D(64, 3x3, same) -> BN -> ReLU
  -> Conv2D(1, 3x3, same)                         # (128, 128, 1)
  -> tanh activation
  -> delta: (128, 128, 1) in range [-1, 1]

new_LL = target_band + delta_scale * delta        # (128, 128, 1)
```

The tanh activation bounds the delta to [-1, 1], and the `delta_scale` (0.55) further limits the maximum perturbation magnitude. The embedding CNN jointly sees the image content and the watermark, allowing it to learn where in the image structure it is safest to hide information without visible artifacts.

### Stage 4: IDWT Reconstruction

```
new_LL (128, 128, 1)
  -> Lambda: new_LL * 2.0                         # Rescale LL to original magnitude
  -> new_LL_scaled: (128, 128, 1)

full_dwt[..., 1:]                                 # Extract LH, HL, HH subbands
  -> rest: (128, 128, 3)

Concatenate([new_LL_scaled, rest], axis=-1)       # (128, 128, 4)
  -> WaveTFFactory().build("haar", dim=2, inverse=True)
  -> embedded_raw: (256, 256, 1)
  -> tf.clip_by_value(embedded_raw, 0.0, 1.0)
  -> embedded_image: (256, 256, 1)   <-- Output 1
```

Only the LL band has been modified; LH, HL, and HH subbands are passed through unchanged. This means the watermark perturbation is entirely confined to the low-frequency component of the image.

### Stage 5: Attack Simulator

```
embedded_image (256, 256, 1) + attack_id (1,)
  -> Lambda(batch_attack_logic): reads attack_id[0,0] as int batch_id
  -> tf.cond chain:
       batch_id == 1 -> salt_pepper_function(image_batch)
       batch_id == 2 -> gaussian_noise_function(image_batch)
       batch_id == 3 -> jpeg_function(image_batch)
       batch_id == 4 -> drop_out_function(image_batch)
       batch_id == 5 -> rotation_function(image_batch)
       batch_id == 6 -> stupid_function(image_batch)
       default      -> image_batch (identity)
  -> Reshape((256, 256, 1))   # Restores shape info lost by Lambda
  -> attacked_image: (256, 256, 1)   <-- Output 3
```

The tf.cond chain is critical for performance. Unlike `tf.switch_case` or `tf.map_fn` with variant tensors, nested `tf.cond` with concrete function bodies is compatible with TensorFlow's graph execution engine and could in principle be compiled under XLA (though XLA compilation is currently disabled due to other incompatibilities described in the trainer). Only the selected branch executes per step; the other five are skipped entirely.

### Stage 6: Extraction CNN

```
attacked_image (256, 256, 1)
  -> WaveTFFactory().build("haar", dim=2)         # Second DWT
  -> extracted_LL: (128, 128, 1)                  # Only LL used

extracted_LL
  -> Conv2D(128, 3x3, stride=2, same) -> BN -> ReLU  # (64, 64, 128)
  -> Conv2D(256, 3x3, stride=2, same) -> BN -> ReLU  # (32, 32, 256)
  -> Conv2D(1, 3x3, stride=2, same)                   # (16, 16, 1)
  -> sigmoid activation
  -> Reshape((256,))
  -> output_watermark: (256,)   <-- Output 2, values in [0, 1]
```

The extraction CNN uses strided convolutions rather than max-pooling. This provides a richer spatial aggregation signal because each output position integrates information from a larger receptive field in a learned rather than hardcoded way. The final sigmoid forces outputs to [0, 1] so they can be thresholded at 0.5 to recover binary bits.

### Model Outputs

| Output name | Shape | Description |
|---|---|---|
| `embedded_image` | (B, 256, 256, 1) | Clean watermarked image; used for image fidelity loss |
| `output_watermark` | (B, 256) | Extracted watermark from the attacked image; used for recovery loss |
| `attacked_image` | (B, 256, 256, 1) | Attacked version of the watermarked image; informational only |

---

## Attack Simulator

### Salt and Pepper Noise

Noise rate: p=0.1 total (0.05 salt, 0.05 pepper). A single random map is generated; pixels below 0.05 become 1.0 and pixels above 0.95 become 0.0. Uses tensor arithmetic to avoid Python loops and runs entirely on GPU.

### Gaussian Noise

Noise standard deviation is sampled uniformly from [0.05, 0.20] per batch. The training upper bound was raised from the paper's evaluation value of 0.15 to 0.20 to create a harder training signal and build in margin for evaluation at sigma=0.15. Uses `tf.random.normal` for GPU-native generation.

### Differentiable JPEG Compression

Quality factor is sampled uniformly from [50, 90] per batch. The full pipeline (DCT quantization, dequantization, IDCT) runs inside TensorFlow graph operations with no NumPy conversions. The straight-through estimator (`x + tf.stop_gradient(round(x) - x)`) makes the quantization step differentiable: in the forward pass it rounds normally, in the backward pass it acts like identity. This allows the model to backpropagate through JPEG compression and directly learn to resist quantization artifacts.

### Pixel Dropout

Drop rate is sampled uniformly from [0.10, 0.50] per batch. Each pixel independently has a chance of being zeroed. The mask is generated with `tf.random.uniform` and cast to float for multiplication with the input.

### Random Rotation

Angular range is [-30 degrees, +30 degrees] per image (extended from the paper's 15 degrees for harder training). Rotation is applied per-image in a batch using `tfa.image.rotate` with bilinear interpolation and reflective fill mode. The reflective fill mode avoids black corners that would create a strong learning signal to avoid rotating rather than becoming robust to it.

### Average Blur (Stupid Attack)

A 3x3 box blur implemented via `tf.nn.avg_pool2d` with stride 1 and SAME padding. This destroys high-frequency watermark components that may survive the embedding even though they are supposed to be confined to the LL band. Being fully differentiable (average pooling has well-defined gradients), it provides a useful robustness training signal at minimal computational cost.

### Identity Pass-Through

Attack ID 0 passes the watermarked image through without modification. Given a 25% probability weight in the training distribution, this ensures the model sees a substantial fraction of clean (non-attacked) examples and is not exclusively trained for robustness, which would hurt PSNR.

### Attack Distribution and Scheduling

The paper uses 1/3 no-attack and 1/6 per attack for four attacks. This implementation uses 2/8 no-attack and 1/8 per attack for six attacks, which is equivalent in spirit. The distribution is implemented as categorical sampling with the weight vector `[2, 1, 1, 1, 1, 1, 1]`.

The attack is selected once per batch, not once per image. This is standard in watermarking research because: (1) it simplifies the data pipeline, (2) over the full training run each attack sees proportional coverage, and (3) it avoids the variant-tensor complications of per-image attack selection inside `tf.map_fn`.

---

## Loss Function and Compilation

```python
model.compile(
    optimizer=Adam(lr=0.0001, clipnorm=1.0),
    loss={
        "embedded_image":  "mse",   # Mean squared error for image fidelity
        "output_watermark": "mae",  # Mean absolute error for bit recovery
        "attacked_image":  "mse"    # For monitoring only
    },
    loss_weights={
        "embedded_image":   80.0,
        "output_watermark":  2.0,
        "attacked_image":    0.0   # Zero weight: no gradient contribution
    },
    metrics={"output_watermark": "binary_accuracy"},
    jit_compile=False
)
```

The large asymmetry between the image loss weight (80.0) and the watermark loss weight (2.0) reflects the core design goal: the watermark must be invisible first (high PSNR) and recoverable second. In practice, 256 watermark bits are much easier to recover than 256x256 = 65,536 image pixels to reconstruct at high fidelity, so a higher weight on the image loss is appropriate.

MAE is used for the watermark loss instead of MSE because the output values are in [0, 1] and MAE provides better gradient behavior for bit-level predictions near 0.5. MSE would weight large errors more heavily, which could cause the model to sacrifice many bit predictions (moving them toward 0.5) to avoid rare large-error outliers.

Gradient clipping at norm 1.0 prevents sudden large weight updates during the early training phase when the model is adapting to attack diversity.

XLA JIT compilation is disabled because the attack simulator contains tensor operations that the XLA GPU JIT backend cannot compile (specifically, the variant tensor handling inside `tf.cond` chains when combined with stateful random ops). Enabling it causes a `FakeParam op unsupported` error during graph compilation.

---

## Training Strategy and Callbacks

**ModelCheckpoint (best_medical_robust.weights.h5):** Saves weights when the total training loss reaches a new minimum. This checkpoint captures the best overall trade-off between image fidelity and watermark robustness.

**ModelCheckpoint (best_medical_psnr.weights.h5):** Saves weights when the `embedded_image_loss` (the MSE image fidelity component alone) reaches a new minimum. This checkpoint captures the best visual quality, useful if you want to prioritize invisible embedding over robustness.

**EarlyStopping:** Monitors total training loss with patience=20. If the loss does not improve for 20 consecutive epochs, training halts and the best weights are restored. This prevents wasting compute on stagnant training and avoids overfitting.

**ReduceLROnPlateau:** Halves the learning rate when training loss does not improve for 3 consecutive epochs, with a minimum floor of 1e-7. This allows the optimizer to escape shallow plateaus without getting stuck in a reduced-gradient regime.

**TensorBoard:** Logs epoch-level metrics to `logs/medical_<timestamp>/`. Histograms and profiling are both disabled because they consume approximately 800 MB and 500 MB of VRAM respectively, which would cause out-of-memory errors on the 4 GB GPU.

**ImageLogger (custom):** At the end of each epoch, saves a three-panel comparison image. The third panel uses the inferno colormap with 50x amplification to make even very subtle watermark signals visible. This gives a continuous visual trace of training progress over all 100 epochs.

---

## Evaluation Modes

| Mode | Attack Parameters | Use Case |
|---|---|---|
| `default` | Randomized (training distribution) | Measure expected performance |
| `paper` | Fixed (sigma=0.15, q=50, p=0.3) | Compare against published benchmarks |
| `stratified` | Multiple fixed strengths per attack | Profile performance vs. attack severity |

The `paper` mode prints an explicit pass/fail table showing whether each metric meets or fails the values reported in the referenced academic paper (PSNR > 30 dB, BER under no attack < 0.001%, BER under salt-and-pepper < 10%, etc.).

---

## Metrics

**PSNR (Peak Signal-to-Noise Ratio):** Computed as `10 * log10(1.0 / MSE)` where MSE is the mean squared error between the original cover image and the clean watermarked output. Higher is better. The academic target is above 30 dB; the paper reports approximately 40.1 dB on average.

**SSIM (Structural Similarity Index Measure):** Computed using `tf.image.ssim` with max_val=1.0. Captures perceptual similarity by comparing luminance, contrast, and structure simultaneously. Range is [0, 1]; 1.0 means identical images. Values above 0.95 are generally considered imperceptible differences.

**BER (Bit Error Rate):** The fraction of extracted bits that differ from the originally embedded bits after rounding the sigmoid output to 0 or 1. Expressed as a percentage. Lower is better. A BER of 50% would mean the extracted watermark is random (no better than chance).

**NC (Normalized Correlation):** The cosine similarity between the original and extracted watermark vectors. Range is [-1, 1]; values near 1 indicate high similarity. Less sensitive than BER to individual bit flip distributions; useful when the overall signal direction is meaningful.

---

## Hardware Constraints and Engineering Decisions

The system was developed on a machine with an NVIDIA RTX 3050 (4 GB VRAM nominal, approximately 3.5 GB usable under WSL2 due to OS overhead). This imposed the following constraints and corresponding architectural decisions:

**Memory growth instead of pre-allocation:** `tf.config.experimental.set_memory_growth(True)` is enabled for all detected GPUs. Without this, TensorFlow would pre-allocate the entire VRAM at startup, leaving nothing for the WSL2 display adapter and causing immediate out-of-memory failures. With memory growth enabled, TensorFlow allocates VRAM incrementally as needed.

**No validation split:** With only 3.5 GB available, maintaining a separate validation dataset in memory alongside the training dataset would exceed the budget. Validation was disabled entirely; early stopping monitors training loss instead.

**No shuffle buffer:** Shuffling 2,048 images in memory would require approximately 500 MB. Shuffling was disabled to conserve VRAM. The training dataset is presented in a deterministic order each epoch. This is partially compensated by the random watermark and random attack selection, which are different for every image in every epoch.

**No disk cache:** Caching the preprocessed dataset to disk would speed up subsequent epochs but was disabled to avoid disk I/O bottlenecks and storage pressure.

**Mixed precision disabled:** TensorFlow's float16 mixed precision policy would halve VRAM requirements for most activations. However, WaveTF's Haar wavelet kernel is implemented using float64 intermediate computations and is incompatible with float16 inputs. Enabling mixed precision causes shape or dtype mismatch errors during the DWT operations.

**XLA disabled:** As described above, the attack simulator's `tf.cond` chain is incompatible with XLA GPU JIT compilation. A warning is printed at startup for transparency.

**TensorBoard profiling and histograms disabled:** Both consume substantial VRAM during the first few batches of training. Disabling them saves approximately 1.3 GB of VRAM.

**Batch size of 24:** Empirically determined to be the largest batch that fits within the VRAM budget with all constraints above in place. Batch size significantly affects training stability; smaller batches would require a lower learning rate.

---

## Saved Weights and Checkpointing

All weight files are saved in HDF5 format (`.weights.h5`) to the `pure_wavelet/` directory. The following files are maintained:

- `best_medical_robust.weights.h5` - Best overall loss checkpoint (updates whenever total loss improves).
- `best_medical_psnr.weights.h5` - Best image fidelity checkpoint (updates whenever embedded_image_loss improves).
- `medical_final_<timestamp>.weights.h5` - Saved at the end of normal training completion.
- `medical_interrupted.weights.h5` - Saved when training is stopped with Ctrl+C.
- `medical_crashed.weights.h5` - Saved when training exits with an unhandled exception.

When `trainer.py` is launched, it automatically selects the most recently modified weight file in `pure_wavelet/` and loads it as a starting point. This means every training run is implicitly a fine-tuning run if any checkpoint exists. To start from scratch, remove or rename the files in `pure_wavelet/`.

---

## Installation and Setup

**Core dependencies** (modern versions, see `requirements_updated.txt`):

```
tensorflow>=2.12.0,<2.16.0
opencv-python>=4.5.0
numpy>=1.21.0,<2.0.0
matplotlib>=3.4.0
scikit-image>=0.19.0
PyWavelets>=1.1.1
tabulate
```

**WaveTF must be installed from source:**

```bash
pip install git+https://github.com/fversaci/WaveTF.git
```

**Optional dependencies:**

- `tensorflow-addons` - Required for per-image continuous rotation. Without it, the rotation attack degrades to 90-degree multiples only.
- `visualkeras` - Required for the layered architecture visualization. Without it, the `visualkeras` architecture diagram is skipped (only the Keras `plot_model` graph is generated).

**Recommended setup:**

```bash
python -m venv venv_watermark
source venv_watermark/bin/activate
pip install -r requirements_updated.txt
pip install git+https://github.com/fversaci/WaveTF.git
```

Verify the environment:

```bash
python check_gpu.py
python check_requirements.py
```

---

## Running the Code

**Training:**

```bash
python trainer.py
```

Training prints configuration details at startup, then begins the epoch loop. Progress is visible in the terminal with per-batch loss values. TensorBoard can be launched in a separate terminal with `tensorboard --logdir logs/` to monitor metrics graphically.

**Evaluation (full suite):**

```bash
# Default mode (training distribution)
python evaluate_model.py

# Paper mode (fixed parameters, comparison table)
python evaluate_model.py --mode paper

# Stratified mode (multiple attack strengths)
python evaluate_model.py --mode stratified

# Quick mode for fast iteration (200 images per attack)
python evaluate_model.py --quick

# Provide weights path directly
python evaluate_model.py --weights pure_wavelet/best_medical_robust.weights.h5
```

**Quick robustness check:**

```bash
python test_robust.py
```

**Text watermark evaluation:**

```bash
python text_eval.py
```

The script presents an interactive model selection menu, then tests the ten predefined texts against all attacks, saves binary visualizations, and prints summary tables.
