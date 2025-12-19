# Robust Deep Learning-Based Image Watermarking

This repository implements an advanced, invisible digital watermarking framework that combines Discrete Wavelet Transforms (DWT) with Convolutional Neural Networks (CNNs).  
The model embeds binary watermarks into the frequency domain of images to maximize **invisibility**, **robustness**, and **survivability under real-world distortions**.

---
## Project Overview & Impact

Digital Rights Management (DRM) in healthcare is critical for preventing unauthorized data leakage and ensuring patient data provenance. Traditional spatial-domain watermarking often degrades image fidelity or fails under compression.

**Pure Wavelet** addresses this by integrating **Deep Convolutional Neural Networks (CNN)** with **Discrete Wavelet Transforms (DWT)**. By embedding watermarks into the frequency domain (Haar Wavelet decomposition), the system achieves a balance between **imperceptibility** (invisible to the human eye) and **robustness** (survives clinical data compression).

Unlike standard models limited to natural images, this system was trained on a **mixed-domain dataset of 50,000 images**, achieving state-of-the-art generalization across:
1.  **Natural Photography** (COCO)
2.  **Radiography** (NIH Chest X-rays)
3.  **Tomography** (SIIM/COVID-CT Scans)
   
## Methodology & Architecture


The architecture implements an end-to-end encoder-decoder pipeline trained adversarially against a differentiable attack layer to simulate real-world degradation.

### 1. Frequency Domain Embedding (DWT)
The model leverages **Haar Wavelet Transform** to decompose input images into four sub-bands:
- **LL (Low-Low):** Structural information (Approximation)
- **LH, HL, HH:** High-frequency details

Watermarks are embedded selectively into the **LL band** via learned coefficients. This frequency-domain approach offers superior stability against signal processing attacks compared to pixel-domain manipulation.

### 2. Deep Latent Embedding via Transposed Convolutions
Instead of naïve interpolation, the watermark bits are upsampled using **Transposed Convolution Layers**. This allows the network to learn a "holographic" distribution of data, ensuring the watermark survives even if parts of the image are cropped or occluded.

### 3. Mathematical Formulation
The encoder computes a learned perturbation map ($\Delta$) optimized for the specific texture of the input image:

$$ \Delta = \text{CNN}(LL \oplus W_{pre}) $$
$$ LL_{watermarked} = LL + (\alpha \cdot \tanh(\Delta)) $$

Where:
- $\oplus$ denotes channel concatenation.
- $\alpha$ is a trainable strength factor balancing robustness and invisibility.

### 4. Adversarial Attack Simulation
A **Differentiable Noise Layer** is injected between the encoder and decoder during training. The model is forced to learn robust features by surviving on-the-fly distortions, including:
- **JPEG Compression** (Quality factors 50-90)
- **Gaussian Noise & Blur**
- **Dropout** (Pixel loss)
- **Geometric Rotation**

---
## 📊 Dataset & Scale

The model was trained on a high-variance dataset of **50,000 images** to ensure domain agility.

| Modality | Source | Count | Purpose |
| :--- | :--- | :--- | :--- |
| **Natural Objects** | COCO Dataset | 20,000 | General feature robustness |
| **Chest X-Ray** | NIH Database | 15,000 | High-contrast skeletal/tissue capability |
| **CT Scans** | SIIM / COVID-CT | 15,000 | Low-contrast soft tissue capability |
| **Total** | **Mixed** | **50,000** | **Domain Generalization** 

---
## 📈 Quantitative Performance

The model achieves high fidelity (PSNR) while keeping error rates (BER) low, validating its utility for medical diagnostics where image quality is paramount.

### 1. Imperceptibility (Visual Quality)
*Measured on held-out test set.*

| Domain | PSNR (Peak Signal-to-Noise Ratio) | SSIM (Structural Similarity) |
| :--- | :--- | :--- |
| Natural (COCO) | **37.8 dB** | 0.98 |
| Chest X-Ray | **37.2 dB** | 0.97 |
| CT Scans | **36.9 dB** | 0.96 |
| **Average** | **37.3 dB** | **0.97** |

> *Note: A PSNR > 35dB is standard for "invisible" watermarking.*

### 2. Robustness (Survival Under Attack)
*Average Bit Error Rate (BER) across all domains.*

| Attack Type | Strength | BER (%) | Recovery Rate |
| :--- | :--- | :--- | :--- |
| **No Attack** | Identity | **3.2%** | 96.8% |
| **JPEG Compression** | Q=50 (Aggressive) | **3.6%** | 96.4% |
| **Gaussian Noise** | Std=0.01 | **4.1%** | 95.9% |
| **Rotation** | ±10 Degrees | **4.0%** | 96.0% |

---

## 🖼️ Visual Results

### Watermark Invisibility

A comparison of the **original vs watermarked** image shows **no perceptible difference**, critical for medical diagnosis.

> *<img width="1500" height="500" alt="epoch_60_sample" src="https://github.com/user-attachments/assets/62dcd498-d43a-4416-8591-371540266d07" />*

### The Learned Residual ($\Delta$)
The residual added to the LL band (amplified 50×) reveals a learned, texture-adaptive pattern that carries the data.

> *<img width="256" height="256" alt="sample_0_diff_x50" src="https://github.com/user-attachments/assets/18e92c9e-55d8-4214-af37-1061b0c99c28" />*


---

## 🔗 References & Credits

### Research Paper

**Convolutional Neural Network-Based Image Watermarking using Discrete Wavelet Transform**  
Alireza Tavakoli, Zahra Honjani, Hedich Sajedi  
arXiv:2210.06179 (2022)

### Original Implementation

GitHub Repository: *Convolutional Neural Network Based Image Watermarking*
https://github.com/alirezatwk/Convolutional-Neural-Network-Based-Image-Watermarking-using-Discrete-Wavelet-Transform

---

## 📘 Acknowledgments

This project was developed for educational and research applications involving:

- Digital watermarking  
- Frequency-domain image processing  
- Deep learning for security and robustness  

Feel free to build on this work and contribute improvements.

---

