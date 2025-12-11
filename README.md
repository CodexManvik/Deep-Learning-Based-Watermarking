# Robust Deep Learning-Based Image Watermarking

This repository implements an advanced, invisible digital watermarking framework that combines Discrete Wavelet Transforms (DWT) with Convolutional Neural Networks (CNNs).  
The model embeds binary watermarks into the frequency domain of images to maximize **invisibility**, **robustness**, and **survivability under real-world distortions**.

---

## 🧠 Methodology & Architecture

### 1. Frequency Domain Embedding (DWT)

The model uses the **Haar Wavelet Transform** to split an image into sub-bands:

- **LL**: Low-frequency (structure)
- **LH, HL, HH**: High-frequency components

The watermark is embedded into the **LL band**, ensuring:

- Minimal visible distortion  
- Strong resistance to JPEG compression and noise  
- Better stability compared to pixel-domain watermarking  

---

### 2. Learned Upsampling via Transposed Convolutions

Instead of naïve interpolation, the model upsamples the watermark using **transposed convolution layers**, allowing:

- Learned filter weights  
- Distributed (holographic) embedding  
- Improved resilience against cropping and dropout attacks  

---

### 3. Embedding Equation

The encoder computes a learned perturbation:

```
Δ = CNN(LL ⊕ W_pre)
LL_watermarked = LL + (α · Δ)
```

Where:

- ⊕ = concatenation  
- Δ is squashed using Tanh into (-1, 1)  
- α controls embedding strength  

---

### 4. End-to-End Robustness Training

A **Differentiable Attack Simulator** is placed between encoder and decoder.

During training, images are randomly distorted with:

- Gaussian noise  
- Salt & Pepper noise  
- Dropout  
- JPEG approximation  

This forces the network to learn **robust watermark features** that survive real-world transformations.

---

## 📊 Performance

| Metric | Result | Description |
|-------|--------|-------------|
| Invisibility | > 30 dB | PSNR of watermarked image |
| JPEG Robustness | 0.00% BER | Bit Error Rate under JPEG compression |
| Noise Robustness | < 0.25% BER | Gaussian / Salt & Pepper |
| Precision | > 0.99 | Normalized Correlation (NC) |

---

## 🖼️ Visual Results

### Watermark Invisibility

A comparison of the **original vs watermarked** image shows **no perceptible difference**.

> *(Add your screenshot at `screenshots/comparison.png`)*

### Δ (Residual) Difference Map

The residual added to the LL band, amplified 50×, reveals a learned grid-like pattern designed for robustness.

> *(Add your screenshot at `screenshots/difference_map.png`)*

---

## 🔗 References & Credits

### Research Paper

**Convolutional Neural Network-Based Image Watermarking using Discrete Wavelet Transform**  
Alireza Tavakoli, Zahra Honjani, Hedich Sajedi  
arXiv:2210.06179 (2022)

### Original Implementation

GitHub Repository: *Convolutional Neural Network Based Image Watermarking*

---

## 📘 Acknowledgments

This project was developed for educational and research applications involving:

- Digital watermarking  
- Frequency-domain image processing  
- Deep learning for security and robustness  

Feel free to build on this work and contribute improvements.

---

