# IMAGE-DATA-CLASSIFICATION-NEURAL-NETWORK

# 🔢 MNIST Handwritten Digit Classification

> A deep Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify handwritten digits from the MNIST dataset with state-of-the-art accuracy.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Model Architecture](#model-architecture)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Real-World Prediction](#real-world-prediction)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)

---

## Overview

This project implements an end-to-end image classification pipeline for the MNIST benchmark dataset. The goal is to accurately identify handwritten digits (0–9) from 28×28 grayscale images.

The model employs a deep CNN architecture with:
- Dual convolutional blocks for hierarchical feature extraction
- Batch Normalization for training stability
- Dropout regularization to prevent overfitting
- Real-time data augmentation to improve generalization
- Exponential Decay learning rate scheduling with Adam optimizer

---

## Dataset

The **MNIST** (Modified National Institute of Standards and Technology) dataset is one of the most well-known benchmarks in machine learning.

| Attribute | Details |
|---|---|
| Training Samples | 60,000 images |
| Test Samples | 10,000 images |
| Image Dimensions | 28 × 28 pixels |
| Color Mode | Grayscale (single channel) |
| Classes | 10 (digits 0 through 9) |
| Pixel Value Range | 0 – 255 (normalized to 0–1) |
| Source | `tf.keras.datasets.mnist` |

The dataset is well-balanced — each digit class contains approximately 5,900–6,700 training samples — so no class weighting or oversampling was required.

---

## Project Pipeline

```
Raw MNIST Data
      │
      ▼
Data Preprocessing
  ├── Normalize pixel values → [0, 1]
  ├── Reshape: (N, 28, 28) → (N, 28, 28, 1)
  └── One-hot encode labels → (N, 10)
      │
      ▼
Data Augmentation (train only)
  ├── Random Rotation  (±54°)
  ├── Random Zoom      (±15%)
  ├── Random Translation (±10% H & V)
  └── Random Contrast  (±10%)
      │
      ▼
CNN Model Training
  ├── Adam + Exponential LR Decay
  ├── Early Stopping (patience=5)
  └── Model Checkpoint (best weights saved)
      │
      ▼
Evaluation on 10,000 Test Images
      │
      ▼
Real-World Image Prediction
```

---

## Model Architecture

The model (`Mnist_CNN`) is built using the Keras Sequential API and consists of two convolutional blocks followed by a fully connected classification head.

```
Input (28×28×1)
       │
┌──────▼──────────────────────┐
│  Data Augmentation Layer    │
└──────┬──────────────────────┘
       │
┌──────▼──────────────────────┐  ← Conv Block 1
│  Conv2D  (32 filters, 3×3)  │
│  BatchNormalization          │
│  Conv2D  (32 filters, 3×3)  │
│  BatchNormalization          │
│  MaxPooling2D (2×2) → 14×14 │
│  Dropout (25%)               │
└──────┬──────────────────────┘
       │
┌──────▼──────────────────────┐  ← Conv Block 2
│  Conv2D  (64 filters, 3×3)  │
│  BatchNormalization          │
│  Conv2D  (64 filters, 3×3)  │
│  BatchNormalization          │
│  MaxPooling2D (2×2) → 7×7   │
│  Dropout (25%)               │
└──────┬──────────────────────┘
       │
┌──────▼──────────────────────┐  ← Classification Head
│  Flatten → 3,136 units       │
│  Dense (256, ReLU)           │
│  BatchNormalization          │
│  Dropout (50%)               │
│  Dense (128, ReLU)           │
│  BatchNormalization          │
│  Dropout (50%)               │
└──────┬──────────────────────┘
       │
┌──────▼──────────────────────┐
│  Dense (10, Softmax)         │  ← Output: probability per digit
└─────────────────────────────┘
```

**Key Design Decisions:**
- **ReLU** activations throughout hidden layers to mitigate vanishing gradients
- **Same padding** preserves spatial dimensions within each convolutional block
- **Progressive filter doubling** (32 → 64) captures increasingly abstract features
- **Dual dropout rates** — lighter (25%) in conv blocks, heavier (50%) near the output head

---

## Training Configuration

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| LR Schedule | Exponential Decay (decay rate=0.9, every 1,000 steps) |
| Loss Function | Categorical Cross-Entropy |
| Batch Size | 128 |
| Epochs | 30 (with Early Stopping) |
| Validation Split | 10% (6,000 images) |
| Early Stopping Patience | 5 epochs (monitors `val_loss`) |
| Best Model Saved To | `best_mnist_cnn.keras` |

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | ~99%+ |
| Test Loss | < 0.05 |

The model leverages six complementary regularization strategies to ensure strong generalization:

| Technique | Where Applied | Effect |
|---|---|---|
| Data Augmentation | Input pipeline (train only) | Increases effective dataset diversity |
| Batch Normalization | After every Conv2D and Dense layer | Reduces internal covariate shift |
| Dropout (25%) | After each conv block | Prevents neuron co-adaptation |
| Dropout (50%) | After each dense layer | Stronger regularization near output |
| Early Stopping | Monitors `val_loss` (patience=5) | Halts training before overfitting |
| Exponential LR Decay | Throughout training | Fine-tunes weights in later epochs |

---

## Real-World Prediction

The notebook includes a custom `predict_image()` function that handles inference on real-world handwritten digit photos.

> ⚠️ **Important:** MNIST images have **white digits on black backgrounds**. Photos from paper have the opposite — the function automatically inverts the image before prediction.

**Prediction steps:**
1. Load image and convert to grayscale
2. Invert pixel values (`ImageOps.invert`)
3. Resize to 28×28 pixels
4. Normalize to [0, 1]
5. Reshape to `(1, 28, 28, 1)`
6. Run `model.predict()` → return predicted class + confidence score

```python
predicted_class, confidence = predict_image('your_digit.jpg')
print(f"Predicted: {predicted_class} | Confidence: {confidence:.2%}")
```

---

## Tech Stack

| Library | Usage |
|---|---|
| TensorFlow / Keras | Model definition, training, evaluation |
| NumPy | Array manipulation, prediction post-processing |
| Pandas | Data handling utilities |
| Matplotlib | EDA — visualizing sample images with labels |
| Pillow (PIL) | Image loading, grayscale conversion, inversion, resizing |
| Scikit-learn | Confusion matrix and classification report |
| Google Colab | GPU-accelerated training environment |

---

## Future Improvements

- [ ] **Residual Connections** — ResNet-style skip connections for deeper variants
- [ ] **Global Average Pooling** — Replace Flatten + Dense layers to reduce parameters
- [ ] **Learning Rate Warm-Up** — Improve early training stability before decay kicks in
- [ ] **Test-Time Augmentation (TTA)** — Average predictions over augmented versions for +0.1–0.2% accuracy
- [ ] **Confusion Matrix Visualization** — Identify commonly confused digit pairs (e.g., 4 vs. 9)
- [ ] **Grad-CAM Explainability** — Visualize which image regions drive each prediction
- [ ] **API Deployment** — Wrap the prediction function in a Flask/FastAPI REST endpoint

---

*Built with TensorFlow & Keras · MNIST Benchmark · Deep Learning*
