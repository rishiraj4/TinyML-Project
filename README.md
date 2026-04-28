# TinyML Anomaly Detection: Slider Dataset

This project implements an anomaly detection system using a subset of the **DCASE 2020 Slider dataset** (part of the MIMII dataset). The goal is to detect machine malfunctions by analyzing sound patterns using a quantized neural network capable of running on edge devices.

## 🚀 Project Overview
* **Model Architecture:** Autoencoder (Dense layers with a bottleneck).
* **Input Features:** Log-Mel Spectrograms (64x64).
* **Optimization Techniques:** Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT).

## 📊 Performance Comparison
One of the core objectives was to reduce the model size and latency for microcontroller deployment while maintaining high accuracy.

| Model Type | Size (KB) | Accuracy (AUC) | Latency (ms)* |
| :--- | :--- | :--- | :--- |
| **Baseline (Float32)** | 4277.10 | 0.8011 | 0.190 |
| **PTQ (Int8)** | 1194.32 | 0.7982 | 0.059 |
| **QAT (Int8)** | **1086.29** | **0.8169** | **0.065** |

*\*Latency measured on Colab CPU. Real-world edge hardware ratios will be similar.*

## 🛠️ Requirements
To run the notebook, you will need:
* Python 3.x
* TensorFlow / TensorFlow Lite
* `tensorflow-model-optimization`
* `librosa` (for audio processing)
* `scikit-learn`

## 📂 Repository Structure
* `Rishi_Raj_TinyML_Course_Project.ipynb`: The main Jupyter notebook containing data setup, feature engineering, training, and quantization.
* `README.md`: Project documentation.

## 📈 Results
The **QAT (Int8)** model outperformed the baseline in both size (75% reduction) and accuracy, proving that quantization-aware training is highly effective for TinyML sound classification tasks.
