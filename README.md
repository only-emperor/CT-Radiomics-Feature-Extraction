# CT Radiomics Feature Extraction & Fractal Analysis 🧠📉

![Python](https://img.shields.io/badge/Language-Python-blue.svg) ![License](https://img.shields.io/badge/License-MIT-green.svg) ![Medical Imaging](https://img.shields.io/badge/Domain-Medical_Imaging-red.svg)

## 📖 Overview

This repository provides a set of Python scripts for advanced radiomics feature extraction from medical images (CT/MRI NIfTI format). It goes beyond standard features by integrating **Frequency Domain Analysis**, **Wavelet Transforms**, and **Fractal Dimension** calculations to quantify tumor heterogeneity.

## ✨ Key Features

1.  **Standard Radiomics**:
    - Uses `pyradiomics` to extract Shape, First-order statistics, GLCM, GLRLM, GLSZM, etc.
    - Supports batch processing with YAML configuration.

2.  **Advanced Frequency & Wavelet Analysis**:
    - Custom implementation of **FFT (Fast Fourier Transform)** features.
    - **Wavelet Transform** (Haar, Db2) decomposition for multi-scale texture analysis.
    - Extracts amplitude spectrum statistics (Energy, Entropy, Centroid).

3.  **Fractal Dimension (FD)**:
    - **3D Box-counting** method for volumetric complexity.
    - **2D Slice-wise** FD calculation (Axial, Sagittal, Coronal planes).
    - Aggregated statistics (Max, Min, Median FD) for comprehensive tumor characterization.

## 🛠️ Requirements

Install the necessary dependencies:

```bash
pip install numpy pandas scipy SimpleITK pyradiomics PyWavelets nibabel tqdm joblib openpyxl
