## Overview
This project implements a Random forest and Support Vector Machine (SVM) to classify Indian Classical Mudras (hand gestures) by combining and standardizing multiple datasets into a unified machine learning pipeline. The core idea of this work is to demonstrate an end-to-end Computer Vision classification workflow using classical machine learning techniques alongside robust feature extraction methods.

## Workflow & Common Idea

The complete workflow consists of three main phases:

### 1. Data Acquisition
Two varied datasets from Kaggle are retrieved programmatically using the `kagglehub` library (`download_kaggle_data.py`). This includes a dataset for generic Indian Classical Mudras and another specifically for Kathak Mudras.

### 2. Data Infusion and Preprocessing
Because the downloaded datasets have different directory layouts and labeling formats (e.g., standard categorical folders vs. YOLO-formatted `.txt` files), `infuse_data.py` acts as an adapter. It:
- Reads the disparate folder structures.
- Extracts and aligns the class names (resolving custom dataset labels).
- Normalizes and copies all the images into a single, unified `data/` directory where each subdirectory represents a unique mudra class.

### 3. Feature Extraction and SVM Training
Once the data is standardized, `train_svm.py` handles the machine learning phase:
- **Feature Engineering:** Since raw pixel data can be noisy and high-dimensional, the images are converted to grayscale, resized to `128x128`, and processed using **Histogram of Oriented Gradients (HOG)**. HOG excels at capturing the distinct shape and structural contours of the hand gestures, which is essential for accurate mudra classification.
- **Model Training:** The extracted HOG features are used to train an SVM model with an RBF (Radial Basis Function) kernel.
- **Evaluation:** The model is evaluated on a dedicated test split to measure its accuracy, producing a detailed classification matrix saved to `classification_results.txt`.

## Getting Started

To run the pipeline from end-to-end, execute the scripts in the following order:

1. **Download the datasets:**
   ```bash
   python download_kaggle_data.py
   ```
2. **Infuse and structure the data:**
   ```bash
   python infuse_data.py
   ```
3. **Train and evaluate the SVM model:**
   ```bash
   python train_svm.py
   ```
