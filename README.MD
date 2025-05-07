# MNIST Digit Recognition Project

## Overview
This project implements and compares two different neural network approaches for handwritten digit recognition using the MNIST dataset:
1. A Multi-Layer Perceptron (MLP) model
2. A Convolutional Neural Network (CNN) model

The MNIST dataset contains 28x28 pixel grayscale images of handwritten digits (0-9), making it a classic benchmark for image classification algorithms.

## Features
- Data loading and preprocessing with proper train/validation/test splits
- Exploratory data analysis with visualizations
- Implementation of both MLP and CNN architectures
- Model training with early stopping
- Comprehensive performance evaluation
- Visualization of results and model comparisons
- Analysis of misclassified examples and model confidence

## Requirements
- Python 3.10
- TensorFlow/Keras
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

## Installation
```bash
# Clone the repository
git clone https://github.com/szala222/digit_recognition.git
cd digit_recognition

# Install required packages
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

## Usage
Simply run the main script:
```bash
python digit_recognition.py
```

This will:
1. Load and preprocess the MNIST dataset
2. Create and train both MLP and CNN models
3. Evaluate and compare model performance
4. Generate visualizations and save the models

## Project Structure
- `digit_recognition.py`: Main script containing all code
- `mnist_mlp_model.h5`: Saved MLP model
- `mnist_cnn_model.h5`: Saved CNN model
- Generated images:
  - `sample_images.png`: Example images from the dataset
  - `class_distribution.png`: Distribution of digits in the training set
  - `model_comparison.png`: Accuracy and loss comparison between models
  - `training_history.png`: Training and validation metrics over epochs
  - `confusion_matrix.png`: Confusion matrix for the CNN model
  - `correct_examples.png`: Examples of correctly classified digits
  - `misclassified_examples.png`: Examples of misclassified digits
  - `uncertain_examples.png`: Examples where the model had low confidence
  - `error_differences.png`: Analysis of possible overfitting

## Implementation Details

### Data Preprocessing
- Images normalized to [0,1] range
- Original training set split into training (80%) and validation (20%) sets
- Data reshaped appropriately for each model type
- Labels one-hot encoded

### MLP Model Architecture
- Input layer: 784 neurons (flattened 28x28 images)
- Hidden layer 1: 128 neurons with ReLU activation
- Dropout layer: 20% dropout rate
- Hidden layer 2: 64 neurons with ReLU activation
- Output layer: 10 neurons with softmax activation

### CNN Model Architecture
- Conv2D: 32 filters, 3x3 kernel, ReLU activation
- MaxPooling2D: 2x2 pool size
- Conv2D: 64 filters, 3x3 kernel, ReLU activation
- MaxPooling2D: 2x2 pool size
- Conv2D: 64 filters, 3x3 kernel, ReLU activation
- Flatten layer
- Dense: 64 neurons with ReLU activation
- Dropout layer: 20% dropout rate
- Output layer: 10 neurons with softmax activation

### Training Details
- Optimizer: Adam
- Loss function: Categorical crossentropy
- Batch size: 128
- Early stopping: Monitoring validation loss with patience of 3 epochs

## Results
The CNN model outperforms the MLP model, achieving higher accuracy and lower loss across all datasets (training, validation, and test). The visualizations in the output directory provide a detailed analysis of model performance.

### Performance Metrics
- CNN Test Accuracy: ~99%
- MLP Test Accuracy: ~97%

The confusion matrix and classification report show that the CNN model performs well across all digit classes, with most errors occurring between visually similar digits like 4/9 and 3/5.

