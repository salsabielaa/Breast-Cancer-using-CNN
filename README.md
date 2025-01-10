# Breast Cancer Detection Using SimpleCNN

This project is a deep learning pipeline to classify breast cancer images using a custom Convolutional Neural Network (SimpleCNN). The model is trained to distinguish between binary classes (e.g., malignant and benign) using a dataset of breast cancer images.

---

## Overview

The goal of this project is to detect breast cancer using X-ray or other imaging data. The neural network processes images through several convolutional layers, followed by pooling, batch normalization, dropout, and fully connected layers. It outputs the probability of the input belonging to a specific class.

Key features:
- Custom Convolutional Neural Network (SimpleCNN).
- Binary classification using `BCEWithLogitsLoss`.
- Includes data preprocessing with resizing, normalization, and augmentation.
- Implements a learning rate scheduler (StepLR).

---

## Dataset

### Dataset Structure:
The dataset should be organized as follows:

```
breastcancer/
├── class_0/  # Images of the first class
├── class_1/  # Images of the second class
```

### Example:
Images should be in `.png` format, and the dataset can include subfolders for each class. 

### Statistics:
- **Training set**: 80% of the dataset
- **Test set**: 20% of the dataset

---

## Model Architecture

### SimpleCNN:
- **Convolutional Layers**:
  - Two convolutional layers (`nn.Conv2d`) with ReLU activation.
  - Batch normalization after each convolutional layer.
  - MaxPooling to reduce spatial dimensions.
- **Fully Connected Layers**:
  - Flatten the output of the final convolutional layer.
  - Three fully connected layers (`nn.Linear`) with Dropout for regularization.
- **Output Layer**:
  - Single neuron output with `Sigmoid` activation for binary classification.

---

## Preprocessing

### Steps:
1. **Resizing**: Images are resized to 128x128 pixels.
2. **Normalization**: Pixel values are normalized to the range [-1, 1].
3. **Transformation**:
   - Composed using `torchvision.transforms.Compose`.

### Transformation Pipeline:
```python
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

---

## Training

- **Loss Function**: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`).
- **Optimizer**: AdamW with weight decay.
- **Learning Rate Scheduler**: StepLR to reduce the learning rate after every 10 epochs.

### Training Process:
1. Forward pass: Compute the outputs using the `SimpleCNN` model.
2. Backward pass: Compute gradients and update weights using AdamW.
3. Evaluate the model on the test set after each epoch.

---

## Evaluation

Metrics used:
- **Training Loss** and **Test Loss**: Indicates how well the model fits the training and test data.
- **Accuracy**: Percentage of correctly predicted samples.
