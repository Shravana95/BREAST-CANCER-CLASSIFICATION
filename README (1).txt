# Project 2 - Breast Cancer Classification Using CNN

## Overview

This project uses a Convolutional Neural Network (CNN) to classify breast histopathology image patches into two classes:

- class0 - IDC-negative
- class1 - IDC-positive

The project was implemented using Python, TensorFlow, Keras, NumPy, Matplotlib and scikit-learn.

## Dataset

The project uses the Breast Histopathology Images dataset from Kaggle.

The original dataset contains 277,524 image patches of size 50 x 50 pixels. Due to storage and computational limitations, a smaller local subset was used for this project.

Final local dataset:

- Class 0: 576 images
- Class 1: 508 images
- Total: 1,084 images

The images are stored as:

dataset/
    class0/
    class1/

## Preprocessing

The images are loaded using TensorFlow's image_dataset_from_directory function.

The dataset is divided using an 80:20 split:

- Training images: 868
- Validation images: 216

Images are resized to 50 x 50 and pixel values are rescaled from 0-255 to 0-1 using 1/255.

## CNN Model

The model contains:

1. Rescaling layer
2. Conv2D - 32 filters
3. MaxPooling2D
4. Conv2D - 64 filters
5. MaxPooling2D
6. Conv2D - 128 filters
7. MaxPooling2D
8. Flatten
9. Dense layer - 128 neurons
10. Dropout - 0.5
11. Sigmoid output layer

Optimizer: Adam

Loss function: Binary Cross-Entropy

Batch size: 16

Training epochs: 10

## Results

Final validation accuracy: 87.5%

Classification report:

- Macro precision: 0.89
- Macro recall: 0.88
- Macro F1-score: 0.87
- Weighted precision: 0.89
- Weighted recall: 0.88
- Weighted F1-score: 0.87

Confusion matrix:

[[104, 3],
 [24, 85]]

The validation accuracy reached approximately 92% around epoch 7 and then decreased slightly by epoch 10.

## Files

Project_2_Breast_Cancer_CNN/
    dataset/
        class0/
        class1/
    output/
        cnn_accuracy.png
        cnn_loss.png
        confusion_matrix.png
        classification_report.txt
        breast_cancer_cnn.keras
    breast_cancer_cnn.py
    download_dataset.py
    algorithm.txt
    README.txt

## How to Run

1. Install Python and the required libraries.

2. Install the required packages:

   pip install tensorflow numpy matplotlib scikit-learn pillow

3. Make sure the dataset folders are present inside the dataset folder.

4. Run the training and evaluation program:

   python breast_cancer_cnn.py

5. The graphs, confusion matrix, classification report and trained model will be saved in the output folder.

## Dataset Download Script

The file download_dataset.py was used to collect the selected image subset from the Kaggle dataset.

Before using it, make sure the Kaggle CLI is installed and authenticated.

Example:

python -m pip install kaggle
python -m kaggle auth login

Then run:

python download_dataset.py
