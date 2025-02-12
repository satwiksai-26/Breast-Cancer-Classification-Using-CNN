# Breast Cancer Classification Using CNN

## Overview
This repository contains a Convolutional Neural Network (CNN) model for classifying breast cancer images. The model is built using TensorFlow and Keras, utilizing deep learning techniques to accurately differentiate between benign and malignant breast cancer images.

## Dataset
The dataset used for training and evaluation consists of labeled breast cancer histopathology images. The images are preprocessed before being fed into the CNN model.

## Features
- Convolutional Neural Network (CNN) architecture for image classification
- Data preprocessing and augmentation
- Model training and evaluation
- Visualization of training metrics

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/satwiksai-26/Breast-Cancer-Classification-Using-CNN.git
   cd Breast-Cancer-Classification-Using-CNN
   ```
2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Prepare the dataset by placing images in the appropriate directories.
2. Run the training script:
   ```sh
   python train.py
   ```
3. Evaluate the model:
   ```sh
   python evaluate.py
   ```
4. To predict on new images:
   ```sh
   python predict.py --image <path_to_image>
   ```

## Model Architecture
The CNN model consists of:
- Convolutional layers with ReLU activation
- MaxPooling layers for dimensionality reduction
- Fully connected dense layers
- Softmax activation for classification

## Results
The trained model achieves high accuracy in classifying breast cancer images. Training and validation accuracy and loss can be visualized using Matplotlib.

## Contributions
Feel free to contribute by opening issues or submitting pull requests.

## License
This project is open-source and available under the MIT License.

## Author
Developed by [Satwik Sai](https://github.com/satwiksai-26).
