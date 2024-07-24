# CUDA MNIST digit classifier

This repository contains code for training and analyzing a simple image classifier for the MNIST dataset using PyTorchon CUDA. The classifier is a fully connected neural network that predicts the digit (0-9) from an input image of a handwritten digit.

## Model

The model is a simple feedforward neural network with the following architecture:
- **Input Layer**: Flattens the 28x28 pixel input image into a 784-dimensional vector.
- **Hidden Layer**: A fully connected layer with 128 neurons and ReLU activation.
- **Output Layer**: A fully connected layer with 10 neurons (one for each digit) and no activation (logits).

## Training Loop

The training loop involves the following steps:
1. **Load Data**: Load the MNIST dataset and prepare DataLoader objects for training and testing.
2. **Initialize Model**: Create an instance of the neural network, define the loss function (CrossEntropyLoss), and the optimizer (Adam).
3. **Training**: For each epoch, iterate over the training data, perform forward and backward passes, and update the model parameters.
4. **Evaluation**: After each epoch, evaluate the model on the test set and print the accuracy.

## Dataset

The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits. Each image is a 28x28 grayscale image. The dataset is loaded using the Hugging Face `datasets` library.

## Analysis

The analysis involves identifying and saving the top 10 misclassified samples from the test set. The misclassified samples are saved as images in the `misclassified_samples` directory.

## Installation and Evaluation

### Prerequisites

- Python 3.10
- `pip` for package management
- `apt` for installing system packages

### Installation

1. **Clone the repository**:
   ```sh
   git clone https://github.com/klang49/mnist-classifier
   cd mnist-classifier
   ```

2. **Install dependencies**:
   ```sh
   make install
   ```

### Training

To train the model, run:
```sh
make train
```

### Analysis

To analyze the model and save the misclassified samples, run:
```sh
make analysis
```

As per the analysis, it is observed that the misclassifications are majorly due to the following reasons:
- Occulsions
- Extra writing
- Italicised Letters
- Wrong proportions

How can we fix it possibly?
- Make model occulsion resistant
- Make model rotation invarient
- Make model independent of the proportions of the components

### Cleanup

To remove the virtual environment, run:
```sh
make remove
```

This README provides an overview of the model, training loop, dataset, analysis, and steps for installation and evaluation.
