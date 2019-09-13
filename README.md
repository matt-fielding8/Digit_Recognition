# Digit_Recognition
## Introduction

The famous MNIST dataset contains thousands of labelled examples of handwritten digits and is commonly used as a benchmark to test algorithm performance. We're going to build a baseline Neural Network algorithm optimised by batch gradient descent then compare its performance with a Convolutional Neural Network optimised by stochastic gradient descent.

## Data

The data has been preprocessed into `train.csv` and `test.csv` files available to download from this [MNIST Dataset: Digit Recognizer](https://www.kaggle.com/ngbolin/mnist-dataset-digit-recognizer/data#Data-cleaning,-normalization-and-selection).

The data files contain pixel intensity data for a total of 70,000 labelled greyscale images of handwritten digits. Each image is  28 pixels in height and 28 pixels in width, 784 pixels in total. Each pixel is represented by a single column in the data files and takes an integer value between 0 and 255, with 0 being the lightest and 255 being the darkest. Each data file also contains a `label` column containing the correct integer values in the range 0-9.

The train:test split is 60:40 (42,000 : 28,000).

## Notebooks
 - [digit_recognition_NN](http://localhost:8888/notebooks/notebooks/digit_recognition_NN.ipynb)
