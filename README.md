# Liquid Time-Constant Network EMG Classifier

The goal of this project is to determine whether LTCs are more accurate and efficient than traditional EMG classification methods (i.e., CNNs and vanilla RNNs), particularly for the SmartHome control project (https://github.com/aaWang27/SmartHomeController).

This project explores the use of Liquid Time-Constant (LTC) networks for EMG classification. LTCs are a type of recurrent neural network developed by Ramin Hasani et. al. (https://arxiv.org/abs/2006.04439). They utilize differential equations to 

Our code uses Hasani et al's Tensorflow implementation of LTC layers (https://github.com/mlech26l/ncps). We combine the LTC layers with convolutional layers to create our classification model.

Data processing is first performed in MATLAB, but we are working on a Python implementation. We are also working on optimizing the LTC network, and will potentially integrate it into live testing for comparison with CNNs.

