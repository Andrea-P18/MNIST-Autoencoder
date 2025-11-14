# MNIST-Autoencoder (PyTorch)
## Overview

This repository contains a simple PyTorch autoencoder trained on the MNIST dataset. It serves as the first step in a personal learning journey into representation learning and different autoencoder architectures. The initial model uses a basic encoder–decoder structure with a deliberately small latent space of 30 neurons, encouraging the network to learn compact and meaningful features from handwritten digits.

## What’s Included

The project currently features the base autoencoder implementation along with a visualization of reconstructed digits (1 through 10), offering a quick look at how well the compressed representations capture the essential structure of the input images. This helps highlight the strengths and limitations of a small latent space and builds intuition for future improvements.

## Learning Goals

The purpose of this project is to gradually expand from a simple baseline to more advanced architectures. Over time, the repository will grow to include variations such as denoising autoencoders, variational autoencoders, RNN-based models, and transformer-based approaches. Each addition will build on the same dataset and workflow, making it easier to compare how different models encode and reconstruct information.

## Future Work

1) Add denoising autoencoder

2) Implement a variational autoencoder (VAE)

3) Explore RNN-based encoding

4) Experiment with transformer-based reconstruction

5) Add visual comparisons between architectures
