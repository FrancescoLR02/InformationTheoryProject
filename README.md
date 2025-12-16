# Information Theoretic Analysis of Variational Autoencoders (VAE)

This repository contains the implementation and analysis of a Variational Autoencoder (VAE), with a specific focus on Information Theory concepts. The project investigates the neural network not just as a generative model, but as a communication channel, analyzing the flow of information and the properties of the latent space.

## Project Objectives

### 1. VAE Implementation
* Implementation and training of a flexible VAE architecture.
* **Dataset:** Primary benchmarks performed on **MNIST**, with an architecture designed to be adaptable for other datasets.

### 2. Mutual Information Analysis
* Study of the **Mutual Information (MI)** dynamics between the network layers.
* Quantification of information flow across the Encoder and Decoder to understand compression and feature extraction.

### 3. Latent Space Characterization
* Analysis of the latent space ($z$) produced by the encoder.
* Investigation of the statistical distribution and **information content** of the generated representations.
* Dimensionality analysis using techniques such as **Principal Component Analysis (PCA)**.

### 4. Channel Coding & Robustness
* Modeling the encoder-decoder link as a communication channel.
* **Noisy Channel:** Analyzing the encoder's behavior and robustness when noise is introduced into the transmission.
* **Coding Schemes:** Evaluation of performance under **lossless** and **lossy** coding constraints during data transfer between encoder and decoder.