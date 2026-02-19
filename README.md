# CNN & Variational Autoencoder for Scientific Anomaly Detection

This repository implements reconstruction-based unsupervised anomaly detection using deep autoencoders, progressing from a controlled image benchmark to the LHC Olympics (LHCO) 2020 jet anomaly detection dataset.
The goal is to study anomaly detection in high-energy physics settings, where potential new physics signals are rare and unlabeled.

# Phase 1: Controlled Benchmark — Fashion-MNIST

A convolutional autoencoder (CNN-AE) is trained on a single “normal” class to validate the anomaly detection pipeline in a controlled environment.

Setup
- Normal data: one selected Fashion-MNIST class
- Anomaly score: reconstruction error (MSE)
- Evaluation metric: ROC-AUC

Purpose
- Establish and validate the reconstruction-based anomaly detection methodology before transitioning to scientific data.

# Phase 2: LHC Olympics 2020 — Jet Anomaly Detection

The framework is extended to the LHC Olympics (LHCO) 2020 dataset, a benchmark for unsupervised new physics searches.

Model
- Variational Autoencoder (VAE)

Training Data
- Background (QCD) jets only

Representation
- Jet-to-image conversion in η–φ space

Physics-Inspired Preprocessing
- Centering on leading particle
- Principal axis alignment (rotation)
- Quadrant flipping for symmetry normalization

Anomaly Score
- Reconstruction loss
- KL divergence term

Evaluation
- ROC-AUC between background and signal events

This phase demonstrates the adaptation of reconstruction-based anomaly detection to structured high-energy physics data.

# Experimental Design

The project is structured to ensure modularity and reproducibility:
```src/
  models/
    cnn_autoencoder.py
    vae.py
    eval.py
  datasets/
    fashion_mnist.py
    scientific_dataset.py
  train.py
  utils.py

scientific_datasets/
  background.h5
  signal.h5
```
The modular design allows:
- Easy benchmarking across models
- Clear separation of data handling, training, and evaluation
- Extension toward alternative architectures

# Results :
Fashion-MNIST Baseline- ROC-AUC: 0.951

LHCO 2020 VAE- ROC-AUC: 0.5001

# Research Motivation

Unsupervised anomaly detection plays a critical role in high-energy physics, where new physics signals are rare and unlabeled. Reconstruction-based methods provide a baseline approach by modeling background distributions and identifying deviations.

This project investigates how such methods behave when transitioning from simple image data to structured jet representations derived from collider events.

# Planned Extension: Graph-Based Anomaly Detection

To better capture relational structure in collision events, this framework will be extended toward:

- Graph representations of jets (particles as nodes, relational edges)
- Graph Neural Networks (GNNs) for graph-level anomaly detection
- Contrastive learning objectives for improved representation learning

This extension aims to move beyond image-based representations and leverage the underlying particle-level structure of collider events.
