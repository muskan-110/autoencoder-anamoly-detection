## CNN & Variational Autoencoder for Scientific Anomaly Detection

This project implements deep learning–based unsupervised anomaly detection using convolutional autoencoders and variational autoencoders. The framework progresses from a controlled benchmark dataset to a real high-energy physics dataset.

# Phase 1: Fashion-MNIST Baseline

A convolutional autoencoder is trained on a single normal class to validate the anomaly detection pipeline.

- Normal data: one selected Fashion-MNIST class

- Anomaly score: reconstruction error (MSE)

- Evaluation metric: ROC-AUC

- Purpose: validate methodology in a controlled environment

This phase establishes the reconstruction-based anomaly detection baseline.

# Phase 2: LHCO 2020 Scientific Dataset (Jet Anomaly Detection)

The framework is extended to a real high-energy physics benchmark from the LHC Olympics 2020 dataset.

- Model: Variational Autoencoder (VAE)

- Training data: background (QCD) jets only

- Representation: jet-to-image conversion in η–φ space

- Preprocessing:

    Centering on leading particle

    Principal axis alignment (rotation)

    Quadrant flipping for symmetry normalization

- Anomaly score:

    Reconstruction loss

    KL divergence

- Evaluation:

    ROC-AUC between background and signal events

This phase demonstrates adaptation of unsupervised anomaly detection to structured scientific data.

## PROJECT STRUCTURE
src/
 ├── models/
 │    ├── cnn_autoencoder.py
 │    └── vae.py
 ├── datasets/
 │    ├── fashion_mnist.py
 │    └── scientific_dataset.py
 ├── train.py
 └── eval.py

notebooks/
results/
requirements.txt
README.md


## INSTALLATION

Create a virtual environment and install dependencies:

python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt

# How to Run
- Train Fashion Baseline
python src/train.py --dataset fashion --model ae

- Evaluate Fashion Baseline
python src/eval.py --dataset fashion --model ae

- Train Scientific VAE
python src/train.py --dataset scientific --model vae --epochs 20 --batch_size 300

- Evaluate Scientific VAE
python src/eval.py --dataset scientific --model vae --batch_size 300

# Datasets

Fashion-MNIST is automatically downloaded.

LHCO 2020 dataset must be placed inside:

scientific_datasets/
  background.h5
  signal.h5

# Research Motivation

Unsupervised anomaly detection is critical in high-energy physics where new physics signals are rare and unlabeled. This project explores reconstruction-based approaches and evaluates their behavior when transitioning from simple image data to structured jet representations.
