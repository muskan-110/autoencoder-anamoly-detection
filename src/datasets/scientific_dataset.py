import numpy as np
import pandas as pd
import h5py
import os

# =============================
# DATA LOADING
# =============================

def load_background(path, n_samples=None):
  
    # Convert .h5 path to .npy path
    npy_path = path.replace('.h5', '.npy')
    
    if not os.path.exists(npy_path):
        raise FileNotFoundError(
            f"NPY file not found: {npy_path}\n"
            f"Please convert your HDF5 files to NPY format first.\n"
            f"See convert_for_windows.py script."
        )
    
    print(f"Loading background from: {npy_path}")
    jets = np.load(npy_path)
    
    if n_samples:
        jets = jets[:n_samples]
    
    print(f" Loaded {len(jets)} jets with shape {jets.shape}")
    return jets


def load_signal(path, n_samples=None):
    
    # Convert .h5 path to .npy path
    npy_path = path.replace('.h5', '.npy')
    
    if not os.path.exists(npy_path):
        raise FileNotFoundError(
            f"NPY file not found: {npy_path}\n"
            f"Please convert your HDF5 files to NPY format first.\n"
            f"See convert_for_windows.py script."
        )
    
    print(f"Loading signal from: {npy_path}")
    jets = np.load(npy_path)
    
    if n_samples:
        jets = jets[:n_samples]
    
    print(f" Loaded {len(jets)} jets with shape {jets.shape}")
    return jets

# =============================
# JET → IMAGE CONVERSION
# =============================

IMG_SIZE = 40
ETA_RANGE = (-1.0, 1.0)
PHI_RANGE = (-np.pi, np.pi)

def preprocess_jet_v3(jet):
    """
    Advanced physics-informed preprocessing:
    1. Centering on the lead particle.
    2. Rotation to align the principal axis.
    3. Flipping to ensure the 2nd maximum is in a fixed quadrant.
    """
    
    jet = jet.copy()

    # 1. Centering: Lead particle (highest pT) to (eta, phi) = (0,0)
    lead_idx = np.argmax(jet[:, 0])
    jet[:, 1] -= jet[lead_idx, 1]
    jet[:, 2] -= jet[lead_idx, 2]

    # 2. Rotation: Align the principal axis
    pts = jet[:, 0]
    etas = jet[:, 1]
    phis = jet[:, 2]

    # Compute the components of the inertia tensor (weighted by pT)
    m11 = np.sum(pts * etas**2)
    m22 = np.sum(pts * phis**2)
    m12 = np.sum(pts * etas * phis)

    # Calculate the rotation angle (theta)
    theta = -np.arctan2(m12, m11)

    # Apply rotation matrix
    eta_new = etas * np.cos(theta) - phis * np.sin(theta)
    phi_new = etas * np.sin(theta) + phis * np.cos(theta)

    jet[:, 1] = eta_new
    jet[:, 2] = phi_new

    # 3. Flipping: Mirror symmetry
    if np.sum(pts[jet[:, 1] < 0]) > np.sum(pts[jet[:, 1] > 0]):
        jet[:, 1] = -jet[:, 1]

    # Check sum of pT in the bottom vs top half
    if np.sum(pts[jet[:, 2] < 0]) > np.sum(pts[jet[:, 2] > 0]):
        jet[:, 2] = -jet[:, 2]

    return jet

def jets_to_images_v2(jets, batch_size=1000, img_size=40):
    """
    Convert jets to images with advanced preprocessing.
    """
    images = []
    for i in range(0, len(jets), batch_size):
        batch = jets[i:i+batch_size]
        for jet in batch:
            processed_jet = preprocess_jet_v3(jet.copy())
            img = jet_to_image(
                processed_jet,
                img_size=img_size,
                eta_range=(-1, 1),  
                phi_range=(-np.pi, np.pi)
            )
            images.append(img)
    return np.array(images)


def jet_to_image(jet, img_size=40, eta_range=(-1, 1), phi_range=(-np.pi, np.pi)):
    """
    Convert a single jet to an image using 2D histogram.
    """
    pt = jet[:, 0]
    eta = jet[:, 1]
    phi = jet[:, 2]

    image, _, _ = np.histogram2d(
        eta,
        phi,
        bins=img_size,
        range=[eta_range, phi_range],
        weights=pt
    )
    return image


def jets_to_images(jets):
    """
    Basic jet to image conversion without advanced preprocessing.
    """
    n = len(jets)
    imgs = np.zeros((n, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    for i, jet in enumerate(jets):
        imgs[i] = jet_to_image(
            jet,
            img_size=IMG_SIZE,
            eta_range=ETA_RANGE,
            phi_range=PHI_RANGE
        )

    return imgs


def normalize_images(images):
    """
    Normalize images by dividing by their sum.
    """
    sums = images.sum(axis=(1, 2), keepdims=True)
    return images / (sums + 1e-8)