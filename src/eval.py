import torch
from torch.utils.data import DataLoader
from datasets.fashion_mnist import get_fashion_mnist
from models.cnn_autoencoder import CNNAutoencoder
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load datasets (NOT loaders)
_, test_dataset = get_fashion_mnist(normal_class=7)

# Create DataLoader here
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

# Load model
model = CNNAutoencoder().to(device)
model.load_state_dict(torch.load("results/model.pth", map_location=device))
model.eval()

# Compute reconstruction errors
scores = []
labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)

        recon = model(x)
        error = torch.mean((x - recon) ** 2, dim=(1, 2, 3))

        scores.append(error.item())
        labels.append(0 if y.item() == 7 else 1)

# ROC / AUC
fpr, tpr, _ = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

print(f"AUC: {roc_auc:.4f}")

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
