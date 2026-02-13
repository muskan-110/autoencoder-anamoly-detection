import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from models.cnn_autoencoder import CNNAutoencoder
from models.vae import JetVAE
from datasets.fashion_mnist import get_fashion_mnist
from datasets.scientific_dataset import (
    load_background,
    jets_to_images,
    jets_to_images_v2,
    normalize_images
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def train_fashion_ae(args):
    train_ds, _ = get_fashion_mnist(normal_class=7)
    loader = DataLoader(train_ds, batch_size=128, shuffle=True)

    model = CNNAutoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        loss_sum = 0
        for x, _ in loader:
            x = x.to(device)
            loss = criterion(model(x), x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()

        print(f"[Fashion AE] Epoch {epoch+1}: {loss_sum/len(loader):.4f}")

    torch.save(model.state_dict(), "results/fashion_ae.pth")


def train_scientific_vae(args):
    # Load and preprocess scientific dataset
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "scientific_datasets", "background.h5")

    jets = load_background(data_path, n_samples=30000)

    model = JetVAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    BATCH = args.batch_size

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0

        for i in range(0, len(jets), BATCH):
            batch_jets = jets[i:i+BATCH]

            batch_imgs = jets_to_images_v2(batch_jets)
            batch_imgs = normalize_images(batch_imgs)

            x = torch.tensor(batch_imgs, dtype=torch.float32).unsqueeze(1).to(device)

            recon, mu, logvar = model(x)

            recon_loss = nn.functional.mse_loss(recon, x, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = (recon_loss + kl_loss) / x.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            del x, batch_imgs

        print(f"[Scientific VAE] Epoch {epoch+1}: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "results/scientific_vae.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="fashion")
    parser.add_argument("--model", type=str, default="ae")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--latent_dim", type=int, default=16)

    args = parser.parse_args()

    if args.dataset == "fashion" and args.model == "ae":
        train_fashion_ae(args)

    elif args.dataset == "scientific" and args.model == "vae":
        train_scientific_vae(args)

    else:
        raise ValueError("Unsupported combination of dataset and model.")
