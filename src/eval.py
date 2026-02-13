import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from models.cnn_autoencoder import CNNAutoencoder
from models.vae import JetVAE
from datasets.fashion_mnist import get_fashion_mnist
from datasets.scientific_dataset import (
    load_background,
    load_signal,
    jets_to_images_v2,
    normalize_images
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================
# Fashion-MNIST Evaluation
# =====================================================

def eval_fashion_ae():
    train_ds, test_ds = get_fashion_mnist(normal_class=7)

    model = CNNAutoencoder().to(device)
    model.load_state_dict(torch.load("results/fashion_ae.pth", map_location=device))
    model.eval()

    errors = []
    labels = []

    with torch.no_grad():
        for x, y in test_ds:
            x = x.unsqueeze(0).to(device)

            recon = model(x)
            err = F.mse_loss(recon, x, reduction='mean').item()

            errors.append(err)
            labels.append(0 if y == 7 else 1)

    auc = roc_auc_score(labels, errors)
    print(f"[Fashion AE] AUC = {auc:.4f}")


# =====================================================
# Scientific Dataset Evaluation
# =====================================================

def eval_scientific_vae(args):

    background = load_background("scientific_datasets/background.npy", n_samples=20000)
    signal = load_signal("scientific_datasets/signal.npy", n_samples=20000)

    model = JetVAE(latent_dim=args.latent_dim).to(device)
    model.load_state_dict(torch.load("results/scientific_vae.pth", map_location=device))
    model.eval()

    def compute_scores(jets):
        scores = []

        with torch.no_grad():
            for i in range(0, len(jets), args.batch_size):
                batch = jets[i:i+args.batch_size]

                imgs = jets_to_images_v2(batch)
                imgs = normalize_images(imgs)

                x = torch.tensor(imgs, dtype=torch.float32).unsqueeze(1).to(device)

                recon, mu, logvar = model(x)

                recon_loss = F.mse_loss(recon, x, reduction='none')
                recon_loss = recon_loss.view(x.size(0), -1).mean(dim=1)

                kl_loss = -0.5 * torch.sum(
                    1 + logvar - mu.pow(2) - logvar.exp(),
                    dim=1
                )

                score = recon_loss + kl_loss
                scores.extend(score.cpu().numpy())

                del x, imgs

        return np.array(scores)

    scores_bkg = compute_scores(background)
    scores_sig = compute_scores(signal)

    y_true = np.concatenate([
        np.zeros_like(scores_bkg),
        np.ones_like(scores_sig)
    ])

    scores = np.concatenate([scores_bkg, scores_sig])

    auc = roc_auc_score(y_true, scores)
    print(f"[Scientific VAE] AUC = {auc:.4f}")


# =====================================================
# Main
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="fashion")
    parser.add_argument("--model", type=str, default="ae")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--latent_dim", type=int, default=16)

    args = parser.parse_args()

    if args.dataset == "fashion" and args.model == "ae":
        eval_fashion_ae()

    elif args.dataset == "scientific" and args.model == "vae":
        eval_scientific_vae(args)

    else:
        raise ValueError("Unsupported dataset/model combination.")
