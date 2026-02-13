import torch
import torch.nn as nn

class JetVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super(JetVAE, self).__init__()

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        # Probabilistic Bottleneck
        self.fc_mu = nn.Linear(32 * 10 * 10, latent_dim)
        self.fc_logvar = nn.Linear(32 * 10 * 10, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 32 * 10 * 10)
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (32, 10, 10)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        features = self.encoder_conv(x)
        mu, logvar = self.fc_mu(features), self.fc_logvar(features)
        z = self.reparameterize(mu, logvar)
        return self.decoder_conv(self.decoder_input(z)), mu, logvar

    def encode(self, x):
        features = self.encoder_conv(x)
        return self.fc_mu(features) # Use mean for latent space evaluation