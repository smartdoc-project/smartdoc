import torch.nn as nn

from architectures.decoder import Decoder
from architectures.encoder import Encoder


class AutoEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y
