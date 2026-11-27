import torch

from architectures.autoencoder import AutoEncoder


def test_autoencoder_shape():
    for latent_dim in (64, 256):
        ae = AutoEncoder(latent_dim=latent_dim)
        x = torch.rand((4, 1, 128, 128))
        assert ae(x).shape == x.shape
        assert ae.encoder(x).shape == (4, latent_dim)
