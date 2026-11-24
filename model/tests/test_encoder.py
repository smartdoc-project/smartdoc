import torch

from architectures.encoder import Encoder


def test_encoder_shape():
    x = torch.rand((4, 1, 128, 128))
    encoder = Encoder()
    z = encoder(x)
    assert z.shape == (4, 256)
