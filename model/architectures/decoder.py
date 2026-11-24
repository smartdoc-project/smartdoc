from torch import nn


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels=1,
        in_channels=(128, 64, 32, 16),
        latent_dim=256,
        image_size=128,
    ):
        super().__init__()
        self.projection = nn.Linear(
            in_features=latent_dim,
            out_features=in_channels[0] * (image_size // 2 ** len(in_channels)) ** 2,
        )
