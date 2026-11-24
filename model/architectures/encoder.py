from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=(16, 32, 64, 128),
        latent_dim=256,
        image_size=128,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        for out_ch in out_channels:
            self.convs.append(
                nn.Conv2d(in_channels, out_ch, kernel_size=3, stride=2, padding=1)
            )
            in_channels = out_ch
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.projection = nn.Linear(
            in_features=(
                out_channels[-1] * (image_size // 2 ** len(out_channels)) ** 2
            ),
            out_features=latent_dim,
        )

    def forward(self, x):
        for conv in self.convs:
            x = self.activation(conv(x))
        x = x.flatten(start_dim=1)  # (B,128,8,8) -> (B,8192)
        return self.projection(x)
