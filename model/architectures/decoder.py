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
        self.initial_channels = in_channels[0]
        self.spatial_size = image_size // (2 ** len(in_channels))
        self.out_features = in_channels[0] * self.spatial_size**2
        self.target_channels = in_channels[1:] + (out_channels,)
        self.projection = nn.Linear(
            in_features=latent_dim, out_features=self.out_features
        )

        self.convs = nn.ModuleList()

        for in_ch, out_ch in zip(in_channels, self.target_channels):
            self.convs.append(
                nn.ConvTranspose2d(
                    in_ch, out_ch, kernel_size=3, stride=2, output_padding=1, padding=1
                )
            )
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.output_activation = nn.Sigmoid()

    def forward(self, z):
        z = self.projection(z)
        z = self.activation(z)
        z = z.reshape((-1, self.initial_channels, self.spatial_size, self.spatial_size))
        for conv in self.convs[:-1]:
            z = self.activation(conv(z))
        z = self.output_activation(self.convs[-1](z))
        return z
