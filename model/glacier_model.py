import torch
import torch.nn as nn
import torch.nn.functional as F
from model.ASPP import ASPP


class GlacierUNET(nn.Module):

    def __init__(self):
        super(GlacierUNET, self).__init__()

        self.n_channels_of_input = 1  # Greyscale
        self.kernel_size = 3
        self.non_linearity = "Leaky_ReLU"
        self.n_layers = 5
        self.features_start = 32
        self.aspp = True

        self.layers, self.bottleneck = self.make_layer_structure()

    def make_layer_structure(self):
        layers = [DoubleConv(in_channels=self.n_channels_of_input, out_channels=self.features_start,
                             kernel_size=self.kernel_size, non_linearity=self.non_linearity)]

        feats = self.features_start
        for _ in range(self.n_layers - 1):
            layers.append(Down(in_channels=feats, out_channels=feats * 2, kernel_size=self.kernel_size,
                               non_linearity=self.non_linearity))
            feats *= 2

        if self.aspp:
            bottleneck = ASPP(feats, [1, 2, 4, 8], feats)
        else:
            bottleneck = None

        for _ in range(self.n_layers - 1):
            layers.append(Up(in_channels=feats, out_channels=feats // 2, kernel_size=self.kernel_size,
                             non_linearity=self.non_linearity))
            feats //= 2

        layers.append(nn.Conv2d(feats, 1, kernel_size=1))
        return nn.ModuleList(layers), bottleneck

    def forward(self, x):
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1:self.n_layers]:
            xi.append(layer(xi[-1]))

        # Bottleneck layers
        if self.bottleneck is not None:
            xi[-1] = self.bottleneck(xi[-1])

        # Up path
        for i, layer in enumerate(self.layers[self.n_layers:-1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, non_linearity: str):
        super().__init__()
        if non_linearity == "ReLU":
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_channels), nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm2d(out_channels), nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, non_linearity: str):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels,
                       kernel_size=kernel_size, non_linearity=non_linearity)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 non_linearity: str):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                               non_linearity=non_linearity)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
