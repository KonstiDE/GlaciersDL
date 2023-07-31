import torch
import torch.nn as nn

from model.unet_layers import (
    DoubleConv, UpConv
)


class GlacierUNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=None):
        super(GlacierUNET, self).__init__()

        if features is None:
            features = [in_channels, 32, 64, 128, 256]

        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up_trans = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.final = nn.Conv2d(features[1], out_channels, kernel_size=(1, 1))

        for i in range(len(features) - 2):
            self.down_convs.append(DoubleConv(features[i], features[i + 1]))

        self.bottleneck = DoubleConv(features[-2], features[-1])

        features = features[::-1]

        for i in range(len(features) - 2):
            self.up_convs.append(DoubleConv(features[i], features[i + 1]))
            self.up_trans.append(UpConv(features[i], features[i + 1]))

    def forward(self, x):
        skip_connections = []

        for down_conv in self.down_convs:
            x = down_conv(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        for i in range(len(self.up_convs)):
            x = self.up_trans[i](x)

            x = torch.cat((x, skip_connections[i]), dim=1)
            x = self.up_convs[i](x)

        return self.final(x)


def test():
    unet = GlacierUNET(in_channels=1, out_channels=1).cuda()

    x = torch.randn(1, 3, 512, 512).cuda()

    out = unet(x)

    print(out.shape)


if __name__ == "__main__":
    test()