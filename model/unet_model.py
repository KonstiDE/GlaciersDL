import torch
import torch.nn as nn

from model.unet_layers import (
    DoubleConv, UpConv
)

from model.unet_aspp import (
    ASPP
)


class GlacierUNET(nn.Module):
    # Init describes the netwrok architectures and the definition it layers
    def __init__(self, in_channels=1, out_channels=1, features=None):
        super(GlacierUNET, self).__init__()

        # Definition of features / level of depth of unet
        if features is None:
            features = [in_channels, 32, 64, 128, 256, 512]

        # Module lists for encoder, decoder double_convs and decoder transposed_convs
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.up_trans = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.final = nn.Conv2d(features[1], out_channels, kernel_size=(1, 1))

        # Append encoder double convolutions
        for i in range(len(features) - 2):
            self.down_convs.append(DoubleConv(features[i], features[i + 1]))

        # Double conv bottleneck with the last features plus the ASPP module
        self.bottleneck_conv = DoubleConv(in_channels=features[-2], out_channels=features[-1])
        self.bottleneck_aspp = ASPP(in_channels=features[-1], out_channels=features[-1], atrous_rates=[1, 2, 4, 8, 16])

        # We revert the features list as we into the decoder
        features = features[::-1]

        # Append decoder double convolutions and transposed convolutions
        for i in range(len(features) - 2):
            self.up_convs.append(DoubleConv(features[i], features[i + 1]))
            self.up_trans.append(UpConv(features[i], features[i + 1]))

    # Forward describes the flow of a tensor being put through the network
    def forward(self, x):
        # Skip connection save space for the decoder
        skip_connections = []

        # Decoding in three steps per level: Double conv, save the result as a skip-connection, pool the result
        for down_conv in self.down_convs:
            x = down_conv(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Put the tensor through the ASPP module serving as a bottleneck
        x = self.bottleneck_conv(x)
        x = self.bottleneck_aspp(x)

        # Reverse skip connections as the decoder uses the opposite order of flow
        skip_connections = skip_connections[::-1]

        # For every level: Do transposed convolution, concat with the skip-connection, double conv
        for i in range(len(self.up_convs)):
            x = self.up_trans[i](x)

            x = torch.cat((x, skip_connections[i]), dim=1)
            x = self.up_convs[i](x)

        # Final level to bring down the features
        return self.final(x)


def test():
    unet = GlacierUNET(in_channels=1, out_channels=1)

    x = torch.randn(4, 1, 256, 256)

    out = unet(x)

    print(out.shape)


if __name__ == "__main__":
    test()
