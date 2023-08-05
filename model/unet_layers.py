import torch.nn as nn


# Transposed convolution with a factor of 2
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=(2, 2))

    def forward(self, x):
        return self.up(x)


# Double convolution block: We use a bigger kernel size and a medium amount of dilation as we what to
# capture a bigger textural context in the image and do not care much about the details.
# As an activation function, LeakyReLU worked best. Furthermore, we normalize our data in between convolutions.
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(5, 5), bias=False, padding=6, padding_mode='replicate', dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 5), bias=False, padding=6, padding_mode='replicate', dilation=3),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)
