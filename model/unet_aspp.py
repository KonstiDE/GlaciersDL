import torch
from torch import nn
from torch.nn import functional as F


# The ASPP module uses several parallel convolutions. Here, the dilation changes within the module, we use a padding
# equal to the dilation plus two to avoid losing pixels while in convolution. We lower the kernel-size, as with bigger
# and very much higher dilation, the padding parameter would explode out of a realistic padding range.
class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__()

        self.aspp_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.aspp_conv(x)


# Inside the ASPP module, we use a 1x1 AverageAdaptivePooling layer to ensure that the ASPP module produces a consistent
# output regardless of the input size and to aggregate over global context information from the entire input feature
# map. To correct out channels to the desired output amount, we use a 1x1 convolution, with BatchNorm and ReLU right
# after.
class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()

        self.aspp_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = self.aspp_pool(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


# TThe ASPP module consists of multiple 1x1 convolutions. The first one with a dilation of one is doubled. The rest, is
# performed only one time with the rates from the unet_model.py file ([(1), 2, 4, 8, 16]). The result is summed
# concatenated afterwards.
class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=512):
        super(ASPP, self).__init__()
        # Standard dilation=1 (default)
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        ]

        # Add all other channels with the desired dilation rate
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # Append our averaging pooling
        modules.append(ASPPPooling(in_channels, out_channels))

        # Convert to module list to easily execute them from forward
        self.convs = nn.ModuleList(modules)

        # One big convolution to
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    # We execute just everything in our convs list
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))

        res = torch.cat(res, dim=1)

        return self.project(res)
