#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchinfo import summary


class SidewaysConv(torch.nn.Module):
    """
    A simple block that doesn't change the number of channels or the pixels (height, width)

    Includes a residual connection
    """

    def __init__(self, in_chan, out_chan, kernel_size=3, padding=0):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_chan, out_chan, kernel_size=kernel_size, padding=padding
            ),
            torch.nn.BatchNorm2d(out_chan),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Conv2d(
                out_chan, out_chan, kernel_size=kernel_size, padding=padding + 1
            ),
            torch.nn.BatchNorm2d(out_chan),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
        )

        self.residual = torch.nn.Conv2d(in_chan, out_chan, kernel_size=1)

    def forward(self, x):

        z = self.net(x)

        _, _, H, W = z.shape
        residual = self.residual(x)[:, :, :H, :W]

        return z + residual


class DownConv(torch.nn.Module):
    """
    Passes the input sideways, then downsamples
    """

    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.sideways = SidewaysConv(in_chan, out_chan)
        self.downsample = torch.nn.MaxPool2d(kernel_size=2)

    def forward(self, x):

        skip = self.sideways(x)

        z = self.downsample(skip)

        return z, skip


class UpConv(torch.nn.Module):
    """
    Upsamples, then passes the input sideways

    """

    def __init__(self, in_chan, kernel_size=2):
        super().__init__()

        self.upsample = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        self.upconv = torch.nn.ConvTranspose2d(
            in_chan * 2,
            in_chan // 2,
            kernel_size=kernel_size,
            dilation=3,
            output_padding=2,
        )
        self.sideways = SidewaysConv(in_chan // 2, in_chan // 2)

    def pad(self, x, skip):
        _, _, Hx, Wx = x.shape
        _, _, Hs, Ws = skip.shape
        diff_in_height = Hs - Hx
        diff_in_width = Ws - Wx

        left = diff_in_width // 2
        right = diff_in_width - left  # if odd, right will be larger
        top = diff_in_height // 2
        bottom = diff_in_height - top  # if odd, bottom will be larger
        return F.pad(x, (left, right, top, bottom))

    def forward(self, x, skip):

        x = self.upsample(x)

        x = self.pad(x, skip)

        x = torch.cat([x, skip], dim=1)
        x = self.upconv(x)

        z = self.sideways(x)

        return z


class Normalize(torch.nn.Module):
    """
    Input normalization
    Numbers below calculated across entire training dataset
    """

    def __init__(self):
        super().__init__()
        self.mean = (0.2788, 0.2657, 0.2628)
        self.std = (0.2058, 0.1943, 0.2246)

        self.transform = transforms.Normalize(self.mean, self.std)

    def forward(self, x):
        return self.transform(x)


class UNet(torch.nn.Module):
    """
    First normalizes input

    Then passes input down four layers, over one layer, and up four layers

    On the way up, passes the matching layer output in as well

    Finally, classifies and trims the output
    """

    def __init__(self):
        super().__init__()

        self.normalize = Normalize()
        self.down1 = DownConv(3, 64)
        self.down2 = DownConv(64, 128)
        self.down3 = DownConv(128, 256)
        self.down4 = DownConv(256, 512)

        self.sideways = SidewaysConv(512, 512)

        self.up1 = UpConv(512)
        self.up2 = UpConv(256)
        self.up3 = UpConv(128)
        self.up4 = UpConv(64)

        self.classify = torch.nn.ConvTranspose2d(32, 5, kernel_size=1)

    def forward(self, x):
        _, _, H, W = x.shape

        x = self.normalize(x)
        x, output1 = self.down1(x)
        x, output2 = self.down2(x)
        x, output3 = self.down3(x)
        x, output4 = self.down4(x)

        x = self.sideways(x)

        x = self.up1(x, output4)
        x = self.up2(x, output3)
        x = self.up3(x, output2)
        x = self.up4(x, output1)

        z = self.classify(x)

        return z[:, :, :H, :W]


if __name__ == "__main__":
    model = UNet()
    summary(model, (32, 3, 96, 128))
