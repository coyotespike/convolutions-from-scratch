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


class SidewaysConvTuple(torch.nn.Module):
    """
    The SidewaysConv again, except it can take a tuple and return the second element unchanged

    This will be used to pass the output of the encoder to the decoder
    """

    def __init__(self, channels, kernel_size=3, padding=0):
        super().__init__()
        self.sideways = SidewaysConv(channels, channels, kernel_size, padding)

    def forward(self, tuple):
        x, output = tuple
        x = self.sideways(x)
        return x, output


class DownConv(torch.nn.Module):
    """
    Passes the input sideways, then downsamples
    """

    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.sideways = SidewaysConv(in_chan, out_chan)
        self.downsample = torch.nn.MaxPool2d(kernel_size=2)

    def forward(self, x, outputs):

        z = self.sideways(x)
        outputs.append(z)
        z = self.downsample(z)

        return z, outputs


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


class Encoder(torch.nn.Module):
    """
    Encodes the input downward, and saves outputs for use later
    """

    def __init__(self, layers=[(3, 128), (128, 256), (256, 512)]):
        super().__init__()

        self.outputs = []

        self.layers = torch.nn.ModuleList(
            [DownConv(in_chan, out_chan) for (in_chan, out_chan) in layers]
        )

    def forward(self, x):
        outputs = self.outputs
        for layer in self.layers:
            x, outputs = layer(x, outputs)

        return x, outputs


class Decoder(torch.nn.Module):
    """
    Decodes input upward

    """

    def __init__(self, layers=[512, 256, 128]):
        super().__init__()

        self.layers = torch.nn.ModuleList([UpConv(chan) for chan in layers])

    def forward(self, x_and_outputs):
        x, outputs = x_and_outputs
        for layer in self.layers:
            out = outputs.pop()
            x = layer(x, out)

        return x


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

    Then passes input down three layers, over one layer, and up three layers

    On the way up, passes the matching layer output in as well

    Finally, classifies and trims the output
    """

    def __init__(self):
        super().__init__()

        self.normalize = Normalize()
        self.encoder = Encoder()
        self.sideways = SidewaysConvTuple(256)
        self.decoder = Decoder()

        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(64, 5, kernel_size=1),
        )

        self.net = torch.nn.Sequential(
            self.normalize,
            self.encoder,
            self.sideways,
            self.decoder,
            self.classifier,
        )

    def forward(self, x):
        _, _, H, W = x.shape

        z = self.net(x)
        return z[:, :, :H, :W]


if __name__ == "__main__":
    model = UNet()
    summary(model, (32, 3, 96, 128))
