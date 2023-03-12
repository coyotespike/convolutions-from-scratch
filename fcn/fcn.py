#!/usr/bin/env python3

import torch
from torchinfo import summary


class UpConv(torch.nn.Module):
    def __init__(
        self, n_input, n_output, stride=1, kernel_size=3, dilation=1, output_padding=0
    ):
        super().__init__()

        padding = kernel_size // 2

        self.net = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                n_input,
                n_output,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride,
                output_padding=output_padding,
                padding=padding,
            ),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.ConvTranspose2d(
                n_output, n_output, kernel_size=kernel_size, padding=padding
            ),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
        )

        self.upsample = None
        if n_input != n_output or stride != 1:
            self.upsample = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(
                    n_input,
                    n_output,
                    kernel_size=1,
                    dilation=dilation,
                    stride=stride,
                    output_padding=output_padding,
                ),
                torch.nn.BatchNorm2d(n_output),
            )

    def forward(self, x):
        identity = x
        if self.upsample is not None:
            identity = self.upsample(x)
            return self.net(x) + identity

        return self.net(x)


class DownConv(torch.nn.Module):
    def __init__(self, n_input, n_output, stride=1, kernel_size=3, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding = kernel_size // 2
        self.dilation = dilation

        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(
                n_input,
                n_output,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                stride=stride,
            ),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Conv2d(
                n_output, n_output, kernel_size=kernel_size, padding=padding
            ),
            torch.nn.BatchNorm2d(n_output),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
        )

        if n_input != n_output or stride != 1:
            self.downsample = torch.nn.Sequential(
                torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(n_output),
            )

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
            return self.net(x) + identity

        return self.net(x)


class FCN(torch.nn.Module):
    def __init__(self, n_input_channels=3, n_output_channels=5):
        super().__init__()

        self.dc1 = DownConv(n_input_channels, 32, stride=2)
        self.dc2 = DownConv(32, 64, stride=2)
        self.dc3 = DownConv(64, 128, stride=2)

        self.uc3 = UpConv(128, 64, stride=2, output_padding=1)
        self.uc2 = UpConv(64, 32, stride=2, output_padding=1)
        self.uc1 = UpConv(32, n_output_channels, stride=2, output_padding=1)

        self.classifier = torch.nn.Conv2d(n_output_channels, n_output_channels, 1)

        self.start = self.dc1
        self.encode_decode = torch.nn.Sequential(self.dc2, self.dc3, self.uc3, self.uc2)
        self.end = self.uc1

    def forward(self, x):

        x1 = self.dc1(x)
        x2 = self.encode_decode(x1)
        x3 = self.uc1(torch.add(x2, x1))

        return self.classifier(x3)


if __name__ == "__main__":
    model = FCN()
    summary(model, (32, 3, 96, 128))
