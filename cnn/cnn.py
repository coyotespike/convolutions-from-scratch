#!/usr/bin/env python3

import torch
from torchvision import transforms
from torchinfo import summary


class Block(torch.nn.Module):
    def __init__(self, n_input, n_output, stride=1, kernel_size=3, dilation=1):
        super().__init__()
        padding = kernel_size // 2

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


class CNN(torch.nn.Module):
    def __init__(
        self,
        layers=[32, 64, 128, 256],
        n_input_channels=3,
        n_output_channels=6,
        kernel_size=3,
    ):
        super().__init__()

        L = [
            # special starter block
            torch.nn.Conv2d(3, 3, kernel_size=7, padding=7 // 2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(3),
            torch.nn.MaxPool2d(kernel_size=3, padding=1),
        ]

        chan_in = n_input_channels
        for chan_out in layers:
            L.append(Block(n_input=chan_in, n_output=chan_out, stride=2, kernel_size=5))
            chan_in = chan_out

        self.network = torch.nn.Sequential(*L)
        # He initialization is better with ReLU, but Xaviet initilization still performed well
        # torch.nn.init.xavier_uniform_(self.network[0].weight)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(chan_in, n_output_channels),
            torch.nn.Dropout(0.5),
        )

    def forward(self, x):
        """
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        ## These weights must be calculated on the training set
        ## Omit this transformation if you have not calculated them
        mean = (0.3234, 0.3310, 0.3444)
        std = (0.2524, 0.2219, 0.2470)

        transf = transforms.Normalize(mean, std)
        x = transf(x)

        z = self.network(x)
        # the last two dimensions are the spatial dimensions
        # So we collapse them into one dimension and average over them
        z = z.mean([2, 3])
        prediction = self.classifier(z)
        return prediction


if __name__ == "__main__":
    model = CNN()
    summary(model, (32, 3, 128, 128))
