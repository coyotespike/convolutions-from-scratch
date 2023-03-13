#!/usr/bin/env python3

import torch


def max_pool_output_shape(
    x, padding=(0, 0), dilation=(1, 1), kernel_size=2, stride=(2, 2)
):
    """
    Takes a torch tensor and returns the output shape of a max pooling layer
    """
    _, _, H, W = x.shape
    h_out = (
        (H + (2 * padding[0]) - (dilation[0] * (kernel_size - 1))) // (stride[0])
    ) + 1
    w_out = (
        (W + (2 * padding[1]) - (dilation[1] * (kernel_size - 1))) // (stride[1])
    ) + 1

    return h_out, w_out


def conv_transpose_output_shape(
    x,
    padding=(0, 0),
    dilation=(1, 1),
    kernel_size=3,
    stride=(1, 1),
    output_padding=(0, 0),
):
    """
    Takes a torch tensor and returns the output shape of a convolutional transpose layer
    """
    _, _, H, W = x.shape
    h_out = (
        ((H - 1) * stride[0])
        - (2 * padding[0])
        + (dilation[0] * (kernel_size - 1))
        + output_padding[0]
        + 1
    )
    w_out = (
        ((W - 1) * stride[1])
        - (2 * padding[1])
        + (dilation[1] * (kernel_size - 1))
        + output_padding[1]
        + 1
    )

    return h_out, w_out


class RoundTrip(torch.nn.Module):
    """
    Demonstrates matching a max_pool with an upsample
    """

    def __init__(self, channels, kernel_size=3):
        super().__init__()

        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, padding=(1, 1))
        self.upsample = torch.nn.ConvTranspose2d(
            channels,
            channels,
            kernel_size=2,
            padding=3 // 2,
            stride=2,
            output_padding=(1, 1),
        )

    def forward(self, x):
        net = torch.nn.Sequential(
            self.max_pool, self.max_pool, self.upsample, self.upsample
        )

        return net(x)


if __name__ == "__main__":
    x = torch.randn(1, 3, 32, 32)
    print(f"Input: {x.shape} \n")

    net = RoundTrip(3)
    y = net(x)

    print(f"RoundTrip: {y.shape} \n")

    print(f"MaxPool: {max_pool_output_shape(x)} \n")

    print(f"ConvTranspose: {conv_transpose_output_shape(x)} \n")

    new_max = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    y = new_max(x)
    print(f"new_max: {y.shape} \n")
