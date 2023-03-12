# Fully Convolutional Network

This model is based on the now-classic [2015 FCN paper](https://arxiv.org/abs/1411.4038).

This image pretty much sums up the architecture:

![](https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-16-at-10.34.02-PM.png)

![](https://csdl-images.computer.org/trans/tp/2017/04/figures/shelh3-2572683.gif)

In contrast to a convolutional net which uses convolutions for much of the computation but then turns to a linear layer for classification, an FCN uses convolutions all the way down. To achieve classification, we upsample the prediction so it is the same size as the input, and then pass a 1x1 kernel layer over it.

This paper also presaged the UNet architecture by utilizing "skip" connections.

As is typical of brilliant papers, it is easy to read and introduces several useful insights all at once.

## Receptive Fields

While several libraries to calculate receptive fields exist, they work better with down-convolutional layers than up-convolutional layers.

There is a gap in the market for a PyTorch library that can take, for instance, a named layer and display its receptive field against a generated field of pixels of an arbitrary input size.

## No Pre-trained Net

In the original paper, and most implementations since then, they used a pre-trained net and took it apart to use some of the layers. By contrast, my implementation here is self-contained.

## Concatenation and Addition

Residual connections, as I use in my `Block` class, add outputs together. Skip connections usually use concatenation, as I do in my `UNet`. What is the difference?

Addition is a linear operation to blur together two layers, so to speak, allowing one layer to influence or correct the next.

Concatenation adds previous information wholesale. Computationally, this is more expensive - we are doubling the computation performed at the layer where we concatenate.

The difference in the end may not matter so much. After all, we we are just wiring information from one layer to the next either way, and the weights will learn via backprop.

[This is a helpful post](https://aman.ai/primers/ai/skip-connections/) discussing the two.

My FCN uses addition rather than concatenation, in order to follow the 2015 paper.

## Layers

The first image contemplates 9 layers, with varying channels: 96, 256, 384, 4096, 21. Mine uses fewer channels.

## Usage

`conda activate <yourenv>`
`pip install -r ../requirements.txt`
`python3 fcn.py`
