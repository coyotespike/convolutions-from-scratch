# U-Net

![](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)

This image might be the best paper summary I've ever seen. Channels, input size, convolutional layers, it's all there.

## Architecture

### Padding

The sideways convolutions do not use padding, allowing the activations to shrink. This follows the original paper, except one of them is padded to get the same output size as the input.

Following the original paper, I take the sideways convolution output - before downsampling - and concatenate it with the input to the upsampling layer.

The original paper cropped outputs to match the input size. The skip outputs were too small for this; instead I padded the upsample outputs.

Thanks [to Milesial](https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py) for the padding calculations. As he notes in a comment to an issue, padding is if anything superior to cropping because it preserves information.

We pad all sides, because a convolution shrinks all sides. Because the right and bottom get cropped when the kernel does not fit the input, we pad those sides a little more.

### Down Sampling

The greatest downsampling takes place with a MaxPool. With a stride of 2, this shrinks the input size by half (default stride equals kernel size, so this is implicit in `DownConv`).

The original paper downsamples via maxpooling 4 times. My images were smaller so I downsampled only 3 times.

### Up Sampling

The greatest upsampling takes place with `Upsample`, which doubles the input size.

In my first implementation, I used a `ConvTranspose2d` layer for learnable upsampling. However, the original paper mentions only "upsampling by a factor of 2," so I switched to `Upsample` with a `mode='bilinear'` argument. [This excellent post](https://machinethink.net/blog/coreml-upsampling/) is a good introduction to upsampling. tldr, PyTorch implements bilinear upsampling correctly, for each new pixel taking the weighted average of the 4 nearest pixels in the input.

There is some evidence that [bilinear upsample avoids checkerboard artifacts](https://distill.pub/2016/deconv-checkerboard/), unlike transposed convolutions.

Upsampling lacks learnable parameters, so it is also a bit more efficient. Because it is a simple interpolation, I do not place ReLU or dropout layers after it.

A third approach would be `MaxUnpool2d`, which attempts the inverse of `MaxPool2d`.
