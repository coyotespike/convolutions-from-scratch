#!/usr/bin/env python3

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
def calculate_dataset_stats(dataset_path, dataset_loader):
    # Load this ourselves to leave shuffle false
    dataset = dataset_loader(dataset_path)

    # works without this just as well
    # train_data = DataLoader(dataset, batch_size=128, shuffle=False)

    # placeholders: Red, Green, Blue
    pixel_sum = torch.tensor([0.0, 0.0, 0.0])
    pixel_sum_sq = torch.tensor([0.0, 0.0, 0.0])

    # we don't need label but we want to destructure, else we get an error
    for img, label in dataset:
        # wif working with dataset directly, shape is torch.Size([3, 64, 64])
        # collapse across height and width, preserving each channel separately
        pixel_sum += img.sum(dim=[1, 2])
        pixel_sum_sq += (img**2).sum(dim=[1, 2])

        # if working with dataloader, shape [128, 3, 64, 64]
        # collapse on dim 0, to get all 128
        # pixel_sum    += img.sum(dim = [0, 2, 3])
        # pixel_sum_sq += (img ** 2).sum(dim = [0, 2, 3])

    # 21000 training images * 64 * 64
    total_pixels = 86016000
    # mean and std
    total_mean = pixel_sum / total_pixels
    total_var = (pixel_sum_sq / total_pixels) - (total_mean**2)
    total_std = torch.sqrt(total_var)

    print("mean: " + str(total_mean))
    print("std:  " + str(total_std))

    """
    for CNN, not dense data
    mean: tensor([0.3234, 0.3310, 0.3444])
    std:  tensor([0.2524, 0.2219, 0.2470])

    hmm now I get this and can't see what changed
    mean: tensor([ 1.3419e-04,  2.2948e-05, -6.5862e-05])
    std:  tensor([0.9998, 1.0002, 1.0002])

    sheit, a day later and back to the first one!!
    performs much better with the first. second one hurts it
    """


def calculate_dense_stats(dataset_path, dataset_loader):
    # Load this ourselves to leave shuffle false
    dataset = dataset_loader(dataset_path)

    # works without this just as well
    # train_data = DataLoader(dataset, batch_size=128, shuffle=False)

    # placeholders: Red, Green, Blue
    pixel_sum = torch.tensor([0.0, 0.0, 0.0])
    pixel_sum_sq = torch.tensor([0.0, 0.0, 0.0])

    # we don't need label but we want to destructure, else we get an error
    for img, label in dataset:
        # wif working with dataset directly, shape is torch.Size([3, 128, 96])
        # collapse across height and width, preserving each channel separately
        pixel_sum += img.sum(dim=[1, 2])
        pixel_sum_sq += (img**2).sum(dim=[1, 2])

        # if working with dataloader, shape [32, 3, 128, 96]
        # collapse on dim 0, to get all 128
        # pixel_sum    += img.sum(dim = [0, 2, 3])
        # pixel_sum_sq += (img ** 2).sum(dim = [0, 2, 3])

    # 10000 training images * 128 * 96
    # don't multiply by channels because we are norming per channel, that would be too big
    num_images = len(dataset)
    total_pixels = num_images * 128 * 96
    # mean and std
    total_mean = pixel_sum / total_pixels
    total_var = (pixel_sum_sq / total_pixels) - (total_mean**2)
    total_std = torch.sqrt(total_var)

    print("mean: " + str(total_mean))
    print("std:  " + str(total_std))

    """
    mean: tensor([0.2788, 0.2657, 0.2628])
    std:  tensor([0.2058, 0.1943, 0.2246])

    """


if __name__ == "__main__":
    calculate_dataset_stats("./data/train")
