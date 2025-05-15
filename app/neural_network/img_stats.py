import os
import torch
from dataset import LatexDataset  # Import the dataset class from dataset.py
import argparse  # Import argparse for command-line arguments
from tqdm import tqdm

def compute_mean_std(data_dir, split="train"):
    """
    Computes the mean and standard deviation of the dataset.

    Args:
        data_dir (str): Path to the dataset directory.
        split (str): Dataset split to use ('train', 'val', or 'test').

    Returns:
        tuple: Mean and standard deviation of the dataset.
    """
    # Load the dataset
    dataset = LatexDataset(data_dir, split=split, augment=False)

    # Initialize variables to compute mean and std
    mean = 0.0
    std = 0.0
    total_pixels = 0

    print(f"Computing mean and std for the {split} split...")

    # Iterate through the dataset
    for idx in tqdm(range(len(dataset))):
        # Load the image (ignore captions and other outputs)
        image, _, _ = dataset[idx]  # image is a tensor of shape [1, H, W] for grayscale

        # Flatten the image to compute statistics
        image = image.view(1, -1)  # Flatten to [1, H * W]
        mean += image.mean()  # Sum mean of the image
        std += image.std()  # Sum std of the image
        total_pixels += image.numel()  # Total number of pixels

    # Compute final mean and std
    mean /= len(dataset)  # Average mean across all images
    std /= len(dataset)  # Average std across all images

    return mean.item(), std.item()


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Compute mean and standard deviation of the dataset.")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the dataset directory.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split to use (default: train).")
    args = parser.parse_args()

    # Compute mean and std for the specified split
    mean, std = compute_mean_std(args.data_dir, split=args.split)

    print(f"Mean: {mean}")
    print(f"Std: {std}")