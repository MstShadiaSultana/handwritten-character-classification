"""Verify the local IAOR project environment."""

import platform
import sys

import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from PIL import Image
from torchvision.datasets import EMNIST, MNIST


def main():
    """Print environment information and run a small tensor test."""

    print("=" * 60)
    print("IAOR LOCAL ENVIRONMENT CHECK")
    print("=" * 60)
    print(f"Operating system : {platform.platform()}")
    print(f"Python version   : {sys.version.split()[0]}")
    print(f"Python executable: {sys.executable}")
    print()

    print(f"PyTorch version    : {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"NumPy version      : {np.__version__}")
    print(f"Matplotlib version : {matplotlib.__version__}")
    print(f"Pandas version     : {pd.__version__}")
    print(f"Scikit-learn       : {sklearn.__version__}")
    print(f"Pillow version     : {Image.__version__}")
    print()

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")

    print(f"CUDA available      : {cuda_available}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")

    if cuda_available:
        print(f"GPU name             : {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs       : {torch.cuda.device_count()}")
    else:
        print("Training device      : CPU")

    print()

    x = torch.rand(100, 100, device=device)
    y = torch.rand(100, 100, device=device)
    result = x @ y

    print(f"Test tensor device   : {result.device}")
    print(f"Test tensor shape    : {tuple(result.shape)}")
    print(f"Test successful      : {torch.isfinite(result).all().item()}")
    print()
    print(f"EMNIST loader available: {EMNIST is not None}")
    print(f"MNIST loader available : {MNIST is not None}")
    print("=" * 60)
    print("SETUP COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()