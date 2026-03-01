# ============================
# Standard Library
# ============================
import os
import random
import pickle
import json
import re
from typing import List

# ============================
# Scientific & Data Handling
# ============================
import numpy as np
import pandas as pd
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors

# ============================
# PyTorch
# ============================
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

# ============================
# Visualization
# ============================
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation

# ============================
# Jupyter / IPython
# ============================
from IPython.display import (
    HTML,
    clear_output,
    Markdown,
    display,
    Image
)

# ============================
# Progress Bars
# ============================
from tqdm import tqdm

# ============================
# Export list
# ============================
__all__ = [
    # Standard Library
    "os", "random", "pickle", "json", "re", "List",

    # Scientific & Data Handling
    "np", "pd", "digamma", "NearestNeighbors",

    # PyTorch
    "torch", "nn", "DataLoader", "torchvision", "transforms",

    # Visualization
    "plt", "cm", "Normalize", "Line2D", "FuncAnimation",

    # Jupyter / IPython
    "HTML", "clear_output", "Markdown", "display", "Image",

    # Progress Bars
    "tqdm"
]

#*****************************************************************************************************************
#*****************************************************************************************************************