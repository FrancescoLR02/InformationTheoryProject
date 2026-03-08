import torch
import numpy as np




# Inside here mutual informations are calculated & mut.info and also activations are stored!
def VAE_info(model, dataset, device, epoch, num_samples, mi_estimator, mi_history, RecorderActivat):

    # -------------------------------- SETTING --------------------------------------

    model.eval()
    model.to(device)

    # load batch of data to evaluate
    loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=False)
    inputs, label = next(iter(loader))
    inputs = inputs.to(device)

    # ---------------------- CALCULATE & STORE ACTIVATIONS ----------------------------

    with torch.no_grad():
        x_hat, z, b, mean, logVar = model(inputs, label) # Foward pass to get the activation value in RecorderActivat.activations

    #print(b)
    #print(b.shape)

    RecorderActivat.save_epoch(epoch) # here we stored activation!

    # ------------------------ CALCULATE & STORE MUT.INFO ------------------------------

    X = inputs.view(inputs.size(0), -1).cpu().numpy()
    Z = RecorderActivat.get("latent_quant") # fixing here it was "latent_space" before introducing quantize latent
    # ACTHUNG
    #print(Z)
    #print(Z.shape)
    Y = RecorderActivat.get("output_space")
        
    mi = {
        "encoder": [],
        "decoder": [],
        "input_latent": None,
        "latent_output": None
    }

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


#*****************************************************************************************************************
#*****************************************************************************************************************

# Inside here mutual informations are calculated & mut.info and also activations are stored!
def VAE_info(model, dataset, device, epoch, num_samples, mi_estimator, mi_history, RecorderActivat, Variational):

    # -------------------------------- SETTING --------------------------------------

    model.eval()
    model.to(device)

    # load batch of data to evaluate
    loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=False)
    inputs, label = next(iter(loader))
    inputs = inputs.to(device)

    # ---------------------- CALCULATE & STORE ACTIVATIONS ----------------------------

    with torch.no_grad():
        x_hat, z, b, mean, logVar = model(inputs, label) # Foward pass to get the activation value in RecorderActivat.activations

    RecorderActivat.save_epoch(epoch) # here we stored activation!

    # ------------------------ CALCULATE & STORE MUT.INFO ------------------------------

    X = inputs.view(inputs.size(0), -1).cpu().numpy()
    Z = RecorderActivat.get("latent_space")
    Y = RecorderActivat.get("output_space")

    if Variational:
        mean = mean.cpu().numpy()
        logVar = logVar.cpu().numpy()
    else:
        mean = Z
        
    mi = {
        "encoder": [],
        "decoder": [],
        "input_latent": None,
        "latent_output": None
    }
    
    # Encoder Layers
    for i in range(len(model.Encoder)):
        layer_name = f"encoder_layer_{i+1}"
        A = RecorderActivat.get(layer_name)
        mi["encoder"].append((
            mi_estimator.mutual_information(A, X), # I(Layer, Input)
            mi_estimator.mutual_information(A, mean)  # I(Layer, Latent)
        ))

    # Decoder Layers
    for i in range(len(model.Decoder)):
        layer_name = f"decoder_layer_{i+1}"
        A = RecorderActivat.get(layer_name)
        mi["decoder"].append((
            mi_estimator.mutual_information(A, mean), # I(Layer, Latent)
            mi_estimator.mutual_information(A, Y)  # I(Layer, Output)
        ))

    if Variational: mi["input_latent"]  = mi_estimator.MutualInfor_Analytical(X, (mean, logVar))
    else: mi["input_latent"]  = mi_estimator.mutual_information(X, Z)
    mi["latent_output"] = mi_estimator.mutual_information(mean, Y)

    if Variational:
        mi['latent_points'] = mean

    else:
        mi['latent_points'] = Z
    
    # Store the mi calculated
    mi_history.append(mi)