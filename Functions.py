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

def VAE_info(model, dataset, device, epoch, num_samples, mi_estimator, RecorderActivat):
    model.eval()
    model.to(device)
    
    # batch of data to evaluate Mutual Info
    loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)
    inputs, _ = next(iter(loader))
    inputs = inputs.to(device)

    with torch.no_grad():
        model(inputs) # Foward pass to get the activation value in RecorderActivat.activations

    RecorderActivat.save_epoch(epoch)

    X = inputs.view(inputs.size(0), -1).cpu().numpy()
    Z = RecorderActivat.get("latent_space")
    Y = RecorderActivat.get("output_space")
        
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
            mi_estimator.mutual_information(A, Z)  # I(Layer, Latent)
        ))

    # Decoder Layers
    for i in range(len(model.Decoder)):
        layer_name = f"decoder_layer_{i+1}"
        A = RecorderActivat.get(layer_name)
        mi["decoder"].append((
            mi_estimator.mutual_information(A, Z), # I(Layer, Latent)
            mi_estimator.mutual_information(A, Y)  # I(Layer, Output)
        ))

    mi["input_latent"]  = mi_estimator.mutual_information(X, Z)
    mi["latent_output"] = mi_estimator.mutual_information(Z, Y)
    
    return mi

#*****************************************************************************************************************
#*****************************************************************************************************************

def LatentVAE_Info(model, dataset, device, epoch, num_samples, mi_estimator, RecorderActivat):
    model.eval()
    model.to(device)

    #Fix the same seed 
    g = torch.Generator()
    g.manual_seed(42)
    
    # batch of data to evaluate Mutual Info
    loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True, generator=g)
    inputs, _ = next(iter(loader))
    inputs = inputs.to(device)

    with torch.no_grad():
        outputs, z, mean, logVar = model(inputs)

    mean = mean.detach().cpu().numpy()
    logVar = logVar.detach().cpu().numpy()

    mi = mi_estimator.LatentMutualInformation(mean, logVar)

    return mi

#*****************************************************************************************************************
#*****************************************************************************************************************

def plot_kde_geometry(recorder, mi_estimator, part="encoder", layer=1, neuron=None, epoch=-1, bins=60):

    if not recorder.history:
        print("No history recorded yet.")
        return

    available_epochs = list(recorder.history.keys())
    if epoch == -1:
        epoch = available_epochs[-1]
    if epoch not in recorder.history:
        print(f"Epoch {epoch} not found in history.")
        return

    if part == "encoder":
        key = f"encoder_layer_{layer}"
    elif part == "decoder":
        key = f"decoder_layer_{layer}"
    elif part == "latent":
        key = "latent_space"
    elif part == "output":
        key = "output_space"
    else:
        print("Part must be 'encoder', 'decoder', 'latent', or 'output'")
        return

    data_dict = recorder.history[epoch]
    if key not in data_dict:
        print(f"Key {key} not found in epoch {epoch}")
        return
        
    X = data_dict[key]

    if neuron is not None:
        X = X[:, neuron:neuron+1]
    
    rho = mi_estimator.density(X)

    X_sq = np.sum(X**2, axis=1, keepdims=True)
    dists_sq = X_sq + X_sq.T - 2 * X @ X.T
    dists = np.sqrt(dists_sq) # dists = np.sqrt(np.maximum(dists_sq, 0))
    tri_idx = np.triu_indices_from(dists, k=1)
    D = dists[tri_idx]

    # for sup-title
    title_str = f"{part}:".upper()
    if part in ["encoder", "decoder"]:
        title_str += f" L{layer}"
    if neuron is not None:
        title_str += f"-N{neuron}"        
    title_str += f" Ep{epoch}"

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title_str, fontsize=14)

    axs[0].hist(D, bins=bins, density=True, alpha=0.7, color='gray', edgecolor='black')
    axs[0].set_title("Pairwise Distances")

    axs[1].hist(rho, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    axs[1].set_title("Estimated Density (KDE)")
    
    plt.tight_layout()
    plt.show()


#*****************************************************************************************************************
#*****************************************************************************************************************


def PlotInfoPlane(mi_history, title_suffix="", suptitle="", start_epoch=1, end_epoch=-1, Step=5):

    mi_history_encoder = mi_history.encoder
    mi_history_decoder = mi_history.decoder

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[6, 6, 0.2], wspace=0.3)
    if suptitle != "": fig.suptitle(suptitle, size=16, weight="bold")

    ax_enc = fig.add_subplot(gs[0, 0])
    ax_dec = fig.add_subplot(gs[0, 1])
    ax_cb  = fig.add_subplot(gs[0, 2])

    total_epochs = len(mi_history_encoder)
    if end_epoch == -1 or end_epoch >= total_epochs:
        end_epoch = total_epochs -1

    # NEW: filter epochs by Step
    epoch_range = [ep for ep in range(start_epoch, end_epoch + 1) if ep % Step == 0]
    epochs = len(epoch_range)

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / max(1, epochs - 1)) for i in range(epochs)]

    encoder_markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', '<', '>']
    decoder_markers = ['>', '*', 'p', 'X', 'H', '>', 'd', 'D', '8', 'o']

    n_layers = len(mi_history_encoder[0])

    # ---------------- ENCODER ----------------
    ax = ax_enc

    for ep_idx, ep in enumerate(epoch_range):
        x = [mi_history_encoder[ep][l][0] for l in range(n_layers)]
        y = [mi_history_encoder[ep][l][1] for l in range(n_layers)]

        ax.plot(x, y, linestyle='-', color=colors[ep_idx],
                alpha=0.6, linewidth=2)

        for l in range(n_layers):
            marker = encoder_markers[l % len(encoder_markers)]
            ax.scatter(x[l], y[l],
                        facecolors=[colors[ep_idx]],
                        edgecolors='black',
                        s=60, marker=marker,
                        linewidths=0.6, zorder=3)

    ax.set_xlabel("I(Layer; Input)", fontsize=14)
    ax.set_ylabel("I(Layer; Latent Z)", fontsize=14)
    ax.set_title(f"Encoder Information Plane {title_suffix}", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)

    enc_handles = []
    enc_labels = []
    for l in range(n_layers):
        mk = encoder_markers[l % len(encoder_markers)]
        handle = Line2D([0], [0], marker=mk, color='black',
                        markerfacecolor='black',
                        markeredgecolor='black',
                        markersize=10, linestyle='None')
        enc_handles.append(handle)
        enc_labels.append(f'Layer {l+1}')

    ax_enc.legend(enc_handles, enc_labels, framealpha=0.9, fontsize=12, title_fontsize=13)

    # ---------------- DECODER ----------------
    ax = ax_dec

    for ep_idx, ep in enumerate(epoch_range):
        x = [mi_history_decoder[ep][l][0] for l in range(n_layers)]
        y = [mi_history_decoder[ep][l][1] for l in range(n_layers)]

        ax.plot(x, y, linestyle='-', color=colors[ep_idx],
                alpha=0.6, linewidth=2)

        for l in range(n_layers):
            marker = decoder_markers[l % len(decoder_markers)]
            ax.scatter(x[l], y[l],
                        facecolors=[colors[ep_idx]],
                        edgecolors='black',
                        s=60, marker=marker,
                        linewidths=0.6, zorder=3)

    ax.set_xlabel("I(Layer; Latent Z)", fontsize=14)
    ax.set_ylabel("I(Layer; Output)", fontsize=14)
    ax.set_title(f"Decoder Information Plane {title_suffix}", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)

    dec_handles = []
    dec_labels = []
    for l in range(n_layers):
        mk = decoder_markers[l % len(decoder_markers)]
        handle = Line2D([0], [0], marker=mk, color='black',
                        markerfacecolor='black',
                        markeredgecolor='black',
                        markersize=10, linestyle='None')
        dec_handles.append(handle)
        dec_labels.append(f'Layer {l+1}')

    ax_dec.legend(dec_handles, dec_labels, framealpha=0.9, fontsize=12, title_fontsize=13)

    # ---------------- COLORBAR ENCODER/DECODER ----------------
    norm = Normalize(vmin=start_epoch, vmax=end_epoch)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax_cb, orientation='vertical')
    cbar.set_ticks(epoch_range)
    cbar.set_ticklabels(epoch_range)
    cbar.set_label("Epoch", fontsize=14)




    # ---------------- GLOBAL: INPUT/LATENT/OUTPUT ----------------

    mi_input_latent  = mi_history.input_latent
    mi_latent_output = mi_history.latent_output
    # --- Extract MI values for selected epochs ---
    X_vals = [mi_input_latent[ep] for ep in epoch_range]
    Y_vals = [mi_latent_output[ep] for ep in epoch_range]

    cmap = plt.get_cmap("Greens")
    colors = [cmap(i / max(1, len(epoch_range) - 1)) for i in range(len(epoch_range))]

    fig, ax = plt.subplots(figsize=(6, 5))

    for i, ep in enumerate(epoch_range):
        ax.scatter(
            X_vals[i], Y_vals[i],
            s=80,
            marker='o',
            facecolors=colors[i],
            edgecolors='black',
            linewidths=0.6,
            label=f"Epoch {ep}"
        )

    # Labels and title
    ax.set_xlabel("MI(Input; Latent)", fontsize=14)
    ax.set_ylabel("MI(Latent; Output)", fontsize=14)
    ax.set_title(f"Global Mutual Information {title_suffix}", fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)

    # --- COLORBAR GLOBAL INPUT/OUTPUT ---
    norm = Normalize(vmin=start_epoch, vmax=end_epoch)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Epoch", fontsize=14)
    cbar.set_ticks(epoch_range)
    cbar.set_ticklabels(epoch_range)

    plt.tight_layout()
    plt.show()

    #plt.show()


#*****************************************************************************************************************
#*****************************************************************************************************************


def read_MI_hist(filename):
    try:
        with open(filename, mode="rb") as f:
            MI_histories = pickle.load(f)
            return MI_histories
    except FileNotFoundError:
        print(f"Errore: {filename} non esiste")


#*****************************************************************************************************************
#*****************************************************************************************************************

def ShowSomeImages(model, testDataset, device):

   model.eval()
   fig, axs = plt.subplots(5, 2, figsize=(6, 12))

   for i in range(5):
      img, _ = random.choice(testDataset)

      x = img.unsqueeze(0).to(device)

      with torch.no_grad():
         recon, _, _, _ = model(x)

      original = img.cpu().squeeze().numpy()
      reconstructed = recon.cpu().squeeze().numpy().reshape(28, 28)

      axs[i, 0].imshow(original, cmap="gist_gray")
      axs[i, 0].set_title("Original")
      axs[i, 0].set_xticks([])
      axs[i, 0].set_yticks([])

      axs[i, 1].imshow(reconstructed, cmap="gist_gray")
      axs[i, 1].set_title("Reconstruction")
      axs[i, 1].set_xticks([])
      axs[i, 1].set_yticks([])

   plt.tight_layout()
   plt.show()


#*****************************************************************************************************************
#*****************************************************************************************************************

def AnimateActivations(epochsActivations, layer):
    
   fig, ax = plt.subplots(figsize=(8, 5))
   
   max_count = 0
   for data in epochsActivations[f'Layer_{layer}']:
      counts, _ = np.histogram(data, bins=100, range=(-1.1, 1.1))
      if counts.max() > max_count:
         max_count = counts.max()
         
   #ax.set_xlim(-1.2, 1.2)
   ax.set_ylim(0, max_count * 1.1) # Add 10% headroom
   
   # Update the function (called for every frame)
   def update(frame):
      ax.clear() 
      # Get data for this epoch
      data = epochsActivations[f'Layer_{layer}'][frame]
      
      # Redraw settings
      ax.hist(data, bins=100, range=(-1.1, 1.1), color='purple', alpha=0.7, edgecolor='none')
      
      # Re-apply limits and labels (clearing wipes them)
      #ax.set_xlim(-1.2, 1.2)
      ax.set_ylim(0, max_count * 1.1)
      ax.set_title(f"Activation Distribution of layer {layer}: Epoch {frame}", fontsize=14)
      ax.set_xlabel("Activation Value (Tanh)", fontsize=12)
      ax.set_ylabel("Count", fontsize=12)
      ax.grid(True, alpha=0.3)

   # Create Animation
   # interval=100 means 100ms per frame (10 fps)
   anim = FuncAnimation(fig, update, frames=len(epochsActivations[f'Layer_{layer}']), interval=200)
   
   plt.close()
   
   # Return the HTML object
   return HTML(anim.to_jshtml())


#*****************************************************************************************************************
#*****************************************************************************************************************

# To generate content index for notebook
def generate_index(file="Restyle.ipynb", title="Index"):
    with open(file, "r", encoding="utf-8") as f:
        nb = json.load(f)

    headers = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            for line in cell["source"]:
                m = re.match(r'^(#+)\s+(.*)', line)
                if m:
                    level = len(m.group(1))
                    text = m.group(2).strip()

                    anchor = re.sub(r'[^a-zA-Z0-9 -]', '', text)
                    anchor = anchor.replace(" ", "-")

                    headers.append((level, text, anchor))

    # HTML style
    md = f"""
<h1 style="color:black; font-size: 38px; font-weight: 700; margin-bottom: 5px;">
    {title}
</h1>

<hr style="border: 1px solid #000;">

<p style="font-size: 18px; color:black; margin-top: 10px;">

</p>
"""

    for level, text, anchor in headers:
        indent = "&nbsp;" * (level - 1) * 6
        size = 20 if level == 1 else 17
        weight = "700" if level == 1 else "500"
        bullet = "•" if level == 1 else "◦"

        md += (
            f'{indent}<span style="font-size:{size}px; color:black; font-weight:{weight};">'
            f'{bullet} <a href="#{anchor}" style="color:black; text-decoration:none;">{text}</a>'
            f'</span><br>\n'
        )

    md += '<br>\n'
    md += '<hr style="border: 1px solid #000;">\n'
    # md += '<hr style="border: 1px solid #000;">\n'
    md += '<br>\n'
    # md += '<br>\n'

    display(Markdown(md))
