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
        model(inputs, label) # Foward pass to get the activation value in RecorderActivat.activations

    RecorderActivat.save_epoch(epoch) # here we stored activation!

    # ------------------------ CALCULATE & STORE MUT.INFO ------------------------------

    X = inputs.view(inputs.size(0), -1).cpu().numpy()
    Z = RecorderActivat.get("latent_space")
    Y = RecorderActivat.get("output_space")
        
    mi = {
        "encoder": [],
        "decoder": [],
        "input_latent": None,
        "latent_output": None
    }

    # mi_method for each pair of layer
    method_in_h  = mi_estimator.method[0]
    method_h_z   = mi_estimator.method[1]
    method_in_z  = mi_estimator.method[2]
    method_z_h   = mi_estimator.method[3]
    method_h_out = mi_estimator.method[4]
    method_z_out = mi_estimator.method[5]

    # recall mi_estimator.method = ["in_h", "h_z" ,"in_z", "z_h", "h_out", "z_out"]
    
    # Encoder Layers
    for i in range(len(model.Encoder)):
        layer_name = f"encoder_layer_{i+1}"
        h = RecorderActivat.get(layer_name)
        mi["encoder"].append((
            mi_estimator.mutual_information(X, h, method_in_h), # I(Input, Layer)
            mi_estimator.mutual_information(h, Z, method_h_z)  # I(Layer, Latent)
        ))

    # Decoder Layers
    for i in range(len(model.Decoder)):
        layer_name = f"decoder_layer_{i+1}"
        h = RecorderActivat.get(layer_name)
        mi["decoder"].append((
            mi_estimator.mutual_information(Z, h, method_z_h), # I(Latent, Layer)
            mi_estimator.mutual_information(h, Y, method_h_out)  # I(Layer, Output)
        ))

    mi["input_latent"]  = mi_estimator.mutual_information(X, Z, method_in_z)  # I(Input, Latent)
    mi["latent_output"] = mi_estimator.mutual_information(Z, Y, method_z_out) # I(Latent, Output)
    
    # Store the mi calculated
    mi_history.append(mi)

#*****************************************************************************************************************
#*****************************************************************************************************************


def PlotInfoPlane(mi_history, title_suffix="", suptitle="", start_epoch=1, end_epoch=-1, Step=5, whichplot="enc/dec"):

    mi_history_encoder = mi_history.encoder
    mi_history_decoder = mi_history.decoder

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[6, 6, 0.2], wspace=0.3)

    if suptitle != "":
        fig.suptitle(suptitle, size=16, weight="bold")

    # Create axes
    ax_enc = fig.add_subplot(gs[0, 0])
    ax_dec = fig.add_subplot(gs[0, 1])
    ax_cb  = fig.add_subplot(gs[0, 2])

    # Hide everything by default
    ax_enc.set_visible(False)
    ax_dec.set_visible(False)
    ax_cb.set_visible(False)

    # Activate depending on whichplot
    if whichplot in ("enc/dec", "all"):
        ax_enc.set_visible(True)
        ax_dec.set_visible(True)

    # Activate ax_cb ONLY for enc/dec or all
    if whichplot in ("enc/dec", "all"):
        ax_cb.set_visible(True)


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

    if whichplot in ("enc/dec", "all"):
        # ---------------- ENCODER ----------------
        ax = ax_enc

        for ep_idx, ep in enumerate(epoch_range):
            x = [mi_history_encoder[ep][l][0] for l in range(n_layers)]
            y = [mi_history_encoder[ep][l][1] for l in range(n_layers)]

            ax.plot(x, y, linestyle='-', color=colors[ep_idx],
                    alpha=0.6, linewidth=2)

            for l in range(n_layers):
                marker = encoder_markers[l % len(encoder_markers)]
                ax.scatter(x[l], y[l], facecolors=[colors[ep_idx]], edgecolors='black', s=60, marker=marker, linewidths=0.6, zorder=3)

        ax.set_xlabel("I(Layer; Input)", fontsize=14)
        ax.set_ylabel("I(Layer; Latent Z)", fontsize=14)
        ax.set_title(f"Encoder Information Plane {title_suffix}", fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)

        enc_handles = []
        enc_labels = []
        for l in range(n_layers):
            mk = encoder_markers[l % len(encoder_markers)]
            handle = Line2D([0], [0], marker=mk, color='black', markerfacecolor='black', markeredgecolor='black', markersize=10, linestyle='None')
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
                ax.scatter(x[l], y[l], facecolors=[colors[ep_idx]], edgecolors='black', s=60, marker=marker, linewidths=0.6, zorder=3)

        ax.set_xlabel("I(Layer; Latent Z)", fontsize=14)
        ax.set_ylabel("I(Layer; Output)", fontsize=14)
        ax.set_title(f"Decoder Information Plane {title_suffix}", fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)

        dec_handles = []
        dec_labels = []
        for l in range(n_layers):
            mk = decoder_markers[l % len(decoder_markers)]
            handle = Line2D([0], [0], marker=mk, color='black', markerfacecolor='black', markeredgecolor='black', markersize=10, linestyle='None')
            dec_handles.append(handle)
            dec_labels.append(f'Layer {l+1}')

        ax_dec.legend(dec_handles, dec_labels, framealpha=0.9, fontsize=12, title_fontsize=13)

        # ---------------- COLORBAR ENCODER/DECODER ----------------
        norm = Normalize(vmin=start_epoch + 1, vmax=end_epoch + 1)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = fig.colorbar(sm, cax=ax_cb, orientation='vertical')
        cbar.set_ticks(np.array(epoch_range)+1)
        cbar.set_ticklabels(np.array(epoch_range)+1)
        cbar.set_label("Epoch", fontsize=14)

    # --------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------

    # ---------------- GLOBAL: INPUT/LATENT/OUTPUT ----------------

    if whichplot in ("in/out", "all"):
        mi_input_latent  = mi_history.input_latent
        mi_latent_output = mi_history.latent_output
        # --- Extract MI values for selected epochs ---
        X_vals = [mi_input_latent[ep] for ep in epoch_range]
        Y_vals = [mi_latent_output[ep] for ep in epoch_range]

        cmap = plt.get_cmap("Greens")
        colors = [cmap(i / max(1, len(epoch_range) - 1)) for i in range(len(epoch_range))]

        fig, ax = plt.subplots(figsize=(6, 5))

        for i, ep in enumerate(epoch_range):
            ax.scatter( X_vals[i], Y_vals[i], s=80, marker='o', facecolors=colors[i], edgecolors='black', linewidths=0.6, label=f"Epoch {ep}" )

        # Labels and title
        ax.set_xlabel("MI(Input; Latent)", fontsize=14)
        ax.set_ylabel("MI(Latent; Output)", fontsize=14)
        ax.set_title(f"Global Mutual Information {title_suffix}", fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)

        # --- COLORBAR GLOBAL INPUT/OUTPUT ---
        norm = Normalize(vmin=start_epoch + 1, vmax=end_epoch + 1)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Epoch", fontsize=14)
        cbar.set_ticks(np.array(epoch_range)+1)
        cbar.set_ticklabels(np.array(epoch_range)+1)

        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------

    # ---------------- GLOBAL: INPUT/LATENT/OUTPUT ----------------

    if whichplot in ("in/out", "all"):
        mi_input_latent  = mi_history.input_latent
        mi_latent_output = mi_history.latent_output
        # --- Extract MI values for selected epochs ---
        X_vals = [mi_input_latent[ep] for ep in epoch_range]
        Y_vals = [mi_latent_output[ep] for ep in epoch_range]

        cmap = plt.get_cmap("Greens")
        colors = [cmap(i / max(1, len(epoch_range) - 1)) for i in range(len(epoch_range))]

        fig, ax = plt.subplots(figsize=(6, 5))

        for i, ep in enumerate(epoch_range):
            ax.scatter( X_vals[i], Y_vals[i], s=80, marker='o', facecolors=colors[i], edgecolors='black', linewidths=0.6, label=f"Epoch {ep}" )

        # Labels and title
        ax.set_xlabel("MI(Input; Latent)", fontsize=14)
        ax.set_ylabel("MI(Latent; Output)", fontsize=14)
        ax.set_title(f"Global Mutual Information {title_suffix}", fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)

        # --- COLORBAR GLOBAL INPUT/OUTPUT ---
        norm = Normalize(vmin=start_epoch + 1, vmax=end_epoch + 1)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Epoch", fontsize=14)
        cbar.set_ticks(np.array(epoch_range)+1)
        cbar.set_ticklabels(np.array(epoch_range)+1)

        plt.tight_layout()
        plt.show()

#*****************************************************************************************************************
#*****************************************************************************************************************

def ShowSomeImages(model, testDataset, device, howmany=5):

   model.eval()
   fig, axs = plt.subplots(howmany, 2, figsize=(4, howmany*2))

   for i in range(howmany):
    img, label = random.choice(testDataset)

    x = img.unsqueeze(0).to(device)

    with torch.no_grad():
         recon, _, _, _, _ = model(x, label)

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

#*****************************************************************************************************************
#*****************************************************************************************************************