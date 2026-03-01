import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib import cm





def PlotInfoPlane(mi_history, title_suffix="", suptitle="", start_epoch=1, end_epoch=-1, Step=5, whichplot="enc/dec"):

    mi_history_encoder = mi_history.encoder
    mi_history_decoder = mi_history.decoder

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[6, 6, 0.2], wspace=0.3)

    if suptitle != "":
        fig.suptitle(suptitle, size=20)#, weight="bold")

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
        ticks = (np.array(epoch_range) + 1)[::5]   # tick every 5
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticks)
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
        ticks = (np.array(epoch_range) + 1)[::5]   # tick every 5
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(ticks)

        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------------------------