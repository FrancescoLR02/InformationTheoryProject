# ============================
# Scientific Computing
# ============================
import numpy as np

# ============================
# Machine Learning Tools
# ============================
from sklearn.neighbors import NearestNeighbors

# ============================
# Special Functions
# ============================
from scipy.special import digamma

# ============================
# Visualization
# ============================
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


#*****************************************************************************************************************
#*******************************ACTIVATION RECORDER
#*****************************************************************************************************************

class ActivationRecorder:

    def __init__(self):
        self.activations = {}     # Current activations (last epoch)
        self.history = {}         # History: history[epoch][layer_name] -> array
        self.is_recording = True  # switch ON/OFF to record or not during the forward pass in the model

    # ----------------------------- HOOKING SYSTEM ----------------------------------------

    def activate_recording(self, state: bool): # to set the recording state (ON or OFF)
        self.is_recording = state

    def hook(self, name):
        def _hook(module, inputs, output):
            if self.is_recording:
                self.activations[name] = output.detach().cpu().numpy()
        return _hook

    def hook(self, name):
        def _hook(module, inputs, output):
            self.activations[name] = output.detach().cpu().numpy()
        return _hook

    # ----------------------------- SETTING THE REGISTER WHEN CREATED  ----------------------------------------

    def InitialRegister(self, model):
        self.activations = {}
        self.history = {}

        # Register hooks for Input
        model.InputSpace.register_forward_hook(self.hook("input_space"))
        if model.Label is not None: 
            model.Label.register_forward_hook(self.hook("label"))
        
        # Register hooks for Encoder
        for i, layer in enumerate(model.Encoder):
            layer.register_forward_hook(self.hook(f"encoder_layer_{i+1}"))

        # Register hooks for Decoder
        for i, layer in enumerate(model.Decoder):
            layer.register_forward_hook(self.hook(f"decoder_layer_{i+1}"))

        # Register Latent and Output
        model.LatentSpace.register_forward_hook(self.hook("latent_space"))
        model.LatentQuant.register_forward_hook(self.hook("latent_quant"))
        model.OutputSpace.register_forward_hook(self.hook("output_space"))

    def save_epoch(self, epoch):
        self.history[epoch] = {k: v.copy() for k, v in self.activations.items()}

    # ----------------------------------GETTERS FOR ACTIVATIONS (last epoch, layer, epoch)------------------------------------------------------

    def get(self, name):
        return self.activations[name]

    # output: an array (indexed by epoch) composed by the activations during training for a specific layer
    def get_layer(self, layer_name):
        
        result = []
        for epoch in self.history.keys():

            if layer_name in self.history[epoch]:
                result.append(self.history[epoch][layer_name])
            else:
                raise ValueError(f"Layer {layer_name} not found in the model")

        return result

    # output: all activations (for all layers) for a specific epoch
    def get_epoch(self, epoch):

        if epoch not in self.history:
            raise ValueError(f"Epoch {epoch} not found in history")

        return self.history[epoch]

    # ----------------------------------SELECT MAIN LAYERS BY LABEL------------------------------------------------------

    def select_by_label(self, target_label):

        # Extract labels
        labels = self.activations["label"]  # shape (N,)

        # Boolean mask
        mask = (labels == target_label)

        # Indices of matching samples
        idx = np.where(mask)[0]

        # Layers to extract
        layers_of_interest = ["input_space", "latent_quant", "output_space"]

        # Output dictionary
        selected = {"idx": idx}

        # Filter labels too
        selected["label"] = labels[idx]

        # Filter each layer
        for layer in layers_of_interest:
            if layer not in self.activations:
                raise ValueError(f"Layer '{layer}' not found in activations")
            selected[layer] = self.activations[layer][idx]

        return selected

    # ----------------------------------PLOT LATENT BIT FREQUENCY BY LABEL------------------------------------------------------

    def plot_bit_freq(self, labels=None):

        # Normalize input
        if labels is None:
            labels = list(range(10))
        elif isinstance(labels, int):
            labels = [labels]

        # Determine latent dimension from any label
        sample = self.select_by_label(labels[0])
        latentDim = sample["latent_quant"].shape[1]

        # Force exactly 2 plots per label
        num_plots = 2
        half = latentDim // 2

        # Color map for different labels
        cmap = plt.get_cmap("tab10")
        label_colors = {lab: cmap(i % 10) for i, lab in enumerate(labels)}

        # Prepare figure
        fig, axes = plt.subplots(len(labels), num_plots,
                                figsize=(num_plots * 9, len(labels) * 4),
                                squeeze=False)

        fig.suptitle("Bit=1 frequency per latent neuron", fontsize=26)

        # Iterate over labels
        for row, digit in enumerate(labels):

            selected = self.select_by_label(digit)
            latent = selected["latent_quant"]

            # Boolean mask: True where bit == 1
            bit_is_one = (latent == 1)

            # Frequency of bit=1 per neuron
            freq = bit_is_one.mean(axis=0)

            # --- LEFT subplot: first half ---
            ax_left = axes[row, 0]
            ax_left.bar(np.arange(0, half), freq[:half],
                        color=label_colors[digit], edgecolor="black")
            ax_left.set_ylim(0, 1)
            ax_left.grid(axis="y", alpha=0.3)
            ax_left.set_xlabel("Neuron index", fontsize=14)
            ax_left.set_ylabel("Fraction bit = 1", fontsize=14)

            # --- RIGHT subplot: second half ---
            ax_right = axes[row, 1]
            ax_right.bar(np.arange(half, latentDim), freq[half:],
                        color=label_colors[digit], edgecolor="black")
            ax_right.set_ylim(0, 1)
            ax_right.grid(axis="y", alpha=0.3)
            ax_right.set_xlabel("Neuron index", fontsize=14)

            # --- Title for the whole row (centered above both plots) ---
            # Use the left axis but center the title across the row
            ax_left.set_title(f"Label {digit}", fontsize=20, weight="bold", color=label_colors[digit], pad=25, loc='right')

            # Titles for individual subplots (optional)
            ax_left.text(0.5, 1.02, f"Neurons 0–{half-1}",
                        transform=ax_left.transAxes, ha='center', fontsize=14)
            ax_right.text(0.5, 1.02, f"Neurons {half}–{latentDim-1}",
                        transform=ax_right.transAxes, ha='center', fontsize=14)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


    # ----------------------------------ANIMATION FOR A LAYER---------------------------------------------------------

    def AnimateActivationLayers(self, layer_name, num_bins=100, Debug=False):
        """
        Create an animation showing how the activation distribution of a specific layer
        evolves across epochs. Uses get_layer(layer_name) to retrieve the data.
        """

        # Retrieve activations for this layer across epochs
        layer_epochs = self.get_layer(layer_name)

        # Prepare figure
        fig, ax = plt.subplots(figsize=(8, 5))

        # # Compute max histogram height across all epochs (for stable y‑axis)
        # max_count = 0
        # for data in layer_epochs:
        #     counts, _ = np.histogram(data, bins=100, range=(-1.1, 1.1))
        #     max_count = max(max_count, counts.max())


        # Compute global histogram height and global min/max activation values
        # setting starting values
        max_count = 0
        global_min = 0
        global_max = 1
        quantile10 = 0.1
        quantile90 = 0.9

        for epoch, data in enumerate(layer_epochs): # data are the activations for a specific epochs
            flat = data.flatten() # data is and array of shape (num_images,num_neurons_layer), need flatten to aggregate all activations
            global_min = min(global_min, flat.min())
            global_max = max(global_max, flat.max())
            quantile10 = min(quantile10, np.quantile(flat, 0.10))
            quantile90 = max(quantile90, np.quantile(flat, 0.90))


            counts, bin_edges = np.histogram(flat, bins=num_bins, range=(quantile10, quantile90))

            if Debug:
                print(f"Epoch: {epoch+1} \t\t\t|\t Activations shape (images,layer size): {data.shape} \t|\t Max hist count:{counts.max()}")
                print(f"Min act: {flat.min():.1f} \t\t|\t Max act: {flat.max():.1f}")
                print(f"10% quantile act: {np.quantile(flat, 0.10):.1f} \t|\t 90% quantile act: {np.quantile(flat, 0.90):.1f}")

            max_count = max(max_count, counts.max())


        ax.set_xlim(quantile10, quantile90)
        ax.set_ylim(0, max_count * 1.2) # extra 20% on y axis for visualization


        # Update function for animation frames
        def update(frame):
            ax.clear()

            data = layer_epochs[frame].flatten()
            ax.hist(data, bins=num_bins, range=(quantile10, quantile90), alpha=0.7, edgecolor='none')

            ax.set_xlim(quantile10, quantile90)
            ax.set_ylim(0, max_count * 1.2) # extra 20% on y axis for visualization

            ax.set_title(f"Activation Distribution - {layer_name} (Epoch {frame+1})", fontsize=14)
            ax.set_xlabel("Activation Value", fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.grid(True, alpha=0.3)

        # Build animation
        anim = FuncAnimation(fig, update, frames=len(layer_epochs), interval=500) #interval is just for frame/s setting

        plt.close()
        return HTML(anim.to_jshtml())

    # ----------------------------------  PLOT DISTANCES BETWEEN NEURONS/IN LAYERS ---------------------------------------------------------

    def plot_activ_distances(self, part="encoder", layer=1, neuron=None, how_many_epoch=5, bins=60):

        # _____________________ CHECK EPOCHS AVAILABLE ________________________
        available_epochs = sorted([ep for ep in self.history.keys() if ep != 1]) # exclude the first epoch which is always problematic

        total_epochs = len(available_epochs)

        if total_epochs < how_many_epoch:
            raise ValueError(f"Requested {how_many_epoch} epochs but only {total_epochs} available.")

        # Always include first and last epoch
        if how_many_epoch == 2:
            selected_epochs = [available_epochs[0], available_epochs[-1]]
        else:
            # Compute equidistant indices
            idxs = np.linspace(0, total_epochs - 1, how_many_epoch, dtype=int)
            selected_epochs = [available_epochs[i] for i in idxs]

        # ______________________________ BUILD THE KEY ____________________________
        if part == "encoder":
            key = f"encoder_layer_{layer}"
        elif part == "decoder":
            key = f"decoder_layer_{layer}"
        elif part == "latent":
            key = "latent_space"
        elif part == "output":
            key = "output_space"
        else:
            raise ValueError("Part must be 'encoder', 'decoder', 'latent', or 'output'")

        # _____________________________ PREPARE PLOT ___________________________
        fig, ax = plt.subplots(figsize=(8, 5))

        cmap = plt.get_cmap("viridis")
        colors = [cmap(i / max(1, how_many_epoch - 1)) for i in range(how_many_epoch)]

        # ________________________ LOOP OVER SELECTED EPOCHS __________________________
        for idx, ep in enumerate(selected_epochs):

            data_dict = self.history[ep]
            if key not in data_dict:
                raise ValueError(f"Key {key} not found in epoch {ep}")

            X = data_dict[key]

            # If a specific neuron is selected
            if neuron is not None:
                X = X[:, neuron:neuron+1]

            # ____________________________ DISTANCES ____________________________
            X_sq = np.sum(X**2, axis=1, keepdims=True)
            dists_sq = X_sq + X_sq.T - 2 * X @ X.T
            dists = np.sqrt(np.maximum(dists_sq, 0))
            tri_idx = np.triu_indices_from(dists, k=1)
            D = dists[tri_idx]

            # _____________________________ PLOT ___________________________
            ax.hist(D, bins=bins, density=True, alpha=0.5, color=colors[idx], edgecolor='black', label=f"Epoch {ep}")

        # __________________________ TITLES & LABELS ___________________________
        title_str = f"{part.upper()}: "
        if part in ["encoder", "decoder"]:
            title_str += f"L{layer}"
        if neuron is not None:
            title_str += f"- N{neuron}"

        ax.set_title(f"{title_str}", fontsize=14)
        ax.set_xlabel("Pairwise Distance", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.show()

    
    def plot_error(self):

        if len(self.history) == 0:
            raise ValueError("No history found. Did you call save_epoch during training?")

        epochs = sorted(self.history.keys())
        mse_values = []

        for ep in epochs:
            data = self.history[ep]

            if "input_space" not in data or "output_space" not in data:
                raise ValueError(f"Missing input/output activations at epoch {ep}")

            inp = data["input_space"]      # shape (N, D)
            out = data["output_space"]     # shape (N, D)

            # Vectorized MSE over all samples and all pixels
            mse = np.mean((inp - out) ** 2)
            mse_values.append(mse)

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, mse_values, marker='o', linewidth=2, color="steelblue")
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("Reconstruction MSE", fontsize=14)
        plt.title("Reconstruction Error per Epoch", fontsize=18, weight="bold")
        plt.grid(alpha=0.3)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.show()

        #return epochs, mse_values



    # ---------------------------------REPRESENTARION (for print)-------------------------------------------------------

    def __repr__(self):
        last_epoch = max(self.history.keys())

        return (
            "ActivationRecorder(\n"
            f"  activations (last epoch): { list(self.activations.keys()) }\n"
            f"  history (activ for all epoch): {last_epoch} epochs\n"
            f"  methods: get_epoch(epoch), get_layer(layer_name)\n"
            ")"
        )