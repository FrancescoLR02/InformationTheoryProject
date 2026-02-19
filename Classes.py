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
#*****************************************************************************************************************

class ActivationRecorder:
    def __init__(self):
        self.activations = {}  # Current activations (last epoch)
        self.history = {}      # History: history[epoch][layer_name] -> array

    def hook(self, name):
        def _hook(module, inputs, output):
            self.activations[name] = output.detach().cpu().numpy()
        return _hook

    def InitialRegister(self, model):
        self.activations = {}
        self.history = {}

        # Register hooks for Input
        model.InputSpace.register_forward_hook(self.hook("input_space"))
        
        # Register hooks for Encoder
        for i, layer in enumerate(model.Encoder):
            layer.register_forward_hook(self.hook(f"encoder_layer_{i+1}"))

        # Register hooks for Decoder
        for i, layer in enumerate(model.Decoder):
            layer.register_forward_hook(self.hook(f"decoder_layer_{i+1}"))

        # Register Latent and Output
        model.LatentSpace.register_forward_hook(self.hook("latent_space"))
        model.OutputSpace.register_forward_hook(self.hook("output_space"))

    def save_epoch(self, epoch):
        self.history[epoch] = {k: v.copy() for k, v in self.activations.items()}

    # --------------------------------------------------------------------------------------------------------
    def get(self, name):
        return self.activations[name]

    # output: all activations (for all layers) for a specific epoch
    def get_epoch(self, epoch):
        if epoch not in self.history:
            raise ValueError(f"Epoch {epoch} not found in history")
        return self.history[epoch]

    # output: an array (indexed by epoch) composed by the activations during training for a specific layer
    def get_layer(self, layer_name):
        result = []
        for epoch in self.history.keys():
            if layer_name in self.history[epoch]:
                result.append(self.history[epoch][layer_name])
            else:
                raise ValueError(f"Layer {layer_name} not found in the model")

        return result

    # --------------------------------------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------------------------------------
    # to get info when this class is called with print()
    def __repr__(self):
        last_epoch = max(self.history.keys())

        return (
            "ActivationRecorder(\n"
            f"  activations (last epoch): { list(self.activations.keys()) }\n"
            f"  history (activ for all epoch): {last_epoch} epochs\n"
            f"  methods: get_epoch(epoch), get_layer(layer_name)\n"
            ")"
        )



#*****************************************************************************************************************
#*****************************************************************************************************************

class MI_History:
    def __init__(self):
        self.encoder = []
        self.decoder = []
        self.input_latent = []
        self.latent_output = []
    
    def append(self, mi_dict):
        self.encoder.append(mi_dict["encoder"])
        self.decoder.append(mi_dict["decoder"])
        self.input_latent.append(mi_dict["input_latent"])
        self.latent_output.append(mi_dict["latent_output"])

    def show(self, what="all"):
        what = what.lower().strip()
        
        if not self.input_latent:
            print("No history available.")
            return

        print(f"{'='*20} MI HISTORY ({what}) {'='*20}")
        
        for epoch in range(len(self.input_latent)):
            print(f"EPOCH {epoch + 1}")
            
            # --- Global Metrics ---
            if what == "global" or what == "all":
                mi_xz = self.input_latent[epoch]
                mi_zy = self.latent_output[epoch]
                print(f"  [Global] I(Input, Z): {mi_xz:.3f} | I(Z, Output): {mi_zy:.3f}")

            # --- Encoder Layers ---
            if what == "encoder" or what == "all":
                print("  [Encoder]")
                for i, (mi_in, mi_lat) in enumerate(self.encoder[epoch]):
                    print(f"    Layer {i+1}: I(Input, L)={mi_in:.3f} | I(L, Z)={mi_lat:.3f}")
            
            # --- Decoder Layers ---
            if what == "decoder" or what == "all":
                print("  [Decoder]")
                for i, (mi_lat, mi_out) in enumerate(self.decoder[epoch]):
                    print(f"    Layer {i+1}: I(Z, L)={mi_lat:.3f}     | I(L, Output)={mi_out:.3f}")
            
            print("-" * 50)
        
    def __repr__(self):
        return (
            "MI_History(\n"
            "  encoder: mutual information for encoder layers\n"
            "  decoder: mutual information for decoder layers\n"
            "  input_latent: I(Input, Z) values\n"
            "  latent_output: I(Z, Output) values\n"
            "  show() available with: all, global, encoder, decoder\n"
            ")"
        )


#*****************************************************************************************************************
#*****************************************************************************************************************

class MI_Estimator:
    def __init__(self, method, sigma=1.0, n_neig=3):
        self.method = method
        self.sigma  = sigma
        self.n_neig = n_neig

    def mutual_information(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)  
        # Reshape 1D arrays
        if X.ndim == 1: X = X.reshape(-1, 1)
        if Y.ndim == 1: Y = Y.reshape(-1, 1)

        if self.method == "kde":
            HX = self.entropy_kde(X)
            HY = self.entropy_kde(Y)
            HXY = self.entropy_kde(np.concatenate([X, Y], axis=1))
            return HX + HY - HXY

        if self.method == "kraskov":
            return self.kraskov_estimation(X, Y)
    
    # ---------------- KDE METHOD ----------------

    def entropy_kde(self, data):
        rho = self.density(data)
        return - float( np.mean(np.log(rho + 1e-10)) )

    def density(self, data):
        N, d = data.shape
        
        data_sq = np.sum(data**2, axis=1, keepdims=True)
        dists_sq = data_sq + data_sq.T - 2 * data @ data.T
        
        #sigma_scaled = self.sigma  # self.sigma * np.sqrt(d) (Scale sigma by dimension) ***********************IMP**********************
        sigma_scaled = self.sigma * np.sqrt(d) 
        kernel = np.exp(-dists_sq / (2 * sigma_scaled**2))
        return np.mean(kernel, axis=1)

    # ---------------- KRASKOV METHOD ---------------- # MAI TESTATO DA VEDERE!!!!!
    def kraskov_estimation(self, X, Y):
        # Add tiny noise to break ties (crucial for KSG)
        X = X + 1e-10 * np.random.rand(*X.shape)
        Y = Y + 1e-10 * np.random.rand(*Y.shape)
        
        N = X.shape[0]
        XY = np.hstack([X, Y])
        
        # 1. Find k-nearest neighbors in Joint Space (max norm)
        knn = NearestNeighbors(n_neighbors=self.n_neig + 1, metric='chebyshev')
        knn.fit(XY)
        dists, _ = knn.kneighbors(XY)
        
        # Distance to the k-th neighbor
        radii = dists[:, -1]
        
        # 2. Count neighbors in marginal spaces within those radii
        # We need efficient search, so we fit new trees
        knn_x = NearestNeighbors(metric='chebyshev').fit(X)
        knn_y = NearestNeighbors(metric='chebyshev').fit(Y)
        
        # radius_neighbors returns array of arrays of indices
        nx_indices = knn_x.radius_neighbors(X, radius=radii, return_distance=False)
        ny_indices = knn_y.radius_neighbors(Y, radius=radii, return_distance=False)
        
        # Count lengths (subtract 1 because query point is included)
        nx = np.array([len(i) - 1 for i in nx_indices])
        ny = np.array([len(i) - 1 for i in ny_indices])
        
        # 3. KSG Formula
        # MI = psi(k) + psi(N) - <psi(nx+1) + psi(ny+1)>
        mi = (digamma(self.n_neig) + digamma(N) - 
              np.mean(digamma(nx + 1) + digamma(ny + 1)))
              
        return max(0, mi)