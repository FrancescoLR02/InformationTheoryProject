import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma


class ActivationRecorder:
    def __init__(self):
        self.activations = {} # Current activations
        self.history = {}     # History: history[epoch][layer_name] -> array

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

    def get(self, name):
        return self.activations[name]

    def save_epoch(self, epoch):
        self.history[epoch] = {k: v.copy() for k, v in self.activations.items()}




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



class MI_Estimator:
    def __init__(self, method, sigma=1.0, n_neig=3):
        self.method = method
        self.sigma  = sigma
        self.n_neig = n_neig
    
    # ---------------- KDE METHOD ----------------
    def density(self, data):
        N, d = data.shape
        
        data_sq = np.sum(data**2, axis=1, keepdims=True)
        dists_sq = data_sq + data_sq.T - 2 * data @ data.T
        
        #sigma_scaled = self.sigma  # self.sigma * np.sqrt(d) (Scale sigma by dimension) ***********************IMP**********************
        sigma_scaled = self.sigma * np.sqrt(d) 
        kernel = np.exp(-dists_sq / (2 * sigma_scaled**2))
        return np.mean(kernel, axis=1)

    def entropy_kde(self, data):
        rho = self.density(data)
        return -np.mean(np.log(rho + 1e-10))

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

    #The idea is to compute:
    #I(X;Z) = H(Z) - H(Z|X)
    def LatentMutualInformation(self, mean, logVar):
        N, d = mean.shape
        eps = 1e-10
        
        #Calculate Conditional Entropy H(Z|X)
        var = np.exp(logVar) + eps
        H_ZgivenX = 0.5 * np.mean(np.sum(np.log(2 * np.pi * np.e * var), axis=1))
        
        # Calculate empirical variance of the means to determine bandwidth
        data_var = np.var(mean, axis=0).mean()
        if data_var < eps: data_var = 1.0
        
        # Silverman's Rule for bandwidth (h)
        bandwidth_sq = data_var * (N ** (-2 / (d + 4))) 
        
        # Compute distances (squared Euclidean)
        squareMean = np.sum(mean**2, axis=1, keepdims=True)
        dists = squareMean + squareMean.T - 2 * mean @ mean.T
        logKernels = -dists / (2 * bandwidth_sq)
    
        log_pdf_sum = np.log(np.sum(np.exp(logKernels), axis=1, keepdims=True) + eps)
        
        # H(Z) = -E[log q(z)]
        normalization = 0.5 * d * np.log(2 * np.pi * bandwidth_sq) + np.log(N)
        h_Z = -np.mean(log_pdf_sum - normalization)

        return h_Z - H_ZgivenX
    
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