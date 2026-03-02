# ============================
# Scientific Computing
# ============================
import numpy as np

# ============================
# Machine Learning Tools
# ============================
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
#from sklearn.neighbors import NearestNeighbors

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
#*******************************MI ESTIMATOR CALCULATOR
#*****************************************************************************************************************

class MI_Estimator:


    def __init__(self, method, sigma=1.0, n_neig=3, default="kde"):
        self.sigma  = sigma
        self.n_neig = n_neig
        #self.method = self.create_method_array(method, default)
        # for mi methods self.method = ["in_h", "h_z" ,"in_z", "z_h", "h_out", "z_out"]

    # def create_method_array(self, method, default):
    #     # if only a string is passed it is replicated 6 times
    #     if isinstance(method, str):
    #         method_array = [method] * 6
    #         return method_array
        
    #     # if a list of string is passed we fill it to 6 methods with the default type
    #     if isinstance(method, (list, tuple)):
    #         method_array = list(method) # convert tuple eventually into list

    #         while len(method_array) < 6:
    #             method_array.append(default)
    #         return method_array[:6] # always a list of 6 strings is returned



    # ------------------------- MUTUAL INFO -------------------------

    def mutual_information(self, X, Y, method_layer = 'kde'):
        X = np.asarray(X)
        Y = np.asarray(Y)  
        # Reshape 1D arrays
        if X.ndim == 1: X = X.reshape(-1, 1)
        if Y.ndim == 1: Y = Y.reshape(-1, 1)

        if method_layer == "kde":
            HX = self.entropy_kde(X)
            HY = self.entropy_kde(Y)
            HXY = self.entropy_kde(np.concatenate([X, Y], axis=1))

            # N_samples = X.shape[0]
            # print(f"Max possible entropy log(N): {np.log(N_samples):.4f}")
            # print(f"H(X): {HX:.4f} | H(Y): {HY:.4f} | H(X,Y): {HXY:.4f}")

            return HX + HY - HXY

        if method_layer == "kraskov":
            return self.mut_info_kraskov(X, Y)

        if method_layer == "vae":
            print("self.entropy_vae(X, Y) TO BE IMPLEMENTED")
            #return self.entropy_vae(X, Y)
    
    # # ------------------------- KDE METHOD -------------------------

    def entropy_kde(self, data):
        rho = self.density(data)
        return - float( np.mean(np.log(rho + 1e-10)) )

    def density(self, data):
        N, d = data.shape
        
        data_sq = np.sum(data**2, axis=1, keepdims=True)
        dists_sq = data_sq + data_sq.T - 2 * data @ data.T
        # sometimes (due to numerical issue) there are small negative squared distances
        dists_sq = np.maximum(dists_sq, 0)
        
        #sigma_scaled = self.sigma  # self.sigma * np.sqrt(d) (Scale sigma by dimension) ***********************IMP**********************
        sigma_scaled = self.sigma# * np.sqrt(d) 
        kernel = np.exp(-dists_sq / (2 * sigma_scaled**2))
        return np.mean(kernel, axis=1)


    # # ------------------------- KDE METHOD -------------------------

    # def entropy_kde(self, data):
    #     rho = self.density(data)
    #     return - float( np.mean(np.log(rho + 1e-10)) )

    # def density(self, data):
    #     N, d = data.shape
        
    #     # Calculate squared pairwise distances
    #     data_sq = np.sum(data**2, axis=1, keepdims=True)
    #     dists_sq = data_sq + data_sq.T - 2 * data @ data.T
    #     dists_sq = np.maximum(dists_sq, 0) # Prevent numerical negative zeros
        
    #     # ---------------------------------------------------------
    #     # DYNAMIC BANDWIDTH (ADAPTIVE SIGMA)
    #     # ---------------------------------------------------------
    #     # 1. Extract unique pairwise distances (upper triangle, ignoring diagonal 0s)
    #     tri_idx = np.triu_indices_from(dists_sq, k=1)
    #     pairwise_dists = np.sqrt(dists_sq[tri_idx])
        
    #     # 2. Calculate the median distance to gauge the current spread of the manifold.
    #     # We use median instead of mean because it is much more robust to outliers.
    #     if len(pairwise_dists) > 0:
    #         median_dist = np.median(pairwise_dists)
    #     else:
    #         median_dist = 1.0 # Fallback for N=1
            
    #     # 3. Scale the base sigma by the median distance.
    #     # We add 1e-8 to prevent division by zero if all points are identical.
    #     sigma_scaled = self.sigma * (median_dist + 1e-8)
    #     # ---------------------------------------------------------

    #     # Compute Gaussian Kernel with the adaptive sigma
    #     kernel = np.exp(-dists_sq / (2 * sigma_scaled**2))
        
    #     return np.mean(kernel, axis=1)


    # ------------------------- KRASKOV METHOD ----------------------------
    def mut_info_kraskov(self, X, Y):

        # If dimensionality is too high, reduce to 30 principal components
        MAX_DIM = 30
        
        if X.shape[1] > MAX_DIM:
            X = PCA(n_components=MAX_DIM).fit_transform(X)
        if Y.shape[1] > MAX_DIM:
            Y = PCA(n_components=MAX_DIM).fit_transform(Y)
        

        # Add tiny noise to break ties (crucial for KSG estimator)
        X = X + 1e-10 * np.random.rand(*X.shape)
        Y = Y + 1e-10 * np.random.rand(*Y.shape)
        
        N = X.shape[0]
        XY = np.hstack([X, Y])
        
        # 1. Find k-nearest neighbors in the joint space (Chebyshev / max-norm)
        tree_xy = KDTree(XY, metric='chebyshev')
        # Query k+1 neighbors because the point itself is included (distance 0)
        dists, _ = tree_xy.query(XY, k=self.n_neig + 1)
        
        # Distance to the k-th neighbor
        radii = dists[:, -1]
        
        # 2. Count neighbors in the marginal spaces (X and Y) within these radii
        # KDTree supports an array of radii directly
        tree_x = KDTree(X, metric='chebyshev')
        tree_y = KDTree(Y, metric='chebyshev')
        
        # query_radius accepts the array 'radii' without issues
        nx_indices = tree_x.query_radius(X, r=radii)
        ny_indices = tree_y.query_radius(Y, r=radii)
        
        # Count how many points fall inside the radius (subtract 1 to exclude the point itself)
        # max(0, ...) protects against extremely rare numerical underflows
        nx = np.array([max(0, len(i) - 1) for i in nx_indices])
        ny = np.array([max(0, len(i) - 1) for i in ny_indices])
        
        # 3. KSG formula:
        # MI = psi(k) + psi(N) - <psi(nx+1) + psi(ny+1)>
        mi = (digamma(self.n_neig) + digamma(N) - 
            np.mean(digamma(nx + 1) + digamma(ny + 1)))
            
        return max(0.0, mi)