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
#*******************************MI ESTIMATOR CALCULATOR
#*****************************************************************************************************************

class MI_Estimator:
    

    def __init__(self, method, sigma=1.0, n_neig=3, default="kde"):
        self.sigma  = sigma
        self.n_neig = n_neig
        self.method = self.create_method_array(method, default)
        # for mi methods self.method = ["in_h", "h_z" ,"in_z", "z_h", "h_out", "z_out"]

    def create_method_array(self, method, default):
        # if only a string is passed it is replicated 6 times
        if isinstance(method, str):
            method_array = [method] * 6
            return method_array
        
        # if a list of string is passed we fill it to 6 methods with the default type
        if isinstance(method, (list, tuple)):
            method_array = list(method) # convert tuple eventually into list

            while len(method_array) < 6:
                method_array.append(default)
            return method_array[:6] # always a list of 6 strings is returned



    # ------------------------- MUTUAL INFO -------------------------

    def mutual_information(self, X, Y, method_layer):
        X = np.asarray(X)
        Y = np.asarray(Y)  
        # Reshape 1D arrays
        if X.ndim == 1: X = X.reshape(-1, 1)
        if Y.ndim == 1: Y = Y.reshape(-1, 1)

        if method_layer == "kde":
            HX = self.entropy_kde(X)
            HY = self.entropy_kde(Y)
            HXY = self.entropy_kde(np.concatenate([X, Y], axis=1))
            return HX + HY - HXY

        if method_layer == "kraskov":
            return self.kraskov_estimation(X, Y)
    
    # ------------------------- KDE METHOD -------------------------

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

    # ------------------------- KRASKOV METHOD ------------------------- # MAI TESTATO DA VEDERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
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