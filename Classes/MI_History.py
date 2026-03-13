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
#*******************************MI HISTORY RECORDER
#*****************************************************************************************************************

class MI_History:

    def __init__(self):
        self.encoder = []
        self.decoder = []
        self.input_latent = []
        self.latent_output = []

        self.latent_points = []
    
    def append(self, mi_dict):
        self.encoder.append(mi_dict["encoder"])
        self.decoder.append(mi_dict["decoder"])
        self.input_latent.append(mi_dict["input_latent"])
        self.latent_output.append(mi_dict["latent_output"])
        self.latent_points.append(mi_dict["latent_points"])

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