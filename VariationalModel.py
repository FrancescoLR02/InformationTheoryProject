# ============================
# PyTorch
# ============================
import torch
from torch import nn

# ============================
# Typing
# ============================
from typing import List, Callable

# ============================
# Visualization
# ============================
import matplotlib.pyplot as plt

# ============================
# Quantization
# ============================
import torch.nn.utils.parametrize as parametrize

# ============================
# Scientific & Data Handling
# ============================
import numpy as np


#*****************************************************************************************************************
#*****************************************************************************************************************

class VariationalAutoEncoder(nn.Module):

    def initialize_weights(self):
        """
        Initialize all Linear layers with Xavier initialization and set all biases to zero.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    #--------------------------------------------------------------------------------------------------------------------------------------
    #--------------------CONSTRUCTOR
    #--------------------------------------------------------------------------------------------------------------------------------------

    def __init__(
        self,
        latentDim: int,
        inputDim: int = 784,
        hiddenDim: List[int] = [512, 256],
        activation_enc: Callable = nn.ReLU,
        activation_dec: Callable = nn.ReLU,
        activation_out: Callable = torch.sigmoid,
        # binarize: str = "no", # if input image are binarize 0-1
        Variational: bool = True
    ):
        super(VariationalAutoEncoder, self).__init__()

        # Validate binarize parameter
        # if binarize not in ["no", "all", "test"]:
        #     raise ValueError(f"binarize must be 'no', 'all', or 'test', got '{binarize}'")

        self.latentDim = latentDim
        self.hiddenDim = hiddenDim
        self.activation_enc = activation_enc
        self.activation_dec = activation_dec
        self.activation_out = activation_out
        self.Variational = Variational
        # self.binarize = binarize

        self.train_loss_history = []
        self.val_loss_history = []

        # train_loss if there is regularizatin/penalty/premium
        self.mse_history = []
        self.penalty_history = []
        self.premium_history = []


        # Identity module for hooking input and output space
        self.InputSpace  = nn.Identity()

        # ---------------- ENCODER ----------------
        currentDim = inputDim
        modules = []

        for h in hiddenDim:
            modules.append(
                nn.Sequential(
                    nn.Linear(currentDim, h),
                    self.activation_enc()
                )
            )
            currentDim = h

        self.Encoder = nn.Sequential(*modules)

        # ---------------- LATENT ----------------
        if self.Variational:
            self.LatentLayerMu = nn.Linear(currentDim, latentDim)
            self.LatentLayerSigma = nn.Linear(currentDim, latentDim)
            # Identity module for hooking latent space
            self.LatentSpace = nn.Identity()
        else:
            # learn latent space directly (no mean/var)
            self.LatentLayer = nn.Linear(currentDim, latentDim)
            self.LatentSpace = nn.Identity()

        # ---------------- DECODER ----------------
        modules = []
        currentDim = latentDim
        reversedDim = hiddenDim[::-1]

        for h in reversedDim:
            modules.append(
                nn.Sequential(
                    nn.Linear(currentDim, h),
                    self.activation_dec()
                )
            )
            currentDim = h

        self.Decoder = nn.Sequential(*modules)
        self.OutputLayer = nn.Linear(currentDim, inputDim)

        # Identity module for hooking output space
        self.OutputSpace = nn.Identity()



        # Initialize weights
        self.initialize_weights()

    #--------------------------------------------------------------------------------------------------------------------------------------
    #--------------------ENCODING
    #--------------------------------------------------------------------------------------------------------------------------------------

    def Encoding(self, x):
        x = x.view(x.size(0), -1)
        x = self.InputSpace(x) # Hook input
        
        h = self.Encoder(x)

        if self.Variational:
            mean = self.LatentLayerMu(h)
            logVar = self.LatentLayerSigma(h)

            std = torch.exp(0.5 *logVar) # because logVar=log(σ²)=2*log(σ) ===> σ=std=exp(0.5*logVar)
            eps = torch.randn_like(std)  # sample ε ~ N(0, 1) (same shape as std, mean=0, std=1), recall std array of length latenDim
            z = mean + std * eps

            # # Binarize latent based on mode
            # should_binarize = (self.binarize == "all") or (self.binarize == "test" and not self.training)
            
            # if should_binarize:
            #     # Apply binarization with temperature-based backward
            #     z = BinarizeWithTemperature.apply(z, self.temperature)

            # Hook latent
            z = self.LatentSpace(z)

            return z, mean, logVar
            
        else:
            z = self.LatentLayer(h)

            # # Binarize latent based on mode
            # should_binarize = (self.binarize == "all") or (self.binarize == "test" and not self.training)
            
            # if should_binarize:
            #     # Apply binarization with temperature-based backward
            #     z = BinarizeWithTemperature.apply(z, self.temperature)

            # Hook latent
            z = self.LatentSpace(z)

            return z, None, None


    #--------------------------------------------------------------------------------------------------------------------------------------
    #--------------------DECODING
    #--------------------------------------------------------------------------------------------------------------------------------------

    def Decoding(self, z):
        y = self.Decoder(z)
        y = self.OutputLayer(y)

        out = self.activation_out(y)

        # # Binarize output based on mode
        # should_binarize = (self.binarize == "all") or \
        #                 (self.binarize == "test" and not self.training)
        
        # if should_binarize:
        #     # Apply binarization with temperature-based backward
        #     out = BinarizeWithTemperature.apply(out, self.temperature)

        # Hook output
        out = self.OutputSpace(out)

        return out

    #--------------------------------------------------------------------------------------------------------------------------------------
    #--------------------FOWARD PASS
    #--------------------------------------------------------------------------------------------------------------------------------------

    def forward(self, x):
        z, mean, logVar = self.Encoding(x)
        out = self.Decoding(z)
        return out, z, mean, logVar

    #--------------------------------------------------------------------------------------------------------------------------------------
    #--------------------PLOT LOSS
    #--------------------------------------------------------------------------------------------------------------------------------------

    def plot_loss(self):
        epochs = range(1, len(self.train_loss_history) + 1)

        # FIGURE 1: Training vs Validation (validation if present)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.train_loss_history, color='blue', linewidth=2, label='Training loss')
        
        if self.val_loss_history:
            plt.plot(epochs, self.val_loss_history, color='red', linewidth=2, label='Validation loss')

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


        # FIGURE 2: Training loss composition MSE + penalty + premium
        if self.penalty_history or self.premium_history:
            plt.figure(figsize=(10, 5))

            plt.plot(epochs, self.train_loss_history, color='blue', linewidth=2, linestyle='--', label='Total loss')

            # MSE
            plt.plot(epochs, self.mse_history, color='green', linewidth=2, label='MSE component')

            # Penalty (if present)
            if self.penalty_history:
                plt.plot(epochs, self.penalty_history, color='orange', linewidth=2, label='Penalty component')

            # Premium (if present)
            if self.premium_history:
                plt.plot(epochs, self.premium_history, color='purple', linewidth=2, label='Premium component')

            plt.xlabel("Epoch")
            plt.ylabel("Loss components")
            plt.title("Loss Composition: MSE vs Penalty vs Premium")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()



    #--------------------------------------------------------------------------------------------------------------------------------------
    #--------------------REPRESENTER (for print)
    #--------------------------------------------------------------------------------------------------------------------------------------

    def __repr__(self):
        return (
            "VariationalAutoEncoder(\n"
            "  modules:\n"
            "    InputSpace, Encoder\n"
            "    LatentLayerMu, LatentLayerSigma, LatentSpace\n"
            "    Decoder, OutputSpace\n"
            "  configuration:\n"
            "    Variational, \n"
            "    latentDim, hiddenDim,\n"
            "    activation_enc, activation_dec, activation_out,\n"
            "    binarize, \n"
            "  training history:\n"
            "    train_loss_history, val_loss_history,\n"
            "    mse_history, penalty_history\n"
            "  methods:\n"
            "    Encoding(), Decoding(), plot_loss()\n"
            ")"
        )