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
        latentDim: int = 50,
        inputDim: int = 784,
        hiddenDim: List[int] = [512, 256],
        activation_enc: Callable = nn.ReLU,
        activation_dec: Callable = nn.ReLU,
        activation_out: Callable = torch.sigmoid,
        bit_type: str = "real",
        Variational: bool = True
    ):
        super(VariationalAutoEncoder, self).__init__()


        self.latentDim = latentDim
        self.hiddenDim = hiddenDim
        self.activation_enc = activation_enc
        self.activation_dec = activation_dec
        self.activation_out = activation_out
        self.Variational = Variational
        self.bit_type = bit_type
        if bit_type not in ["real", "restricted", "discrete"]:
            raise ValueError(f"bit_type must be 'real', 'restricted', or 'discrete', you wrote '{bit_type}' not a valid choice.")

        self.train_loss_history = []
        self.val_loss_history = []

        # train_loss if there is regularizatin/penalty/premium
        self.mse_history = []
        self.kl_history = []
        self.penalty_history = []
        self.premium_history = []

        # Initialize weights
        self.initialize_weights()

        # ---------------- INPUT ----------------

        # Identity module for hooking input and output space
        self.InputSpace  = nn.Identity()
        self.Label       = nn.Identity()

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

        else:
            # learn latent space directly (no mean/var)
            self.LatentLayer = nn.Linear(currentDim, latentDim)
        
        # Identity modules for hooking latent space
        self.LatentSpace = nn.Identity()
        self.LatentQuant= nn.Identity()

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

        # ---------------- OUTPUT ----------------

        # Identity module for hooking output space
        self.OutputSpace = nn.Identity()

    #--------------------------------------------------------------------------------------------------------------------------------------
    #--------------------ENCODING METHOD
    #--------------------------------------------------------------------------------------------------------------------------------------

    def Encoding(self, x, label=None):

        if label is not None: self.Label(label)

        x = x.view(x.size(0), -1)
        x = self.InputSpace(x) # Hook input

        h = self.Encoder(x)

        mean  = None
        logVar= None

        if self.Variational:
            mean = self.LatentLayerMu(h)
            logVar = self.LatentLayerSigma(h)
            std = torch.exp(0.5 *logVar) # because logVar=log(σ²)=2*log(σ) ===> σ=std=exp(0.5*logVar)
            eps = torch.randn_like(std)  # sample ε ~ N(0, 1) (same shape as std, mean=0, std=1), recall std array of length latenDim
            z = mean + std * eps
            
        else:
            z = self.LatentLayer(h)

        # Hook latent (passing through identity layers)
        z = self.LatentSpace(z)

        if self.bit_type == "discrete":
            bit = ( 2 * (z > 0).float() ) -1       # b is -1/1
            # bit = (z > 0).float() # here b is 0/1
            
            # in forward pass we get b (-1/1)
            # in backward, because of .detach() we have the gradient of z, namely the identity
            b = z + (bit - z).detach()
        else:
            b = z

        # Hook latent quantize (passing through identity layers)
        b = self.LatentQuant(b)

        return z, b, mean, logVar


    #--------------------------------------------------------------------------------------------------------------------------------------
    #--------------------DECODING METHOD
    #--------------------------------------------------------------------------------------------------------------------------------------

    def Decoding(self, z):
        y = self.Decoder(z)
        y = self.OutputLayer(y)

        out = self.activation_out(y)

        # Hook output
        out = self.OutputSpace(out)

        return out

    #--------------------------------------------------------------------------------------------------------------------------------------
    #--------------------FOWARD PASS
    #--------------------------------------------------------------------------------------------------------------------------------------

    def forward(self, x, label=None):

        z, b, mean, logVar = self.Encoding(x, label)

        out = self.Decoding(b)

        return out, z, b, mean, logVar
 

    #--------------------------------------------------------------------------------------------------------------------------------------
    #--------------------PLOT LOSS
    #--------------------------------------------------------------------------------------------------------------------------------------

    def plot_loss(self, start_epoch = 1):
        epochs = range(1, len(self.train_loss_history) + 1)

        # FIGURE 1: Training vs Validation (validation if present)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs[start_epoch:], self.train_loss_history[start_epoch:], color='blue', linewidth=2, label='Training loss')
        
        if self.val_loss_history:
            plt.plot(epochs[start_epoch:], self.val_loss_history[start_epoch:], color='red', linewidth=2, label='Validation loss')

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

            plt.plot(epochs[start_epoch:], self.train_loss_history[start_epoch:], color='blue', linewidth=2, linestyle='--', label='Total loss')

            # MSE
            plt.plot(epochs[start_epoch:], self.mse_history[start_epoch:], color='green', linewidth=2, label='MSE component')

            # KL component
            if self.Variational:
                plt.plot(epochs[start_epoch:], self.kl_history[start_epoch:], color='yellow', linewidth=2, label='KL div. component')

            # Penalty (if present)
            if self.penalty_history:
                plt.plot(epochs[start_epoch:], self.penalty_history[start_epoch:], color='orange', linewidth=2, label='Penalty component')

            # Premium (if present)
            if self.premium_history:
                plt.plot(epochs[start_epoch:], self.premium_history[start_epoch:], color='purple', linewidth=2, label='Premium component')

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
            "    InputSpace, Label, Encoder\n"
            "    LatentLayerMu, LatentLayerSigma, LatentSpace, LatentQuant\n"
            "    Decoder, OutputSpace\n"
            "  configuration:\n"
            "    Variational, \n"
            "    latentDim, hiddenDim,\n"
            "    activation_enc, activation_dec, activation_out,\n"
            "    bit_type, \n"
            "  training history:\n"
            "    train_loss_history, val_loss_history,\n"
            "    mse_history, penalty_history\n"
            "  methods:\n"
            "    Encoding(), Decoding(), plot_loss()\n"
            ")"
        )