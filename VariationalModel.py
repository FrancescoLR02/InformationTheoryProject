import torch
from torch import nn
from typing import List, Callable
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms


class DiscretizeWithTemperature(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, temperature, n_values):
        ctx.save_for_backward(input)
        ctx.temperature = temperature
        ctx.n_values = n_values
        
        # Forward: discretize to n_values in range [0, 1]
        # Map input through sigmoid to [0, 1]
        #x_sigmoid = torch.sigmoid(input)
        x_normalized = (torch.tanh(input) + 1) / 2  # mappa [-∞,∞] → [0,1] più uniformemente
        
        # Discretize to n_values: 0, 1/(n-1), 2/(n-1), ..., 1
        # Example: n=2 → {0, 1}, n=3 → {0, 0.5, 1}, n=4 → {0, 0.33, 0.66, 1}
        discrete_vals = torch.round(x_normalized * (n_values - 1)) / (n_values - 1)
        
        return discrete_vals
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        temperature = ctx.temperature
        n_values = ctx.n_values
        
        # Backward: gradient of sigmoid with temperature
        sig = torch.sigmoid(input / temperature)
        grad_input = grad_output * sig * (1 - sig) / temperature
        
        return grad_input, None, None


class VariationalAutoEncoder(nn.Module):

    def __init__(
        self,
        latentDim: int,
        hiddenDim: List[int] = [512, 256],
        inputDim: int = 784,
        sigmaVAE: float = 0.5,
        activation_enc: Callable = nn.ReLU,
        activation_dec: Callable = nn.ReLU,
        activation_out: Callable = torch.sigmoid,
        binarize: str = "no",
        n_discretize: int = 2,
        temperature: float = 1,
        Variational: bool = True
    ):
        super(VariationalAutoEncoder, self).__init__()

        # Validate binarize parameter
        if binarize not in ["no", "all", "test"]:
            raise ValueError(f"binarize must be 'no', 'all', or 'test', got '{binarize}'")
        
        # Validate n_discretize parameter
        if n_discretize < 2:
            raise ValueError(f"n_discretize must be >= 2, got {n_discretize}")

        self.latentDim = latentDim
        self.hiddenDim = hiddenDim
        self.sigma = sigmaVAE
        self.activation_out = activation_out
        self.Variational = Variational
        self.binarize = binarize
        self.n_discretize = n_discretize
        self.temperature = temperature

        self.train_loss_history = []
        self.val_loss_history = []


        # Identity module for hooking input and output space
        self.InputSpace  = nn.Identity()

        # ---------------- ENCODER ----------------
        currentDim = inputDim
        modules = []

        for h in hiddenDim:
            modules.append(
                nn.Sequential(
                    nn.Linear(currentDim, h),
                    activation_enc()
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
                    activation_dec()
                )
            )
            currentDim = h

        self.Decoder = nn.Sequential(*modules)
        self.OutputLayer = nn.Linear(currentDim, inputDim)

        # Identity module for hooking output space
        self.OutputSpace = nn.Identity()



    def Encoding(self, x):
        x = x.view(x.size(0), -1)
        x = self.InputSpace(x) # Hook input
        
        h = self.Encoder(x)

        if self.Variational:
            mean = self.LatentLayerMu(h)
            logVar = self.LatentLayerSigma(h)

            std = torch.exp(0.5 * logVar)
            eps = torch.randn_like(std) #* self.sigma
            z = mean + std * eps

            # Discretize latent based on mode
            should_discretize = (self.binarize == "all") or \
                              (self.binarize == "test" and not self.training)
            
            if should_discretize:
                # Apply discretization with temperature-based backward
                z = DiscretizeWithTemperature.apply(z, self.temperature, self.n_discretize)

            # Hook latent
            z = self.LatentSpace(z)

            return z, mean, logVar
            
        else:
            z = self.LatentLayer(h)

            # Discretize latent based on mode
            should_discretize = (self.binarize == "all") or \
                              (self.binarize == "test" and not self.training)
            
            if should_discretize:
                # Apply discretization with temperature-based backward
                z = DiscretizeWithTemperature.apply(z, self.temperature, self.n_discretize)

            # Hook latent
            z = self.LatentSpace(z)

            return z, None, None


    def Decoding(self, z):
        y = self.Decoder(z)
        y = self.OutputLayer(y)

        out = self.activation_out(y)

        # Discretize output based on mode
        should_discretize = (self.binarize == "all") or \
                          (self.binarize == "test" and not self.training)
        
        if should_discretize:
            # Apply discretization with temperature-based backward
            out = DiscretizeWithTemperature.apply(out, self.temperature, self.n_discretize)

        # Hook output
        out = self.OutputSpace(out)

        return out


    def forward(self, x):
        z, mean, logVar = self.Encoding(x)
        out = self.Decoding(z)
        return out, z, mean, logVar
    

    def plot_loss(self):
        epochs = range(1, len(self.train_loss_history) + 1)

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