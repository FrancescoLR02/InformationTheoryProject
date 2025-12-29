import torch
from torch import nn
from typing import List, Callable
import numpy as np
import matplotlib.pyplot as plt

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
        Variational: bool = True
    ):
        super(VariationalAutoEncoder, self).__init__()

        self.latentDim = latentDim
        self.hiddenDim = hiddenDim
        self.sigma = sigmaVAE
        self.activation_out = activation_out
        self.Variational = Variational

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
            self.LatentSpace = nn.Linear(currentDim, latentDim)

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

            # Hook latent
            z = self.LatentSpace(z)

            return z, mean, logVar
        else:
            z = self.LatentSpace(h)

            return z, None, None


    def Decoding(self, z):
        y = self.Decoder(z)
        y = self.OutputLayer(y)

        out = self.activation_out(y)

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