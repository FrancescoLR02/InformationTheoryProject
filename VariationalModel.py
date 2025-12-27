import torch
from torch import nn
from typing import List, Callable


class VariationalAutoEncoder(nn.Module):

    def __init__(
        self,
        latentDim: int,
        hiddenDim: List[int] = [512, 256],
        inputDim: int = 784,
        sigma: float = 0.5,
        activation_enc: Callable = nn.ReLU,
        activation_dec: Callable = nn.ReLU,
        activation_out: Callable = torch.sigmoid,
        Variational: bool = True
    ):
        super(VariationalAutoEncoder, self).__init__()

        self.latentDim = latentDim
        self.hiddenDim = hiddenDim
        self.sigma = sigma
        self.activation_out = activation_out
        self.Variational = Variational

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
        # Latent layers: either variational (mu/logVar) or direct linear mapping
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

        # Identity module for hooking input and output space
        self.InputSpace  = nn.Identity()
        self.OutputSpace = nn.Identity()


    def Encoding(self, x):

        x = x.view(x.size(0), -1)
        x = self.InputSpace(x) # So hook can take also the input
        
        h = self.Encoder(x)

        if self.Variational:
            mean = self.LatentLayerMu(h)
            logVar = self.LatentLayerSigma(h)

            std = torch.exp(0.5 * logVar)
            eps = torch.randn_like(std) * self.sigma
            z = mean + std * eps

            # Pass through identity module so hooks can capture it
            z = self.LatentSpace(z)
        else:
            # direct mapping to latent space (no sampling)
            z = self.LatentSpace(h)

        return z


    def Decoding(self, z):
        y = self.Decoder(z)
        y = self.OutputLayer(y)

        out = self.activation_out(y)

        # Pass through identity module so hooks can capture it
        out = self.OutputSpace(out)

        return out


    def forward(self, x):
        z = self.Encoding(x)
        out = self.Decoding(z)
        return out, z