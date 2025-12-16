import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
from typing import List



class VariationalAutoEncoder(nn.Module):

   def __init__(self, latentDim: int, hiddenDim: List = [512, 256], inputDim: int = 784):
      super(VariationalAutoEncoder, self).__init__()

      self.latentDim = latentDim
      self.hiddenDim = hiddenDim
      
      ####--------------ENCODER--------------

      #Initialize the initial dimension to be inputDim
      currentDim = inputDim

      modules = []

      #Define the architecture dynamically
      for h in self.hiddenDim:
         modules.append(
            nn.Sequential(
               nn.Linear(currentDim, h),
               nn.ReLU()
            )
         )
         #update the value of the current dimesion
         currentDim = h

      self.Encoder = nn.Sequential(*modules)

      #Define the latent space 
      self.EncoderMu = nn.Linear(hiddenDim[-1], latentDim)
      self.EncoderSigma = nn.Linear(hiddenDim[-1], latentDim)


      ####--------------DECORDER--------------
      modules = []
      currentDim = latentDim

      #Layers in the decoder are inverted w.r.t encoder
      reversedDim = hiddenDim[::-1]

      for h in reversedDim:
         modules.append(
            nn.Sequential(
               nn.Linear(currentDim, h),
               nn.ReLU()
            )
         )
         currentDim = h

      self.Decoder = nn.Sequential(*modules)
      self.finalLayer = nn.Linear(currentDim, inputDim)

   
   def Encoding(self, x):

      #Flatten the data
      x = x.view(x.size(0), -1)
      x = self.Encoder(x)
      mean, logVar = self.EncoderMu(x), self.EncoderSigma(x)
      return mean, logVar
   
   def Reparametrization(self, mean, logVar):
      std = torch.exp(0.5*logVar)
      epsilon = torch.rand_like(std)
      return mean + std*epsilon
   
   def Decoding(self, z):
      z = self.Decoder(z)
      result = self.finalLayer(z)
      return torch.sigmoid(result)
   
   def forward(self, x):
      mean, logVar = self.Encoding(x)
      z = self.Reparametrization(mean, logVar)
      result = self.Decoding(z)

      return result, z, mean, logVar