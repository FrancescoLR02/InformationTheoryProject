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

#*****************************************************************************************************************
#*****************************************************************************************************************

class BitwiseWeightQuantizer(nn.Module):
    """
    Simulates hardware quantization using Straight-Through Estimator (STE).
    Gradients flow through the un-quantized path during backprop.
    """
    def __init__(self, nBits=8):
        super().__init__()
        self.nBits = nBits
        self.q_min = -(2 ** (nBits - 1))
        self.q_max = (2 ** (nBits - 1)) - 1

    def forward(self, w):
        # 1. Scale factor based on max absolute value
        maxVal = torch.max(w.abs().max(), torch.tensor(1e-8, device=w.device))
        scale = maxVal / self.q_max
        
        # 2. Quantize (Round & Clamp)
        wInt = torch.clamp(torch.round(w / scale), self.q_min, self.q_max)
        
        # 3. Dequantize (Fake Quantization for Training)
        # We bring back int number from [ -2^(nbit-1), +2^(nbit-1) - 1 ] in float number near 0
        w_quant = wInt * scale
        
        # 4. STE Magic: Forward uses w_quant, Backward uses w
        # For problem in backward propagation round() in not differentiable
        return w + (w_quant - w).detach()


#*****************************************************************************************************************
#*****************************************************************************************************************

class BinarizeWithTemperature(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, temperature):
        ctx.save_for_backward(input)
        ctx.temperature = temperature
        # Forward: sigmoid then hard binarization (0 or 1)
        x_sigmoid = torch.sigmoid(input)
        return (x_sigmoid > 0.5).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        temperature = ctx.temperature
        # Backward: gradient of sigmoid with temperature
        sig = torch.sigmoid(input / temperature)
        grad_input = grad_output * sig * (1 - sig) / temperature
        return grad_input, None

#*****************************************************************************************************************
#*****************************************************************************************************************

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
        binarize: str = "no", # if input image are binarize 0-1
        temperature: float = 1,
        Variational: bool = True,
        quantize_bits: int = None  # for quantize weights up to nbits (see BitwiseWeightQuantizer)
    ):
        super(VariationalAutoEncoder, self).__init__()

        # Validate binarize parameter
        if binarize not in ["no", "all", "test"]:
            raise ValueError(f"binarize must be 'no', 'all', or 'test', got '{binarize}'")

        self.latentDim = latentDim
        self.hiddenDim = hiddenDim
        self.sigma = sigmaVAE
        self.activation_out = activation_out
        self.Variational = Variational
        self.binarize = binarize
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


        # ---------------- WEIGHT QUANTIZER ----------------
        self.quantize_bits = quantize_bits
        if self.quantize_bits is not None:
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    if parametrize.is_parametrized(module, "weight"):
                        parametrize.remove_parametrizations(module, "weight")
                    parametrize.register_parametrization(module, "weight", BitwiseWeightQuantizer(self.quantize_bits))


    def Encoding(self, x):
        x = x.view(x.size(0), -1)
        x = self.InputSpace(x) # Hook input
        
        h = self.Encoder(x)

        if self.Variational:
            mean = self.LatentLayerMu(h)
            logVar = self.LatentLayerSigma(h)

            std = torch.exp(logVar)
            eps = torch.randn_like(std) #* self.sigma
            z = mean + std * eps

            # Binarize latent based on mode
            should_binarize = (self.binarize == "all") or (self.binarize == "test" and not self.training)
            
            if should_binarize:
                # Apply binarization with temperature-based backward
                z = BinarizeWithTemperature.apply(z, self.temperature)

            # Hook latent
            z = self.LatentSpace(z)

            return z, mean, logVar
            
        else:
            z = self.LatentLayer(h)

            # Binarize latent based on mode
            should_binarize = (self.binarize == "all") or \
                            (self.binarize == "test" and not self.training)
            
            if should_binarize:
                # Apply binarization with temperature-based backward
                z = BinarizeWithTemperature.apply(z, self.temperature)

            # Hook latent
            z = self.LatentSpace(z)

            return z, None, None


    def Decoding(self, z):
        y = self.Decoder(z)
        y = self.OutputLayer(y)

        out = self.activation_out(y)

        # Binarize output based on mode
        should_binarize = (self.binarize == "all") or \
                        (self.binarize == "test" and not self.training)
        
        if should_binarize:
            # Apply binarization with temperature-based backward
            out = BinarizeWithTemperature.apply(out, self.temperature)

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