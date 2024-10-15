# Conditional GAN Script
# Created by: Deep Ray, University of Maryland
# Date: 27 August 2023


import os
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import importlib
from torch import (
    add,
    cat,
    rsqrt,
    rand,
    randn,
    autograd,
    ones_like,
    norm,
    pow,
    square,
    sqrt,
    sum,
    Tensor,
    empty,
)
import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial


# Conditional Instance Normalization
class CondInsNorm(nn.Module):
    def __init__(self, x_dim, z_dim, eps=1.0e-6, act_param=0.1):
        super(CondInsNorm, self).__init__()
        self.eps = eps
        self.z_shift = nn.Sequential(
            nn.Conv1d(z_dim, x_dim, kernel_size=1),
            nn.ELU(alpha=act_param)
        )
        self.z_scale = nn.Sequential(
            nn.Conv1d(z_dim, x_dim, kernel_size=1),
            nn.ELU(alpha=act_param)
        )

    def forward(self, x, z):
        z_upsampled = F.interpolate(z, size=x.size(
            2), mode='linear', align_corners=True)
        shift = self.z_shift(z_upsampled)
        scale = self.z_scale(z_upsampled)

        x_mean = x.mean(dim=2, keepdim=True)
        x_var = x.var(dim=2, keepdim=True)
        x_norm = (x - x_mean) / torch.sqrt(x_var + self.eps)
        return x_norm * scale + shift

# U-Net with Conditional Instance Normalization


class UNetCondInsNorm(nn.Module):
    def __init__(self, z_dim):
        super(UNetCondInsNorm, self).__init__()

        # Encoder (Downsampling)
        self.down_conv1 = nn.Conv1d(1, 2, kernel_size=3, padding=1)
        self.down_conv2 = nn.Conv1d(2, 4, kernel_size=3, padding=1)
        self.down_conv3 = nn.Conv1d(4, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)

        # Conditional Instance Normalization
        self.cond_norm1 = CondInsNorm(2, z_dim)
        self.cond_norm2 = CondInsNorm(4, z_dim)
        self.cond_norm3 = CondInsNorm(8, z_dim)

        # Decoder (Upsampling)
        self.up_trans1 = nn.ConvTranspose1d(8, 4, kernel_size=2, stride=2)
        self.up_trans2 = nn.ConvTranspose1d(4, 2, kernel_size=2, stride=2, output_padding=1)
        self.up_trans3 = nn.ConvTranspose1d(2, 1, kernel_size=2, stride=2)

        self.upsample = nn.ConvTranspose1d(1, 1, kernel_size=2, stride=2, output_padding=0)  # Upsample from 50 to 100

        self.cond_norm4 = CondInsNorm(4, z_dim)
        self.cond_norm5 = CondInsNorm(2, z_dim)
        self.cond_norm6 = CondInsNorm(1, z_dim)

        self.elu = nn.ELU()

    def forward(self, y, z):
        # Encoder
        y1 = self.elu(self.down_conv1(y))  # [50, 1] -> [50, 2]
        y1 = self.cond_norm1(y1, z)
        y1_pooled = self.pool(y1)          # [50, 2] -> [25, 2]

        y2 = self.elu(self.down_conv2(y1_pooled))  # [25, 2] -> [25, 4]
        y2 = self.cond_norm2(y2, z)
        y2_pooled = self.pool(y2)                # [25, 4] -> [12, 4]

        y3 = self.elu(self.down_conv3(y2_pooled))  # [12, 4] -> [12, 8]
        y3 = self.cond_norm3(y3, z)
        y3_pooled = self.pool(y3)  # Corrected: [12, 8] -> [6, 8]

        # Decoder
        y_up1 = self.elu(self.up_trans1(y3_pooled))  # [6, 8] -> [12, 4]
        y_up1 = self.cond_norm4(y_up1, z)

        y_up2 = self.elu(self.up_trans2(y_up1))      # [12, 4] -> [25, 2]
        y_up2 = self.cond_norm5(y_up2, z)

        y_up3 = self.elu(self.up_trans3(y_up2)) # [25, 2] -> [50, 1]
        y_up3 = self.cond_norm6(y_up3, z)

        # Final upsample 
        out = self.elu(self.upsample(y_up3)) # [50, 1] -> [100, 1]

        print("OUTPUT", out.shape)
        return out


# Initialize model
generator = UNetCondInsNorm(z_dim=5)

# Example input
y = torch.randn(1, 1, 50)
z = torch.randn(1, 5, 5)

# Forward pass
output = generator(y, z)


''' Critic Model: Idea here is to reduce x dimension to y dimension, then stack them and reduce to a scalar. '''
''' Assumption: x_dim = 100, y_dim = 50'''

class CriticModel(nn.Module):
    def __init__(self, x_dim=100, y_dim=50):
        super(CriticModel, self).__init__()

        self.downsample_x = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1) 

        self.conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1) # stack x and y 
        

        # MLP to reduce to scalar 
        self.fc1 = nn.Linear(50, 25)  
        self.fc2 = nn.Linear(25, 1) 

        self.elu = nn.ELU()


    def forward(self, x, y):
        x = x.unsqueeze(1)  
        y = y.unsqueeze(1)  

        x = self.downsample_x(x)

        # Combined shape after concatenation: [batch_size, 2, 50]
        combined = torch.cat((x, y), dim=1)

        # Pass through convolution layer
        combined = self.conv(combined)  # [batch_size, 2, 50] -> [batch_size, 1, 50]
        combined = combined.squeeze(1)  # Remove channel dimension: [batch_size, 1, 50] -> [batch_size, 50]

        # Pass through MLP
        combined = self.elu(self.fc1(combined)) 
        output = self.elu(self.fc2(combined)) 

        return output


x = torch.randn(10, 100)
y = torch.randn(10, 50)

critic = CriticModel()
output = critic(x, y)

print("Critic model shape", output.shape)

# Custom forward wrapper


class ModelSummaryWrapper(nn.Module):
    def __init__(self, model, z_dim):
        super(ModelSummaryWrapper, self).__init__()
        self.model = model
        self.z_dim = z_dim

    def forward(self, y):
        z = torch.randn(y.size(0), self.z_dim, 1).to(
            y.device)
        return self.model(y, z)


class CriticSummaryWrapper(nn.Module):
    def __init__(self, model, y_size):
        super(CriticSummaryWrapper, self).__init__()
        self.model = model
        self.y_size = y_size

    def forward(self, x):
        # Create a dummy y tensor with the correct size and pass it to the model
        batch_size = x.size(0)
        y = torch.randn(batch_size, self.y_size).to(
            x.device)  # Generate dummy y tensor
        return self.model(x, y)


# Ensure y_size matches your model
critic = CriticSummaryWrapper(critic, y_size=50)
summary(critic, input_size=(100,))

generator = ModelSummaryWrapper(generator, z_dim=5)
summary(generator, input_size=(1, 50))
