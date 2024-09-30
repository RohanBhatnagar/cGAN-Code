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


class CondInsNorm(nn.Module):
    ''' Implementing conditional instance normalization
        where input_x is normalized wrt input_z
        input_x is assumed to have the shape (N, x_dim, H)
        input_z is assumed to have the shape (N, z_dim, 5)
    '''

    def __init__(self, x_dim, z_dim, eps=1.0e-6, act_param=0.1):
        super(CondInsNorm, self).__init__()
        self.eps = eps
        self.z_shift = nn.Sequential(
            nn.Conv1d(in_channels=z_dim,
                      out_channels=x_dim,
                      kernel_size=1,
                      stride=1),
            nn.ELU(alpha=act_param)
        )
        self.z_scale = nn.Sequential(
            nn.Conv1d(in_channels=z_dim,
                      out_channels=x_dim,
                      kernel_size=1,
                      stride=1),
            nn.ELU(alpha=act_param)
        )

    def forward(self, x, z):
        x_size = x.size()
        assert len(x_size) == 3
        assert len(z.size()) == 3

        shift = self.z_shift(z)
        scale = self.z_scale(z)

        # print("Shift shape: ", shift.shape)
        # print("Scale shape: ", scale.shape)

        x_reshaped = x.view(x_size[0], x_size[1], x_size[2])
        x_mean = x_reshaped.mean(2, keepdim=True)
        x_var = x_reshaped.var(2, keepdim=True)
        x_rstd = torch.rsqrt(x_var + self.eps)  # reciprocal sqrt
        x_s = ((x_reshaped - x_mean) * x_rstd).view(*x_size)
        output = x_s * scale + shift

        # print("Shape of output: ", output.shape)

        return output


''' Generator model. We want to introduce the latent variable Z at every layer, as per the NACO paper.'''


class UNetWithCIN1D(nn.Module):
    def __init__(self, y_dim, x_dim, z_dim, activation=torch.nn.ELU()):
        super(UNetWithCIN1D, self).__init__()

        self.down1 = self.conv_block(y_dim, 64)
        self.down2 = self.conv_block(64, 128)
        self.down3 = self.conv_block(128, 256)

        self.up3 = self.conv_block(256, 128)
        self.up2 = self.conv_block(128, 64)
        self.up1 = self.conv_block(64, x_dim)

        self.cin1 = CondInsNorm(64, z_dim)
        self.cin2 = CondInsNorm(128, z_dim)
        self.cin3 = CondInsNorm(256, z_dim)
        self.cin4 = CondInsNorm(128, z_dim)
        self.cin5 = CondInsNorm(64, z_dim)
        self.cin6 = CondInsNorm(x_dim, z_dim)

        self.final_conv = nn.Conv1d(100, x_dim, kernel_size=1)
        self.activation = activation

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3,
                      padding=1, stride=1),
            torch.nn.ELU(),
        )

    def cin_block(self, out_channels, z_dim):
        return nn.Sequential(
            CondInsNorm(out_channels, z_dim)
        )

    def forward(self, y, z):
        d1 = self.down1(y)  # (N, 64, length)
        d1 = self.cin1(d1, z)  # Apply CIN after down1

        d2 = self.down2(d1)  # (N, 128, length)
        d2 = self.cin2(d2, z)  # Apply CIN after down2

        d3 = self.down3(d2)  # (N, 256, length)
        d3 = self.cin3(d3, z)  # Apply CIN after down3

        # Upsampling path
        u3 = self.up3(d3)  # (N, 128, length)
        u3 = self.cin4(u3, z)  # Apply CIN after up3

        u2 = self.up2(u3 + d2)  # (N, 64, length)
        u2 = self.cin5(u2, z)  # Apply CIN after up2

        u1 = self.up1(u2 + d1)  # (N, x_dim, length)
        u1 = self.cin6(u1, z)  # Apply CIN after up1

        output = self.final_conv(u1)  # Final convolution

        return self.activation(output)


# Test generator model forward pass here

y_dim = 50
x_dim = 100
z_dim = 5

y = torch.randn(10, 50, 1)
z = torch.randn(10, 5, 1)

generator = UNetWithCIN1D(y_dim=y_dim, x_dim=x_dim,
                          z_dim=z_dim)

output = generator(y, z)
print("Generator model shape: ", output.shape)


''' Critic Model: Idea here is to reduce x dimension to y dimension, then stack them and reduce to a scalar. '''
''' Assumption: x_dim = 100, y_dim = 50'''


class CriticModel(nn.Module):
    def __init__(self, x_dim=100, y_dim=50):
        super(CriticModel, self).__init__()

        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
        # stack X and Y into channel and feed into 1d convolution
        self.conv = nn.Conv1d(in_channels=2, out_channels=1,
                              kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(50, 1)  # dense layer to reduce to scalar

    def forward(self, x, y):
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)

        x = self.avg_pool(x)

        # combined shape: [batch_size, 2, 50]
        combined = torch.cat((x, y), dim=1)

        combined = self.conv(combined)
        combined = combined.squeeze(1)

        # Pass through the fully connected layer to reduce to scalar
        output = self.fc(combined)

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
        y = torch.randn(batch_size, self.y_size).to(x.device)  # Generate dummy y tensor
        return self.model(x, y)

critic = CriticSummaryWrapper(critic, y_size=50)  # Ensure y_size matches your model
summary(critic, input_size=(100,))

generator = ModelSummaryWrapper(generator, z_dim=5)
summary(generator, input_size=(50, 1))
