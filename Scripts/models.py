# Conditional GAN Script
# Created by: Deep Ray, University of Maryland
# Date: 27 August 2023


import os
import numpy as np
import matplotlib.pyplot as plt
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


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_widths, activation):
        super(MLP, self).__init__()

        layers = []

        # Add input layer
        layers.append(
            nn.Linear(input_dim, hidden_widths[0])
        )  # Linear transformation: input_dim -> first hidden layer width
        layers.append(activation)  # Activation function

        # Add hidden layers
        for i in range(1, len(hidden_widths)):
            layers.append(
                nn.Linear(hidden_widths[i - 1], hidden_widths[i])
            )  # Linear transformation: previous width -> current width
            layers.append(activation)  # Activation function

        # Add output layer
        layers.append(
            nn.Linear(hidden_widths[-1], output_dim)
        )  # Linear transformation: last hidden layer width -> output_dim

        # Construct the network as a sequence of layers
        self.network = nn.Sequential(*layers)

        # Initialize weights and biases for each linear layer
        # for layer in self.network:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.uniform_(
        #             layer.weight, -1, 1
        #         )  # Initialize weights with uniform distribution between -1 and 1
        #         nn.init.uniform_(
        #             layer.bias, -1, 1
        #         )  # Initialize biases with uniform distribution between -1 and 1

    def forward(self, x):
        # Define how input data passes through the network
        return self.network(x)


class G_model_CNN(nn.Module):
    def __init__(self, z_dim, y_dim, activation):
        super(G_model_CNN, self).__init__()

        self.input_dim = y_dim + z_dim

        # upscaling from 55 to 100
        self.transpose_input = nn.ConvTranspose1d(
            in_channels=1,
            out_channels=1,
            kernel_size=6,
            stride=2,
            padding=1
        )

        # padded Conv1D layers to maintain shape for X output
        self.network = nn.Sequential(
            nn.ConvTranspose1d(in_channels=1, out_channels=2,
                               kernel_size=3, stride=1),
            activation,
            nn.ConvTranspose1d(in_channels=2, out_channels=4,
                               kernel_size=3, stride=1),
            activation,
            nn.Conv1d(in_channels=4, out_channels=8,
                      kernel_size=3, stride=1, padding=1),
            activation,
            nn.Conv1d(in_channels=8, out_channels=4,
                      kernel_size=3, stride=1, padding=1),
            activation,
            nn.Conv1d(in_channels=4, out_channels=2,
                      kernel_size=3, stride=1, padding=1),
            activation,
            nn.Conv1d(in_channels=2, out_channels=1,
                      kernel_size=3, stride=1, padding=1)
        )

    def forward(self, y, z):
        x = torch.cat((y, z), dim=1)  # stack y, z
        x = x.unsqueeze(1)  # add channel
        x = self.tranpose_input(x)  # increase input dimension to 100
        return x.squeeze(1)  # remove channel
# 1d convolution layer 
# input size 100, conv layer size 3 stride 1
# see how many layers you need 
#  , increase number of channels to 2, then 4, then 8, 4,2, back to 1 
# 


# 150 down to 1, using 1d convolution layers with elu activation 
# use avg pooling to shrink 


# input y, z stacked 
# tranpose conv layer to inc dimension to 100
# followed by elu 
# another tranpose 
# elu 

# k = 3, 5, stide = 2, 1

class D_model_CNN(nn.Module):
    def __init__(self, x_dim, y_dim, activation):
        super(D_model_CNN, self).__init__()

        self.input_dim = x_dim + y_dim

        self.network = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3, stride=1),
            activation,
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, stride=1),
            activation,
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=1),
            activation,
            nn.AvgPool1d(kernel_size=2),
            nn.Flatten(),  # Flatten the output for the final linear layer
            # Adjust 150 based on the actual size after pooling
            nn.Linear(in_features=150, out_features=1)
        )

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)  # Concatenate x and y
        x = x.unsqueeze(1)  # Add channel dimension
        return self.network(x)


def get_lat_var(batch_size, z_dim):
    """This function generates latent variables"""
    z = randn((batch_size, z_dim))
    return z


def gradient_penalty_Adler(fake_X, true_X, true_Y, model, device, p=2, c0=1.0):
    """Evaluates gradient penalty term"""
    batch_size, *other_dims = true_X.size()
    epsilon = rand([batch_size] + [1 for _ in range(len(other_dims))])
    epsilon = epsilon.expand(-1, *other_dims).to(device)
    x_hat = epsilon * true_X + (1 - epsilon) * fake_X
    x_hat.requires_grad = True
    d_hat = model(x_hat, true_Y)
    grad = autograd.grad(
        outputs=d_hat,
        inputs=x_hat,
        grad_outputs=ones_like(d_hat).to(device),
        create_graph=True,
        retain_graph=True,
    )[0]
    grad = grad.view(batch_size, -1)
    grad_norm = sqrt(1.0e-8 + sum(square(grad), dim=1))
    grad_penalty = pow(grad_norm - c0, p).mean()
    return grad_penalty


def full_gradient_penalty(
    fake_X, true_X, true_Y, model, device, p=2, c0=1.0, concat=False
):
    """Evaluates full gradient penalty term"""
    batch_size, *other_dims = true_X.size()
    epsilon = rand([batch_size] + [1 for _ in range(len(other_dims))])
    epsilon = epsilon.expand(-1, *other_dims).to(device)
    x_hat = epsilon * true_X + (1 - epsilon) * fake_X
    x_hat.requires_grad = True
    true_Y.requires_grad = True
    if concat:
        input_xy = cat((x_hat, true_Y), dim=1)
        d_hat = model(input_xy)
    else:
        d_hat = model(x_hat, true_Y)
    grad = autograd.grad(
        outputs=d_hat,
        inputs=(x_hat, true_Y),
        grad_outputs=ones_like(d_hat).to(device),
        create_graph=True,
        retain_graph=True,
    )
    grad_x, grad_y = grad[0], grad[1]
    grad_x = grad_x.view(batch_size, -1)
    grad_y = grad_y.view(batch_size, -1)
    grad_norm = sqrt(
        1.0e-8 + add(sum(square(grad_x), dim=1), sum(square(grad_y), dim=1))
    )
    grad_penalty = pow(grad_norm - c0, p).mean()
    return grad_penalty
