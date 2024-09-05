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


def get_lat_var(batch_size, z_dim):
    """This function generates latent variables"""
    z = randn((batch_size, z_dim))*0
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
