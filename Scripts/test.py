# Conditional GAN Script
# Created by: Deep Ray, University of Maryland
# Date: 27 August 2023

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchinfo import summary
from functools import partial
import time
import random
from config import cla
from utils import *
from models import *

device = torch.device("cpu")


PARAMS = cla()


print('\n ============== RUNNING TESTING SCRIPT =================\n')

print('\n --- Loading conditional GAN generator from checkpoint\n')
if PARAMS.GANdir == None:
    GANdir = get_GAN_dir(PARAMS)
else:
    GANdir = PARAMS.GANdir

if PARAMS.act_func == "tanh":
    activation_function = torch.nn.Tanh()
elif PARAMS.act_func == "elu":
    activation_function = torch.nn.ELU()
elif PARAMS.act_func == "relu":
    activation_function = torch.nn.ReLU()

G_model = MLP(
    input_dim=1+PARAMS.z_dim,
    output_dim=1,
    hidden_widths=(128, 256, 64, 32),
    activation=activation_function,
)

G_state = torch.load(
    f"{GANdir}/checkpoints/G_model_{PARAMS.ckpt_id}.pth", map_location=torch.device(device))
G_model.load_state_dict(G_state)
G_model.eval()  # NOTE: switching to eval mode for generator

# Creating sub-function to generate latent variables
glv = partial(get_lat_var, z_dim=PARAMS.z_dim)

# Creating results directory
results_dir = f"{GANdir}/{PARAMS.results_dir}"
print(f"\n --- Generating results directory: {results_dir}")
if os.path.exists(results_dir):
    print('\n     *** Folder already exists!\n')
else:
    os.makedirs(results_dir)

z = glv(PARAMS.n_test)

y_list = true_stats(PARAMS.dataset)

if PARAMS.y_pert_sigma == 0.0:
    y_pert = torch.zeros((PARAMS.n_test, 1))
else:
    y_pert = torch.normal(std=PARAMS.y_pert_sigma, size=(PARAMS.n_test, 1))

fig, axs = plt.subplots(1, len(y_list))
for i, y in enumerate(y_list):
    y_in = y*torch.ones((PARAMS.n_test, 1)) + y_pert
    x_pred = G_model(torch.cat((y_in, z), dim=1)).detach().numpy().squeeze()
    axs[i].hist(x_pred, bins=100, density=True)
    axs[i].set_xlabel('x')
    axs[i].set_title(f"y={y:.2f}")


plt.savefig(
    f'{results_dir}/test_cond_hist_ckpt_{PARAMS.ckpt_id}.pdf', bbox_inches='tight')


print("----------------------- DONE --------------------------")
