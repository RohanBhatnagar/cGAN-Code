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


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


PARAMS = cla()

# assert PARAMS.z_dim == None or PARAMS.z_dim > 0

random.seed(PARAMS.seed_no)
np.random.seed(PARAMS.seed_no)
torch.manual_seed(PARAMS.seed_no)
metric_save_freq = max(PARAMS.save_freq // 100, 1)

print("\n ============== LAUNCHING TRAINING SCRIPT =================\n")


print("\n --- Creating network folder \n")
savedir = make_save_dir(PARAMS)


print("\n --- Loading training data from file\n")
# Assuming dataset if of size N x 2, with the first column corresponding to x (the inferred variable)
# and the second column corresponding to y (the measured variable)
dataset_full = np.load(f"../Data/dataset.npy")
train_data = torch.tensor(
    dataset_full[:PARAMS.n_train, :], dtype=torch.float32)
valid_data = torch.tensor(dataset_full[PARAMS.n_train:(
    PARAMS.n_train+PARAMS.n_valid), :], dtype=torch.float32)

# Creating data loader for training data
loader = DataLoader(train_data, batch_size=PARAMS.batch_size,
                    shuffle=True, drop_last=True)


print("\n --- Creating conditional GAN models\n")

if PARAMS.act_func == "tanh":
    activation_function = torch.nn.Tanh()
elif PARAMS.act_func == "elu":
    activation_function = torch.nn.ELU()
elif PARAMS.act_func == "relu":
    activation_function = torch.nn.ReLU()

G_model = MLP(
    input_dim=50+PARAMS.z_dim,
    output_dim=100,
    hidden_widths=(128, 256, 64, 32),
    activation=activation_function,
)

D_model = MLP(
    input_dim=150,
    output_dim=1,
    hidden_widths=(128, 256, 64, 32),
    activation=activation_function,
)


# Moving models to correct device and adding optimizers
G_model.to(device)
D_model.to(device)
G_optim = torch.optim.Adam(
    G_model.parameters(), lr=0.001, betas=(0.5, 0.9), weight_decay=PARAMS.reg_param
)
D_optim = torch.optim.Adam(
    D_model.parameters(), lr=0.001, betas=(0.5, 0.9), weight_decay=PARAMS.reg_param
)

# Creating sub-function to generate latent variables
glv = partial(get_lat_var, z_dim=PARAMS.z_dim)


# ============ Training ==================
print("\n --- Initiating GAN training\n")

n_iters = 1
G_loss_log = []
D_loss_log = []
rel_L2_error_log = []
wd_loss_log = []


for i in range(PARAMS.n_epoch):
    for true in loader:
        true_X = true[:, 0:100].to(device)
        true_Y = true[:, 100:150].to(device)

        # ---------------- Updating critic -----------------------
        D_optim.zero_grad()
        z = glv(PARAMS.batch_size)
        z = z.to(device)

        fake_X = G_model(torch.cat((true_Y, z), dim=1)).detach()
        fake_val = D_model(torch.cat((fake_X, true_Y), dim=1))
        true_val = D_model(torch.cat((true_X, true_Y), dim=1))
        gp_val = full_gradient_penalty(
            fake_X=fake_X,
            true_X=true_X,
            true_Y=true_Y,
            model=D_model,
            device=device,
            concat=True,
        )
        fake_loss = torch.mean(fake_val)
        true_loss = torch.mean(true_val)
        wd_loss = true_loss - fake_loss
        D_loss = -wd_loss + PARAMS.gp_coef * gp_val

        D_loss.backward()
        D_optim.step()
        D_loss_log.append(D_loss.item())
        wd_loss_log.append(wd_loss.item())
        print(
            f"     *** (epoch,iter):({i},{n_iters}) ---> d_loss:{D_loss.item():.4e}, gp_term:{gp_val.item():.4e}, wd:{wd_loss.item():.4e}"
        )

        # ---------------- Updating generator -----------------------
        if n_iters % PARAMS.n_critic == 0:
            G_optim.zero_grad()
            z = glv(PARAMS.batch_size)
            z = z.to(device)
            fake_X = G_model(torch.cat((true_Y, z), dim=1))
            fake_val = D_model(torch.cat((fake_X, true_Y), dim=1))
            G_loss = -torch.mean(fake_val)

            G_loss.backward()
            G_optim.step()
            G_loss_log.append(G_loss.item())
            print(f"     ***           ---> g_loss:{G_loss.item():.4e}")

        n_iters += 1

    # Saving intermediate output and generator checkpoint
    if (i + 1) % metric_save_freq == 0:
        true_X = valid_data[:, 0:100]
        true_Y = valid_data[:, 100:150].to(device)
        z = glv(PARAMS.n_test)
        z = z.to(device)

        print(true_X.shape, true_Y.shape, z.shape)     

        fake_X = G_model(torch.cat((true_Y, z), dim=1)).cpu().detach().numpy()
        rel_L2_error = calc_rel_L2_error(
            true_X.detach().numpy(), true_Y.cpu().detach().numpy(), fake_X
        )
        rel_L2_error_log.append(rel_L2_error)

        if (i + 1) % PARAMS.save_freq == 0:
            torch.save(G_model.state_dict(),
                       f"{savedir}/checkpoints/G_model_{i+1}.pth")
            plt.figure()
            plt.hist2d(fake_X.squeeze(), true_Y.cpu().detach(
            ).numpy().squeeze(), density=True, bins=200)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(f'{savedir}/val_hist2d_{i+1}.pdf', bbox_inches='tight')
            plt.close()


save_loss(G_loss_log, "g_loss", savedir, PARAMS.n_epoch)
save_loss(D_loss_log, "d_loss", savedir, PARAMS.n_epoch)
save_loss(wd_loss_log, "wd_loss", savedir, PARAMS.n_epoch)
save_loss(rel_L2_error_log, "rel_L2_error", savedir, PARAMS.n_epoch)


print("\n ============== DONE =================\n")
