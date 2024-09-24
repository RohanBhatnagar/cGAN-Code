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


class NumpyToTorchDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data, device):
        self.x_data = torch.tensor(x_data, dtype=torch.float).to(device)
        self.y_data = torch.tensor(y_data, dtype=torch.float).to(device)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


random.seed(PARAMS.seed_no)
np.random.seed(PARAMS.seed_no)
torch.manual_seed(PARAMS.seed_no)
metric_save_freq = max(PARAMS.save_freq // 100, 1)

print("\n ============== LAUNCHING TRAINING SCRIPT =================\n")

print("\n --- Creating network folder \n")
savedir = make_save_dir(PARAMS)


print("\n --- Loading training datasets \n")
X_dataset_full = np.load(PARAMS.X_dataset)
Y_clean_dataset_full = np.load(PARAMS.Y_clean_dataset)
Y_noisy_dataset_full = np.load(PARAMS.Y_noisy_dataset)

print(X_dataset_full.shape)
print(Y_clean_dataset_full.shape)
print(Y_noisy_dataset_full.shape)


X_train = torch.tensor(X_dataset_full[:PARAMS.n_train, :], dtype=torch.float32)
Y_noisy_train = torch.tensor(
    Y_noisy_dataset_full[:PARAMS.n_train, :], dtype=torch.float32)

X_valid = torch.tensor(X_dataset_full[PARAMS.n_train:(
    PARAMS.n_train+PARAMS.n_valid), :], dtype=torch.float32)
Y_noisy_valid = torch.tensor(Y_noisy_dataset_full[PARAMS.n_train:(
    PARAMS.n_train+PARAMS.n_valid), :], dtype=torch.float32)

print(X_valid.shape, Y_noisy_valid.shape)

# Creating data loader for training data
data_object = NumpyToTorchDataset(X_train, Y_noisy_train, dev)
loader = DataLoader(data_object, batch_size=PARAMS.batch_size,
                    shuffle=True, drop_last=True)


print("\n --- Creating conditional GAN models\n")
#use elu
if PARAMS.act_func == "tanh":
    activation_function = torch.nn.Tanh()
elif PARAMS.act_func == "elu":
    activation_function = torch.nn.ELU()
elif PARAMS.act_func == "relu":
    activation_function = torch.nn.ReLU()

# G_model = MLP(
#     input_dim=Y_noisy_train.shape[1]+PARAMS.z_dim,
#     output_dim=X_train.shape[1],
#     # hidden_widths=(128, 256, 64, 32),
#     hidden_widths=(100, 100, 100),
#     activation=activation_function,
# )

G_model = G_model_CNN(
    x_dim=X_train.shape[1],
    y_dim=Y_noisy_train.shape[1],
    activation=activation_function,
)

# D_model = MLP(
#     input_dim=X_train.shape[1] + Y_noisy_train.shape[1],
#     output_dim=1,
#     # hidden_widths=(128, 256, 64, 32),
#     hidden_widths=(100, 50, 20, 10),
#     activation=activation_function,
# )

D_model = D_model_CNN(
    x_dim=X_train.shape[1],
    y_dim=Y_noisy_train.shape[1],
    activation=activation_function,
)

# Moving models to correct device and adding optimizers
G_model.to(device)
D_model.to(device)
G_optim = torch.optim.AdamW(
    G_model.parameters(), lr=0.001, weight_decay=PARAMS.reg_param
)
D_optim = torch.optim.AdamW(
    D_model.parameters(), lr=0.001, weight_decay=PARAMS.reg_param
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
    for x, y in loader:
        true_X = x.to(device)
        true_Y = y.to(device)

        # ---------------- Updating critic -----------------------
        D_optim.zero_grad()
        z = glv(PARAMS.batch_size)
        z = z.to(device)

        # print(true_Y.shape, z.shape)

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
        true_X = X_valid
        true_Y = Y_noisy_valid.to(device)
        z = glv(PARAMS.n_test)
        z = z.to(device)

        # MC insteead of test - tile true_Y, then generate corresponding X to get a distribution
        # # then plot the mean of the Xs, mean +- stddev

        fake_X = G_model(torch.cat((true_Y, z), dim=1)).cpu().detach().numpy()
        rel_L2_error = calc_rel_L2_error(
            true_X.detach().numpy(), true_Y.cpu().detach().numpy(), fake_X
        )
        rel_L2_error_log.append(rel_L2_error)

        if (i + 1) % PARAMS.save_freq == 0:
            torch.save(G_model.state_dict(),
                       f"{savedir}/checkpoints/G_model_{i+1}.pth")

        if (i + 1) % PARAMS.plot_freq == 0:
            one_Y = torch.tensor(np.tile(true_Y[0], (PARAMS.z_n_MC, 1))).to(device)
            z = glv((PARAMS.z_n_MC)).to(device)

            fake_X_dist = G_model(
                torch.cat((one_Y, z), dim=1)).cpu().detach().numpy()

            mean_fake_X = np.mean(fake_X_dist, axis=0)
            stddev_fake_X = np.std(fake_X_dist, axis=0)

            # print(f'*___________________{stddev_fake_X}___________________*')
            plt.figure()
            plt.plot(true_X[0], label=f"True X Sample {i+1}", color="blue")
            # plt.plot(true_Y[0], label=f"True Y Sample {i+1}", linestyle='--', color="green")
            plt.plot(
                mean_fake_X, label=f"Predicted Mean X Sample {i+1}", color="red")
            plt.fill_between(range(len(mean_fake_X)),
                             mean_fake_X - stddev_fake_X,
                             mean_fake_X + stddev_fake_X,
                             color='red', alpha=0.3, label=f"Predicted X ± StdDev")
            
            plt.plot(
                fake_X_dist[0,:], label=f"X_1")
            plt.plot(
                fake_X_dist[1,:], label=f"X_2")
            plt.plot(
                fake_X_dist[2,:], label=f"X_3")
            
            plt.xlabel("Index")
            plt.ylabel("Value")
            plt.legend()
            plt.title(
                f"True X, True Y, and Predicted Mean X with StdDev for Sample {i+1}")
            plt.savefig(f'{savedir}/plot{i+1}.pdf', bbox_inches='tight')
            plt.close()

save_loss(G_loss_log, "g_loss", savedir, PARAMS.n_epoch)
save_loss(D_loss_log, "d_loss", savedir, PARAMS.n_epoch)
save_loss(wd_loss_log, "wd_loss", savedir, PARAMS.n_epoch, scale="log")
save_loss(rel_L2_error_log, "rel_L2_error", savedir, PARAMS.n_epoch)

# plot network weights (abs)
plot_network_weights(G_model, "G_model", savedir)
plot_network_weights(D_model, "D_model", savedir)


print("\n ============== DONE =================\n")