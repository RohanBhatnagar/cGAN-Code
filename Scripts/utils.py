# Conditional GAN Script
# Created by: Deep Ray, University of Maryland
# Date: 27 August 2023
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt


def make_save_dir(PARAMS):
    """This function creates the results save directory"""

    savedir = get_GAN_dir(PARAMS)

    if os.path.exists(savedir):
        print("\n     *** Folder already exists!\n")
    else:
        os.makedirs(savedir)

    # Creating directory to save generator checkpoints
    if os.path.exists(f"{savedir}/checkpoints"):
        print("\n     *** Checkpoints directory already exists\n")
    else:
        os.makedirs(f"{savedir}/checkpoints")

    print("\n --- Saving parameters to file \n")
    param_file = savedir + "/parameters.txt"
    with open(param_file, "w") as fid:
        for pname in vars(PARAMS):
            fid.write(f"{pname} = {vars(PARAMS)[pname]}\n")

    return savedir


def get_GAN_dir(PARAMS):
    savedir = (
        f"../trained_models/"
        f"{PARAMS.dataset}"
        f"_Nsamples{PARAMS.n_train}"
        f"_Ncritic{PARAMS.n_critic}_Zdim{PARAMS.z_dim}"
        f"_BS{PARAMS.batch_size}_Nepoch{PARAMS.n_epoch}"
        f"_actf_{PARAMS.act_func}"
        f"_GP{PARAMS.gp_coef}{PARAMS.sdir_suffix}"
    )
    return savedir


def save_loss(loss, loss_name, savedir, n_epoch, scale='linear'):
    np.savetxt(f"{savedir}/{loss_name}.txt", loss)
    fig, ax1 = plt.subplots()
    
    loss = [abs(l) for l in loss]
    ax1.plot(loss)
    ax1.set_xlabel("Steps")
    ax1.set_ylabel(loss_name)
    ax1.set_xlim([1, len(loss)])

    ax1.set_yscale(scale)

    ax2 = ax1.twiny()
    ax2.set_xlim([0, n_epoch])
    ax2.set_xlabel("Epochs")

    plt.tight_layout()
    plt.savefig(f"{savedir}/{loss_name}.png", dpi=200)
    plt.close()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calc_rel_L2_error(true_X, true_Y, fake_X):
    hist2d_fake, e1, e2 = np.histogram2d(
        fake_X[:, 0],
        true_Y[:, 0],
        bins=[75, 50],
        range=np.array([[-3, 3], [-2, 2]]),
        density=True,
    )
    hist2d_fake = hist2d_fake / np.sum(hist2d_fake)
    hist2d_true, e1, e2 = np.histogram2d(
        true_X[:, 0],
        true_Y[:, 0],
        bins=[75, 50],
        range=np.array([[-3, 3], [-2, 2]]),
        density=True,
    )
    hist2d_true = hist2d_true / np.sum(hist2d_true)
    relative_L2_error = np.sqrt(
        np.sum(np.power(hist2d_fake - hist2d_true, 2))
    ) / np.sum(np.power(hist2d_true, 2))

    return relative_L2_error

# True conditionals at fixed y's


def true_stats(dataset):
    if dataset == "tanh":
        y_list = [-1.0, 1.0]

    elif dataset == "bimodal":
        y_list = [-1.0, 1.0]

    elif dataset == "swissroll":
        y_list = [-0.6, 0.6]

    return y_list
