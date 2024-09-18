# Conditional GAN Script
# Created by: Deep Ray, University of Maryland
# Date: 27 August 2023

import os, shutil
import numpy as np
import matplotlib.pyplot as plt
import random

seed_no = 1008
random.seed(seed_no)
np.random.seed(seed_no)

def gp_samples1D(NN, n_samples, length_scale, output_scale):
    # GRF sample generation
    def RBF1d(x1, x2, length_scale, output_scale):
        diffs = np.expand_dims(x1 / length_scale, 1) - \
                np.expand_dims(x2 / length_scale, 0)
        r2 = np.sum(diffs**2, axis=2)
        return output_scale * np.exp(-0.5 * r2)

    X = np.linspace(0, 1, NN)[:, None]
    K = RBF1d(X, X, length_scale, output_scale)
    L = np.linalg.cholesky(K + 1e-10*np.eye(NN))
    gp_samples = L @ np.random.randn(NN, n_samples)
    del L
    return gp_samples


print("\n ============== RUNNING SCRIPT TO GENERATE DATA =================\n")

N_samples = 4000
X_data    = gp_samples1D(100,N_samples,0.1,1).T
Y_data    = X_data**2 #+ np.sin(X_data*np.pi)
Y_noisy   = Y_data + np.random.normal(0, 0.1, Y_data.shape)

print(X_data.shape)

np.save('X_data.npy',X_data)
np.save('Y_clean_data.npy',Y_data)
np.save('Y_noisy_data.npy',Y_noisy)


print("\n ============== DONE =================\n")
