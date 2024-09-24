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


def generate_bimodal(interval=(-3.0, 3.0), N_total=1000):
    dataset = np.ones((N_total , 2))
    dataset[:, 0] = np.random.normal(0, 1.0, N_total)
    dataset[:, 1] = np.cbrt(
        dataset[:, 0] + np.random.normal(0, 1, size=N_total)
    )
    dataset[:,[0,1]] = dataset[:,[1,0]]

    return dataset


def generate_swissroll(noise, N_total=1000):
    t = 3 * np.pi / 2 * (1 + 2 * np.random.rand(1, N_total))
    dataset = np.concatenate(
             (0.1 * t * np.cos(t), 0.1 * t * np.sin(t))
             ) + noise * np.random.randn(2, N_total)

    dataset = np.transpose(dataset)
    dataset[:,[0,1]] = dataset[:,[1,0]]
    return dataset


def generate_tanh(interval=(-3.0, 3.0), N_total=100):
    dataset = np.ones((N_total, 2))
    dataset[:, 0] = np.random.uniform(interval[0], interval[1], N_total)
    dataset[:, 1] = np.tanh(dataset[:, 0]) + np.random.gamma(
        shape=1.0, scale=0.3, size=N_total)

    dataset[:,[0,1]] = dataset[:,[1,0]]
    return dataset

print("\n ============== RUNNING SCRIPT TO GENERATE DATA =================\n")

N_samples = 2000

# print("\n --- Creating bimodal data\n")
# dataset = generate_bimodal(interval=(-3.0, 3.0), N_total=N_samples)
# np.save('bimodal_data.npy',dataset)
# plt.figure()
# plt.hist2d(dataset[:,0],dataset[:,1],density=True,bins=200)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.savefig('bimodal_data.pdf',bbox_inches='tight')
# plt.close()

# print("\n --- Creating tanh data\n")
# dataset = generate_tanh(interval=(-3.0, 3.0), N_total=N_samples)
# np.save('tanh_data.npy',dataset)
# plt.figure()
# plt.hist2d(dataset[:,0],dataset[:,1],density=True,bins=200)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.savefig('tanh_data.pdf',bbox_inches='tight')
# plt.close()

# print("\n --- Creating swissroll data\n")
# dataset = generate_swissroll(N_total=N_samples, noise=0.1)
# np.save('swissroll_data.npy',dataset)
# plt.figure()
# plt.hist2d(dataset[:,0],dataset[:,1],density=True,bins=200)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.savefig('swissroll_data.pdf',bbox_inches='tight')
# plt.close()




print("\n ============== DONE =================\n")
