#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np

## Read files 
with open('expected_alice.dat', 'r') as f:
    l = [[float(num) for num in line.split()] for line in f]
exp0 = np.array(l)

with open('expected_cavity.dat', 'r') as f:
    l = [[float(num) for num in line.split()] for line in f]
exp1 = np.array(l)

with open('population_alice.dat', 'r') as f:
    l = [[float(num) for num in line.split()] for line in f]
pop0 = np.array(l)


# Plot
fig, axs = plt.subplots(1,2, figsize=(10,7))
fig.suptitle("Expected energy level")
axs[0].set_title("Alice")
axs[1].set_title("Cavity")

for i in range(exp0.shape[1]-1):
    axs[0].plot(exp0[:,0], exp0[:, i+1])
    axs[1].plot(exp1[:,0], exp1[:, i+1])
for ax in axs.flat:
    ax.set(xlabel='duration')
plt.show()


fig, axs = plt.subplots(1,3, figsize=(20,7))
fig.suptitle("Alice population aka photon number")
for i in range(exp0.shape[1]-1):  # i = 0,...,8
    k = i*3
    axs[0].plot(pop0[:,0], pop0[:, k+1])
    axs[1].plot(pop0[:,0], pop0[:, k+2])
    axs[2].plot(pop0[:,0], pop0[:, k+3])
for ax in axs.flat:
    ax.set(xlabel='duration')
plt.show()

