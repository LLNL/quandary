#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

############
# This script gathers Quandary results (nhalf and magnetization) from subfolders from each sample and writes the (averaged) Nhalf and (averaged) local magnetization to a files. 
############

# Number of spin sites
N=8
#N=12

# Number of samples to average over
nsamples=10

# Specify the prefix and postfix of folder names where data is located
folder_prefix  = "spinchain" + str(N)   # everything before _sample<x>
folder_postfix = "/data_out"            # everything after _sample<x>, including path to sub-data folder
#ncores = 1
#folder_postfix = "_ncores" + str(ncores) + "/data_out"    

# Final time
T=10.0

# Number of time steps
ntime = 1000    # for N=8
#ntime = 1500    # for N=12

# Choose to plot Nhalf and local magnetization 
do_plots = True

# Average nhalf over each sample
nhalf_avg = np.zeros(ntime+1)
magnet_avg = np.zeros((N,ntime+1))
for isample in range(nsamples):

        foldername =  folder_prefix +"_sample" + str(isample) + folder_postfix

        # Add to nhalf for all times points
        filename = foldername + "/nhalf.iinit0001.dat"
        print("Reading from file ", filename) 
        with open(filename) as f:
            lines = f.readlines()
        for nt in range(ntime+1):
            words = lines[nt].split()
            nhalf_avg[nt] += float(words[1])

        # Add to local magnetization for all time points
        filename = foldername + "/magnetization.iinit0001.dat"
        print("Reading from file ", filename) 
        with open(filename) as f:
            lines = f.readlines()
        for nt in range(ntime+1):
            words = lines[nt].split()
            for i in range(N):
                magnet_avg[i,nt] += float(words[i+1])

# Finalize averaging
for nt in range(ntime+1):
    nhalf_avg[nt] = nhalf_avg[nt] / nsamples
    for i in range(N):
        magnet_avg[i,nt] = magnet_avg[i,nt] / nsamples

# Write nhalf and magnetization averages to files
dt = float(T) / float(ntime)
times = np.zeros(ntime+1)
for i in range(ntime+1):
    times[i] = i*dt

fnhalf = open("nhalf_avg.dat", "w")
fmagnet = open("magnetization_avg.dat", "w")
fnhalf.write("# time    nhalf_avg\n")
fmagnet.write("# time    magnetization site j\n")
for nt in range(ntime+1):
    fnhalf.write(str(times[nt])+ " "+ str(nhalf_avg[nt])+"\n")
    fmagnet.write(str(times[nt])+ " ")
    for i in range(N):
        fmagnet.write(str(magnet_avg[i,nt])+ " ")
    fmagnet.write("\n")
fnhalf.close()
fmagnet.close()
print("File written: nhalf_avg.dat")
print("File written: magnetization_avg.dat")

if do_plots:
    # Plot average nhalf over time
    print("Plotting averaged Nhalf...")
    plt.plot(times, nhalf_avg)
    plt.show()
    
    # Plot a heatmap for averaged magnetization 
    print("Plotting averaged magnetization...")
    fig, ax = plt.subplots(figsize=(6,4))
    mycmap = plt.get_cmap('coolwarm')
    plt.imshow(magnet_avg, interpolation='none', aspect='auto', cmap=mycmap)
    plt.colorbar()
    plt.title(r"Heat Map of Spin Chain $\langle \sigma^z_j \rangle$")
    plt.xlabel("Time-step index")
    plt.ylabel("Spin Site Index $j$")
    plt.show()
