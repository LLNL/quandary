#!/usr/bin/env python
import sys
# Set path to Quandary's utility folder (so that python finds config.py, ordered_bunch.py...)
sys.path.append('/Users/guenther5/Numerics/quandary/util/')
import os
import copy
from batch_job import submit_job
from batch_job import submit_job_local
from config import *
from util import *
import numpy as np
import matplotlib.pyplot as plt


# Specify the global config file 
inputname = "spinchain"
configfile = Config(inputname + ".cfg")

# Number of spin sites
N=8

# Number of samples to average over
nsamples=10

# Time domain
konfig = copy.deepcopy(configfile)
ntime = konfig.ntime
dt = konfig.dt
times = np.zeros(ntime+1)
for i in range(ntime+1):
    times[i] = i*dt

# Average nhalf over each sample
nhalf_avg = np.zeros(ntime+1)
magnet_avg = np.zeros((N,ntime+1))
for isample in range(nsamples):

        jobname =  inputname+str(N) +"_" + str(isample)

        # Add to nhalf for all times points
        filename = jobname + "/data_out/nhalf.iinit0001.dat"
        print("Reading from file ", filename) 
        with open(filename) as f:
            lines = f.readlines()
        for nt in range(ntime+1):
            words = lines[nt].split()
            nhalf_avg[nt] += float(words[1])


        # Add to local magnetization for all time points
        filename = jobname + "/data_out/magnetization.iinit0001.dat"
        print("Reading from file ", filename) 
        with open(filename) as f:
            lines = f.readlines()
        for nt in range(ntime+1):
            words = lines[nt].split()
            for i in range(N):
                magnet_avg[i,nt] += float(words[i+1])
#Average
for nt in range(ntime+1):
    nhalf_avg[nt] = nhalf_avg[nt] / nsamples
    for i in range(N):
        magnet_avg[i,nt] = magnet_avg[i,nt] / nsamples

# Create heatmap
print(np.shape(magnet_avg))

fig, ax = plt.subplots(figsize=(6,4))
mycmap = plt.get_cmap('coolwarm')
plt.imshow(magnet_avg, interpolation='none', aspect='auto', cmap=mycmap)
plt.colorbar()
plt.title(r"Heat Map of Spin Chain $\langle \sigma^z_j \rangle$")
plt.xlabel("Time-step index")
plt.ylabel("Spin Site Index $j$")
plt.show()



# Write avgs to file
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

# Plot
plt.plot(times, nhalf_avg)
plt.show()

