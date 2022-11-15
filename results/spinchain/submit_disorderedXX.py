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

# Location and name of the Quandary executable
executable = "/Users/guenther5/Numerics/quandary/main"

N = 8 # Number of spin sites

# Specify h and J amplitudes (frequency domain, quandary multiplies by 2pi!)
hamp = 1.0   # GHz?
Jamp = 1.0   

# Specify the initial condition. Here: domain wall |111...000>
initstate= np.zeros(N)
for i in range(int(N/2)):
   initstate[i] = 1.0 

# Specify the number of samples for h
nsamples = 1 # 10


# Submit the job(s).
for isample in range(nsamples):

        # Sample h uniform random, J fixed
        #h = np.random.uniform(-hamp, hamp, N)
        h = np.ones(N)
        J = np.ones(N)
        for i in range(N):
            h[i] = h[i] / np.pi
            J[i] = J[i] / np.pi
    
        # Set up configuration option strings
        transfreqstr = ""
        rotfreqstr = ""
        Jklstr = ""
        targetstr = "pure, "
        id = 0
        for i in range(N):
            transfreqstr += str(h[i]) + ", "
            rotfreqstr += "0.0, "
            targetstr += "0, "
            for j in range(i+1,N):
                if j == i+1:
                    Jklstr += str(J[i]) + ", "
                else :
                    Jklstr += "0.0, "
        #print("Jkl", Jklstr)
        #print("transfreq", transfreqstr)
        #print("target", targetstr)

        # Specify the jobname 
        jobname =  inputname+str(N) +"_" + str(isample)

        # create folder for the job
        if os.path.exists(jobname):
           pass
        else:
           os.mkdir(jobname)

        # Create a config file
        konfig = copy.deepcopy(configfile)
        konfig.transfreq = transfreqstr
        konfig.rotfreq = rotfreqstr
        konfig.optim_target= targetstr
        konfig.Jkl = Jklstr
        newconfigfile = jobname + ".cfg"
        konfig.dump(jobname + "/" + newconfigfile)

        # submit the job
        os.chdir(jobname)
        print("Submitting job ", jobname)  #, ":  h / 2pi = ", h, ", J / 2pi = ", J)
        submit_job_local(jobname, executable, newconfigfile, True) 
        os.chdir("../")


# Gather results from each job

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
    nhalf_avg[nt] / nsamples
    for i in range(N):
        magnet_avg[i,nt] / nsamples

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

