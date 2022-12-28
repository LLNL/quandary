#!/usr/bin/env python
import sys
# Set path to Quandary's utility folder (so that python finds config.py, ordered_bunch.py...)
sys.path.append('/g/g90/guenther/Numerics/quandary/util/')

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
executable = "../main"
runcommand = "srun -n "

# Number of spin sites
#N=8
N=24

# Number of time steps
#ntime = 1000 
ntime = 3000

# Final time
T=10.0

# Specify h and J amplitudes (frequency domain, quandary multiplies by 2pi!)
hamp = 1.0   # GHz?
Jamp = 1.0   

# Specify the initial condition. Here: domain wall |111...000>
initstate= np.zeros(N)
for i in range(int(N/2)):
   initstate[i] = 1

# Specify the number of samples for h
nsamples = 1 # 10


# Specify number of cores and runtime
#npt = [1,2,4,8]
#npt = [16,32,64, 128, 256, 512, 1024]   #N=20
#npt = [1,2,4,8,16,32]
npt = [32,64, 128, 256, 512, 1024, 2048, 4096]
#npt = [1,2,4,8,16,32,64, 128, 256, 512, 1024, 2048]
runtime = ["04:00:00","02:30:00","01:50:00","01:00:00","00:50:00","00:40:00","00:30:00","00:30:00"]

# Submit job for each number of spacial processors
for inpt in range(len(npt)):

    # Submit job for each sample
    for isample in range(nsamples):
    
            # Sample h uniform random, J fixed
            #h = np.random.uniform(-hamp, hamp, N)
            h = np.ones(N)
            J = np.ones(N)
            for i in range(N):
                h[i] = h[i] / np.pi
                J[i] = J[i] / np.pi
        
            # Set up configuration option strings
            nlevels = ""
            transfreqstr = ""
            rotfreqstr = ""
            Jklstr = ""
            targetstr = "pure, "
            initialconditionstr = "pure, "
            id = 0
            for i in range(N):
                nlevels += "2, "
                transfreqstr += str(h[i]) + ", "
                rotfreqstr += "0.0, "
                initialconditionstr += str(int(initstate[i])) + ", "
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
            jobname =  inputname+str(N) +\
                       "_sample" + str(isample) +\
                       "_npt" + str(npt[inpt])
    
            # create folder for the job
            if os.path.exists(jobname):
               pass
            else:
               os.mkdir(jobname)
    
            # Create a config file
            konfig = copy.deepcopy(configfile)
            konfig.nlevels = nlevels
            konfig.nessential = nlevels
            konfig.transfreq = transfreqstr
            konfig.rotfreq = rotfreqstr
            konfig.optim_target= targetstr
            konfig.initialcondition = initialconditionstr
            konfig.ntime= ntime
            konfig.dt = float(T) / float(ntime)
            konfig.Jkl = Jklstr
            newconfigfile = jobname + ".cfg"
            konfig.dump(jobname + "/" + newconfigfile)
    
            # submit the job
            os.chdir(jobname)
            #print("Submitting job ", jobname)  #, ":  h / 2pi = ", h, ", J / 2pi = ", J)
            submit_job(jobname, runcommand, npt[inpt], runtime[inpt], executable, newconfigfile, "qude", True) 
            os.chdir("../")



