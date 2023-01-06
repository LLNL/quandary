#!/usr/bin/env python
import sys
# Set path to Quandary's utility folder (so that python finds config.py, ordered_bunch.py...)
sys.path.append('/g/g90/guenther/Numerics/quandary/util/')

import os
import copy
from batch_job import submit_job
from batch_job import run_local
from config import *
from util import *
import numpy as np
import matplotlib.pyplot as plt

###############
# This script samples h from Uniform distribution and executes Quandary for each sample, either locally in a terminal or submitting a batch job to LC. 
# This script can also be used to perform a parallel scaling study on LC.
###############

# Specify the global configuration file for Quandary (each run copies global config options from this file, overwriting the relevant config options for the new h sample)
inputname = "spinchain"
configfile = Config(inputname + ".cfg")

# Specify the location of the Quandary executable
executable = "/g/g90/guenther/Numerics/quandary/main"

# Specify the mpi runcommand
runcommand = "srun -n "      # on LC Quartz 
#runcommand = "mpirun -np "

# Choose to submit jobs to LC (do_submit_LC=True) rather than executing locally in this terminal (do_submit_LC=False). 
do_submit_LC = False

# If submitting to LC: Specify a bank
bank = "qude"

# Specify the number of spin sites
N=8
#N=12
#N=16
#N=20

# Set the final time
T=10.0

# Set the Number of time steps
ntime = 1000   # for N=8
#ntime = 1500    # for N=12
#ntime = 2000   # for N=16
#ntime = 3500   # for N=20

# Specify the initial state at time t=0. Here: domain wall |111...000>
initstate= np.zeros(N)
for i in range(int(N/2)):
   initstate[i] = 1

# Specify the number of samples for randomizing h
nsamples = 10

# Specify h and J amplitudes (frequency domain)
hamp = 1.0  
Jamp = 1.0   

# Specify number of cores to run on, and potentially the corresponding runtime that is to be allocated when submitting this job to LC. 
# Note: 'ncores' and 'runtime' should have the same length!
# This script will submit <nsamples> jobs for each selected number of cores in the below 'ncores' variable.
# To perform a parallel scaling study, set nsamples=1 and choose the number of cores and runtimes for the study. 
# If not performing a scaling study, choose nsamples > 1, and ncores=[<x>] for x begin the number of cores to run on (length(ncores)=1)
ncores = [1]                          # best for N=8
#ncores = [32]                        # best for N=12
#ncores = [1,2,4,8, 16, 32]           # scaling study for N=12
#ncores = [16,32,64, 128, 256, 512]   # scaling study for N=20
runtime = ["00:00:05"]       # for N=8 on 1 core
#runtime = ["00:00:15"]      # for N=12 on 32 cores
#runtime = ["00:01:00", "00:00:40", "00:00:30", "00:00:20","00:00:10", "00:00:10"]      # scaling study for N=12 (over-estimated)

# Submit <nsample> jobs for each number of spacial processors
for icores in range(len(ncores)):

    # Submit one job for each sample
    for isample in range(nsamples):
    
            # Sample h uniform random, J fixed
            h = np.random.uniform(-hamp, hamp, N)
            #h = np.ones(N)
            J = np.ones(N)
            for i in range(N):
                h[i] = h[i] / np.pi   # Quandary multiplies by 2pi internally!
                J[i] = Jamp*J[i] / np.pi
        
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
    
            # Specify the jobname 
            jobname =  inputname+str(N) +\
                       "_sample" + str(isample) +\
                       "_ncores" + str(ncores[icores])
    
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
    
            # Execute Quandary locally, or submit the job on LC
            os.chdir(jobname)
            if do_submit_LC:
                print("Submitting job ", jobname)
                submit_job(jobname, runcommand, ncores[icores], runtime[icores], executable, newconfigfile, bank, True) # submit batch job
            else:
                print("Running Quandary: ", runcommand + str(ncores[icores]) + " " + executable + " " + newconfigfile)
                run_local(runcommand, ncores[icores], executable, newconfigfile, True) # Run locally in this terminal
            os.chdir("../")
