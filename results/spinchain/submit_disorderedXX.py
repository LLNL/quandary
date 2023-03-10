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

# Specify the global configuration file for Quandary (each run copies global configuration options from this file, and overwrites the relevant config options for the new h-sample)
inputname = "spinchain"
configfile = Config(inputname + ".cfg")

# Specify the location of the Quandary executable
executable = "/g/g90/guenther/Numerics/quandary/main"  

# Specify the mpi runcommand
# runcommand = "srun -n "         # on LC Quartz 
runcommand = "mpirun -np "      # Locally on Stefanie's Mac

# Choose to submit jobs to LC ('True') or run locally within this terminal ('False'). 
# If submitting to LC, specify your bank
do_submit_LC = False
bank = "qude"       # Stefanie's UDE bank. 

# FOR DEBUGGING: Setting this to 'True' will not execute quandary (nor submit a job), but will only create the subfolders and dump the configuration file and potentially the batch script into it.  
debug = False

# Specify the number of spin sites
N=8
#N=12
#N=16
#N=20

# Set the final time and number of time steps
T=10.0
ntime = 1000   # for N=8
#ntime = 1500    # for N=12
#ntime = 2000    # for N=16
#ntime = 3500    # for N=20

# Specify the initial state at time t=0. Here: domain wall |111...000>
initstate= np.zeros(N)
for i in range(int(N/2)):
   initstate[i] = 1


# Specify the number of samples for randomizing h as well as the amplitude for the random number generator
nsamples = 10
hamp = 1.0 / np.pi      # Quandary multiplies by 2pi internally!

# Specify J for each spin site. Here constant
Jamp = 1.0 / np.pi   
J = np.ones(N)
for i in range(N):
    J[i] = Jamp*J[i] 

# Specify U for each spin site. Here U=0
Uamp = 0.0 / np.pi
U = np.ones(N)
for i in range(N):
    U[i] = Uamp*U[i] 
 
# Specify number of cores to run on, and potentially the corresponding runtime that is to be allocated when submitting this job to LC. 
# This script will submit <nsamples> jobs for each selected number of cores.
# To perform a parallel scaling study, set nsamples=1 and choose the number of cores and runtimes for the study. 
# If not performing a scaling study, choose nsamples > 1, and ncores=[<x>] for x begin the number of cores to run on. 
ncores = [1]                          # best for N=8
#ncores = [32]                        # best for N=12
#ncores = [1,2,4,8, 16, 32]           # scaling study for N=12
#ncores = [16,32,64, 128, 256, 512]   # scaling study for N=20
runtime = ["00:00:05"]       # for N=8 on 1 core
#runtime = ["00:00:15"]      # for N=12 on 32 cores
#runtime = ["00:01:00", "00:00:40", "00:00:30", "00:00:20","00:00:10", "00:00:10"]      # scaling study for N=12 (over-estimated)

# Submit <nsample> jobs for each number of spatial processors
for icores in range(len(ncores)):

    # Submit one job for each sample
    for isample in range(nsamples):
    
            # Sample h uniform random for each spin site
            h = np.random.uniform(-hamp, hamp, N)
            #h = np.ones(N)
       
            # Set up configuration option strings
            nlevels = ""
            transfreq = ""
            Jkl = ""
            crosskerr = ""
            initialcondition = "pure, "
            id = 0
            for i in range(N):
                nlevels += "2, "
                transfreq += str(h[i] + 2*U[i]) + ", "
                for j in range(i+1,N):
                    if j == i+1:
                        Jkl += str(J[i]) + ", "
                        crosskerr += str(2*U[i]) + ", "
                    else :
                        Jkl += "0.0, "
                        crosskerr += "0.0, "
                initialcondition += str(int(initstate[i])) + ", "
    
            # Specify the jobname 
            jobname =  inputname+str(N) + "_sample" + str(isample)
            if (len(ncores)>1):
                       jobname += "_ncores" + str(ncores[icores])
    
            # create folder for the job
            if os.path.exists(jobname):
               pass
            else:
               os.mkdir(jobname)
    
            # Create a config file
            konfig = copy.deepcopy(configfile)
            konfig.nlevels = nlevels
            konfig.transfreq = transfreq
            konfig.crosskerr = crosskerr
            konfig.Jkl = Jkl
            konfig.initialcondition = initialcondition
            konfig.ntime= ntime
            konfig.dt = float(T) / float(ntime)
            newconfigfile = jobname + ".cfg"
            konfig.dump(jobname + "/" + newconfigfile)
    
            # Execute Quandary locally, or submit the job on LC
            os.chdir(jobname)
            if do_submit_LC:
                print("Submitting job ", jobname)
                submit_job(jobname, runcommand, ncores[icores], runtime[icores], executable, newconfigfile, bank, not debug) # submit batch job
            else:
                print("Running Quandary: ", runcommand + str(ncores[icores]) + " " + executable + " " + newconfigfile)
                run_local(runcommand, ncores[icores], executable, newconfigfile, not debug) # Run locally in this terminal
            os.chdir("../")
