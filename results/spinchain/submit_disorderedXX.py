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

# Specify the number of spin sites
N = 8 

# Specify the initial condition. Here: domain wall |111...000>
initstate= np.zeros(N)
for i in range(int(N/2)):
   initstate[i] = 1.0 

# Specify h and J amplitudes
hamp = 1.0
Jamp = 1.0   

# Specify the number of samples for h
nsamples = 10

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
        jobname =  inputname+str(N) + "_Lindblad_" + str(isample)

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



