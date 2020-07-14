#!/usr/bin/env python
import sys
import os
import copy
from batch_job import submit_job
from config import *
from util import *

# Set path to utility folder (so that python finds config.py, ordered_bunch.py...)
sys.path.append('../util/')

# Specify runcommand for the cluster ("srun -n" on quartz or "mpirun -np" on elwe)
runcommand = "srun -n"

# Specify the global config file 
configfile = Config("AxC.cfg")

# Specify the varying parameters
cfactor = [2, 5, 10]

# Specify number of cores
npt = [2,4,8,16,32,64,128, 256, 512, 1024]

# Submit a job for each parameter setup
for cf in range(len(cfactor)):

    for np in range(len(npt)):

        # Copy config file and set new config file options
        konfig = copy.deepcopy(configfile)
        konfig.braid_cfactor = cfactor[cf]
        konfig.np_braid = npt[np]

        # Specify jobname 
        jobname =  \
                  "cf"  + str(konfig.braid_cfactor)  +\
                  "npt"  + str(konfig.np_braid) 

        # create folder for the job
        if os.path.exists(jobname):
           pass
        else:
           os.mkdir(jobname)

        # Create a config file
        newconfigfile = jobname + ".cfg"
        konfig.dump(jobname + "/" + newconfigfile)

        # submit the job
        os.chdir(jobname)
        submit_job(jobname, runcommand, npt[np], "00:02:00", "../main", newconfigfile, "sqoc", False)
        os.chdir("../")


