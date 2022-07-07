#!/usr/bin/env python
import sys
# Set path to utility folder (so that python finds config.py, ordered_bunch.py...)
sys.path.append('/Users/guenther5/Numerics/quandary/util/')

import os
import copy
from batch_job import submit_job
from config import *
from util import *

# Specify runcommand for the cluster ("srun -n" on quartz or "mpirun -np" on elwe)
runcommand = "srun -n"

# Specify the global config file 
configfile = Config("AxC.cfg")

# Specify the varying parameters
nlevels = ["2,2", "4,4", "8,8"]    # nlevels is handled as a string option in util/config.py. Other options might be handled as floats or ints, check the util/config.py file.

# Specify number of cores
npt = [2,4,8]

# Submit a job for each parameter setup
for i in range(len(nlevels)):

        # Copy config file and set new config file options
        konfig = copy.deepcopy(configfile)
        konfig.nlevels = nlevels[i]   

        # Specify jobname 
        jobname =  \
                  "nlevels"  + konfig.nlevels  +\
                  "npt"  + str(npt[i]) 

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
        submit_job(jobname, runcommand, npt[i], "00:02:00", "../main", newconfigfile, "sqoc", False) # sqoc is the slurm account name. check util/batch_job.py for this function!
        #submit_job(jobname, runcommand, npt[i], "00:02:00", "../main", newconfigfile, "sqoc", True) 
        os.chdir("../")


