#!/usr/bin/env python

# Set path to quandary/util folder so that python finds config.py, ordered_bunch.py...
import sys
sys.path.append('/Users/guenther5/Numerics/quandary_master/util/')

import os
import copy
from batch_job import submit_job
from config import *
from util import *
import matplotlib.pyplot as plt
import numpy as np

# Specify runcommand
runcommand = "../../../main"

# Specify the global config file 
configfile = Config("AxC.cfg")

# Specify different cavity amplitudes
optim_init_ampl = [1,2,3,4,5,6,7,8,9]

# Set number and spacing of time-duration
nsteps = 40
dsteps = 0.05

# Prepare output
time   = np.zeros(nsteps)
exp0   = np.zeros(len(optim_init_ampl)*nsteps).reshape(len(optim_init_ampl), nsteps)
exp1   = np.zeros(len(optim_init_ampl)*nsteps).reshape(len(optim_init_ampl), nsteps)
pop0_0 = np.zeros(len(optim_init_ampl)*nsteps).reshape(len(optim_init_ampl), nsteps)
pop0_1 = np.zeros(len(optim_init_ampl)*nsteps).reshape(len(optim_init_ampl), nsteps)
pop0_2 = np.zeros(len(optim_init_ampl)*nsteps).reshape(len(optim_init_ampl), nsteps)

# For each cavity amplitude, submit jobs for all drive durations from (0,2us] with spacing 0.01us (200 jobs per cavity amplitude)
for amp in range(len(optim_init_ampl)):

    # Iterate over durations t0=0.01, to tf=2.0
    for i in range(1,nsteps+1):

        # Number of time steps to simulate
        N = i*dsteps / configfile.dt + 1040

        # pipulse start and stop
        tstart = i * dsteps
        tstop = tstart + 0.104

        # string for pipulse config option
        pipulsestr = "0, " + str(tstart) + ", " + str(tstop) + ", " + str(15.10381)

        # String for control strength config option
        amp_float = optim_init_ampl[amp]*1.0
        conf_str= "0.0, " + str(amp_float)

        # Copy config file and set new config options
        konfig = copy.deepcopy(configfile)
        konfig.optim_init_ampl = conf_str
        konfig.ntime = N
        konfig.apply_pipulse = pipulsestr
        konfig.output_frequency = N

        # Specify jobname, this is also the name of the folder where this job will run
        jobname =  \
                  "amp"  + str(optim_init_ampl[amp]) + \
                  "tpulse"  + str(tstart)

        # create a folder for this job
        if os.path.exists(jobname):
           #pass
           continue
        else:

           # Create and dump the config file to the folder
           os.mkdir(jobname)
           newconfigfile = jobname + ".cfg"
           konfig.dump(jobname + "/" + newconfigfile)
    
           # Submit the job
           os.chdir(jobname)
           #submit_job(jobname, runcommand, 1, "00:00:30", "../main", newconfigfile, "sqoc", False)
           os.system(runcommand + " " + newconfigfile)
           os.chdir("../")





        ##############
        # Read result from output files
        ##############
        with open(jobname+"/data_out/expected0.iinit-001.rank0000.dat") as file1:
            lastline = file1.readlines()[-1].split()
            exp0[amp,i-1] = float(lastline[1])
        with open(jobname+"/data_out/expected1.iinit-001.rank0000.dat") as file1:
            lastline = file1.readlines()[-1].split()
            exp1[amp, i-1] = float(lastline[1])
        with open(jobname+"/data_out/population0.iinit-001.rank0000.dat") as file1:
            lastline = file1.readlines()[-1].split()
            pop0_0[amp, i-1] = float(lastline[1])
            pop0_1[amp, i-1] = float(lastline[2])
            pop0_2[amp, i-1] = float(lastline[3])
        
        # Set duration
        time[i-1] = float(lastline[0]) - 0.104


# Save to file
fexp0 = open("expected_alice.dat", 'w')
fexp1 = open("expected_cavity.dat", 'w')
fpop0 = open("population_alice.dat", 'w')
for i in range(nsteps):
    exp0str = str(time[i])
    exp1str = str(time[i])
    pop0str = str(time[i])
    for amp in range(len(optim_init_ampl)):
        exp0str += "   " + str(exp0[amp, i])
        exp1str += "   " + str(exp1[amp, i])
        pop0str += "   " + str(pop0_0[amp, i]) + " " + str(pop0_1[amp, i]) + " " + str(pop0_2[amp, i])

    exp0str += "\n"
    exp1str += "\n"
    pop0str += "\n"

    fexp0.write(exp0str)
    fexp1.write(exp1str)
    fpop0.write(pop0str)
fexp0.close()
fexp1.close()
fpop0.close()


# Plot
fig, axs = plt.subplots(1,2, figsize=(10,7))
fig.suptitle("Expected energy level")
axs[0].set_title("Alice")
axs[1].set_title("Cavity")

for i in range(len(optim_init_ampl)):
    axs[0].plot(time, exp0[i, :], label="p=q="+str(optim_init_ampl[i]))
    axs[1].plot(time, exp1[i, :], label="p=q="+str(optim_init_ampl[i]))
for ax in axs.flat:
    ax.set(xlabel='duration')
plt.legend()
plt.show()


fig, axs = plt.subplots(1,3, figsize=(20,7))
fig.suptitle("Alice population aka photon number")
for i in range(len(optim_init_ampl)):
    axs[0].plot(time, pop0_0[i, :], label="p=q="+str(optim_init_ampl[i]))
    axs[1].plot(time, pop0_1[i, :], label="p=q="+str(optim_init_ampl[i]))
    axs[2].plot(time, pop0_2[i, :], label="p=q="+str(optim_init_ampl[i]))
for ax in axs.flat:
    ax.set(xlabel='duration')
plt.legend()
plt.show()

