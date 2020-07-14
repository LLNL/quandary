import os, sys, shutil, copy
import subprocess
from config import Config


batch_args_mapping_slurm = {"NAME"  : "--job-name",
                            "ERROR" : "--error",
                            "OUTPUT" : "--output",
                            "TIME"  : "--time",
                            "NTASKS"  : "--ntasks",
                            "ACCOUNT"  : "--account",
                            "NTASKSPERNODE"  : "--ntasks-per-node",
                            "CONSTRAINT" : "--constraint",
                            "PARTITION" : "--partition"}

# use slurm argument names
batch_args_mapping = batch_args_mapping_slurm

default_batch_args = {batch_args_mapping["NAME"]          : "default",
                      batch_args_mapping["OUTPUT"]        : "default-%j.out",
                      batch_args_mapping["ERROR"]         : "default-%j.err",
                      batch_args_mapping["ACCOUNT"]       : "sqoc",
                      batch_args_mapping["TIME"]          : "01:00:00",
                      batch_args_mapping["NTASKS"]        : 1}


def submit_job(jobname, runcommand, ntasks, time_limit, executable, arguments, account, run=True):

    batch_args = copy.deepcopy(default_batch_args)

    batch_args[batch_args_mapping["NAME"]]            = jobname
    batch_args[batch_args_mapping["ERROR"]]           = jobname+".err"
    batch_args[batch_args_mapping["OUTPUT"]]          = jobname+".out"
    batch_args[batch_args_mapping["NTASKS"]]          = ntasks 
    batch_args[batch_args_mapping["ACCOUNT"]]         = account
    batch_args[batch_args_mapping["TIME"]]            = time_limit
    

    command =  runcommand  + " " + str(ntasks) + " " + executable + " " + arguments
    scriptname = jobname+".batch"
    assemble_batch_script(scriptname, command, batch_args)

    if run:
      subprocess.call("sbatch " + jobname + ".batch", shell=True)

def assemble_batch_script(name, run_command, args):
    
    outfile = open(name, 'w')

    outfile.write("#!/usr/bin/bash\n")

    for arg,value in args.iteritems():
        outfile.write("#SBATCH " + arg + "=" + str(value) + "\n")

    outfile.write(run_command)
    outfile.close()
