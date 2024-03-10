# Make sure you have the location of quandary.py in your PYTHONPATH. E.g. with
#   > export PYTHONPATH=/path/to/quandary/:$PYTHONPATH
from quandary import * 

## Two qubit test case: CNOT gate, two levels each, no guard levels, dipole-dipole coupling 5KHz ##

# 01 transition frequencies [GHz] per oscillator
freq01 = [4.80595, 4.8601] 

# Coupling strength [GHz] (Format [0<->1, 0<->2, ..., 1<->2, ... ])
Jkl = [0.005]  # Dipole-Dipole coupling of qubit 0<->1

# Frequency of rotations for computational frame [GHz] per oscillator
favg = sum(freq01)/len(freq01)
rotfreq = favg*np.ones(len(freq01))

# # If Lindblad solver: Specify decay (T1) and dephasing (T2) [ns]. Make sure to also pass those to the QuandaryConfig constructor if you uncomment this.
# T1 = [100000.0, 110000.0]
# T2 = [80000.0, 90000.0]

# Set the pulse duration (ns)
T = 200.0

# Set up the CNOT target gate
unitary = np.identity(4)
unitary[2,2] = 0.0
unitary[3,3] = 0.0
unitary[2,3] = 1.0
unitary[3,2] = 1.0
# print("Target gate: ", unitary)

# Flag for printing out more information
verbose = False

# For reproducability: Random number generator seed
rand_seed=1234

# Set up the Quandary configuration for this test case
myconfig = QuandaryConfig(freq01=freq01, Jkl=Jkl, rotfreq=rotfreq, T=T, targetgate=unitary, verbose=verbose, rand_seed=rand_seed) # potentially add T1=T1, T2=T2 for Lindblad solver with decay and decoherence

# Set some run options for Quandary
runtype = "optimization"    # "simulation", or "gradient", or "optimization"
quandary_exec="/Users/guenther5/Numerics/quandary/quandary" # Absolute path to Quandary's executable
ncores = 4  		    # Number of cores. Up to 8 for Lindblad solver, up to 4 for Schroedinger solver
datadir = "./CNOT_run_dir"  # Compute and output directory 

# Potentially, load initial control parameters from a file. 
# myconfig.pcof0_filename = os.getcwd() + "/"+datadir+"/params.dat"  # absolute path!

# Execute quandary
t, pt, qt, infidelity, expectedEnergy, population = quandary_run(myconfig, quandary_exec=quandary_exec, ncores=ncores, datadir=datadir, runtype=runtype)
print(f"Fidelity = {1.0 - infidelity}")

# Plot the control pulse and expected energy level evolution
if True:
	plot_pulse(myconfig.Ne, t, pt, qt)
	plot_expectedEnergy(myconfig.Ne, t, expectedEnergy) # if T1 or T2 decoherence (Lindblad solver), also pass the argument 'lindblad_solver=True' to this function.
	# plot_population(myconfig.Ne, myconfig.time, population)