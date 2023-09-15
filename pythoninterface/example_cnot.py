from quandary import * 

## Two qubit test case: CNOT gate, two levels each, no guard levels ##

# 01 transition frequencies [GHz] per oscillator
freq01 = [4.80595, 4.8601] 

# Coupling strength [GHz] (Format [0<->1, 0<->2, ..., 1<->2, ... ])
Jkl = [0.005]  # Jaynes-Cumming coupling of qubit 0<->1

# Frequency of rotations for computational frame [GHz] per oscillator
favg = sum(freq01)/len(freq01)
rotfreq = favg*np.ones(len(freq01))

# # If Lindblad solver: Specify decay (T1) and dephasing (T2) [ns]. Make sure to also pass those to the QuandaryConfig constructor if you uncomment this.
# T1 = [100000.0, 110000.0]
# T2 = [80000.0, 90000.0]

# Set the time duration (ns)
T = 200.0

# Set bounds on the control pulse amplitude (in rotational frame, p and q) [MHz] per oscillator
maxctrl_MHz = [10.0 for _ in range(len(freq01))]

# Set the amplitude of initial control vector for each oscillator [MHz]
initctrl_MHz = [10.0 for _ in range(len(freq01))]  
randomize_init_ctrl = False     # Use constant initial control pulse

# Set up the CNOT target gate
unitary = np.identity(4)
unitary[2,2] = 0.0
unitary[3,3] = 0.0
unitary[2,3] = 1.0
unitary[3,2] = 1.0
# print("Target gate: ", unitary)

# Flag for printing out more information
verbose = False

# Set up the Quandary configuration for this test case
myconfig = QuandaryConfig(freq01=freq01, Jkl=Jkl, rotfreq=rotfreq, T=T, initctrl_MHz=initctrl_MHz, randomize_init_ctrl=randomize_init_ctrl, maxctrl_MHz=maxctrl_MHz, targetgate=unitary, verbose=verbose) # potentially add T1=T1, T2=T2 for Lindblad solver with decay and decoherence

# Set some run options for Quandary
runtype = "optimization"    # "simulation", or "gradient", or "optimization"
quandary_exec="/Users/guenther5/Numerics/quandary/main" # Absolute path to Quandary's executable
ncores = 4  		    # Number of cores. Up to 8 for Lindblad solver, up to 4 for Schroedinger solver
datadir = "./CNOT_run_dir"  # Compute and output directory 

# Potentially, load initial control parameters from a file. 
# myconfig.pcof0_filename = os.getcwd() + "/"+datadir+"/params.dat"  # absolute path!

# Execute quandary
pt, qt, expectedEnergy, infidelity = quandary_run(myconfig, quandary_exec=quandary_exec, ncores=ncores, datadir=datadir, runtype=runtype)
print(f"Fidelity = {1.0 - infidelity}")

# Plot the control pulse and expected energy level evolution
if True:
	plot_pulse(myconfig.Ne, myconfig.time, pt, qt)
	plot_expectedEnergy(myconfig.Ne, myconfig.time, expectedEnergy) # if T1 or T2 decoherence (Lindblad solver), also pass the argument 'lindblad_solver=True' to this function.