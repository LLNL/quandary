from quandary import * 

## Two qubit test case: CNOT gate, two levels each, no guard levels ##

# 01 transition frequencies [GHz] per oscillator
freq01 = [4.80595, 4.8601] 

# Coupling strength [GHz] (Format [0<->1, 0<->2, ..., 1<->2, ... ])
Jkl = [0.005]  # Jaynes-Cumming coupling of qubit 0<->1

# Frequency of rotations for computational frame [GHz] per oscillator
favg = sum(freq01)/len(freq01)
rotfreq = favg*np.ones(len(freq01))

# If Lindblad solver: Specify decay (T1) and dephasing (T2) [ns]. Make sure to also pass those to the QuandaryConfig constructor if you uncomment this.
# T1 = [] # [100.0, 110.0]
# T2 = [] # [80.0, 90.0]

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

# Quandary run options
runtype = "optimization" # "simulation", or "gradient", or "optimization"
quandary_exec="/Users/guenther5/Numerics/quandary/main"
# quandary_exec="/cygdrive/c/Users/scada-125/quandary/main.exe"
ncores = 4  # Number of cores 
datadir = "./CNOT_run_dir"  # Compute and output directory 
verbose = False

# Prepare Quandary configuration
myconfig = QuandaryConfig(freq01=freq01, Jkl=Jkl, rotfreq=rotfreq, T=T, initctrl_MHz=initctrl_MHz, randomize_init_ctrl=randomize_init_ctrl, maxctrl_MHz=maxctrl_MHz, targetgate=unitary, verbose=verbose)

# Potentially load initial control coefficient from a file
# myconfig.pcof0_filename = os.getcwd() + "/"+datadir+"/params.dat"

# Execute quandary
pt, qt, expectedEnergy, infidelity = quandary_run(myconfig, quandary_exec=quandary_exec, ncores=ncores, datadir=datadir, runtype="optimization")


print(f"Fidelity = {1.0 - infidelity}")

# Plot the control pulse and expected energy level evolution
if True:
	plot_pulse(myconfig.Ne, myconfig.time, pt, qt)
	plot_expectedEnergy(myconfig.Ne, myconfig.time, expectedEnergy)

