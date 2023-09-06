from quandary import * 

## Two qubit test case: CNOT gate, two levels each, no guard levels ##

# 01 transition frequencies [GHz] per oscillator
freq01 = [4.10595, 4.8152] 

# Anharmonicities [GHz] per oscillator
selfkerr = [0.2198, 0.2252]

# Coupling strength [GHz] (Format [0<->1, 0<->2, ..., 1<->2, ... ])
crosskerr = [0.01]  # Crossker coupling of qubit 0<->1

# Frequency of rotations for computational frame [GHz] per oscillator
favg = sum(freq01)/len(freq01)
rotfreq = favg*np.ones(len(freq01))

# If Lindblad solver: Specify decay (T1) and dephasing (T2) [ns]. Make sure to also pass those to the QuandaryConfig constructor if you uncomment this.
# T1 = [] # [100.0, 110.0]
# T2 = [] # [80.0, 90.0]

# Set the time duration (ns)
T = 150.0

# Bounds on the control pulse (in rotational frame, p and q) [MHz] per oscillator
maxctrl_MHz = 10.0*np.ones(len(freq01))  

# Set the amplitude of initial (randomized) control vector for each oscillator 
amp_frac = 0.9
initctrl_MHz = [amp_frac * maxctrl_MHz[i] for i in range(len(freq01))]

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
myconfig = QuandaryConfig(freq01=freq01, selfkerr=selfkerr, crosskerr=crosskerr, rotfreq=rotfreq, T=T, maxctrl_MHz=maxctrl_MHz, initctrl_MHz=initctrl_MHz, targetgate=unitary, verbose=verbose)

# Execute quandary
pt, qt, expectedEnergy, infidelity = quandary_run(myconfig, quandary_exec=quandary_exec, ncores=ncores, datadir=datadir)


print(f"Fidelity = {1.0 - infidelity}")

# Plot the control pulse and expected energy level evolution
if True:
	plot_pulse(myconfig.Ne, myconfig.time, pt, qt)
	plot_expectedEnergy(myconfig.Ne, myconfig.time, expectedEnergy)

