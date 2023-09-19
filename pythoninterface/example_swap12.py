from quandary import * 

## Two qubit test case for the SWAP gate, two essential levels each, no guard levels ##

# 01 transition frequencies [GHz] per oscillator
freq01 = [5.12, 5.06] 
# Coupling strength [GHz] (Format [0<->1, 0<->2, ..., 1<->2, ... ])
Jkl = [5.0e-3]         # Jaynes-Cummings coupling

# Frequency of rotations for computational frame [GHz] per oscillator
favg = sum(freq01)/len(freq01)
rotfreq = favg*np.ones(len(freq01))

# Set the time duration (ns)
T = 200.0

# Bspline spacing (ns) for control pulse parameterization. // The number of Bspline basis functions is then T/dtau + 2.
dtau = 10.0 # 3.33

# Number of points to resolve the shortest period of the dynamics
Pmin = 40  # 60 # 40 # 80

# Bounds on the control pulse (in rotational frame, p and q) [MHz] per oscillator
maxctrl_MHz = 30.0*np.ones(len(freq01))  

# Set the amplitude of initial (randomized) control vector for each oscillator 
amp_frac = 0.9
initctrl_MHz = [amp_frac * maxctrl_MHz[i] for i in range(len(freq01))]

# Set up a SWAP target gate (in essential level dimensions)
unitary = np.identity(4)
unitary[1,1] = 0.0
unitary[1,2] = 1.0
unitary[2,1] = 1.0
unitary[2,2] = 0.0
# print("Target gate: ", unitary)

# Quandary run options
runtype = "optimization"        # "simulation" # "simulation", or "gradient", or "optimization"
# quandary_exec="/Users/guenther5/Numerics/quandary/main"
quandary_exec="/Users/petersson1/src/quandary/main"
ncores = 4  # Number of cores 
datadir = "./SWAP12_run_dir"  # Compute and output directory 
verbose = False

# Prepare Quandary
myconfig = QuandaryConfig(freq01=freq01, Jkl=Jkl, rotfreq=rotfreq, T=T, maxctrl_MHz=maxctrl_MHz, initctrl_MHz=initctrl_MHz, targetgate=unitary, verbose=verbose, dtau=dtau, Pmin=Pmin)

# Potentially load initial control parameters from a file
# myconfig.pcof0_filename="./SWAP12_params.dat"

# Execute quandary
pt, qt, infidelity, expectedEnergy, population = quandary_run(myconfig, quandary_exec=quandary_exec, ncores=ncores, datadir=datadir)

print(f"Fidelity = {1.0 - infidelity}")
print("\n Quandary data directory: ", datadir)


# Plot the control pulse and expected energy level evolution
if True:
	plot_pulse(myconfig.Ne, myconfig.time, pt, qt)
	plot_expectedEnergy(myconfig.Ne, myconfig.time, expectedEnergy)