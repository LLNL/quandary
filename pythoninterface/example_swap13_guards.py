from quandary import * 

## Three qubit chain test case. Swap first and last qubit ##

Ne = [2,2,2]  # Number of essential energy levels
Ng = [1,1,1]  # Number of extra guard levels
# Ng = [0,0,0]

# 01 transition frequencies [GHz] per oscillator
freq01 = [5.18, 5.12, 5.06] 
# Anharmonicities [GHz] per oscillator
selfkerr = [0.34, 0.34, 0.34]
# Coupling strength [GHz] (Format [0<->1, 0<->2, ..., 1<->2, ... ])
Jkl = [5.0e-3, 0.0, 5.0e-3] # dispersive coupling between qubit 0 and 1 as well as between 1 and 2 (chain) 

# Frequency of rotations for computational frame [GHz] per oscillator
favg = sum(freq01)/len(freq01)
rotfreq = favg*np.ones(len(Ne))

# Set the time duration (ns)
T = 500.0

# Bspline spacing (ns) for control pulse parameterization. // The number of Bspline basis functions is then T/dtau + 2.
dtau = 10.0 # 3.33

# Bounds on the control pulse (in rotational frame, p and q) [MHz] per oscillator
maxctrl_MHz = 40.0*np.ones(len(Ne))  

# Set the amplitude of initial (randomized) control vector for each oscillator 
amp_frac = 0.9
initctrl_MHz = [amp_frac * maxctrl_MHz[i] for i in range(len(Ne))]

# Number of points to resolve the shortest period of the dynamics
Pmin = 40  # 60 # 40 # 80

cw_amp_thres = 5e-2 # growth rate threshold
cw_prox_thres = 1e-3 # proximity threshold

# Set up a target gate (in essential level dimensions)
# Here: SWAP gate
unitary = np.identity(np.prod(Ne))
# Note 0-based indices
unitary[1,1] = 0.0
unitary[1,4] = 1.0
unitary[3,3] = 0.0
unitary[3,6] = 1.0
unitary[4,4] = 0.0
unitary[4,1] = 1.0
unitary[6,6] = 0.0
unitary[6,3] = 1.0
print("Target gate: ", unitary)

# Quandary run options
runtype = "optimization" # "simulation" # "simulation", or "gradient", or "optimization"
# quandary_exec="/Users/guenther5/Numerics/quandary/main"
quandary_exec="/Users/petersson1/src/quandary/main"
ncores = 4 # np.prod(Ne)  # Number of cores 
datadir = "./SWAP13_guard_run_dir"  # Compute and output directory 
verbose = True


# Prepare Quandary
myconfig = QuandaryConfig(Ne=Ne, Ng=Ng, freq01=freq01, selfkerr=selfkerr, Jkl=Jkl, rotfreq=rotfreq, T=T, maxctrl_MHz=maxctrl_MHz, initctrl_MHz=initctrl_MHz, targetgate=unitary, verbose=verbose, dtau=dtau, Pmin=Pmin, cw_amp_thres=cw_amp_thres, cw_prox_thres=cw_prox_thres)

# Execute quandary
t, pt, qt, infidelity, uT, expectedEnergy, population = quandary_run(myconfig, quandary_exec=quandary_exec, ncores=ncores, datadir=datadir, runtype=runtype)

print(f"Fidelity = {1.0 - infidelity}")
print("\n Quandary data directory: ", datadir)

# Plot the control pulse and expected energy level evolution
if True:
	plot_pulse(myconfig.Ne, t, pt, qt)
	plot_expectedEnergy(myconfig.Ne, t, expectedEnergy)