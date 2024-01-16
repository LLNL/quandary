from quandary import * 

## Three-qubit chain test case to swap the first and last qubit. One guard level each. ##

Ne = [2,2,2]  # Number of essential energy levels
Ng = [1,1,1]  # Number of extra guard levels

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

# Bounds on the control pulse (in rotational frame, p and q) [MHz] per oscillator
maxctrl_MHz = 40.0*np.ones(len(Ne))  

# Amplitude of initial (randomized) control vector for each oscillator, here 90% of allowed maximum amplitude
amp_frac = 0.9
initctrl_MHz = [amp_frac * maxctrl_MHz[i] for i in range(len(Ne))]

# Options for selecting carrier waves: 
cw_amp_thres = 5e-2  # Min. theshold on growth rate for each carrier
cw_prox_thres = 1e-3 # Max. threshold on carrier proximity

# Set up a target gate (in essential level dimensions), here SWAP13 gate
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
quandary_exec="/Users/guenther5/Numerics/quandary/quandary"
datadir = "./SWAP13_guard_run_dir"  # Compute and output directory 
verbose = False

# Increase default number of iterations
maxiter=300


# Prepare Quandary
myconfig = QuandaryConfig(Ne=Ne, Ng=Ng, freq01=freq01, selfkerr=selfkerr, Jkl=Jkl, rotfreq=rotfreq, T=T, maxctrl_MHz=maxctrl_MHz, targetgate=unitary, verbose=verbose, cw_amp_thres=cw_amp_thres, cw_prox_thres=cw_prox_thres, maxiter=maxiter)

# Execute quandary
t, pt, qt, infidelity, expectedEnergy, population = quandary_run(myconfig, quandary_exec=quandary_exec, datadir=datadir, runtype=runtype)

print(f"Fidelity = {1.0 - infidelity}")
print("\n Quandary data directory: ", datadir)

# Plot the control pulse and expected energy level evolution
if True:
	plot_pulse(myconfig.Ne, t, pt, qt)
	plot_expectedEnergy(myconfig.Ne, t, expectedEnergy)