#  Quandary's python interface functions are defined in /path/to/quandary/quandary.py. Import them here. 
from quandary import * 

## Two qubit test case for the SWAP gate, two essential levels each, no guard levels ##

# 01 transition frequencies [GHz] per oscillator
freq01 = [5.12, 5.06] 
# Coupling strength [GHz] (Format [0<->1, 0<->2, ..., 1<->2, ... ])
Jkl = [5.0e-3]         # Dipole-Dipole coupling

# Frequency of rotations for computational frame [GHz] per oscillator
favg = sum(freq01)/len(freq01)
rotfreq = favg*np.ones(len(freq01))

# Set the time duration (ns)
T = 200.0

# Bounds on the control pulse (in rotational frame, p and q) [MHz] per oscillator
maxctrl_MHz = 30.0*np.ones(len(freq01))  

# Set up a SWAP target gate (in essential level dimensions)
unitary = np.identity(4)
unitary[1,1] = 0.0
unitary[1,2] = 1.0
unitary[2,1] = 1.0
unitary[2,2] = 0.0
# print("Target gate: ", unitary)

# You can enable more output by passing to Quandary
verbose = True 

# Prepare Quandary
quandary = Quandary(freq01=freq01, Jkl=Jkl, rotfreq=rotfreq, T=T, maxctrl_MHz=maxctrl_MHz, targetgate=unitary, verbose=verbose)

# Execute quandary
datadir = "SWAP12_run_dir"
t, pt, qt, infidelity, expectedEnergy, population = quandary.optimize(datadir=datadir)
print(f"Fidelity = {1.0 - infidelity}")
print("\n Quandary data directory: ", datadir)

# Plot the control pulse and expected energy level evolution
if True:
	plot_pulse(quandary.Ne, t, pt, qt)
	plot_expectedEnergy(quandary.Ne, t, expectedEnergy)


# # Adding some decoherence and simulate again
# quandary.T1 = [10000.0]
# quandary.T2 = [8000.0]
# quandary.update()
# quandary.pcof0 = quandary.popt[:]
# t, pt, qt, infidelity, expectedEnergy, population = quandary.simulate(datadir=datadir, maxcores=8)
# print(f"Fidelity under decoherence = {1.0 - infidelity}")