#  Quandary's python interface functions are defined in /path/to/quandary/quandary.py. Import them here. 
from quandary import * 

## Two qubit test case: CNOT gate, two levels each, no guard levels, dipole-dipole coupling 5KHz ##

# 01 transition frequencies [GHz] per oscillator
freq01 = [4.80595, 4.8601] 

# Coupling strength [GHz] (Format [0<->1, 0<->2, ..., 1<->2, ... ])
Jkl = [0.005]  # Dipole-Dipole coupling of qubit 0<->1

# Frequency of rotations for computational frame [GHz] per oscillator
favg = sum(freq01)/len(freq01)
rotfreq = favg*np.ones(len(freq01))

# Set the pulse duration (ns)
T = 200.0

# Set up the CNOT target gate
unitary = np.identity(4)
unitary[2,2] = 0.0
unitary[3,3] = 0.0
unitary[2,3] = 1.0
unitary[3,2] = 1.0
# print("Target gate: ", unitary)

# Flag for printing out more information to screen
verbose = False

# For reproducability: Random number generator seed
rand_seed=1234

# Set up the Quandary configuration for this test case. Make sure to pass all of the above to the corresponding fields, compare help(Quandary)!
quandary = Quandary(freq01=freq01, Jkl=Jkl, rotfreq=rotfreq, T=T, targetgate=unitary, verbose=verbose, rand_seed=rand_seed) 

# Optionally, if you already have control parameters, load them from a file. 
# quandary.pcof0_filename = os.getcwd() + "/CNOT_params.dat"  # absolute path!

# Execute quandary
t, pt, qt, infidelity, expectedEnergy, population = quandary.optimize()
print(f"Fidelity = {1.0 - infidelity}")

# Plot the control pulse and expected energy level evolution
if True:
	plot_pulse(quandary.Ne, t, pt, qt)
	plot_expectedEnergy(quandary.Ne, t, expectedEnergy) 
	# plot_population(quandary.Ne, t, population)


# You can predict the decoherence error of optimized dynamics:
print("Evaluate accuracy under decay and dephasing decoherence:\n")
T1 = [100000.0, 10000.0] #[ns] decay for each qubit
T2 = [80000.0 , 80000.0] #[ns] dephase for each qubit
quandary_lblad = Quandary(freq01=freq01, Jkl=Jkl, rotfreq=rotfreq, T=T, targetgate=unitary, verbose=verbose, T1=T1, T2=T2)
quandary_lblad.pcof0 = quandary.popt[:]
t, pt, qt, infidelity, expect, _ = quandary_lblad.simulate(maxcores=8) # Running on 8 cores

