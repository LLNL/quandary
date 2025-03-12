#  Quandary's python interface functions are defined in /path/to/quandary/quandary.py. Import them here. 
from quandary import * 

## Two qubit test case: CNOT gate, two levels each, 1 or 2 guard levels, dipole-dipole coupling 5MHz ##
Ne = [2, 2]
Ng = [2, 2] # [1, 1]

# 01 transition frequencies [GHz] per oscillator
freq01 = [4.80595, 4.8601] 
selfkerr = [0.2, 0.2]

# Coupling strength [GHz] (Format [0<->1, 0<->2, ..., 1<->2, ... ])
Jkl = [0.005]  # Dipole-Dipole coupling of qubit 0<->1

# Frequency of rotations for computational frame [GHz] per oscillator
favg = sum(freq01)/len(freq01)
rotfreq = favg*np.ones(len(freq01))

# Set the pulse duration (ns)
T = 300.0

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

# Piecewise constant B-spline
spline_order = 0 
spline_knot_spacing = 1.0  # [ns] Width of the constant control segments (dist. between basis functions)

# Piecewise quadratic B-spline
# spline_order = 2
# spline_knot_spacing = 10.0/3 # [ns] Distance between basis functions

# In order get less noisy control functions, activate the penalty term for variation of the control parameters
gamma_variation = 1.0

# Optionally: let controls functions start and end near zero
control_enforce_BC = True

# Max # optimization iterations
maxiter = 500

# Set up the Quandary configuration for this test case. Make sure to pass all of the above to the corresponding fields, compare help(Quandary)!
quandary = Quandary(Ne=Ne, Ng=Ng, freq01=freq01, selfkerr=selfkerr, Jkl=Jkl, rotfreq=rotfreq, T=T, targetgate=unitary, verbose=verbose, rand_seed=rand_seed, spline_order=spline_order, spline_knot_spacing=spline_knot_spacing, gamma_variation=gamma_variation, control_enforce_BC=control_enforce_BC, maxiter = maxiter) 

# Execute quandary
t, pt, qt, infidelity, expectedEnergy, population = quandary.optimize()
print(f"Schroedinger Fidelity = {1.0 - infidelity}")

# Plot the control pulse and expected energy level evolution
if True:
	plot_pulse(quandary.Ne, t, pt, qt)
	plot_expectedEnergy(quandary.Ne, t, expectedEnergy) 
	# plot_population(quandary.Ne, t, population)
