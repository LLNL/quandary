# Make sure you have the location of quandary.py in your PYTHONPATH. E.g. with
#   > export PYTHONPATH=/path/to/quandary/:$PYTHONPATH
# Further, make sure that your quandary executable is in your $PATH variable. E.g. with
#   > export PATH=/path/to/quandary/:$PATH
from quandary import * 

## Two qubit test case, demonstrating the use of piecewise constant control functions with total variation penalty term. CNOT gate, two levels each, no guard levels, dipole-dipole coupling 5KHz ##

# 01 transition frequencies [GHz] per oscillator
freq01 = [4.80595, 4.8601] 
# Coupling strength [GHz] (Format [0<->1, 0<->2, ..., 1<->2, ... ])
Jkl = [0.005]  # Dipole-Dipole coupling of qubit 0<->1
# Frequency of rotations for computational frame [GHz] per oscillator
favg = sum(freq01)/len(freq01)
rotfreq = favg*np.ones(len(freq01))

# Set the pulse duration (ns)
T = 140.0

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

# For piecewise constant control functions, choose spline order of 0 (Default spline order would be 2, being 2nd order Bsplines). Note, the number spline basis functions for piecewise constant controls has to be much larger than if you use 2nd order Bsplines. Also note that if the spline order is zero, it is recommended not to use any carrier frequencies, which is already the default.
spline_order = 0
nsplines = 140

# In order get less noisy control functions, activate the penalty term for variation of the control parameters
gamma_variation = 1.0

# Optionally: let controls functions start and end near zero
control_enforce_BC = True

# Set up the Quandary configuration for this test case. Make sure to pass all of the above to the corresponding fields, compare help(Quandary)!
quandary = Quandary(freq01=freq01, Jkl=Jkl, rotfreq=rotfreq, T=T, targetgate=unitary, verbose=verbose, rand_seed=rand_seed, spline_order=spline_order, nsplines=nsplines, gamma_variation=gamma_variation, control_enforce_BC=control_enforce_BC) 

# Optimize with quandary
t, pt, qt, infidelity, expectedEnergy, population = quandary.optimize()

print(f"Optimized Fidelity = {1.0 - infidelity}")

# Plot the optimized control pulse and expected energy level evolution
if True:
	plot_pulse(quandary.Ne, t, pt, qt)
	plot_expectedEnergy(quandary.Ne, t, expectedEnergy) 

# the optimized parameter vector is stored in quandary.popt as a 1-dimensioan numpy.ndarray
Ncoeff = np.size(quandary.popt)
pcof_opt = quandary.popt[:]

# setup the random number generator
rng = np.random.default_rng()

# perturb the pcof vector by random numbers
amp = 2.0*np.pi*1e-3 # units for amp are in rad/ns = 2*pi*GHz
pcof_sim = pcof_opt + 2.0*amp*(rng.random(Ncoeff)-0.5)

# simulate with quandary
t_2, pt_2, qt_2, infidelity_2, expectedEnergy_2, population_2 = quandary.simulate(pcof0=pcof_sim)

print(f"Perturbed Fidelity = {1.0 - infidelity_2}")

# Plot the perturbed control pulse and expected energy level evolution
if True:
	plot_pulse(quandary.Ne, t_2, pt_2, qt_2)
	plot_expectedEnergy(quandary.Ne, t_2, expectedEnergy_2) 
