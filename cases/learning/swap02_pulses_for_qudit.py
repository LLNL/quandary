# Make sure you have the location of quandary.py in your PYTHONPATH. E.g. with
#   > export PYTHONPATH=/path/to/quandary/:$PYTHONPATH
# Further, put your quandary executable is in your $PATH variable. E.g. with
#   > export PATH=/path/to/quandary/:$PATH
#   or specify its location within the call to quandary.optimize(quandary_exec=/path/to/Quandary/quandary)
from quandary import * 

## One qudit test case: Swap the 0 and 2 state of a three-level qudit ##

Ne = [3]  # Number of essential energy levels
Ng = [1]  # Number of extra guard levels

# Frequency scaling factor relative to GHz and ns (1e-6 sec)
freq_scale = 1.0 # 
time_scale = 1/freq_scale

#  Transition frequencies [GHz] from the device: 
# 2025, Jan 06
# f01 = 3422.625432
# f12=  3213.617052
# July 25, 2025: Transition frequencies [GHz]: f01 = 3.416682744 f12= 3.2074712470000004
# Aug 4, 2025:   Transition frequencies [GHz]: f01 = 3.416634567 f12= 3.2074712470000004
f01 = 3.416634567*freq_scale
f12 = 3.2074712470000004*freq_scale

# 01 transition frequencies [GHz] per oscillator
freq01 = [f01] 
# Anharmonicities [GHz] per oscillator
selfkerr = [f01-f12]

# Set the total time duration (us)
# T = 0.360*time_scale
T = 360.0*time_scale

# Bounds on the control pulse (in rotational frame, p and q) [MHz] per oscillator
maxctrl = 7.0e-3*freq_scale
initctrl = 1.0e-3*freq_scale

# Set up a target gate (in essential level dimensions)
# Set up a target gate (in essential level dimensions)
if Ne[0] == 3:
	unitary = [[0,0,1],[0,1,0],[1,0,0]]  # 3 essential levels: Swaps first and third levels
elif Ne[0] == 4:
	unitary = [[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]]  # 4 essential levels: Swaps first and third levels
else:
	print("Wrong number of essential levels")
	stop

rand_seed = 1235
# Prepare Quandary with those options. This sets default options for all member variables and overwrites those that are passed through the constructor here. Use help(Quandary) to see all options.
quandary = Quandary(Ne=Ne, Ng=Ng, freq01=freq01, selfkerr=selfkerr, initctrl= initctrl, maxctrl=maxctrl, targetgate=unitary, T=T, control_enforce_BC=True, rand_seed=rand_seed, cw_prox_thres=0.5*abs(selfkerr[0]), gamma_leakage=600.0, verbose=True)

# Turn off verbosity after the carrier frequencies have been reported
quandary.verbose = False

quandary.carrier_frequency[0] = quandary.carrier_frequency[0][0:2]
print("Carrier freq: ", quandary.carrier_frequency) 

# Execute quandary. Default number of executing cores is the essential Hilbert space dimension. Limit the number of cores by passing ncores=<int>. Use help(quandary.optimize) to see all arguments.
datadir="./SWAP02_run_dir"
t, pt, qt, infidelity, expectedEnergy, population = quandary.optimize(datadir=datadir)
print(f"\nFidelity = {1.0 - infidelity}")

# Plot the control pulse and expected energy level evolution
if True:
    # If one oscillator, you can also use the plot_results function to plot everything in one figure.
    plot_results_1osc(quandary, pt[0], qt[0], expectedEnergy[0], population[0])

# Other optimization results can be accessed from Quandary, in particular:
#   quandary.popt         :  Optimized control parameters
#   quandary.optim_hist   :  Dictionary of relevant optimization convergence history fields
#   quandary.time         :  Time points where the expected energy is stored

# You can simulate the dynamics using the optimized control parameters by passing the optimized result 'quandary.popt' as an initial 'pcof0' vector:
t, pt, qt, infidelity, expectedEnergy, population = quandary.simulate(datadir=datadir, pcof0 = quandary.popt)
print(f"\nSimulated Fidelity = {1.0 - infidelity}")

print("T=", quandary.T, "nsteps=", quandary.nsteps, "dt=", quandary.dT)

# You can evaluate the control pulses on different time grid using a specific sampling rate with
points_per_ns = 64
eval_datadir = datadir + "_eval"
t1, p1_list, q1_list = quandary.evalControls(pcof0=quandary.popt, points_per_ns=points_per_ns, datadir=eval_datadir)

# Remove last time point (just to be consistent with TensorFlow results)
t1 = t1[0:-1]
p1 = p1_list[0][0:-1]
q1 = q1_list[0][0:-1]

# can we make it cell-centered instead?

# Keep all data points (breaks the Quandary_pulse.ipynb notebook)
# p1 = p1_list[0]
# q1 = q1_list[0]

# save p1 and q1 arrays on separate ASCII files
np.savetxt('p_ctrl.dat', p1, fmt='%20.10e')
np.savetxt('q_ctrl.dat', q1, fmt='%20.10e')

print("Saved control arrays on files")


