# Make sure you have the location of quandary.py in your PYTHONPATH. E.g. with
#   > export PYTHONPATH=/path/to/quandary/:$PYTHONPATH
# Further, put your quandary executable is in your $PATH variable. E.g. with
#   > export PATH=/path/to/quandary/:$PATH
#   or specify its location within the call to quandary.optimize(quandary_exec=/path/to/Quandary/quandary)
from quandary import * 

unitMHz = True
## One qudit test case: Swap the 0 and 2 state of a three-level qudit ##

Ne = [3]  # Number of essential energy levels
Ng = [1]  # Number of extra guard levels

#  Transition frequencies [GHz] from the device: 2025, Jan 06
f01 = 3422.625432
f12=  3213.617052

# 01 transition frequencies [GHz] per oscillator
freq01 = [f01] 
# Anharmonicities [GHz] per oscillator
selfkerr = [f01-f12]

# Set the total time duration (us)
T = 0.240

# Bounds on the control pulse (in rotational frame, p and q) [MHz] per oscillator
maxctrl_MHz = 4.0

# Set up a target gate (in essential level dimensions)
unitary = [[0,0,1],[0,1,0],[1,0,0]]  # Swaps first and last level
# print(unitary)

# Prepare Quandary with those options. This set default options for all member variables and overwrites those that are passed through the constructor here. Use help(Quandary) to see all options.
quandary = Quandary(Ne=Ne, Ng=Ng, freq01=freq01, selfkerr=selfkerr, maxctrl_MHz=maxctrl_MHz, targetgate=unitary, T=T, rand_seed=1234, unitMHz= unitMHz)

# Execute quandary. Default number of executing cores is the essential Hilbert space dimension. Limit the number of cores by passing ncores=<int>. Use help(quandary.optimize) to see all arguments.
datadir="./SWAP02_run_dir"
t, pt, qt, infidelity, expectedEnergy, population = quandary.optimize(datadir=datadir)
print(f"\nFidelity = {1.0 - infidelity}")

# Plot the control pulse and expected energy level evolution
if True:
    # plot_pulse(quandary.Ne, t, pt, qt)
    # plot_expectedEnergy(quandary.Ne, t, expectedEnergy)
    # plot_population(quandary.Ne, t, population)

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
t1, p1_list, q1_list = quandary.evalControls(pcof0=quandary.popt, points_per_ns=points_per_ns, datadir=datadir)

print("After evalControls")

print("T=", quandary.T, "nsteps=", quandary.nsteps, "dt=", quandary.dT)

# Remove last time point (just to be consistent with TensorFlow results)
t1 = t1[0:-1]
p1 = p1_list[0][0:-1]
q1 = q1_list[0][0:-1]

# Sanity check (requires Qutip): Plot the time evolution using mesolve, starting from initial ground state. This plot should look a lot like the plot in the lower left corner of the above figure (population from |0>)
# nstates = 4
# starting = 0

# Htot = generate_Hamiltonian(nstates)
# xpath, ypath = np.array(p1), np.array(q1)
# samplerate = 64
# t = np.arange(0, len(xpath), 1)/samplerate

# concat_pulse = np.stack((xpath,ypath), axis=-1) *2*np.pi*1e-3    
# prob_me_time, prob_me_gate, numgate = qt_mesolve(Htot, nstates, starting, concat_pulse, t, cop=True)

# fig, ax = plt.subplots()
# for i in range(3):
#     ax.plot(t, prob_me_time[i].real, '--', label=str(i))
#     ax.grid()
#     ax.set_xlabel('Time (ns)')
#     ax.set_ylabel('Population') 


# save p1 and q1 arrays on separate ASCII files
np.savetxt('p_ctrl.dat', p1, fmt='%20.10e')
np.savetxt('q_ctrl.dat', q1, fmt='%20.10e')

print("Saved control arrays on files")
