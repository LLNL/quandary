#  Quandary's python interface functions are defined in /path/to/quandary/quandary.py. Import them here. 
from quandary import * 

## One qudit test case: Swap the 0 and 2 state of a three-level qudit ##

Ne = [3]  # Number of essential energy levels
Ng = [1]  # Number of extra guard levels

# 01 transition frequencies [GHz] per oscillator
freq01 = [4.10595] 
# Anharmonicities [GHz] per oscillator
selfkerr = [0.2198]

# Set the total time duration (ns)
T = 100.0

# Bounds on the control pulse (in rotational frame, p and q) [MHz] per oscillator
maxctrl_MHz = 8.0

# Set up a target gate (in essential level dimensions)
unitary = [[0,0,1],[0,1,0],[1,0,0]]  # Swaps first and last level
# print(unitary)

# Prepare Quandary with those options. This set default options for all member variables and overwrites those that are passed through the constructor here. Use help(Quandary) to see all options.
quandary = Quandary(Ne=Ne, Ng=Ng, freq01=freq01, selfkerr=selfkerr, maxctrl_MHz=maxctrl_MHz, targetgate=unitary, T=T, rand_seed=1234)

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



# You can also load an initial control parameters from a file, and simulate or optimize on it.
#   quandary.pcof0_filename = os.getcwd() + "/" + datadir + "/params.dat" # Use absolute path!
#   t, pt, qt, infidelity, expectedEnergy, population = quandary.simulate(datadir=datadir)

# You can evaluate the control pulses on different time grid using a specific sampling rate with
#   points_per_ns = 4
#   t, pt, qt = quandary.evalControls(pcof=quandary.popt, points_per_ns=points_per_ns,datadir=datadir)

# Any Quandary configuration option can also be directly without creating a new Quandary instance. In most cases however, it is advised to call quandary.update() afterwards to ensure that number of time steps and carrier wave frequencies are being re-computed. For example, if you want to change the pulse length, do this:
#    quandary.T = 200.0
#    quandary.update()
#    t, pt, qt, infidelity, expectedEnergy, population = quandary.optimize()

# # Now let's simulate the optimized dynamics under decoherent noise:
# quandary.T1 = [10000.0]     #[ns]. One number per qubit
# quandary.T2 = [8000.0]      #[ns]. One number per qubit
# quandary.pcof0 = quandary.popt[:]
# quandary.update()
# t, pt, qt, infidelity, expectedEnergy, population = quandary.simulate(datadir=datadir, maxcores=8)