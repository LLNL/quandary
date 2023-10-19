from quandary import * 

## One qudit test case: Swap the 0 and 2 state of a three-level qudit ##

Ne = [3]  # Number of essential energy levels

# 01 transition frequencies [GHz] per oscillator
freq01 = [4.10595] 
# Anharmonicities [GHz] per oscillator
selfkerr = [0.2198]

# Set the total time duration (ns)
T = 200.0

# Bounds on the control pulse (in rotational frame, p and q) [MHz] per oscillator
maxctrl_MHz = [4.0]  

# Set up a target gate (in essential level dimensions)
unitary = [[0,0,1],[0,1,0],[1,0,0]]  # Swaps first and last level
# print(unitary)

# Flag for debugging: prints out more information during initialization and quandary execution if set to 'true'
verbose = False

# Prepare Quandary configuration. The 'QuandaryConfig' dataclass gathers all configuration options and sets defaults for those member variables that are not passed through the constructor here. It is advised to compare what other defaults are set in the QuandaryConfig constructor (beginning of quandary.py)
myconfig = QuandaryConfig(Ne=Ne, freq01=freq01, selfkerr=selfkerr, maxctrl_MHz=maxctrl_MHz, targetgate=unitary, T=T, verbose=verbose)

# Set the location of the quandary executable (absolute path!)
quandary_exec="/Users/guenther5/Numerics/quandary/main"

# Potentially load initial control parameters from a file. 
# myconfig.pcof0_filename = os.getcwd() + "/SWAP02_params.dat" # Use absolute path!

# # Execute quandary. Default number of executing cores is the essential Hilbert space dimension.
t, pt, qt, infidelity, expectedEnergy, population = quandary_run(myconfig, quandary_exec=quandary_exec, datadir="./SWAP02_run_dir")
print(f"\nFidelity = {1.0 - infidelity}")

# Plot the control pulse and expected energy level evolution
if True:
    # plot_pulse(myconfig.Ne, t, pt, qt)
    # plot_expectedEnergy(myconfig.Ne, t, expectedEnergy)
    # plot_population(myconfig.Ne, t, population)

    # If one oscillator, you can also use the plot_results function to plot everything in one figure.
    plot_results_1osc(myconfig, pt[0], qt[0], expectedEnergy[0], population[0])


# Other optimization results can be accessed from the myconfig, in particular:
#   myconfig.popt         :  Optimized control parameters
#   myconfig.optim_hist   :  Dictionary of relevant optimization convergence history fields
#   myconfig.time         :  Time points where the expected energy is stored


# You can change configuration options directly, without creating a new QuandaryConfig instance. In most cases however, it is advised to call myconfig.update() afterwards to ensure that number of time steps and carrier wave frequencies are being re-computed.
# E.g. if you want to change the pulse length, do this:
#    myconfig.T = 200.0
#    myconfig.update()
# t, pt, qt, infidelity, expectedEnergy, population = quandary_run(myconfig, quandary_exec=quandary_exec)

# If you want to run Quandary using previously optimized control parameters, this will do it:
#    myconfig.pcof0= myconfig.popt
#    t, pt, qt, infidelity, expectedEnergy, population = quandary_run(myconfig, quandary_exec=quandary_exec, runtype="simulation")     # Note the runtype.
# [myconfig.update() is not needed in this case]

# Evaluate control pulse on different sampling rate
# points_per_ns = 1
# t, pt, qt = evalControls(myconfig, pcof=myconfig.popt, points_per_ns=points_per_ns, quandary_exec=quandary_exec, datadir="./SWAP02_run_dir")

