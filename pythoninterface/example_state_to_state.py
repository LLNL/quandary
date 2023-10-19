from quandary import * 

## One qudit test case: State preparation (state-to-state) of the GHZ state ##

Ne = [3]  # Number of essential energy levels
Ng = [0]  # Number of extra guard levels

# 01 transition frequencies [GHz] per oscillator
freq01 = [4.10595] 
# Anharmonicities [GHz] per oscillator
selfkerr = [0.2198]

# Set the total time duration (ns)
T = 100.0

# Bounds on the control pulse (in rotational frame, p and q) [MHz] per oscillator
maxctrl_MHz = [4.0]  

# Set up the target state (in essential level dimensions)
targetstate =  [1.0/np.sqrt(2), 1.0/np.sqrt(2), 0.0] 
print("target state = ", targetstate)

# Set an initial condition (pure state). 
initialcondition = "pure, 1"

# Flag for debugging: prints out more information during initialization and quandary execution if set to 'true'
verbose = False

# Prepare Quandary configuration. The 'QuandaryConfig' dataclass gathers all configuration options and sets defaults for those member variables that are not passed through the constructor here. It is advised to compare what other defaults are set in the QuandaryConfig constructor (beginning of quandary.py)
myconfig = QuandaryConfig(Ne=Ne, freq01=freq01, selfkerr=selfkerr, maxctrl_MHz=maxctrl_MHz, targetstate=targetstate, T=T, verbose=verbose, Ng=Ng, initialcondition=initialcondition, tol_infidelity=1e-5)

# Set the location of the quandary executable (absolute path!)
quandary_exec="/Users/guenther5/Numerics/quandary/main"

# # Execute quandary on one core.
ncores=1
t, pt, qt, infidelity, uT, expectedEnergy, population = quandary_run(myconfig, quandary_exec=quandary_exec, datadir="./run_dir", ncores=ncores)
print(f"\nFidelity = {1.0 - infidelity}")

# Plot the control pulse and expected energy level evolution
if True:
    plot_results_1osc(myconfig, pt[0], qt[0], expectedEnergy[0], population[0])
