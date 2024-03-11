# Make sure you have the location of quandary.py in your PYTHONPATH. E.g. with
#   > export PYTHONPATH=/path/to/quandary/:$PYTHONPATH
# Further, make sure that your quandary executable is in your $PATH variable. E.g. with
#   > export PATH=/path/to/quandary/:$PATH
from quandary import * 

## One qudit test case: State preparation (state-to-state) of the GHZ state ##

Ne = [2]  # Number of essential energy levels
Ng = [1]  # Number of extra guard levels

# 01 transition frequencies [GHz] per oscillator
freq01 = [4.10595] 
# Anharmonicities [GHz] per oscillator
selfkerr = [0.2198]

# Set the total time duration (ns)
T = 50.0

# Bounds on the control pulse (in rotational frame, p and q) [MHz] per oscillator
maxctrl_MHz = 4.0  

# Set up the target state (in essential level dimensions)
targetstate =  [1.0/np.sqrt(2), 1.0/np.sqrt(2)] 
print("target state = ", targetstate)

# Set an initial condition (must be a pure state, here ground state). 
initialcondition = "pure, 0"

# Prepare Quandary configuration. The 'QuandaryConfig' dataclass gathers all configuration options and sets defaults for those member variables that are not passed through the constructor here. It is advised to compare what other defaults are set in the QuandaryConfig constructor (beginning of quandary.py)
myconfig = QuandaryConfig(Ne=Ne, Ng=Ng, freq01=freq01, selfkerr=selfkerr, maxctrl_MHz=maxctrl_MHz, targetstate=targetstate, T=T,  initialcondition=initialcondition, tol_infidelity=1e-5)

# # Execute quandary.
t, pt, qt, infidelity, expectedEnergy, population = quandary_run(myconfig)
print(f"\nFidelity = {1.0 - infidelity}")

# Plot the control pulse and expected energy level evolution
if True:
    plot_results_1osc(myconfig, pt[0], qt[0], expectedEnergy[0], population[0])
