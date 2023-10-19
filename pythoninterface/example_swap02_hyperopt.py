from quandary import * 

## One qudit test case: Swap the 0 and 2 state of a three-level qudit ##

Ne = [3]  # Number of essential energy levels

# 01 transition frequencies [GHz] per oscillator
freq01 = [4.10595] 
# Anharmonicities [GHz] per oscillator
selfkerr = [0.2198]
# Frequency of rotations for computational frame [GHz] per oscillator
rotfreq = freq01

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
myconfig = QuandaryConfig(freq01=freq01, rotfreq=rotfreq, maxctrl_MHz=maxctrl_MHz, targetgate=unitary, T=T, nsplines=20, tol_infidelity=1e-8, tol_costfunc=1e-8, gamma_energy=1.0, control_enforce_BC=False)

# Set the location of the quandary executable (absolute path!)
quandary_exec="/Users/guenther5/Numerics/quandary/main"

# # Execute quandary. Default number of executing cores is the essential Hilbert space dimension.
t, pt, qt, infidelity, expectedEnergy, population = quandary_run(myconfig, quandary_exec=quandary_exec, datadir="./SWAP02_run_dir")
print(f"\nFidelity = {1.0 - infidelity}")

# # Plot the control pulse and expected energy level evolution
# if True:
#     plot_pulse(myconfig.Ne, t, pt, qt)
#     plot_expectedEnergy(myconfig.Ne, t, expectedEnergy)
#     plot_population(myconfig.Ne, t, population)

#     # If one oscillator, you can also use the plot_results function to plot everything in one figure.
#     plot_results_1osc(myconfig, pt[0], qt[0], expectedEnergy[0], population[0])


#
# Hyperopt: Hyperparameter tuning
#

from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval, Trials

# Define the space of Hyperparameters
space = {
    'nsplines':     hp.randint('nsplines', 100),
    'initctrl_MHz': hp.uniform('initctrl_MHz', 1e-2, maxctrl_MHz[0]),
    'gamma_energy': hp.loguniform('gamma_energy', -10.0, 0),
    'gamma_tik0': hp.loguniform('gamma_tik0', -10.0, 0)
}

# Define the objective function. <params> is a dictionary containing fields for each hyperparameter as determined through the 'space' field. 
def objective(params):
    
    # Unpack the hyperparameters from the params dictionary
    nsplines, initctrl_MHz, gamma_energy, gamma_tik0 = params['nsplines'], params['initctrl_MHz'], params['gamma_energy'], params['gamma_tik0']
    print("\n Execute nsplines, initctrl_MHz, gamma_energy, gamma_tik0 =", nsplines, initctrl_MHz, gamma_energy, gamma_tik0)
    
    # Set up Quandary's configuration file
    maxiter=100
    randomize_init_ctrl = False     # One should probably call Quandary multiple times for different random initial guesses!
    myconfig = QuandaryConfig(freq01=freq01, maxiter=maxiter, gamma_energy=gamma_energy, gamma_tik0=gamma_tik0, rotfreq=rotfreq, maxctrl_MHz=maxctrl_MHz, targetgate=unitary, T=T, nsplines=nsplines, control_enforce_BC=False, randomize_init_ctrl=randomize_init_ctrl)

    # Execute quandary. 
    t, pt, qt, infidelity, expectedEnergy, population = quandary_run(myconfig, quandary_exec=quandary_exec, datadir="./SWAP02_run_dir")
    print(f"-> Fidelity = {1.0 - infidelity}")
    
    # Return the infidelity. Hyperopt allows to pass more information back. Check the documentation.
    return {'loss': infidelity, 'status': STATUS_OK }

# Call Hyperopt's fmin to optimize across the space. The optimizing algorithm here is the default (which?), but can be changed, see the documentation.
trials = Trials()
max_evals=100   
best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

# Print best result
print(space_eval(space, best))


# Analyze search space values that have been visited. There is probably a better way of doing this...

# trials.trials - a list of dictionaries representing everything about the search
# trials.results - a list of dictionaries returned by 'objective' during the search
# trials.losses() - a list of losses (float for each 'ok' trial)
# trials.statuses() - a list of status strings
for i in range(len(trials.trials)):
    if trials.losses()[i] > 1e-3:
        print(f"Loss({i})={trials.losses()[i]}")
        print('  nsplines = ', trials.trials[i]['misc']['vals']['nsplines'])
        print('  initctrl_MHz = ', trials.trials[i]['misc']['vals']['initctrl_MHz'])
        print('  gamma_energy = ', trials.trials[i]['misc']['vals']['gamma_energy'])
        print('  gamma_tik0 = ', trials.trials[i]['misc']['vals']['gamma_tik0'])