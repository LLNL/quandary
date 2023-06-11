import numpy as np
from quandary import quandary

## One qubit test case ##
Ne = [2]  # Number of essential energy levels
Ng = [2]  # Number of extra guard levels

# 01 transition frequencies
freq01 = [5.12] 
# Anharmonicities
selfkerr = [0.34]
# Coupling
Jkl = []        # no Jaynes-Cummings coupling
crosskerr = []  # no crossker coupling
# Setup frequency of rotations in computational frame
rotfreq = freq01
# If Lindblad solver: Specify decay (T1) and dephasing (T2) 
T1 = [10.0]
T2 = [20.0]

# Set the initial duration (ns)
T = 80.0
# Bspline spacing (ns)
dtau = 3.33
# Points per shortest period
Pmin = 40  # 60 # 40 # 80

# Bounds on the ctrl vector (rot frame) for each qubit
maxctrl_MHz = [5.0]  # Will be divided by Nfreq internally

# Maximum amplitude of initial control vector
amp_frac = 0.9
initctrl_MHz = [amp_frac * maxctrl_MHz[i] for i in range(len(maxctrl_MHz))]
rand_seed = 1234
randomize_init_ctrl = True


# Set up a target gate 
# TODO: Set up in ESSENTIAL rather than full dimensions
dim = np.prod([Ne[i] + Ng[i] for i in range(len(Ne))])
targetgate = np.zeros((dim,dim))
targetgate[0,1] = 1.0
targetgate[1,0] = 1.0
# print(targetgate)

# Optimization options
costfunction = "Jtrace"
initialcondition = "basis"  # TODO? 
gamma_tik0 = 1e-4
gamma_energy = 0.01
tol_infidelity = 1e-3
tol_costfunc= 1e-3
maxiter = 100

# Quandary run options
quandary_exec="./main"
runtype = "simulation"
ncores = np.prod(Ne)
datadir = "./run_dir"

# Execute quandary
popt, infidelity, optim_hist = quandary.run(Ne, Ng, freq01, selfkerr, crosskerr, Jkl, rotfreq, maxctrl_MHz, T, initctrl_MHz, rand_seed, randomize_init_ctrl, targetgate,  dtau=dtau, Pmin=Pmin, datadir=datadir, tol_infidelity=tol_infidelity, tol_costfunc=tol_costfunc, maxiter=maxiter, gamma_tik0=gamma_tik0, gamma_energy=gamma_energy, costfunction=costfunction, initialcondition=initialcondition, T1=T1, T2=T2, runtype=runtype, ncores=ncores, quandary_exec=quandary_exec, verbose=True)
# Other keyword arg defaults
# cw_amp_thres = 6e-2
# cw_prox_thres = 1e-3

# TODO:
#   * Initial conditions passed via string to quandary config
#   * Don't set up pcof0, use Quandaries initialization instead
#   * Test init_control loading from file
#   * All function call arguments should be keyword only.  
#   * Add dpdm regularization. Is that in the 'juqbox_interface' branch?
#   * Let user specify dtau or nsplines
#   * Gather all configuration in a dictionary (or other struct) that contains all defaults and allows for changes.
#   * Change quandary's leakage term scaling: Potentially use same scaling as in Juqbox (exponentially increasing)

# Note: 
#   * leakage_weights = [0.0, 0.0] is disabled.
#   * "use_eigenbasis" disabled.
#   * Init controls for standard control parameterization (no growthrate). Always use initctrl_MHz to specify the amplitude.

