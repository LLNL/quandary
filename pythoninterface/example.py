from quandary import * 


## One qubit test case ##
Ne = [3]  # Number of essential energy levels
Ng = [1]  # Number of extra guard levels

# 01 transition frequencies [GHz]
freq01 = [4.10595] 
# Anharmonicities [GHz]
selfkerr = [0.2198]
# Coupling
Jkl = []        # no Jaynes-Cummings coupling
crosskerr = []  # no crossker coupling
# Setup frequency of rotations for computational frame
rotfreq = freq01
# If Lindblad solver: Specify decay (T1) and dephasing (T2) [ns]
T1 = [] # [100.0]
T2 = [] # [80.0]

# Set the time duration (ns)
T = 100.0
# Number of points to resolve the shortest period of the dynamics
Pmin = 40  # 60 # 40 # 80

# Bounds on the control pulse (in rotational frame, p and q) 
maxctrl_MHz = [10.0]  
# Bspline spacing (ns). The number of Bsplines is T/dtau + 2.
dtau = 3.33

# Amplitude of initial (randomized) control vector
amp_frac = 0.9
initctrl_MHz = [amp_frac * maxctrl_MHz[i] for i in range(len(maxctrl_MHz))]
rand_seed = 1234
randomize_init_ctrl = True

# Set up a target gate 
unitary = [[0,0,1],[0,1,0],[1,0,0]]  # Swap first and last level
# print(unitary)

# Optimization options
costfunction = "Jtrace"
initialcondition = "basis"  # "basis", "diagonal", "pure, 0,0,1,...", "file, /path/to/file" 
gamma_tik0 = 1e-4 	# Tikhonov regularization
gamma_energy = 0.01	# Penality: Integral over control pulse energy
tol_infidelity = 1e-3   # Stopping tolerance based on the infidelity
tol_costfunc = 1e-3	# Stopping criterion based on the objective function
maxiter = 100 		# Maximum number of optimization iterations

# Quandary run options
runtype = "simulation"  # "simulation", or "gradient", or "optimization"
quandary_exec="/Users/guenther5/Numerics/quandary/main"
ncores = np.prod(Ne)  # Number of cores 
# ncores = 1
datadir = "./run_dir"  # Compute and output directory 
verbose = True

# # Load pcof0 from file
with open('./params.dat', 'r') as f:
    pcof0 = [float(line.strip()) for line in f if line]
# pcof0=[]

# Execute quandary
popt, infidelity, optim_hist = pulse_gen(Ne, Ng, freq01, selfkerr, crosskerr, Jkl, rotfreq, maxctrl_MHz, T, initctrl_MHz, rand_seed, randomize_init_ctrl, unitary,  dtau=dtau, Pmin=Pmin, datadir=datadir, tol_infidelity=tol_infidelity, tol_costfunc=tol_costfunc, maxiter=maxiter, gamma_tik0=gamma_tik0, gamma_energy=gamma_energy, costfunction=costfunction, initialcondition=initialcondition, T1=T1, T2=T2, runtype=runtype, quandary_exec=quandary_exec, ncores=ncores, verbose=verbose, pcof0=pcof0)
# Other keyword arg defaults
# cw_amp_thres = 6e-2
# cw_prox_thres = 1e-3

print(f"Fidelity = {1.0 - infidelity}")



# TODO:
#   * Check Quandary control initialization: Is the initial amplitude scaled inside quandary? 
#   * Check quandary for target gate dimension (essential?) when reading from file. If the target gate is read from a file, it needs to be the 'final' target gate (full dimensions, potentially rotated).
#   * All function call arguments should be keyword only.  
#   * Add dpdm regularization. Is that in the 'juqbox_interface' branch?
#   * Gather all configuration in a dictionary (or other struct) that contains all defaults and allows for changes.
#   * Change quandary's leakage term scaling: Potentially use same scaling as in Juqbox (exponentially increasing)

# Note: 
#   * pcof0 uses Quandaries initialization
#   * leakage_weights = [0.0, 0.0] is disabled.
#   * "use_eigenbasis" disabled.