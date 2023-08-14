from quandary import * 


## One qubit test case ##
Ne = [3]  # Number of essential energy levels
Ng = [0]  # Number of extra guard levels

# 01 transition frequencies [GHz]
freq01 = [4.10595] 
# Anharmonicities [GHz]
selfkerr = [0.2198]
# Coupling strength
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

# Bounds on the control pulse (in rotational frame, p and q) [MHz] 
maxctrl_MHz = [10.0]  
# Bspline spacing (ns) for control pulse parameterization. // The number of Bspline basis functions is then T/dtau + 2.
dtau = 3.33

# Set the amplitude of initial (randomized) control vector
amp_frac = 0.9
initctrl_MHz = [amp_frac * maxctrl_MHz[i] for i in range(len(Ne))]
rand_seed = 1234
randomize_init_ctrl = True

# Set up a target gate (in essential level dimensions)
unitary = [[0,0,1],[0,1,0],[1,0,0]]  # Swaps first and last level
# print(unitary)

# Optimization options
costfunction = "Jtrace"     # "Jtrace", "Jfrobenius"
initialcondition = "basis"  # "basis", "diagonal", "pure, 0,0,1,...", "file, /path/to/file" 
gamma_tik0 = 1e-4 	# Tikhonov regularization
gamma_energy = 0.01	# Penality: Integral over control pulse energy
gamma_dpdm = 0.01	# Penality: Integral over second state derivative
tol_infidelity = 1e-3   # Stopping tolerance based on the infidelity
tol_costfunc = 1e-3	# Stopping criterion based on the objective function
maxiter = 100 		# Maximum number of optimization iterations

# Quandary run options
runtype = "optimization"  # "simulation", or "gradient", or "optimization"
quandary_exec="/Users/guenther5/Numerics/quandary/main"
# quandary_exec="/cygdrive/c/Users/scada-125/quandary/main.exe"
ncores = np.prod(Ne)  # Number of cores 
# ncores = 1
datadir = "./run_dir"  # Compute and output directory 
verbose = True

# Potentially load initial control parameters from a file
# with open('./params.dat', 'r') as f:
    # pcof0 = [float(line.strip()) for line in f if line]
pcof0=[]

# Execute quandary
popt, infidelity, optim_hist = pulse_gen(Ne, Ng, freq01, selfkerr, crosskerr, Jkl, rotfreq, maxctrl_MHz, T, initctrl_MHz, rand_seed, randomize_init_ctrl, unitary,  dtau=dtau, Pmin=Pmin, datadir=datadir, tol_infidelity=tol_infidelity, tol_costfunc=tol_costfunc, maxiter=maxiter, gamma_tik0=gamma_tik0, gamma_energy=gamma_energy, gamma_dpdm=gamma_dpdm, costfunction=costfunction, initialcondition=initialcondition, T1=T1, T2=T2, runtype=runtype, quandary_exec=quandary_exec, ncores=ncores, verbose=verbose, pcof0=pcof0)
# Other keyword arg defaults
# cw_amp_thres = 6e-2
# cw_prox_thres = 1e-3

print(f"Fidelity = {1.0 - infidelity}")



# TODO:
#   * All function call arguments should be keyword only.  
#   * Add dpdm regularization and energy integral penalty term. Those are in the 'juqbox_interface' branch.
#   * Gather all configuration in a dictionary (or other struct) that contains all defaults and allows for changes.
#   * Change quandary's leakage term scaling: Potentially use same scaling as in Juqbox (exponentially increasing)
# get_resonance should remove non-essential level transitions!

# Note: 
#   * pcof0 uses Quandaries initialization
#   * leakage_weights = [0.0, 0.0] is disabled.
#   * "use_eigenbasis" disabled.

# Reading non-standard hamiltonian from file:
#   * Hsys must be real
#   * No transfer functions
#   * No time-depended system Hamiltonian
#   * Only one control operator per oscillator (complex)