from quandary import * 


## One qubit test case ##
Ne = [3]  # Number of essential energy levels
Ng = [0]  # Number of extra guard levels

# 01 transition frequencies [GHz]
freq01 = [5.12] 
# Anharmonicities [GHz]
selfkerr = [0.34]
# Coupling
Jkl = []        # no Jaynes-Cummings coupling
crosskerr = []  # no crossker coupling
# Setup frequency of rotations for computational frame
rotfreq = freq01
# If Lindblad solver: Specify decay (T1) and dephasing (T2) [ns]
T1 = [100.0]
T2 = [80.0]

# Set the time duration (ns)
T = 80.0
# Number of points to resolve the shortest period of the dynamics
Pmin = 40  # 60 # 40 # 80

# Bounds on the control pulse (in rotational frame, p and q) 
maxctrl_MHz = [10.0]  
# Bspline spacing (ns)
dtau = 3.33

# Amplitude of initial (randomized) control vector
amp_frac = 0.9
initctrl_MHz = [amp_frac * maxctrl_MHz[i] for i in range(len(maxctrl_MHz))]
rand_seed = 1234
randomize_init_ctrl = True

# Set up a target gate 
# TODO: Set up in ESSENTIAL rather than full dimensions
unitary = [[0,0,1],[0,1,0],[1,0,0]]  # Swap first and last level
# print(unitary)

# Optimization options
costfunction = "Jtrace"
initialcondition = "basis"  # TODO? 
gamma_tik0 = 1e-4 	# Tikhonov regularization
gamma_energy = 0.01	# Penality: Integral over control pulse energy
tol_infidelity = 1e-3   # Stopping tolerance based on the infidelity
tol_costfunc = 1e-3	# Stopping criterion based on the objective function
maxiter = 100 		# Maximum number of optimization iterations

# Quandary run options
runtype = "simulation"  # "simulation", or "gradient", or "optimization"
# ncores = np.prod(Ne)  # Number of cores 
ncores = 1
datadir = "./quandary_data"  # Compute and output directory 
verbose = True 

# Execute quandary
popt, infidelity, optim_hist = pulse_gen(Ne, Ng, freq01, selfkerr, crosskerr, Jkl, rotfreq, maxctrl_MHz, T, initctrl_MHz, rand_seed, randomize_init_ctrl, unitary,  dtau=dtau, Pmin=Pmin, datadir=datadir, tol_infidelity=tol_infidelity, tol_costfunc=tol_costfunc, maxiter=maxiter, gamma_tik0=gamma_tik0, gamma_energy=gamma_energy, costfunction=costfunction, initialcondition=initialcondition, T1=T1, T2=T2, runtype=runtype, ncores=ncores, verbose=True)


print(f"{infidelity}")
