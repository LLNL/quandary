from quandary import * 

## Two qubit test case ##

Ne = [2,2]  # Number of essential energy levels
Ng = [2,2]  # Number of extra guard levels

# 01 transition frequencies [GHz] per oscillator
freq01 = [4.914, 5.114] 
# Anharmonicities [GHz] per oscillator
selfkerr = [0.33, 0.23]
# Coupling strength [GHz] (Format [0<->1, 0<->2, ..., 1<->2, ... ])
Jkl = [5.0e-3]       # dispersive coupling
crosskerr = [0.0]  # No Crossker coupling of qubit 0<->1

# Frequency of rotations for computational frame [GHz] per oscillator
favg = sum(freq01)/len(freq01)
rotfreq = favg*np.ones(len(Ne))

# If Lindblad solver: Specify decay (T1) and dephasing (T2) [ns]
T1 = [] # [100.0, 110.0]
T2 = [] # [80.0, 90.0]

# Set the time duration (ns)
T = 150.0

# Bspline spacing (ns) for control pulse parameterization. // The number of Bspline basis functions is then T/dtau + 2.
dtau = 10.0 # 3.33

# Bounds on the control pulse (in rotational frame, p and q) [MHz] per oscillator
maxctrl_MHz = 30.0*np.ones(len(Ne))  

# Set the amplitude of initial (randomized) control vector for each oscillator 
amp_frac = 0.5

initctrl_MHz = [amp_frac * maxctrl_MHz[i] for i in range(len(Ne))]

# Number of points to resolve the shortest period of the dynamics
Pmin = 40  # 60 # 40 # 80

rand_seed = 1234
randomize_init_ctrl = True

cw_amp_thres = 1e-7 # Include cross-resonance
cw_prox_thres = 1e-2 # 1e-2 # 1e-3

# Set up a target gate (in essential level dimensions)
# Here: SWAP gate
# Note 0-based indices
unitary = np.identity(np.prod(Ne))
unitary[1,1] = 0.0
unitary[2,2] = 0.0
unitary[1,2] = 1.0
unitary[2,1] = 1.0
# print("Target gate: ", unitary)

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
runtype = "optimization" # "simulation" # "simulation", or "gradient", or "optimization"
#quandary_exec="/Users/guenther5/Numerics/quandary/main"
quandary_exec="/Users/petersson1/src/quandary/main"
# quandary_exec="/cygdrive/c/Users/scada-125/quandary/main.exe"
ncores = np.prod(Ne)  # Number of cores 
# ncores = 1
datadir = "./CNOT_run_dir"  # Compute and output directory 
verbose = False

# Potentially load initial control parameters from a file
# with open('./CNOT_params.dat', 'r') as f:
    # pcof0 = [float(line.strip()) for line in f if line]
pcof0=[]

# Potentially create custom Hamiltonian model, if different from standard model
Ntotal = [sum(x) for x in zip(Ne, Ng)]
Hsys, Hc_re, Hc_im = hamiltonians(Ntotal, freq01, selfkerr, crosskerr, Jkl, rotfreq=rotfreq, verbose=verbose)

cw_amp_thres = 1e-7
cw_prox_thres = 1e-2

# Execute quandary
popt, infidelity, optim_hist = quandary_run(Ne, Ng, freq01, selfkerr, crosskerr, Jkl, rotfreq, maxctrl_MHz, T, initctrl_MHz, rand_seed, randomize_init_ctrl, unitary,  dtau=dtau, Pmin=Pmin, cw_amp_thres=cw_amp_thres,  cw_prox_thres=cw_prox_thres, datadir=datadir, tol_infidelity=tol_infidelity, tol_costfunc=tol_costfunc, maxiter=maxiter, gamma_tik0=gamma_tik0, gamma_energy=gamma_energy, gamma_dpdm=gamma_dpdm, costfunction=costfunction, initialcondition=initialcondition, T1=T1, T2=T2, runtype=runtype, quandary_exec=quandary_exec, ncores=ncores, verbose=verbose, pcof0=pcof0,  Hsys=Hsys, Hc_im=Hc_im, Hc_re=Hc_re)
# Other keyword arg defaults


print(f"Fidelity = {1.0 - infidelity}")
print("\n Quandary data directory: ", datadir)
