from quandary import * 

## QFT gates on qubit chain, two levels, no guard levels, dipole-dipole coupling 5KHz (chain topology) ##

# ALL PROBLEMS encoded here
freq01_all = [5.18, 5.12, 5.06, 5.0, 4.94] 
T_all = [50, 200, 300, 600, 1500]
Jkl_coupling = 5e-3  	# Dipole-Dipole for CHAIN topology

## Number of qubits ##
nqubits = 4

# Number of essential elements in state vector
nstates = 2**nqubits

# verbosity
opt_verbose = False # False adds --quiet to run-time args
verbose = True # used during setup

# Runtypes
do_optim = True 
do_plot = False # Warning: buggy

rand_seed = 1234

# penalty coefficients
gamma_energy = 1e-4
gamma_tik0 = 0.0 
gamma_dpdm = 0.0

# Additional options
print_frequency_iter = 50

# Multiple Shooting options
nwindows = 32 # 8 # 32 # 24 # 16 # 32 # 64 # 16  # Using -1 defaults to using nwindows = number of time-tasks.

# Set to true for AL method, false for quadratic penalty
update_lagrangian = False

# Misc optimization options
tao_warmstart = False # True
unitarize = False

# Max number of outer AL/QP iterations
maxouter = 5

# Inner iteration
maxiter = 500
interm_tol = 1e-4
tol_infidelity = 1e-5

tol_grad = 1e-1 # 5e-2 # 1e-4

# initial penalty strength
mu = 1.0/nstates # 1e-3
mu_factor = 5.0 # 10.0 # 1.5

# MPI options 
maxcores = nstates*nwindows

# (for quartz)
nnodes = ceil(maxcores/36)

# (for dane)
#nnodes = ceil(maxcores/112)

print("nstates =", nstates, "nwindows =", nwindows, "ncores =", maxcores, "nnodes =", nnodes)

# batch job args
if nqubits == 2:
	batchargs = ["0:15:00", "qude", 1, "pdebug"]
else:
	batchargs = ["0:05:00", "qude", nnodes, "pbatch"]
#	batchargs = ["0:15:00", "qude", nnodes, "pdebug"]

        
# Set directory for the results
datadir = "./qft_"+str(nqubits)+"_win_" + str(nwindows) + "_run_dir"
#             str(time), str(acct), int(Nodes), str(partition)

# Carrier wave thresholds
cw_amp_thres = 5e-2  # Min. theshold on growth rate for each carrier
cw_prox_thres = 1e-3 # Max. threshold on carrier proximity

# coupling topo
fully_coupled = False

# Set up frequency vector
freq01 =  []	# 01 transition frequencies [GHz] 
Ne = []
Ng = []
for i in range(nqubits):
	freq01.append(freq01_all[len(freq01_all)-i-1])
	Ne.append(2)
	Ng.append(0)
print("Frequencies: ", freq01)
print("Ne: ", Ne)

# Set up qubit CHAIN coupling
Jkl=[]
for i in range(len(Ne)):
	for j in range(i+1, len(Ne)):
		if fully_coupled or j == i+1:
			val = Jkl_coupling
		else:
			val = 0.0
		Jkl.append(val)
print("Coupling: ", Jkl)

# Set the pulse duration (ns)
T = T_all[nqubits-1]
dtau = 10.0  	# Bspline spacing [ns]
scalefactor_states = 1.0/18.0 # 18.0 comes from estimating norm(dS/dalpha_k)
print("T=", T, "scalefactor_states=", scalefactor_states)

# Set up rotational frame frequency
favg = sum(freq01)/len(freq01) 
rotfreq = favg*np.ones(len(freq01))

# Define d-dimensional Discrete Fourier Transform gate 
def get_QFT_gate(dim):
	gate_Hd =  np.zeros((dim, dim), dtype=complex)
	om_d = np.exp(1j*2*np.pi/dim)

	for j in range(dim):
		for k in range(dim):
			gate_Hd[j,k] = om_d**(j*k) / np.sqrt(dim)

	return gate_Hd
unitary = get_QFT_gate(np.prod(Ne))
# print("Target gate: ", unitary)

# Set up the Quandary configuration for this test case
quandary = Quandary(Ne=Ne,
                    Ng=Ng,
                    freq01=freq01,
                    Jkl=Jkl,
                    rotfreq=rotfreq,
                    T=T,
                    targetgate=unitary,
                    dtau=dtau,
                    verbose=verbose,
                    cw_amp_thres=cw_amp_thres,
                    cw_prox_thres=cw_prox_thres,
                    rand_seed=rand_seed,
                    maxiter=maxiter,
                    gamma_energy=gamma_energy,
                    gamma_tik0=gamma_tik0,
                    gamma_dpdm=gamma_dpdm,
                    nwindows = nwindows,
                    mu = mu,
                    mu_factor = mu_factor,
                    update_lagrangian = update_lagrangian,
                    tao_warmstart = tao_warmstart,
                    tol_infidelity = tol_infidelity,
                    interm_tol = interm_tol,
                    tol_grad = tol_grad,
                    unitarize = unitarize,
                    maxouter = maxouter,
                    scalefactor_states = scalefactor_states,
                    print_frequency_iter = print_frequency_iter)

if do_optim:
	quandary.verbose = opt_verbose
	t, pt, qt, infidelity, expectedEnergy, population = quandary.optimize(maxcores=maxcores, datadir=datadir, batchargs=batchargs)
	print("Results in folder:", datadir)

#print(f"Fidelity = {1.0 - infidelity}")



# AFTER THE JOBS FINISHED, load results into python if needed.
# Plot the control pulse and expected energy level evolution
if do_plot:
	t, pt, qt, uT, expectedEnergy, population, popt, infidelity, optim_hist = quandary.get_results(datadir=datadir, ignore_failure=True)
	print("Fidelity = ", 1.0 - infidelity)
	plot_pulse(quandary.Ne, t, pt, qt)

