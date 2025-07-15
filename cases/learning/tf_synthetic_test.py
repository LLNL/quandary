# Make sure you have the location of quandary.py in your PYTHONPATH. E.g. with
#   > export PYTHONPATH=/path/to/quandary/:$PYTHONPATH
# Further, make sure that your quandary executable is in your $PATH variable. E.g. with
#   > export PATH=/path/to/quandary/:$PATH
from quandary import * 
np.set_printoptions( linewidth=800)


do_datageneration = True 
do_training = True # False 
do_extrapolate = False
do_analyze = False
do_prune = False

maxcores=3 # for unitary optimization

# Standard Hamiltonian and Lindblad model setup
Ne = [3]			# Number of essential levels
Ng = [0]			# Number of guard levels

#  Transition frequencies [GHz] from the device: 2025, Jan 06
f01 = 3422.625432e-3
f12=  3213.617052e-3

# 01 transition frequencies [GHz] per oscillator
freq01 = [f01] 
# Anharmonicities [GHz] per oscillator
selfkerr = [f01-f12]
# Rotating frame frequency
rotfreq = freq01

# Set the gate duration (us)
T = 0.240e3

# Bounds on the control pulse (in rotational frame, p and q) [MHz] per oscillator
maxctrl = 4.0e-3

# Amplitude of randomized initial control vector
initctrl = 10.0e-3

# Set up a target gate (in essential level dimensions)
unitary = [[0,0,1],[0,1,0],[1,0,0]]  # Swaps first and last level

verbose = False
rand_seed=1234

# Prepare Quandary with the above options. This set default options for all member variables and overwrites those that are passed through the constructor here. Use help(Quandary) to see all options.
quandary = Quandary(Ne=Ne, Ng=Ng, freq01=freq01, rotfreq=rotfreq, selfkerr=selfkerr, initctrl= initctrl, maxctrl=maxctrl, targetgate=unitary, T=T, verbose=False, rand_seed=rand_seed)

# Execute quandary. Default number of executing cores is the essential Hilbert space dimension. Limit the number of cores by passing ncores=<int>. Use help(quandary.optimize) to see all arguments.
datadir="./SWAP02_optimize"
t, pt, qt, infidelity, expectedEnergy, population = quandary.optimize(datadir=datadir, maxcores=maxcores)
print(f"\nFidelity = {1.0 - infidelity}")
pcof_opt = quandary.popt # get the optimized control vector

print("Optimized pulse with Schroedinger's eqn, dir = ", datadir)
plot_results_1osc(quandary, pt[0], qt[0], expectedEnergy[0], population[0])

# Modify quandary options for data generation & training (Use Lindblad's eqn)
initialcondition =  "basis" # "pure, 0" "diagonal" "basis" # Initial condition at t=0: Groundstate
T1 = [100.0] # Decoherence times [us]
T2 = [40.0]
output_frequency = 1  # write every x-th timestep
dirprefix = "SWAP02_basis" # "SWAP02_diag" "SWAP02_pure" # add a prefix for run directories

quandary2 = Quandary(Ne=Ne, Ng=Ng, freq01=freq01, rotfreq=rotfreq, selfkerr=selfkerr, maxctrl=maxctrl, targetgate=unitary, T=T, pcof0=pcof_opt, verbose=verbose, rand_seed=rand_seed,  initialcondition=initialcondition, output_frequency=output_frequency, T1=T1, T2=T2)

cwd = os.getcwd()
datadir_test = cwd+"/"+dirprefix+"_asmeasured" # NOTE: not an array
pfact = [0.8, 1.2] # [0.75, 1.5] #

if do_datageneration:
	# Perturb the controls: Scale each carrier wave component
	# One system, 2 frequencies in this test
	Nfirst = round(len(pcof_opt)/2) # Number of elements for the first carrier frequency

	pcof_pert = np.zeros(2*Nfirst)
	pcof_pert[0:Nfirst] = [pcofi * pfact[0] for pcofi in pcof_opt[0:Nfirst]]
	pcof_pert[Nfirst:2*Nfirst] = [pcofi * pfact[1] for pcofi in pcof_opt[Nfirst:2*Nfirst]]	# original control vector in pcof_opt

	# Generate training data: Simulate perturbed controls
	t, pt, qt, infidelity_pert, expectedEnergy, population = quandary2.simulate(pcof0=pcof_pert, maxcores=maxcores, datadir=datadir_test)
	print("Rescaled pulse, dir = ", datadir_test)
	plot_results_1osc(quandary2, pt[0], qt[0], expectedEnergy[0], population[0])

	print("-> Generated trajectory for perturbed pulse amplitude with perturbation factor:", pfact)
	# print("->   Unperturbed pulse trajectory directory:", datadir_orig, " Fidelity:", 1.0 - infidelity)
	print("->   Perturbed pulse (training data) directory:", datadir_test, " Fidelity:", 1.0 - infidelity_pert,"\n")
 
################
# NOW DO TRAINING! 
################
# Make sure to use the same basemodel Hamiltonian as above

# Set the UDE model: List of learnable terms, containing "hamiltonian" and/or "lindblad" and/or "transferLinear"
UDEmodel = "transferLinear"
maxcores = 1 # Note: Only have one pulse

# Set the training time domain
T_train = T	  
# Add data type specifier to the first element of the data list
trainingdatadir = [] # needs to be in a list
trainingdatadir.append("syntheticRho, " + datadir_test)
print("trainingdatadir = ", trainingdatadir, " len = ", len(trainingdatadir))

# Switch between tikhonov regularization norms (L1 or L2 norm)
tik0_onenorm = True 			#  Use L1 for sparsification property
# Factor to scale the loss objective function value
loss_scaling_factor = 1e3

# Output directory for training
UDEdatadir = cwd + "/" + dirprefix+ "_UDE"

# Set training optimization parameters
quandary2.gamma_tik0 = 1e-9
quandary2.gamma_tik0_onenorm = tik0_onenorm
quandary2.loss_scaling_factor = loss_scaling_factor
quandary2.tol_grad_abs = 1e-7
quandary2.tol_grad_rel = 1e-7
quandary2.tol_costfunc = 1e-8
quandary2.tol_infidelity = 1e-7
quandary2.gamma_leakage = 0.0
quandary2.gamma_energy = 0.0
quandary2.gamma_dpdm = 0.0
quandary2.maxiter = 500

learnparams_perturb = pfact # correct scaling factors
learnparams_identity = [1.0, 1.0] # factor of one for each carrier wave

### TEST: Simulate with transfer functions set to the exact pertubation from above -> Loss should be large!
quandary2.UDEsimulate(pcof0=pcof_opt, trainingdatadir=trainingdatadir, UDEmodel=UDEmodel, learn_params=learnparams_perturb, maxcores=maxcores, datadir=UDEdatadir+"_scaledtransfer")
print("learnparams = ", learnparams_perturb, " CHECK: Loss should be zero (small)!\n")

## TEST: Simulate with transfer functions set to the identity -> Loss should be zero!
# quandary2.UDEsimulate(pcof0=pcof_opt, trainingdatadir=trainingdatadir, UDEmodel=UDEmodel, learn_params=learnparams_identity, maxcores=maxcores, datadir=UDEdatadir+"_identitytransfer")
# print("learnparams = ", learnparams_identity, " CHECK: Loss should be large!\n")

if do_training:
	print("\nStarting UDE training for UDE model = ", UDEmodel, " initial_params: ", learnparams_identity, "...")

	# Start training, use the unperturbed control parameters in pcof_opt
	quandary2.training(pcof0=pcof_opt, trainingdatadir=trainingdatadir, UDEmodel=UDEmodel, datadir=UDEdatadir, T_train=T_train, learn_params=learnparams_identity, maxcores=maxcores) # maxcores defaults to 75?

	filename = UDEdatadir + "/params.dat"
	learnparams_opt = np.loadtxt(filename)
	print("Training finished. Learned transfer function parameters: ", learnparams_opt, "\n")

	# Simulate forward with optimized paramters to write out the Training data evolutions and the learned evolution
	print("\n -> Eval loss of optimized UDE model.")
	quandary2.UDEsimulate(trainingdatadir=trainingdatadir, UDEmodel=UDEmodel, datadir=UDEdatadir+"/FWD_opt", T_train=quandary2.T, learn_params=learnparams_opt, maxcores=maxcores)

