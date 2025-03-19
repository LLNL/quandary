# Make sure you have the location of quandary.py in your PYTHONPATH. E.g. with
#   > export PYTHONPATH=/path/to/quandary/:$PYTHONPATH
# Further, make sure that your quandary executable is in your $PATH variable. E.g. with
#   > export PATH=/path/to/quandary/:$PATH
from quandary import * 
np.set_printoptions( linewidth=800)


do_datageneration = True 
do_training = True 
do_extrapolate = False
do_analyze = False
do_prune = False

# Standard Hamiltonian and Lindblad model setup
unitMHz = True
Ne = [3]			# Number of essential levels
Ng = [0]			# Number of guard levels
freq01 = [4105.95]  # 01- transition [MHz]
rotfreq = freq01	# rotating frame frequency
selfkerr = [219.8]  # 1-2 transition [MHz]
N = np.prod([Ne[i] + Ng[i] for i in range(len(Ne))])

# Set the duration (us)
T = 0.2		  # 2.0 
# dt = 3.00e-05
dt = 0.0001
output_frequency = 1  # write every x-the timestep

# For testing, can add a prefix for run directories. 
dirprefix = "data_out"

# Decoherence [us]
T1 = [100.0]
T2 = [40.0]

# Set up a dummy gate
unitary = np.identity(np.prod(Ne))

# Initial condition at t=0: Groundstate
initialcondition = "pure, 0"

verbose = False
rand_seed=1234


# Generate training data trajectories for various pulse strength (one trajectory for each element in the initctrl_MHz vector)
initctrl_MHz = [8.0] # [1.0, 0.5, 0.25]
perturbfac = 1.5
randomize_init_ctrl = True 
trainingdatadir = []
for ctrlMHz in initctrl_MHz:
	cwd = os.getcwd()
	trainingdatadir.append(cwd+"/"+dirprefix+"_ctrlMHz"+str(ctrlMHz)+"_asmeasured")

	# Setup quandary object using the standard Hamiltonian model.
	quandary = Quandary(Ne=Ne, Ng=Ng, freq01=freq01, rotfreq=rotfreq, selfkerr=selfkerr, T=T, targetgate=unitary, verbose=verbose, rand_seed=rand_seed, T1=T1, T2=T2, initialcondition=initialcondition, dT=dt, randomize_init_ctrl=randomize_init_ctrl, initctrl_MHz=ctrlMHz, output_frequency=output_frequency, unitMHz=unitMHz) 

	if do_datageneration:

		# First simulate the unperturbed controls to get the params of the unperturbed controls
		datadir_orig = cwd+"/"+dirprefix+"_ctrlMHz"+str(ctrlMHz)+"_origpulse"
		quandary.simulate(maxcores=8, datadir=datadir_orig)
		pcof_org = np.loadtxt(datadir_orig+"/params.dat")

		# Now perturb the controls: Scale each carrier wave component. Here: two carrier waves, but choose scaling g1 = g2 = const. TODO. 
		pcof_pert = [pcofi * perturbfac for pcofi in pcof_org]

		# Generate training data: Simulate perturbed controls
		quandary.simulate(pcof0=pcof_pert, maxcores=8, datadir=trainingdatadir[-1])

		print("-> Generated trajectory for pulse amplitude ", ctrlMHz, " with perturbation ", perturbfac)
		print("->   Unperturbed pulse trajectory directory:", datadir_orig)
		print("->   Perturbed pulse (training data) directory:", trainingdatadir[-1])
 

################
# NOW DO TRAINING! 
################
# Make sure to use the same above assumed basemodel Hamiltonian

# Set the UDE model: List of learnable terms, containing "hamiltonian" and/or "lindblad" and/or "transferLinear"
UDEmodel = "transferLinear"

# Set the training time domain
T_train = T	  
# Add data type specifyier to the first element of the data list
trainingdatadir[0] = "synthetic, "+trainingdatadir[0]

# Switch between tikhonov regularization norms (L1 or L2 norm)
tik0_onenorm = True 			#  Use L1 for sparsification property
# Factor to scale the loss objective function value
loss_scaling_factor = 1e3

# Output directory for training
UDEdatadir = cwd+"/" + dirprefix+ "_UDE"

# Set training optimization parameters
quandary.gamma_tik0 = 1e-9
quandary.gamma_tik0_onenorm = tik0_onenorm
quandary.loss_scaling_factor = loss_scaling_factor
quandary.tol_grad_abs = 1e-7
quandary.tol_grad_rel = 1e-7
quandary.tol_costfunc = 1e-8
quandary.tol_infidelity = 1e-7
quandary.gamma_leakage = 0.0
quandary.gamma_energy = 0.0
quandary.gamma_dpdm = 0.0
quandary.maxiter = 500

### TEST: Simulate with transfer functions set to the identity -> Loss should be large!
# learnparams_identity = [1.0, 1.0] # one for each carrier wave
# quandary.UDEsimulate(pcof0=pcof_org, trainingdatadir=trainingdatadir, UDEmodel=UDEmodel, learn_params=learnparams_identity, maxcores=8, datadir=UDEdatadir+"_identitytransfer")
# print(" CHECK: Loss should be large!\n")
### TEST: Simulate with transfer functions set to the exact pertubation from above -> Loss should be zero!
# learnparams_perturb = [id*perturbfac for id in learnparams_identity]
# quandary.UDEsimulate(pcof0=pcof_org, trainingdatadir=trainingdatadir, UDEmodel=UDEmodel, learn_params=learnparams_perturb, maxcores=8, datadir=UDEdatadir+"_scaledtransfer")
# print(" CHECK: Loss should be zero!\n")

if do_training:
	print("\n Starting UDE training for UDE model = ", UDEmodel, ")...")

	# Start training, use the unperturbed control parameters in pcof_org
	quandary.training(pcof0=pcof_org, trainingdatadir=trainingdatadir, UDEmodel=UDEmodel, datadir=UDEdatadir, T_train=T_train)

	filename = UDEdatadir + "/params.dat"
	learnparams_opt = np.loadtxt(filename)
	print("Training finished. Learned transfer function parameters: ", learnparams_opt, "\n")

	# Simulate forward with optimized paramters to write out the Training data evolutions and the learned evolution
	print("\n -> Eval loss of optimized UDE model.")
	quandary.UDEsimulate(trainingdatadir=trainingdatadir, UDEmodel=UDEmodel, datadir=UDEdatadir+"/FWD_opt", T_train=quandary.T, learn_params=learnparams_opt)

	# Simulate forward the baseline model using identity transfer function
	identityinit = np.ones(len(learnparams_opt))
	print("\n -> Eval loss of initial guess UDE model.")
	quandary.UDEsimulate(trainingdatadir=trainingdatadir, UDEmodel=UDEmodel, datadir=UDEdatadir+"/FWD_identityinit", T_train=quandary.T, learn_params=identityinit)
