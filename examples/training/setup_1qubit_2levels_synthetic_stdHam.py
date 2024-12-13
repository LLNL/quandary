# Make sure you have the location of quandary.py in your PYTHONPATH. E.g. with
#   > export PYTHONPATH=/path/to/quandary/:$PYTHONPATH
# Further, make sure that your quandary executable is in your $PATH variable. E.g. with
#   > export PATH=/path/to/quandary/:$PATH
from quandary import * 
np.set_printoptions( linewidth=800)


do_datageneration = True 
do_training = True 
do_extrapolate = True 
do_analyze = True 

# Standard Hamiltonian and Lindblad model setup
unitMHz = True
Ne = [3]			# Number of essential levels
Ng = [0]			# Number of guard levels
freq01 = [4805.95]  # 01- transition [MHz]
rotfreq = freq01	# rotating frame frequency
selfkerr = [100.0]  # 1-2 transition [MHz]
N = np.prod([Ne[i] + Ng[i] for i in range(len(Ne))])

# Set the duration (us)
T = 1.0		  # 2.0 
dt = 0.002    
# dt = 0.0002    
nsteps = T/dt
output_frequency = 1  # write every x-the timestep

# For testing, can add a prefix for run directories. 
# dirprefix = "TESTdt"+str(dt) +"_"
# dirprefix = "dt"+str(dt) +"_"
dirprefix = ""

# Decoherence [us]
T1 = [20.0]
T2 = [4.0]

# Set up a dummy gate
unitary = np.identity(N)

# Initial condition at t=0: Groundstate
initialcondition = "pure, 0"

# Initial controls. Will be overwritten from the training data files. 
randomize_init_ctrl = False
initctrl_MHz = 1.0
carrier_frequency = [[0.0] for _ in range(len(Ne))]

verbose = False
rand_seed=1234

# Setup quandary object using the standard Hamiltonian model, this is the baseline model used during training.
quandary = Quandary(Ne=Ne, Ng=Ng, freq01=freq01, rotfreq=rotfreq, selfkerr=selfkerr, T=T, targetgate=unitary, verbose=verbose, rand_seed=rand_seed, T1=T1, T2=T2, initialcondition=initialcondition, randomize_init_ctrl=randomize_init_ctrl, initctrl_MHz=initctrl_MHz, nsteps=nsteps, output_frequency=output_frequency, unitMHz=unitMHz, carrier_frequency=carrier_frequency) 

# forget about the decay times. Relearn them later.
quandary.T1 = [0.0]
quandary.T2 = [0.0]

# Perturb the standard system Hamiltonian for data generation. This will be the Hamiltonian that is to be learned, alongside the decay and dephasing operators.
Hsys_mod = np.zeros((N,N), dtype=complex)
import random
random.seed(10)
rand_amp_MHz = 1.0  # Amplitude of the perturbation [MHz]
rand_amp_radus = rand_amp_MHz * (2*np.pi) 
for i in range(Hsys_mod.shape[0]):
	for j in range(i+1,Hsys_mod.shape[1]):
		val = random.uniform(-rand_amp_radus, rand_amp_radus) + random.uniform(-rand_amp_radus, rand_amp_radus)*1j
		Hsys_mod[i,j] = quandary.Hsys[i,j] + val
		Hsys_mod[j,i] = Hsys_mod[i,j].conj()
	val = random.uniform(-rand_amp_radus, rand_amp_radus)
	Hsys_mod[i,i] = quandary.Hsys[i,i] + val
# Diagonal shift so that 0,0 element is 0:
Id = np.identity(N, dtype=complex)
a = Hsys_mod[0,0]
Hsys_mod = Hsys_mod - a*Id

# Pertubed model uses the standard control operators (a+a' and a-a')
Hc_re_mod = quandary.Hc_re.copy()
Hc_im_mod = quandary.Hc_im.copy()

# Generate training data trajectories for various pulse strength (one trajectory for each element in constctrl_MHz)
constctrl_MHz = [1.0, 0.5]  	# Amplitudes of constant-control training data trajectories
trainingdatadir = []
for ctrlMHz in constctrl_MHz:
	cwd = os.getcwd()
	trainingdatadir.append(cwd+"/"+dirprefix+"stdHam_perturb_"+str(rand_amp_MHz)+"MHz_ctrlP"+str(ctrlMHz) + "_ctrlQ"+str(ctrlMHz) +"_rundir")

	# Quandary object for perturbed Hamiltonian for each data trajectory
	quandary_mod = Quandary(Ne=Ne, Ng=Ng, T=T, targetgate=unitary, verbose=verbose, rand_seed=rand_seed, T1=T1, T2=T2, standardmodel=False, Hc_re=Hc_re_mod, Hc_im=Hc_im_mod, Hsys=Hsys_mod, carrier_frequency=quandary.carrier_frequency, initialcondition=initialcondition, randomize_init_ctrl=randomize_init_ctrl, initctrl_MHz=ctrlMHz, nsteps=nsteps, output_frequency=output_frequency, unitMHz=True) 

	if do_datageneration:
		t, pt, qt, infidelity, expectedEnergy, population = quandary_mod.simulate(maxcores=8, datadir=trainingdatadir[-1])
		print("-> Generated trajectory for constant pulse amplitude ", ctrlMHz, ", data_dir=", trainingdatadir[-1])
 

print("Perturbation Hamiltonian (to be learned!), unit MHz: ") 
Ham_org = (quandary_mod.Hsys - quandary.Hsys) / (2*np.pi)
print(Ham_org)
print("")
print("Assumed standard Hamiltonian, unit MHz: ") 
print(quandary.Hsys/ (2*np.pi))
print("")


################
# NOW DO TRAINING! 
################
# Make sure to use the same above assumed basemodel Hamiltonian

# Set the UDE model: "hamiltonian", or "lindblad", or "both"
UDEmodel = "both"   
# Set the training time domain
T_train = T	  
# Add data type specifyier to the first element of the data list
trainingdatadir[0] = "synthetic, "+trainingdatadir[0]

# Switch between tikhonov regularization norms (L1 or L2 norm)
tik0_onenorm = True 			#  Use L1 for sparsification property
# Factor to scale the loss objective function value
loss_scaling_factor = 1e3

# Output directory for training
UDEdatadir = cwd+"/" + dirprefix+ "UDE_stdHam_perturb_"+str(rand_amp_MHz)+"MHz_npulses"+str(len(constctrl_MHz))+"_rundir"

# Set training optimization parameters
quandary.gamma_tik0 = 1e-7
quandary.gamma_tik0_onenorm = tik0_onenorm
quandary.loss_scaling_factor = loss_scaling_factor
quandary.tol_grad_abs = 1e-7
quandary.tol_grad_rel = 1e-7
quandary.tol_costfunc = 1e-7
quandary.tol_infidelity = 1e-7
quandary.gamma_leakage = 0.0
quandary.gamma_energy = 0.0
quandary.gamma_dpdm = 0.0
quandary.maxiter = 150

if do_training:
	print("\n Starting UDE training (UDEmodel=", UDEmodel, ")...")
	quandary.training(trainingdatadir=trainingdatadir, UDEmodel=UDEmodel, datadir=UDEdatadir, T_train=T_train)

	# Simulate forward with optimized paramters to write out the Training data evolutions and the learned evolution
	filename = UDEdatadir + "/params.dat"
	params = np.loadtxt(filename)
	quandary.UDEsimulate(trainingdatadir=trainingdatadir, UDEmodel=UDEmodel, datadir=UDEdatadir+"/FWD_opt", T_train=quandary.T, learn_params=params)

	# Simulate forward the baseline model using zero learnable parameters to print out evolution
	zeroinit = np.zeros(len(params))
	quandary.UDEsimulate(trainingdatadir=trainingdatadir, UDEmodel=UDEmodel, datadir=UDEdatadir+"/FWD_zeroinit", T_train=quandary.T, learn_params=zeroinit)

# stop

#########
# Extrapolate to random controls
########

if do_extrapolate:
	# Get learned parameters
	filename = UDEdatadir + "/params.dat"
	params = np.loadtxt(filename)

	# Simulate the learned UDE model on a random control pulse
	datadir=UDEdatadir + "_FWDrandcontrol_rundir"
	quandary.initctrl_MHz = [1.0]
	quandary.randomize_init_ctrl = True
	quandary.rand_seed = 12344
	quandary.nsplines=30
	t_UDE, pt_UDE, qt_UDE, infid_UDE, expect_UDE, popul_UDE = quandary.simulate(UDEmodel=UDEmodel, learn_params=params, datadir=datadir)

	# Simulate the exact (to-be-learned) model using the same random controls
	datadir = "./stdHam_perturb_"+str(rand_amp_MHz)+"MHz_FWDrandcontrol_rundir"
	quandary_mod.initctrl_MHz = quandary.initctrl_MHz
	quandary_mod.randomize_init_ctrl = quandary.randomize_init_ctrl
	quandary_mod.rand_seed = quandary.rand_seed
	quandary_mod.nsplines=quandary.nsplines
	t_mod, pt_mod, qt_mod, infid_mod, expect_mod, popul_mod = quandary_mod.simulate(datadir=datadir)

	# Compare the solutions at the final time
	print("\nExtrapolation random control: ")
	print("  Infidelity (UDE) = ", infid_UDE, ", Infidelity (exact) = ", infid_mod, " -> rel. error = ", abs((infid_UDE - infid_mod)/infid_mod))
	print("  Final-time rho error (Frob-norm) = ", np.linalg.norm(quandary.uT - quandary_mod.uT))
	print("")
	

################
# Analyze the trained Hamiltonian and Lindblad terms 
################



## Now read trained Operators from file and compare
if do_analyze:
	# Loading learned operators data from 'UDEdatadir'

	# Read learned Hamiltonian (N x N) and compute error
	Ham_learned = loadLearnedHamiltonian(UDEdatadir, N)
	print("TO-BE-Learned Hamiltonian [MHz]:") 
	print(Ham_org)
	print("Learned Hamiltonian [MHz]:") 
	print(Ham_learned)
	Ham_error = Ham_org - Ham_learned
	print("\n Hamiltonian error norm [MHz]:", np.linalg.norm(Ham_error))

	if UDEmodel == "lindblad" or UDEmodel == "both":
		# Load learned operators from file
		LearnedOps = loadLearnedLindbladOperators(UDEdatadir)
		assert(len(LearnedOps) == N**2-1)
		# # Print learned Lindblad operators
		# for i in range(len(LearnedOps_re)):
		# 	print(f"Lindblad Operator {i}:\n{LearnedOps[i]}")

		# Set up the learned system Matrix
		LindbladSys_learned = np.zeros((N**2, N**2), dtype=complex)
		for i in range(N**2-1):
			addme = systemmat_lindblad(LearnedOps[i], LearnedOps[i])
			LindbladSys_learned += addme

		## TEST: Make sure that the matrix as was set up above equals the system matrix when assembled with the double sum coefficients and basis operators. 
		# Get coefficients of the learned lindblad operators
		filename = UDEdatadir + "/params.dat"
		skiprowsHam = N**2-1
		paramsL = np.loadtxt(filename, skiprows=skiprowsHam)
		# Basis matrices used during learning
		BasisMats = getEijBasisMats(N)
		# Set up the double-sum system matrix
		Lsys_double = np.zeros((N**2, N**2), dtype=complex)
		A = np.zeros((N**2-1, N**2-1), dtype=complex)
		for i in range(N**2-1):
			for j in range(N**2-1):
				addme = systemmat_lindblad(BasisMats[i], BasisMats[j])
				A[i,j] = assembleAij(i,j,len(BasisMats), paramsL)
				Lsys_double += A[i,j] * addme
		assert(np.linalg.norm(Lsys_double - LindbladSys_learned) < 1e-12)

		# Set up the original system Lindblad matrix using Decay and Decoherence
		a = lowering(N)
		L01 = a					# Qubit 0  decay
		L02 = a.transpose() @ a 	# Qubit 0 dephasing
		Lops_org = [L01, L02]
		decoherencetimes = [T1[0], T2[0]] # us
		LindbladSys_org = np.zeros((N**2, N**2), dtype=complex)
		LopsSys = []
		for i in range(len(Lops_org)):
			addme = systemmat_lindblad(Lops_org[i], Lops_org[i])
			LopsSys.append(addme)
			LindbladSys_org += 1./decoherencetimes[i] * addme
		# print("Original Lindblad System matrix")
		# print(LindbladSys_org)
		# print("Learned Lindblad System matrix")
		# print(LindbladSys_learned)
		Lindblad_error = LindbladSys_org - LindbladSys_learned
		# print(" Lindblad System matrix error norm:", np.linalg.norm(Lindblad_error))

		# Find best match to Lops ansatz min 1/2 || G - sum x_i Li||^2
		nops = len(LopsSys)
		M = np.zeros((nops,nops), dtype=complex) 	# Hessian
		rhs = np.zeros(nops, dtype=complex)
		for i in range(nops):
			for j in range(nops):
				M[i,j] = np.trace(LopsSys[i].transpose().conj()@LopsSys[j])
			rhs[i] = np.trace(LindbladSys_learned.transpose().conj()@LopsSys[i])
		# Solve linear system
		x = np.linalg.inv(M) @ rhs
		for xi in x:
			assert(abs(xi.imag)<1e-16)
		x = [xi.real for xi in x]

		# Residual in the objective function 
		f = 0.0
		err = LindbladSys_learned.copy()
		for i in range(nops):
			err -= x[i] * LopsSys[i]
		err = 1/2*np.linalg.norm(err)**2

		print("\nBest match decay and decoherence times (residual = ", err,")")
		print([1.0/xi for xi in x])
		print("Original decay and decoherence times:")
		print(decoherencetimes)

