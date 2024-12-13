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
do_prune = True 

# Standard Hamiltonian and Lindblad model setup
unitMHz = True
Ne = [2, 2]
Ng = [0, 0]
freq01 = [4805.95, 4860.1] 	 # 01 transition for each qubit [MHz]
favg = sum(freq01)/len(freq01)
rotfreq = favg*np.ones(len(freq01))
Jkl = [5.0]  	# Dipole-Dipole coupling of qubit 0<->1 [MHz]
N = np.prod([Ne[i] + Ng[i] for i in range(len(Ne))])

# Set the pulse duration (us)
T = 0.5
dt = 0.0002
nsteps = T/dt
output_frequency = 5  # write trajectories every 1ns

# Decoherence [us]
T1 = [20.0, 30.0]	# Decay per qubit
T2 = [4.0, 8.0]		# Dephasing per qubit
# T1 = [0.0, 0.0]
# T2 = [0.0, 0.0]

# Set up a dummy target gate. Not used.
unitary = np.identity(N)

# Initial condition at t=0: Groundstate
initialcondition = "pure, 0,0"		

# Initial controls. Will be overwritten from the training data files. 
randomize_init_ctrl = False			
initctrl_MHz = [1.0, 1.0]

verbose = False
rand_seed=1234

# Setup quandary object using the standard Hamiltonian model, this is the baseline model used during training.
quandary = Quandary(Ne=Ne, Ng=Ng, freq01=freq01, Jkl=Jkl, rotfreq=rotfreq, T=T, targetgate=unitary, verbose=verbose, rand_seed=rand_seed, T1=T1, T2=T2, initialcondition=initialcondition, randomize_init_ctrl=randomize_init_ctrl, initctrl_MHz=initctrl_MHz, nsteps=nsteps, output_frequency=output_frequency, unitMHz=unitMHz) 

# forget about the decay times (relearn them!)
quandary.T1 = [0.0, 0.0]
quandary.T2 = [0.0, 0.0]

# Perturb the standard system Hamiltonian for data generation. This will be the Hamiltonian that is to be learned, alongside the decay and dephasing operators.
Hsys_mod = np.zeros((N,N), dtype=complex)
import random
random.seed(10)
rand_amp_MHz = 1.0	# Amplitude of the perturbation [MHz]
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
	trainingdatadir.append(cwd+"/stdHam_perturb_"+str(rand_amp_MHz)+"MHz_ctrlP"+str(ctrlMHz) + "_ctrlQ"+str(ctrlMHz) +"_rundir")

	# Quandary object for perturbed Hamiltonian for each data trajectory
	quandary_mod = Quandary(Ne=Ne, Ng=Ng, T=T, targetgate=unitary, verbose=verbose, rand_seed=rand_seed, T1=T1, T2=T2, standardmodel=False, Hc_re=Hc_re_mod, Hc_im=Hc_im_mod, Hsys=Hsys_mod, carrier_frequency=quandary.carrier_frequency, initialcondition=initialcondition, randomize_init_ctrl=randomize_init_ctrl, initctrl_MHz=[ctrlMHz, ctrlMHz], nsteps=nsteps, output_frequency=output_frequency, unitMHz=True) 

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

# stop

################
# NOW DO TRAINING! 
################
# Make sure to use the same initially assumed standard Hamiltonian as above (here: all zero)

# Set the UDE model: "hamiltonian", or "lindblad", or "both"
UDEmodel = "both"   
# UDEmodel = "hamiltonian"   
# Set the training time domain
T_train = T
# Add data type specifyier to the first element of the data list
trainingdatadir[0] = "synthetic, "+trainingdatadir[0]

# Switch between tikhonov regularization norms (L1 or L2 norm)
tik0_onenorm = True 			#  Use L1 for sparsification property
# Factor to scale the loss objective function value
loss_scaling_factor = 1e3

# Output directory for training
UDEdatadir = cwd + "/UDE_stdHam_perturb_"+str(rand_amp_MHz)+"MHz_npulses"+str(len(constctrl_MHz))+"_rundir"

# Set training optimization parameters
quandary.gamma_tik0 = 1e-8
quandary.gamma_tik0_onenorm = tik0_onenorm
quandary.loss_scaling_factor = loss_scaling_factor
quandary.tol_grad_abs = 1e-6
quandary.tol_grad_rel = 1e-7
quandary.tol_costfunc = 1e-7
quandary.tol_infidelity = 1e-7
quandary.gamma_leakage = 0.0
quandary.gamma_energy = 0.0
quandary.gamma_dpdm = 0.0

if do_training:
	print("\n Starting UDE training (UDEmodel=", UDEmodel, ")...")

	if do_prune:
		# Re-start training from pruned sparsified parameters 
		filename = UDEdatadir + "/params.dat"
		params = np.loadtxt(filename)
		cutoff = 1e-1
		params = [0.0 if abs(p) < cutoff else p for p in params]
		# print("SPARSE ", params)
		quandary.training(trainingdatadir=trainingdatadir, UDEmodel=UDEmodel, datadir=UDEdatadir, T_train=T_train, learn_params=params)
	else:
		# Start training from scratch
		quandary.training(trainingdatadir=trainingdatadir, UDEmodel=UDEmodel, datadir=UDEdatadir, T_train=T_train)

	# Simulate forward with optimized paramters to write out the Training data evolutions and the learned evolution
	filename = UDEdatadir + "/params.dat"
	params = np.loadtxt(filename)
	quandary.UDEsimulate(trainingdatadir=trainingdatadir, UDEmodel=UDEmodel, datadir=UDEdatadir+"/FWD_opt", T_train=quandary.T, learn_params=params)

	# Simulate forward the baseline model using zero learnable parameters to print out evolution
	zeroinit = np.zeros(len(params))
	quandary.UDEsimulate(trainingdatadir=trainingdatadir, UDEmodel=UDEmodel, datadir=UDEdatadir+"/FWD_zeroinit", T_train=quandary.T, learn_params=zeroinit)


#########
# Extrapolate to random controls
########

if do_extrapolate:
	# Get learned parameters
	filename = UDEdatadir + "/params.dat"
	params = np.loadtxt(filename)

	# Simulate the learned UDE model on a random control pulse
	datadir=UDEdatadir + "_FWDrandcontrol_rundir"
	quandary.initctrl_MHz = [1.0, 1.0]
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

if do_analyze:
	# Loading learned operators data from 'UDEdatadir'

	# Read learned Hamiltonian (N x N)
	filename_re = UDEdatadir + "/LearnedHamiltonian_Re.dat"
	filename_im = UDEdatadir + "/LearnedHamiltonian_Im.dat"
	Ham_test_re = np.loadtxt(filename_re, usecols=range(N), skiprows=2)
	Ham_test_im = np.loadtxt(filename_im, usecols=range(N), skiprows=2)
	Ham_test = Ham_test_re + 1j*Ham_test_im	
	print("TO-BE-Learned Hamiltonian [MHz]:") 
	print(Ham_org)
	print("Learned Hamiltonian [MHz]:") 
	print(Ham_test)
	Ham_error = Ham_org - Ham_test
	print("\n Hamiltonian error norm [MHz]:", np.linalg.norm(Ham_error))

	if UDEmodel == 'lindblad' or UDEmodel=='both':
		# Read learned Lindblad operators (N^2-1 many, each NxN complex)
		filename_re = UDEdatadir + "/LearnedLindbladOperators_Re.dat"
		filename_im = UDEdatadir + "/LearnedLindbladOperators_Im.dat"
		LearnedOps_re = []
		LearnedOps_im = []
		current_block = []
		skip_next = False  # Flag to track if the next line should be skipped
		with open(filename_re, "r") as file:
			for i,line in enumerate(file):
				if skip_next:  # Skip the current line
					skip_next = False
					continue
				if line.startswith("Mat"):  # Check if the line starts with "Mat"
					if i>0:
						LearnedOps_re.append(np.array(current_block, dtype=complex))
					current_block = []
					skip_next = True  # Skip the next line as well
					continue
				# filtered_lines.append(line.strip())
				row = np.fromstring(line, sep=" ")
				current_block.append(row)
		if current_block:
			LearnedOps_re.append(np.array(current_block, dtype=complex))
		with open(filename_im, "r") as file:
			for i,line in enumerate(file):
				if skip_next:  # Skip the current line
					skip_next = False
					continue
				if line.startswith("Mat"):  # Check if the line starts with "Mat"
					if i>0:
						LearnedOps_im.append(np.array(current_block, dtype=complex))
					current_block = []
					skip_next = True  # Skip the next line as well
					continue
				# filtered_lines.append(line.strip())
				row = np.fromstring(line, sep=" ")
				current_block.append(row)
		if current_block:
			LearnedOps_im.append(np.array(current_block, dtype=complex))
		LearnedOps = LearnedOps_re.copy()
		for i in range(len(LearnedOps_re)):
			LearnedOps[i] += 1j*LearnedOps_im[i]

		# Print learned Lindblad operators
		# for i in range(len(LearnedOps_re)):
			# print(f"Lindblad Operator {i}:\n{LearnedOps[i]}")

		# Set up the original system Lindblad matrix using Decay and Decoherence and the learned system Lindblad Matrix
		a = lowering(Ne[0] + Ng[0])
		Id2 = np.identity(Ne[0] + Ng[0])
		a0 = np.kron(a, Id2)
		a1 = np.kron(Id2, a)
		# Decoherence operators
		L01 = a0					# Qubit 0  decay
		L11 = a1 					# Qubit 1  decay
		L02 = a0.transpose() @ a0 	# Qubit 0 dephasing
		L12 = a1.transpose() @ a1	# Qubit 1 dephasing
		Lops_org = [L01, L11, L02, L12]
		decoherencetimes = [T1[0], T1[1], T2[0], T2[1]] # us
		LindbladSys_org = np.zeros((N**2, N**2), dtype=complex)
		LindbladSys_learned = np.zeros((N**2, N**2), dtype=complex)
		LopsSys = []
		for i in range(N):
			addme = systemmat_lindblad(Lops_org[i], Lops_org[i])
			LopsSys.append(addme)
			LindbladSys_org += 1./decoherencetimes[i] * addme
		for i in range(N**2-1):
			addme = systemmat_lindblad(LearnedOps[i], LearnedOps[i])
			LindbladSys_learned += addme
		# print("Original Lindblad System matrix")
		# print(LindbladSys_org)
		# print("Learned Lindblad System matrix")
		# print(LindbladSys_learned)

		## TEST: Make sure the learned system matrix as set up above matches to the system matrix when assembled with the double sum and basis operators. 
		# Get coefficients of the learned lindblad operators
		filename = UDEdatadir + "/params.dat"
		skiprowsHam = N**2-1
		paramsL = np.loadtxt(filename, skiprows=skiprowsHam)
		# Basis matrices used during learning
		BasisMats = getEijBasisMats(N)
		Lsys_double = np.zeros((N**2, N**2), dtype=complex)
		A = np.zeros((N**2-1, N**2-1), dtype=complex)
		for i in range(N**2-1):
			for j in range(N**2-1):
				addme = systemmat_lindblad(BasisMats[i], BasisMats[j])
				A[i,j] = assembleAij(i,j,len(BasisMats), paramsL)
				Lsys_double += A[i,j] * addme

		assert(np.linalg.norm(Lsys_double - LindbladSys_learned) < 1e-12)

		# Find best match to Lops ansatz min 1/2 || G - sum x_i Li||^2
		nops = len(Lops_org)
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

		print("Best match decay and decoherence times:")
		print([1.0/xi for xi in x])
		print("Original decay and decoherence times:")
		print(decoherencetimes)

		# Residual in the objective function 
		f = 0.0
		err = LindbladSys_learned.copy()
		for i in range(nops):
			err -= x[i] * LopsSys[i]
		print("Match objective function value: ", 1/2*np.linalg.norm(err)**2)



