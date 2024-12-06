# Make sure you have the location of quandary.py in your PYTHONPATH. E.g. with
#   > export PYTHONPATH=/path/to/quandary/:$PYTHONPATH
# Further, make sure that your quandary executable is in your $PATH variable. E.g. with
#   > export PATH=/path/to/quandary/:$PATH
from quandary import * 
np.set_printoptions( linewidth=800)

do_datageneration = True 
do_training = True 
do_analyze = True 
do_extrapolate = True 
do_testansatz = False

# Standard Hamiltonian and Lindblad model setup
unitMHz = True
Ne = [2, 2]
freq01 = [4805.95, 4860.1] 	 # MHz
favg = sum(freq01)/len(freq01)
rotfreq = favg*np.ones(len(freq01))
Jkl = [0.005]  # Dipole-Dipole coupling of qubit 0<->1 [MHz]

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
unitary = np.identity(4)
unitary[2,2] = 0.0
unitary[3,3] = 0.0
unitary[2,3] = 1.0
unitary[3,2] = 1.0
# print("Target gate: ", unitary)

# Initial condition at t=0: Groundstate
initialcondition = "pure, 0,0"		

# Initial controls. Will be overwritten from the training data files. 
randomize_init_ctrl = False			
initctrl_MHz = [1.0, 1.0]

verbose = False
rand_seed=1234
quandary = Quandary(Ne=Ne, freq01=freq01, Jkl=Jkl, rotfreq=rotfreq, T=T, targetgate=unitary, verbose=verbose, rand_seed=rand_seed, T1=T1, T2=T2, initialcondition=initialcondition, randomize_init_ctrl=randomize_init_ctrl, initctrl_MHz=initctrl_MHz, nsteps=nsteps, output_frequency=output_frequency, unitMHz=unitMHz) 
# t, pt, qt, infidelity, expectedEnergy, population = quandary.simulate(maxcores=8, datadir="./Ham_original_rundir")

# SET UP CUSTOM PERTURBED system Hamiltonian. 
Hsys_mod = np.zeros((4,4), dtype=complex)
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
Id = np.zeros((4,4), dtype=complex)
Id[0,0] = 1.0
Id[1,1] = 1.0
Id[2,2] = 1.0
Id[3,3] = 1.0
a = Hsys_mod[0,0]
Hsys_mod = Hsys_mod - a*Id

# Use the same control Hamiltonian as used in quandary (standard: a+a' and a-a')
Hc_re_mod = quandary.Hc_re.copy()
Hc_im_mod = quandary.Hc_im.copy()

# Generate training data trajectories
constctrl_MHz = [5.0, 6.0]  	# Amplitudes of constant-control training data trajectories
trainingdatadir = []
for ctrlMHz in constctrl_MHz:
	cwd = os.getcwd()
	trainingdatadir.append(cwd+"/stdHam_perturb_"+str(rand_amp_MHz)+"MHz_ctrlP"+str(ctrlMHz) + "_ctrlQ"+str(ctrlMHz) +"_rundir")

	quandary_mod = Quandary(Ne=Ne, T=T, targetgate=unitary, verbose=verbose, rand_seed=rand_seed, T1=T1, T2=T2, standardmodel=False, Hc_re=Hc_re_mod, Hc_im=Hc_im_mod, Hsys=Hsys_mod, carrier_frequency=quandary.carrier_frequency, initialcondition=initialcondition, randomize_init_ctrl=randomize_init_ctrl, initctrl_MHz=[ctrlMHz, ctrlMHz], nsteps=nsteps, output_frequency=output_frequency, unitMHz=True) 

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

# Output directory for training
UDEdatadir = cwd + "/UDE_stdHam_perturb_"+str(rand_amp_MHz)+"MHz_npulses2_rundir"

# forget about the decay times (relearn them!)
quandary.T1 = [0.0, 0.0]
quandary.T2 = [0.0, 0.0]

# Set training optimization parameters
quandary.gamma_tik0 = 1e-9
quandary.tol_grad_abs = 1e-6
quandary.tol_grad_rel = 1e-7
quandary.tol_costfunc = 1e-7
quandary.tol_infidelity = 1e-7
quandary.gamma_leakage = 0.0
quandary.gamma_energy = 0.0
quandary.gamma_dpdm = 0.0

if do_training:
	print("\n Starting UDE training (UDEmodel=", UDEmodel, ")...")
	quandary.training(trainingdatadir=trainingdatadir, UDEmodel=UDEmodel, datadir=UDEdatadir, T_train=T_train)

# stop


################
# Analyze the trained Hamiltonian and Lindblad terms 
################

# Construct the vectorized Lindblad system matrix for term  A rho B' - 1/2{B'A, rho}:
#  -> vectorized S = \bar B \kron A - 1/2(I \kron B'A + (B'A)^T \kron I)
def systemmat_lindblad(A,B):
	dim = A.shape[0]
	Ident = np.identity(dim)
	S = np.kron(B.conjugate(), A)		
	BdA = B.transpose().conjugate() @ A 
	S -= 0.5 * np.kron(Ident, BdA)
	S -= 0.5 * np.kron(BdA.transpose(), Ident)
	return S


# Set up the original decay and dephasing operators
a = lowering(2)
Id2 = np.identity(2)
a0 = np.kron(a, Id2)
a1 = np.kron(Id2, a)
# Decoherence operators
L01 = a0					# Qubit 0  decay
L11 = a1 					# Qubit 1  decay
L02 = a0.transpose() @ a0 	# Qubit 0 dephasing
L12 = a1.transpose() @ a1	# Qubit 1 dephasing

N = 4

## Now read trained Operators from file and compare
if do_analyze:
	# Loading learned operators data from 'UDEdatadir'

	# Read learned Hamiltonian (N x N)
	filename_re = UDEdatadir + "/LearnedHamiltonian_Re.dat"
	filename_im = UDEdatadir + "/LearnedHamiltonian_Im.dat"
	Ham_test_re = np.loadtxt(filename_re, usecols=range(N), skiprows=2)
	Ham_test_im = np.loadtxt(filename_im, usecols=range(N), skiprows=2)
	Ham_test = Ham_test_re + 1j*Ham_test_im	
	print("Learned Hamiltonian [MHz]:") 
	print(Ham_test)
	Ham_error = Ham_org - Ham_test
	print("Hamiltonian error [MHz]:", np.linalg.norm(Ham_error))

	if UDEmodel == 'lindblad' or UDEmodel=='both':
		# Read learned system Lindblad matrix (N^2 x N^2)
		filename = UDEdatadir + "/LearnedLindbladSystemMat.dat"
		Lindblad_learned = np.loadtxt(filename, skiprows=2)
		# print("Learned Lindblad System matrix:")
		# print(Lindblad_learned)

		# Set up the original system Lindblad matrix, using Decay and Decoherence: \bar L\kron L - 1/2 (I \kron L'L + (L'L)^T\kron I)
		Lops = [L01, L11, L02, L12]
		decoherencetimes = [T1[0], T1[1], T2[0], T2[1]] # us
		Lindblad_org = np.zeros((N**2, N**2))
		LopsSys = []
		for i in range(N):
			addme = systemmat_lindblad(Lops[i], Lops[i])
			LopsSys.append(addme)
			Lindblad_org += 1./decoherencetimes[i] * addme
		print("Original Lindblad System matrix")
		print(Lindblad_org)
		print("Learned Lindblad System matrix")
		print(Lindblad_learned)
		Lindblad_error = Lindblad_org - Lindblad_learned
		print(" Lindblad System matrix error norm:", np.linalg.norm(Lindblad_error))


		# Find best match to Lops ansatz min 1/2 || G - sum x_i Li||^2
		M = np.zeros((N,N)) 	# Hessian
		rhs = np.zeros(N)
		for i in range(N):
			for j in range(N):
				M[i,j] = np.trace(LopsSys[i].transpose()@LopsSys[j])
			rhs[i] = np.trace(Lindblad_learned.transpose()@LopsSys[i])
		# Solve linear system
		x = np.linalg.inv(M) @ rhs

		print("Best match decay and decoherence times:")
		print([1.0/xi for xi in x])
		print("Original decay and decoherence times:")
		print(decoherencetimes)

		# Residual in the objective function 
		f = 0.0
		err = Lindblad_learned.copy()
		for i in range(4):
			err -= x[i] * LopsSys[i]
		print("Match objective function value: ", 1/2*np.linalg.norm(err)**2)


		# Get coefficients of the learned lindblad operators
		filename = UDEdatadir + "/params.dat"
		skiprowsHam = N**2-1
		paramsL = np.loadtxt(filename, skiprows=skiprowsHam)

#########
# Extrapolate to random controls
########

if do_extrapolate:
	# Get learned parameters
	filename = UDEdatadir + "/params.dat"
	params = np.loadtxt(filename)

	# Simulate the evolution for random control pulses using the learned UDE model
	datadir=UDEdatadir + "_FWDrandcontrol"
	quandary.initctrl_MHz = [3.0, 3.0]
	quandary.randomize_init_ctrl = True
	quandary.rand_seed = 1234
	quandary.nsplines=30
	t_UDE, pt_UDE, qt_UDE, infid_UDE, expect_UDE, popul_UDE = quandary.simulate(UDEmodel=UDEmodel, learn_params=params, datadir=datadir)

	# Simulate the exact (to-be-learned) model with same random controls
	datadir = "./stdHam_perturb_"+str(rand_amp_MHz)+"MHz_randcontrol"
	quandary_mod.initctrl_MHz = [3.0, 3.0]
	quandary_mod.randomize_init_ctrl = True
	quandary_mod.rand_seed = 1234
	quandary_mod.nsplines=30
	t_mod, pt_mod, qt_mod, infid_mod, expect_mod, popul_mod = quandary_mod.simulate(datadir=datadir)

	# Compare solution at final time
	print("Extrapolation random control TEST: infid_UDE=", infid_UDE, ",infid_org=", infid_mod, ", rel. error = ", abs((infid_UDE - infid_mod)/infid_mod))
	print("Final time error: ", np.linalg.norm(quandary.uT - quandary_mod.uT))
	


###################
### TEST what can be learned with different basis matrices
###################

##### Access an element in lower-trianglular matrix X for the Lindblad coefficients. The coefficients are stored vectorized row-wise in params
def getXij(i,j, nbasis, params):
	# Mapping for accessing row-wise vectorized X_ij coefficients for lower-triangular matrix X
	def mapID(i,j, nbasis):
		return int(i*nbasis - i*(i+1)/2 + j)
	xij = 0.0
	if (i<=j):
		xij = params[mapID(i,j, nbasis)]
	return xij

##### Compute elements in A = X*X':  aij = sum_l x_il * x_jl
def assembleAij(i,j, nbasis, params):
		aij = 0.0
		for l in range(nbasis):
			xil = getXij(i,l,nbasis,params)
			xjl = getXij(j,l,nbasis,params)
			aij += xil*xjl
		return aij
#####

if do_testansatz:
	doublesum = True 
	real_coeffs = True 

	# # Basis mats on full system
	BasisMats = getGellmanMats(N, upperonly=False, realonly=True, shifted=True)
	# BasisMats = getGellmanMats(N, upperonly=False, realonly=False, shifted=True)
	# BasisMats= getEijBasisMats(N, upperonly=True)

	# # Basis mats per qubit:
	# # Those can learn the original decay and dephasing
	# BasisMats = []
	# BasisM_0 = getGellmanMats(Ne[0], realonly=True, upperonly=True, shifted=True)
	# BasisM_1 = getGellmanMats(Ne[1], realonly=True, upperonly=True, shifted=True)
	# Id2 = np.zeros(Ne)
	# Id2[0,0] = 1.0
	# Id2[1,1] = 1.0
	# # Lift to full dimension
	# for B in BasisM_0:
	# 	BasisMats.append(np.kron(B, Id2))
	# for B in BasisM_1:
	# 	BasisMats.append(np.kron(Id2, B))

	# TEST: First make sure that the loaded system matrix is the same as when assembled here:
	nbasis = len(BasisMats)
	Lindblad_learned_test = np.zeros(Lindblad_learned.shape)
	if doublesum:
		for i in range(len(BasisMats)):
			for j in range(len(BasisMats)):
				aij = assembleAij(i,j, nbasis, paramsL)
				Lindblad_learned_test += aij * systemmat_lindblad(BasisMats[i], BasisMats[j])
	else:
		for i in range(len(BasisMats)):
			j=i
			aij = paramsL[i]
			Lindblad_learned_test += aij * systemmat_lindblad(BasisMats[i], BasisMats[j])
	# assert(np.linalg.norm(Lindblad_learned - Lindblad_learned_test) < 1e-14)
	print("ERROR = ", np.linalg.norm(Lindblad_learned - Lindblad_learned_test))


	# Set up coefficient matrix A for double sum of Lindblad terms
	if doublesum:
		if real_coeffs:
			X = np.zeros((N**2-1, N**2-1), dtype=float)
		else:
			X = np.zeros((N**2-1, N**2-1), dtype=complex)
		for i in range(N**2-1):
			for j in range(i+1):
				val = random.uniform(-1.0, 1.0) + random.uniform(-1.0, 1.0)*1j
				if i==j:
					X[i,j] = val.real 
				else:
					if real_coeffs:
						X[i,j] = val.real
					else:
						X[i,j] = val
		A = X @ X.transpose().conjugate()

	# assemble Gellmann system matrix
	if real_coeffs:
		G_test = np.zeros((N**2, N**2), dtype=float)
	else:
		G_test = np.zeros((N**2, N**2), dtype=complex)
	BasisSystemMats = []
	for i in range(len(BasisMats)):
		if doublesum:
			for j in range(len(BasisMats)):
				addme = systemmat_lindblad(BasisMats[i], BasisMats[j])
				BasisSystemMats.append(addme)
				coeff = A[i,j]
				G_test += coeff * addme
		else:
			j=i
			addme = systemmat_lindblad(BasisMats[i], BasisMats[j])
			BasisSystemMats.append(addme)
			coeff = random.uniform(-1.0, 1.0) + random.uniform(-1.0, 1.0)*1j
			print(coeff)
			if real_coeffs:
				G_test += coeff.real * addme
			else:
				G_test += coeff * addme
	learnable = abs(G_test)>0
	print("System matrix elements that can be learned with the basis matrices ")
	print(learnable) 

	print("System matrix elements that SHOULD be learned (Lindblad orig)")
	should_be_learned = abs(Lindblad_org)>0
	print(should_be_learned)

	# # Test diagonalizing A and convert to a single sum
	# evals, evecs = np.linalg.eig(A)
	# U = evecs.transpose().conjugate()
	# D = np.diag(evals)
	# # test: 
	# diag_err = np.linalg.norm(evecs.conjugate().transpose()@evecs - np.identity(15))
	# # print("Diagonalization error: ", diag_err)

	# # transform the operators using the unitary U
	# BasisMats_new = []
	# for i in range(N**2 -1):
	# 	Gadd = np.zeros((4, 4), dtype=complex)
	# 	for j in range(len(BasisMats)):
	# 		Gadd += U[i,j].conjugate() * BasisMats[j]
	# 	BasisMats_new.append(Gadd)

	# G_test_single = np.zeros((N**2, N**2), dtype=complex)
	# for i in range(len(BasisMats_new)):
	# 	addme = systemmat_lindblad(BasisMats_new[i], BasisMats_new[i])
	# 	G_test_single += D[i,i] * addme
	
	# print("System matrix elements that can be learned with the basis matrices (SINGLE SUM)")
	# print(G_test_single - G_test)
	# print("ERROR = ", np.linalg.norm(G_test - G_test_single))
	# assert(np.linalg.norm(G_test - G_test_single))



##############
# expand a basis Lin=[B1, B2, ...] to span the entire Hilbertspace
def expandBasis(N, Lin=[]):
	Eij = getEijBasisMats(4, upperonly=False, shifted=False)
	testid = 0
	for testid in range(len(Eij)):
		# Set up span(Lin, Eij)
		dimA = N**2
		AA = np.zeros( (dimA, len(Lin)+1))
		for i in range(len(Lin)):
			AA[:,i] = Lin[i].flatten(order='F') # Vectorize columnwise
		AA[:,len(Lin)] = Eij[testid].flatten(order='F') # Vectorize columnwise
		# Test rank
		rank = np.linalg.matrix_rank(AA)
		if rank == AA.shape[1]:
			Lin.append(Eij[testid])
			# print("-> appending ")
			# print(Eij[testid])

	# TEST if B spans a basis
	dimA = N**2
	vecL= np.zeros( (dimA, dimA-1))
	for i in range(len(Lin)):
		vecL[:,i] = Lin[i].flatten(order='F')
	assert(np.linalg.matrix_rank(vecL) == N**2-1)
	assert(len(Lin) == N**2-1)
	# print("Done setting up a basis that contains decay and decoherence")


