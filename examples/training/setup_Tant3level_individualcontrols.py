# Make sure you have the location of quandary.py in your PYTHONPATH. E.g. with
#   > export PYTHONPATH=/path/to/quandary/:$PYTHONPATH
# Further, make sure that your quandary executable is in your $PATH variable. E.g. with
#   > export PATH=/path/to/quandary/:$PATH
from quandary import * 
np.set_printoptions( linewidth=800)

do_training = True 
do_loadtrained = True 
do_plot2D = False
do_plot3D = False
do_analyze = True 
do_extrapolate = False

unitMHz = True 

# Best-guess system description
Ne = [3]
Ng = [0]
N = np.prod([Ne[i] + Ng[i] for i in range(len(Ne))])
freq01 = [3423.0] 
rotfreq = freq01 		# Best-guess detuning: freq01 - rotfreq = 0.0
selfkerr = [200.0]		# best-guess for anharmonicity
# Best-guess decoherence [ns]
T1 = [0.0] # [130.0]
T2 = [0.0] # [6.0]

# Set the duration (us) and time-step size
T = 4.0  	# 4.0  # 8.0
dT = 0.002
nsteps = int(np.ceil(T/dT))
T = nsteps*dT
output_frequency = 1  

# Set up a dummy gate
unitary = np.identity(Ne[0]+Ng[0])

# Initial ground state
initialcondition = "pure, 0"

# Initial controls - will be overwritten by data
randomize_ctrl = False
initctrl_MHz = 1.0
carrier_freq = [[0.0]]
verbose = False
rand_seed=1234

# Output lists
delta_w01 = []
delta_selfkerr = []
delta_p = []
delta_q = []
pMHz = []
qMHz = []

# Iterate over control pulse trajectories, training each pulse separately
# pvals = [0.015, -0.015, 0.018, -0.018, 0.026, -0.026]
# pvals = [0.015, 0.018, 0.026, -0.015, -0.018, -0.026]
# qvals = [0.015, 0.018, 0.026, -0.015, -0.018, -0.026]
pvals = [0.015] 
qvals = [-0.015] 
for pval in pvals:
	for qval in qvals:

		# Set up Quandary with standard Hamiltonian model 
		quandary = Quandary(Ne=Ne, Ng=Ng, freq01=freq01, rotfreq=rotfreq, selfkerr=selfkerr, T=T, nsteps=nsteps, targetgate=unitary, verbose=verbose, rand_seed=rand_seed, T1=T1, T2=T2, initialcondition=initialcondition, output_frequency=output_frequency, randomize_init_ctrl=randomize_ctrl, initctrl_MHz=initctrl_MHz, carrier_frequency=carrier_freq, unitMHz=unitMHz) 

		# Set the UDE learning model: "hamiltonian", or "lindblad", or "both"
		UDEmodel = "both"   
		# Set the training time domain
		T_train = T # T/2.0	  
		# Switch to use mitigated data (corrected with confusion matrix), or raw data
		trainingdata_corrected = True
		# Switch between tikhonov regularization norms (L1 or L2 norm)
		tik0_onenorm = True 			#  Use L1 for sparsification property
		# Factor to scale the loss objective function value
		loss_scaling_factor = 1.0

		# Set training data directories and specifier. For Tant3level, only one pulse per training currently. TODO.
		trainingdatadir = ["Tant3level, /Users/guenther5/Numerics/quantum-udes-database/experiment/240715/Rabi_const_p"+str(pval)+"_q"+str(qval)+"_0711_0715_2024.dat"]

		# Set output directory for training
		UDEdatadir = "./UDE_Tant3level_p"+str(pval)+"_q"+str(qval)+"_rundir"

		# Set training optimization parameters
		quandary.gamma_tik0 = 1e-6		# Tikhonov regularization
		quandary.gamma_tik0_onenorm = tik0_onenorm
		quandary.loss_scaling_factor = loss_scaling_factor
		quandary.tol_grad_abs = 1e-8
		quandary.tol_grad_rel = 1e-8
		quandary.tol_costfunc = 1e-8
		quandary.tol_infidelity = 1e-8
		quandary.gamma_leakage = 0.0
		quandary.gamma_energy = 0.0
		quandary.gamma_dpdm = 0.0
		quandary.maxiter = 100
		# quandary.maxiter = 1

		if do_training:
			print("\n Starting UDE training (UDEmodel=", UDEmodel, ")...")
			out = quandary.training(trainingdatadir=trainingdatadir, trainingdata_corrected=trainingdata_corrected, UDEmodel=UDEmodel, datadir=UDEdatadir, T_train=T_train)
			time, pt, qt, infidelity, expectedEnergy, population = out

			# Simulate forward with optimized paramters to write Training data evolutions and learned evolution on a longer time domain
			filename = UDEdatadir + "/params.dat"
			params = np.loadtxt(filename)
			quandary.T = 16.0
			quandary.nsteps =  int(np.ceil(quandary.T/dT))
			quandary.update()
			
			quandary.UDEsimulate(trainingdatadir=trainingdatadir, trainingdata_corrected=trainingdata_corrected, UDEmodel=UDEmodel, datadir=UDEdatadir+"/FWD_opt", T_train=quandary.T, learn_params=params)

			# Simulate forward the baseline model using zero learnable parameters to print out evolution
			zeroinit = np.zeros(len(params))
			quandary.UDEsimulate(trainingdatadir=trainingdatadir, trainingdata_corrected=trainingdata_corrected, UDEmodel=UDEmodel, datadir=UDEdatadir+"/FWD_zeroinit", T_train=quandary.T, learn_params=zeroinit)

  
		########
		# Load learned Hamiltonian and LIndblad terms from 'UDEdatadir'
		########

		if do_loadtrained:
			# Read learned Hamiltonian (N x N)
			filename_re = UDEdatadir + "/LearnedHamiltonian_Re.dat"
			filename_im = UDEdatadir + "/LearnedHamiltonian_Im.dat"
			Ham_learn_re = np.loadtxt(filename_re, usecols=range(N), skiprows=2)
			Ham_learn_im = np.loadtxt(filename_im, usecols=range(N), skiprows=2)
			Ham_learn = Ham_learn_re + 1j*Ham_learn_im	
			print("Learned Hamiltonian [MHz]:") 
			print(Ham_learn)
			# Ham_error = Ham_org - Ham_test
			# print("Hamiltonian error [MHz]:", np.linalg.norm(Ham_error))

			# Extract 01-frequency, selfkerr and p/q correction
			delta_w01.append(Ham_learn[1,1].real)
			delta_selfkerr.append(Ham_learn[2,2].real)
			delta_p.append(Ham_learn[0,1].real)
			delta_q.append(Ham_learn[0,1].imag)

			# Get pulse strength used in training
			conversion_factor = 47.90850565409482  # conversion factor: Volt to MHz
			ppulse_MHz = pval*conversion_factor 
			qpulse_MHz = qval*conversion_factor
			pMHz.append(ppulse_MHz)
			qMHz.append(qpulse_MHz)

			print("\nLearned Hamiltonian corrections for pulse strength p=", ppulse_MHz, ", q=", qpulse_MHz, " MHz:")
			print("Delta w01 [MHz] = ", delta_w01[-1])
			print("Delta xi  [MHz] = ", delta_selfkerr[-1])
			print("Delta p   [MHz] = ", delta_p[-1], " (", round((ppulse_MHz+delta_p[-1])/ppulse_MHz*100.0,2), "%)")
			print("Delta q   [MHz] = ", delta_q[-1], " (", round((qpulse_MHz+delta_q[-1])/qpulse_MHz*100.0,2), "%)")

			# Get learned coefficients 
			filename = UDEdatadir + "/params.dat"
			params = np.loadtxt(filename)
			rowsHam = Ne[0]**2-1
			paramsH = params[:rowsHam]
			paramsL = params[rowsHam:]
# end of loop over training pulses

# Print all Hamiltonian corrections
print("Learned corrections:")
thetas = []
thetacorrs = []
amps = []
ampcorrs = []
thetafac= []
thetaerr = []
ampfac= []
amperr = []

for i in range(len(delta_w01)):

	# Compute lab-frame pulse amplitude and phase correction
	theta = np.angle(pMHz[i] +1j*qMHz[i])
	if theta < 0:
		theta = 2*np.pi - abs(theta)
	thetacorr = np.angle( (pMHz[i] + delta_p[i]) + 1j*(qMHz[i]+delta_q[i])) 
	if thetacorr < 0:
		thetacorr = 2*np.pi - abs(thetacorr)
	amp = np.sqrt(pMHz[i]**2 + qMHz[i]**2)
	ampcorr = np.sqrt((pMHz[i]+delta_p[i])**2 + (qMHz[i]+delta_q[i])**2)
	thetas.append(theta)
	thetacorrs.append(thetacorr)
	thetafac.append(thetacorr/theta)
	thetaerr.append(round(abs(thetacorr-theta)/abs(theta)*100.0,2))
	amps.append(amp)
	ampcorrs.append(ampcorr)
	ampfac.append(ampcorr/amp)
	amperr.append(round(abs(ampcorr-amp)/abs(amp)*100.0,2))

	print("\n* For pulse trajectory p=", round(pMHz[i], 5), "MHz, q=", round(qMHz[i],5), "MHz:")
	print("	 Delta w01 [MHz] = ", delta_w01[i])
	print("	 Delta xi  [MHz] = ", delta_selfkerr[i])
	print("	 Delta p   [MHz] = ", delta_p[i], " (", round((pMHz[i] + delta_p[i])/pMHz[i]*100.0,2), "%)")
	print("	 Delta q   [MHz] = ", delta_q[i], " (", round((qMHz[i] + delta_q[i])/qMHz[i]*100.0,2), "%)")
	print("  Lab-pulse ampl = ", amps[i], ", corrected =",  ampcorrs[i], "(fac= ", ampfac[i], ", rel. err = ", amperr[i],"%)") 
	print("  Lab-pulse phase = ", thetas[i], ", corrected =",  thetacorrs[i], "(fac=", thetafac[i], ", rel. err = ", thetaerr[i] ,"%)") 

# Plot of Hamiltonian corrections
if do_plot2D:
	# 2D Plot of Hamiltonian corrections
	ax = plt.axes() 
	irange = [i for i in range(len(delta_w01))]
	##
	# plt.ylabel("Learned Hamiltonian correction [MHz]")
	# plt.scatter(irange, delta_w01, color="green", label='delta w01') 
	# plt.scatter(irange, delta_selfkerr, color="blue", label='delta xi') 
	##
	# plt.ylabel("Learned rot-frame pulse correction [MHz]")
	# plt.scatter(irange, delta_p, color="black", marker="x", label='delta p') 
	# plt.scatter(irange, delta_q, color="red", marker="x", label='delta q') 
	##
	# plt.ylabel("Rot-frame pulse amplitude ratio (p+dp)/p (%)")
	# plt.scatter(irange, [(pMHz[i]+delta_p[i])/pMHz[i]*100.0 for i in range(len(delta_w01))], color="black", marker="x", label='p factor (%)') 
	# plt.scatter(irange, [(qMHz[i]+delta_q[i])/qMHz[i]*100.0 for i in range(len(delta_w01))], color="red", marker="x", label='q factor (%)') 
	##
	# plt.ylabel("Lab-pulse correction factor ")
	# plt.scatter(irange, ampfac, color="blue", marker="x", label='amplitude') 
	# plt.scatter(irange, thetafac, color="green", marker="o", label='phase') 
	##
	plt.ylabel("Lab-pulse rel. error (%)")
	plt.scatter(irange, amperr, color="blue", marker="x", label='amplitude') 
	plt.scatter(irange, thetaerr, color="green", marker="o", label='phase') 
	##
	plt.xlabel("Control pulse [MHz]")
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=4)
	plt.grid()
	ax.set_xticks(irange) 
	ax.set_xticklabels(["p="+str(round(pMHz[i],2))+", q="+str(round(qMHz[i],2)) for i in range(len(delta_w01))]) 
	ax.tick_params(axis='x', labelrotation=90)
	plt.show() 

if do_plot3D:
	# 3D plot of Hamiltonian corrections
	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')
	##
	# ax.set_zlabel('Learned Hamiltonian correction [MHz]')
	# ax.scatter(pMHz, qMHz, delta_w01, marker='x', color='green', label='delta w01')
	# ax.scatter(pMHz, qMHz, delta_selfkerr, marker='x', color='blue', label='delta xi')
	##
	# ax.set_zlabel('Delta control [MHz]')
	# ax.scatter(pMHz, qMHz, delta_p, marker='x', color='black', label='delta p')
	# ax.scatter(pMHz, qMHz, delta_q, marker='x', color='red', label='delta q')
	##
	# ax.set_zlabel("Lab-pulse correction factor ")
	# ax.scatter(pMHz, qMHz, ampfac, marker='x', color='blue', label='amplitude')
	# ax.scatter(pMHz, qMHz, thetafac, marker='o', color='green', label='phase')
	##
	ax.set_zlabel("Lab-pulse rel. error (%)")
	ax.scatter(pMHz, qMHz, amperr, marker='x', color='blue', label='amplitude')
	ax.scatter(pMHz, qMHz, thetaerr, marker='o', color='green', label='phase')
	##
	ax.set_xlabel('p MHz')
	ax.set_ylabel('q MHz')
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=4)
	plt.grid()
	plt.show()


###########
# Analyze the Lindblad operators (MIGHT NOT BE WORKING... Todo.)
##########
if do_analyze:
	def systemmat_lindblad(A,B):
	# Construct the vectorized Lindblad system matrix for term  A rho B' - 1/2{B'A, rho}:
	#  -> vectorized S = \bar B \kron A - 1/2(I \kron B'A + (B'A)^T \kron I)
		dim = A.shape[0]
		Ident = np.identity(dim)
		S = np.kron(B.conjugate(), A)		
		BdA = B.transpose().conjugate() @ A 
		S -= 0.5 * np.kron(Ident, BdA)
		S -= 0.5 * np.kron(BdA.transpose(), Ident)
		return S

	# Set up the original system Lindblad matrix, using Decay and Decoherence
	a = lowering(Ne[0])
	a0 = a
	L01 = a0					# Qubit 0  decay
	L02 = a0.transpose() @ a0 	# Qubit 0 dephasing
	Lops_org = [L01, L02]
	# decoherencetimes = [T1[0], T2[0]] # us
	## Best guess decoherence times
	decoherencetimes = [130.0, 6.0] # us
	LindbladSys_org = np.zeros((N**2, N**2), dtype=complex)
	LopsSys = []
	for i in range(len(Lops_org)):
		addme = systemmat_lindblad(Lops_org[i], Lops_org[i])
		LopsSys.append(addme)
		LindbladSys_org += 1./decoherencetimes[i] * addme

	# Set up the learned system Lindblad matrix. 
	##### Compute elements in A = X*X':  aij = sum_l x_i^l * x_j^l
	def assembleAij(i,j, nbasis, params):
		# Mapping for accessing column-wise vectorized X_i^l coefficients in lower-triangular matrix X
		def mapID(i,j, nbasis):
			return int(i*nbasis - i*(i+1)/2 + j)
		# Sum up
		aij = 1j*0.0
		for l in range(nbasis):
			xil = 1j*0.0
			if (l<=i):
				xil  =     params[mapID(l,i, nbasis)]
				xil += 1j* params[mapID(l,i, nbasis)+int(len(params)/2)] 
			xjl = 0.0
			if (l<=j):
				xjl  =     params[mapID(l,j, nbasis)]
				xjl += 1j* params[mapID(l,j, nbasis)+int(len(params)/2)]
			aij += xil*xjl.conjugate()
		return aij
	#####
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


	# TEST: Same system matrix should be constructed when reading the learned Lindblad operators from file and assembling them in a single-sum (N^2-1 many, each NxN complex)
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
	for i in range(len(LearnedOps_re)):
		print(f"Lindblad Operator {i}:\n{LearnedOps[i]}")

	# Now assemble lindblad system matrix using those operators
	LindbladSys_learned = np.zeros((N**2, N**2), dtype=complex)
	for i in range(N**2-1):
		addme = systemmat_lindblad(LearnedOps[i], LearnedOps[i])
		LindbladSys_learned += addme
	# Make sure those two system matrices are the same
	assert(np.linalg.norm(Lsys_double - LindbladSys_learned) < 1e-11)

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

	# Residual in the objective function 
	f = 0.0
	err = LindbladSys_learned.copy()
	for i in range(nops):
		err -= x[i] * LopsSys[i]
	err = 1/2*np.linalg.norm(err)**2

	print("Best match decay and decoherence times (residual = ", err,")")
	# print([1.0/np.sqrt(xi) for xi in x])
	print([1.0/xi for xi in x])
	print("Best-guess decay and decoherence times:")
	print(decoherencetimes)



#########
# Extrapolate to random controls. TODO!
########

if do_extrapolate:
	# Get learned parameters
	filename = UDEdatadir + "/params.dat"
	params = np.loadtxt(filename)

	# Load random pulse sequence
	ipulse = 0
	randpulsefile = "/Users/guenther5/Numerics/quantum-udes-database/experiment/240715/Random_pulses/randpulse"+str(ipulse)+"_20240716.dat"
	randpulsedata = np.loadtxt(randpulsefile, skiprows=1, usecols=[1,2,3])
	t = randpulsedata[:,0]  # ns
	pt = randpulsedata[:,1]
	qt = randpulsedata[:,2]

	quandary.nsteps= len(t)-1
	quandary.T = t[-1]/1e+3 # us
	dt = quandary.T/quandary.nsteps
	quandary.spline_order=0
	quandary.spline_knot_spacing=dt
	stop

	# Simulate constant control using UDE model
	datadir=UDEdatadir + "_FWDrandcontrol"
