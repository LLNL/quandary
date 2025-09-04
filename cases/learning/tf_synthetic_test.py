# Make sure you have the location of quandary.py in your PYTHONPATH. E.g. with
#   > export PYTHONPATH=/path/to/quandary/:$PYTHONPATH
# Further, make sure that your quandary executable is in your $PATH variable. E.g. with
#   > export PATH=/path/to/quandary/:$PATH
from quandary import * 
np.set_printoptions( linewidth=800)

maxcores=8 # Maximum number of cores to be used

do_datageneration = True 
do_training = True # False 
do_extrapolate = False
do_analyze = False
do_prune = False

do_lindblad = False # Switch to run Lindblad solver rather than Schroedinger

cwd = os.getcwd()
dirprefix = cwd + "/SWAP02" # Choose a prefix for all data directories 

# Standard Hamiltonian and Lindblad model setup
Ne = [3]			# Number of essential levels
Ng = [0]			# Number of guard levels
Ntot = [Ne[i] + Ng[i] for i in range(len(Ne))]

# Frequency scaling factor relative to MHz and us (1e-6 sec)
freq_scale = 1e-2 # 1e-2 (100 MHz); 1e-3 (GHz)
time_scale = 1/freq_scale

#  Transition frequencies [GHz] from the device: 2025, Jan 06
f01 = 3422.625432*freq_scale
f12=  3213.617052*freq_scale

# 01 transition frequencies [GHz] per oscillator
freq01 = [f01] 
# Anharmonicities [GHz] per oscillator
selfkerr = [f01-f12]
# Rotating frame frequency
rotfreq = freq01

# Set the gate duration (us)
T = 0.240*time_scale

# Bounds on the control pulse (in rotational frame, p and q) [MHz] per oscillator
maxctrl = 4.0*freq_scale

# Amplitude of randomized initial control vector
initctrl = 10.0*freq_scale

# Set up a target gate (in essential level dimensions)
unitary = [[0,0,1],[0,1,0],[1,0,0]]  # Swaps first and last level

# Decoherence times. Only used if do_lindblad=True
T1 = [100.0*time_scale] 
T2 = [40.0*time_scale]

verbose = False
rand_seed=1234

######
# Optimize for pulses that realize the unitary gate, or load from previouly optimized pulses
#####
quandary = Quandary(Ne=Ne, Ng=Ng, freq01=freq01, rotfreq=rotfreq, selfkerr=selfkerr, initctrl= initctrl, maxctrl=maxctrl, targetgate=unitary, T=T, verbose=False, rand_seed=rand_seed)
datadir= dirprefix + "_optimize"
t, pt, qt, infidelity, expectedEnergy, population = quandary.optimize(datadir=datadir, maxcores=maxcores)
print(f"\nFidelity = {1.0 - infidelity}")
print("Optimized pulse with Schroedinger's eqn, dir = ", datadir)
# plot_results_1osc(quandary, pt[0], qt[0], expectedEnergy[0], population[0])

pcof_opt = quandary.popt  # Get the optimized control pulse vector

##### 
# Data generation with modified control pulses and modified Hamiltonian 
#####

# Set up modified quandary object for data generation and learning
initialcondition =  "diagonal"  # "pure, 0" "diagonal" "basis" # Initial condition at t=0: Groundstate
output_frequency = 1  # write every x-th timestep
if do_lindblad:
	quandary2 = Quandary(Ne=Ne, Ng=Ng, freq01=freq01, rotfreq=rotfreq, selfkerr=selfkerr, maxctrl=maxctrl, targetgate=unitary, T=T, pcof0=pcof_opt, verbose=verbose, rand_seed=rand_seed,  initialcondition=initialcondition, output_frequency=output_frequency, T1=T1, T2=T2)
else: 
	quandary2 = Quandary(Ne=Ne, Ng=Ng, freq01=freq01, rotfreq=rotfreq, selfkerr=selfkerr, maxctrl=maxctrl, targetgate=unitary, T=T, pcof0=pcof_opt, verbose=verbose, rand_seed=rand_seed,  initialcondition=initialcondition, output_frequency=output_frequency)


# Perturb the controls: Scale each carrier wave component. One system, 2 frequencies in this test
pfact = [0.8, 1.2] # [0.75, 1.5] # These are the transfer parameters that are to be learned
Nfirst = round(len(pcof_opt)/2) # Number of elements for the first carrier frequency
pcof_pert = np.zeros(2*Nfirst)
pcof_pert[0:Nfirst] = [pcofi * pfact[0] for pcofi in pcof_opt[0:Nfirst]]
pcof_pert[Nfirst:2*Nfirst] = [pcofi * pfact[1] for pcofi in pcof_opt[Nfirst:2*Nfirst]]	# original control vector in pcof_opt

# Perturb the Hamiltonian in the quandary2 object (order MHz)
pert_freq01 = -1*freq_scale
pert_selfkerr = -2*freq_scale
quandary2.freq01 = [freqi  + pert_freq01 for freqi in freq01] 
quandary2.selfkerr= [selfi + pert_selfkerr for selfi in selfkerr]


datadir_test = dirprefix+"_asmeasured" 
if do_datageneration:
	# Generate training data: Simulate perturbed controls
	print("Generate data in ", datadir_test)
	t, pt, qt, infidelity_pert, expectedEnergy, population = quandary2.simulate(pcof0=pcof_pert, maxcores=maxcores, datadir=datadir_test)
	# plot_results_1osc(quandary2, pt[0], qt[0], expectedEnergy[0], population[0])

	print("-> Generated trajectory for perturbed pulse amplitude with perturbation factor:", pfact, " AND perturbed Hamiltonian with freq01 += ", pert_freq01, " selfkerr += ", pert_selfkerr)
	print("-> Perturbed pulse (training data) directory:", datadir_test, " Fidelity:", 1.0 - infidelity_pert,"\n")
 

################
# NOW DO TRAINING! 
################

# Set the UDE model: List of learnable terms, containing "hamiltonian" and/or "lindblad" and/or "transferLinear"
UDEmodel = "hamiltonian, transferLinear"

# Set the training time domain
T_train = T	  

# Set training data identifier and all filenames (all initial conditions)
data_identifier = "syntheticPop" # synthetic (Quandary) data for population data
data_filenames = ["population0.iinit0000.dat",\
				  "population0.iinit0001.dat",\
				  "population0.iinit0002.dat"]

# Switch between tikhonov regularization norms (L1 or L2 norm)
tik0_onenorm = True 			#  Use L1 for sparsification property
# Factor to scale the loss objective function value
loss_scaling_factor = 1e3

# Output directory for training
UDEdatadir = cwd + "/" + dirprefix+ "_UDE"

# Prepare the trainingdata list that is passed to Quandary. Format:
# ["identifyier, dir1, filename1, filename2, ...",  # 1st pulse
#  "identifyier, dir2, filename1, filename2, ...", 	# 2nd pulse
#  ...] 
trainingdata = [data_identifier + ", " + datadir_test]
for filename in data_filenames:
	trainingdata[0] +=  ", " + filename

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


### TEST: Simulate with the exact control pertubation from above and using the exact (same) Hamiltonian as in the data generation
###  -> Loss should be zero!
learnparams_perturb = 0e-4*np.ones(10) # No perturbation
learnparams_perturb[8] = pfact[0] # correct scaling factors
learnparams_perturb[9] = pfact[1] # correct scaling factors
quandary2.UDEsimulate(pcof0=pcof_opt, trainingdata=trainingdata, UDEmodel=UDEmodel, learn_params=learnparams_perturb, maxcores=maxcores, datadir=UDEdatadir+"_exact")
print("learnparams = ", learnparams_perturb, " CHECK: Above Loss should be zero (small)!\n")

# Reset the Hamiltonian to the original model:  
quandary2.freq01[:] = freq01[:]
quandary2.selfkerr[:] = selfkerr[:]

# Set up initial learning parameters
learnparams_init = 1e-4*np.ones(10) 
learnparams_init[8] = 0.9 
learnparams_init[9] = 1.1 

## TEST: Simulate with initial learning parameters -> Loss should be large!
quandary2.UDEsimulate(pcof0=pcof_opt, trainingdata=trainingdata, UDEmodel=UDEmodel, learn_params=learnparams_init, maxcores=maxcores, datadir=UDEdatadir+"_init")
print("Initial learnparams = ", learnparams_init, " CHECK: Above Loss should be large!\n")

if do_training:
	print("\nStarting UDE training for UDE model = ", UDEmodel, " initial_params: ", learnparams_init, "...")

	# Start training, use the unperturbed control parameters in pcof_opt
	quandary2.training(pcof0=pcof_opt, trainingdata=trainingdata, UDEmodel=UDEmodel, datadir=UDEdatadir, T_train=T_train, learn_params=learnparams_init, maxcores=maxcores) 

	filename = UDEdatadir + "/params.dat"
	learnparams_opt = np.loadtxt(filename)
	print("Training finished. Learned transfer function parameters: ", learnparams_opt, "\n")

	# Simulate forward with optimized paramters to write out the Training data evolutions and the learned evolution
	print("\n -> Eval loss of optimized UDE model.")
	quandary2.UDEsimulate(trainingdata=trainingdata, UDEmodel=UDEmodel, datadir=UDEdatadir+"/FWD_opt", T_train=quandary2.T, learn_params=learnparams_opt, maxcores=maxcores)

