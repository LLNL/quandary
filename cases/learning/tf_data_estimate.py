# Make sure you have the location of quandary.py in your PYTHONPATH. E.g. with
#   > export PYTHONPATH=/path/to/quandary/:$PYTHONPATH
# Further, make sure that your quandary executable is in your $PATH variable. E.g. with
#   > export PATH=/path/to/quandary/:$PATH
from quandary import * 
np.set_printoptions( linewidth=800)

maxcores=8 # Maximum number of cores to be used

do_sanityTest = True # False
do_training = True # False 

do_extrapolate = False
do_analyze = False
do_prune = False
do_plotResults = True

# NOTE: The setup of the quandary object needs to be identical to that in 'swap02_pulses_for_qudit.py'

# Both 3 & 4 essential levels work. Data was generated with 4 essential levels
Ne = [3]  # Number of essential energy levels
Ng = [1]  # Number of extra guard levels

# Frequency scaling factor relative to GHz and ns (1e-9 sec)
freq_scale = 1.0 # 
time_scale = 1/freq_scale

#  Transition frequencies [GHz] from the device: 
# 2025, Jan 06
# f01 = 3422.625432
# f12=  3213.617052
# July 25, 2025: Transition frequencies [GHz]: f01 = 3.416682744 f12= 3.2074712470000004
# Aug 4, 2025:   Transition frequencies [GHz]: f01 = 3.416634567 f12= 3.2074712470000004
f01 = 3.416634567*freq_scale
f12 = 3.2074712470000004*freq_scale

# 01 transition frequencies [GHz] per oscillator
freq01 = [f01] 
# Anharmonicities [GHz] per oscillator
selfkerr = [f01-f12]
# Rotating frame frequency
rotfreq = freq01

# Set the total time duration (us)
# T = 0.360*time_scale
T = 360.0*time_scale

# Bounds on the control pulse (in rotational frame, p and q) [MHz] per oscillator
maxctrl = 7.0e-3*freq_scale
initctrl = 1.0e-3*freq_scale

# Set up a target gate (in essential level dimensions)
if Ne[0] == 3:
	unitary = [[0,0,1],[0,1,0],[1,0,0]]  # 3 essential levels: Swaps first and third levels
elif Ne[0] == 4:
	unitary = [[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]]  # 4 essential levels: Swaps first and third levels
else:
	print("Wrong number of essential levels")
	stop

# print(unitary)

rand_seed = 1235

cwd = os.getcwd()
datadir= cwd + "/SWAP02_optimize"
pcof_opt_file = datadir + "/params.dat"
# Prepare Quandary with those options. This sets default options for all member variables and overwrites those that are passed through the constructor below. Use help(Quandary) to see all options.
quandary = Quandary(Ne=Ne, Ng=Ng, freq01=freq01, selfkerr=selfkerr, pcof0_filename=pcof_opt_file, maxctrl=maxctrl, targetgate=unitary, T=T, control_enforce_BC=True, rand_seed=rand_seed, cw_prox_thres=0.5*abs(selfkerr[0]), gamma_leakage=600.0, verbose=True)

# Turn off verbosity after the carrier frequencies have been reported
quandary.verbose = False

# Only use 2 carrier frequencies
quandary.carrier_frequency[0] = quandary.carrier_frequency[0][0:2]
print("Carrier freq: ", quandary.carrier_frequency) 

# Execute quandary. Default number of executing cores is the essential Hilbert space dimension. Limit the number of cores by passing ncores=<int>. Use help(quandary.optimize) to see all arguments.

t, pt, qt, infidelity, expectedEnergy, population = quandary.simulate(datadir=datadir+"_FWD", maxcores=maxcores)
print(f"\nSimulated Fidelity = {1.0 - infidelity}")

pcof_opt = quandary.popt # get the optimized control vector

if do_plotResults:
	plot_results_1osc(quandary, pt[0], qt[0], expectedEnergy[0], population[0])

# Modify quandary options for data generation & training (Use Lindblad's eqn)
initialcondition =  "diagonal"  # "pure, 0" "diagonal" "basis" # Initial condition at t=0: Groundstate
T1 = [100.0] # Decoherence times [us]
T2 = [40.0]
output_frequency = 1  # write every x-th timestep
dirprefix = "SWAP02_" + initialcondition # add a prefix for run directories

verbose = False

# We only have data for 3 levels
# Ne = [3]  # Number of essential energy levels
# Ng = [1]  # Number of extra guard levels
# unitary = [[0,0,1],[0,1,0],[1,0,0]] 


datadir_test = cwd+"/vibranium_data/Aug8-25" # Data directory

################
# NOW DO TRAINING! 
################

# Set the UDE model: List of learnable terms, containing "hamiltonian" and/or "lindblad" and/or "transferLinear"
UDEmodel = "transferLinear"

# Set the training time domain
T_train = T	  

# Set training data identifier and all filenames (all initial conditions)
data_identifier = "Tant3Pop" # synthetic (Quandary) data for population data
data_filenames = ["init_0_pop_cor.dat",\
				  "init_1_pop_cor.dat",\
				  "init_2_pop_cor.dat"]

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

learnparams_identity = np.zeros(2) 
learnparams_identity[0] = 1.0 # 0.9 
learnparams_identity[1] = 1.0 # 1.1 

if do_sanityTest:
	# maxcores=1

	quandary.UDEsimulate(pcof0=pcof_opt, trainingdata=trainingdata, UDEmodel=UDEmodel, learn_params=learnparams_identity, maxcores=maxcores, datadir=UDEdatadir+"_identitytransfer")
	print("learnparams = ", learnparams_identity, " CHECK: Above loss!\n")

if do_training:
	print("\nStarting UDE training for UDE model = ", UDEmodel, " initial_params: ", learnparams_identity, " result-directory = ", UDEdatadir, "...")

	# Start training, use the unperturbed control parameters in pcof_opt
	quandary.training(pcof0=pcof_opt, trainingdata=trainingdata, UDEmodel=UDEmodel, datadir=UDEdatadir, T_train=T_train, learn_params=learnparams_identity, maxcores=maxcores) 

	filename = UDEdatadir + "/params.dat"
	learnparams_opt = np.loadtxt(filename)
	print("Training finished. Learned transfer function parameters: ", learnparams_opt, "\n")

	# Simulate forward with optimized paramters to write out the Training data evolutions and the learned evolution
	fwd_dir = UDEdatadir+"/FWD_opt"
	print("\n -> Eval loss of optimized UDE model. Results (populations) in dir: ", fwd_dir)
	quandary.UDEsimulate(trainingdata=trainingdata, UDEmodel=UDEmodel, datadir=fwd_dir, T_train=quandary.T, learn_params=learnparams_opt, maxcores=maxcores)

