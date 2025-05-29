# -------------------------------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------------------------------


# Make sure you have the location of quandary.py in your PYTHONPATH. E.g. with
#   > export PYTHONPATH=/path/to/quandary/:$PYTHONPATH
# Further, make sure that your quandary executable is in your $PATH variable. E.g. with
#   > export PATH=/path/to/quandary/:$PATH
from quandary import * 
np.set_printoptions( linewidth = 800)

do_datageneration   : bool          = True 
do_training         : bool          = True 
do_extrapolate      : bool          = False
do_analyze          : bool          = False
do_prune            : bool          = False



# Standard Hamiltonian and Lindblad model setup
unitMHz             : bool          = True
Ne                  : list[int]     = [3]           # len = Q. k'th element = Number of essential levels in the k'th sub-system    = n^e_k in the paper
Ng                  : list[int]     = [0]           # len = Q. k'th element = Number of guard levels in the k'th sub-system
freq01              : list[float]   = [4105.95]     # len = Q, k'th element = 01- transition [MHz] for the k'th sub-system         = \omega_k in the paper
rotfreq             : list[float]   = freq01        # len = Q. k'th element = rotating frame frequency for the k'th sub-system     = \varepsilon_k in the paper
selfkerr            : list[float]   = [219.8]       # 1-2 transition [MHz]
N                   : np.ndarray    = np.prod([Ne[i] + Ng[i] for i in range(len(Ne))])

# Set the duration (us)
T                   : float         = 0.2           # 2.0 
dt                  : float         = 0.0001
output_frequency    : int           = 1             # write every x-the timestep

# For testing, can add a prefix for run directories. 
dirprefix           : str           = "data_out"

# Decoherence [us]
T1                  : list[float]   = [100.0]
T2                  : list[float]   = [40.0]

# Set up a dummy gate
unitary             : np.ndarray    = np.identity(np.prod(Ne))

# Initial condition at t = 0: Ground state
initialcondition    : str           = "pure, 0"

verbose             : bool          = False
rand_seed           : int           = 1234



# -------------------------------------------------------------------------------------------------
# Generate training data trajectories for various pulse strength (one trajectory for each element 
# in the initctrl_MHz vector)

initctrl_MHz        : list[float]   = [8.0] # [1.0, 0.5, 0.25]
pert_alpha_scale    : float         = 1.5
pert_sigma_offset   : float         = 1.0*2*np.pi

randomize_init_ctrl : bool          = True 
trainingdatadir     : list[str]     = []

# Number of carrier frequencies per pulse strength, per oscillator
n_carrier_freq      : list[list[int]] = [];

# Loop over the different pulse strengths
for ctrlMHz in initctrl_MHz:
    cwd = os.getcwd()
    trainingdatadir.append(cwd + "/" + dirprefix + "_ctrlMHz" + str(ctrlMHz) + "_asmeasured")

    # Setup quandary object using the standard Hamiltonian model.
    quandary = Quandary(Ne                  = Ne,
                        Ng                  = Ng, 
                        freq01              = freq01, 
                        rotfreq             = rotfreq, 
                        selfkerr            = selfkerr, 
                        T                   = T, 
                        targetgate          = unitary, 
                        verbose             = verbose, 
                        rand_seed           = rand_seed, 
                        T1                  = T1, 
                        T2                  = T2, 
                        initialcondition    = initialcondition, 
                        dT                  = dt, 
                        randomize_init_ctrl = randomize_init_ctrl, 
                        initctrl_MHz        = ctrlMHz, 
                        output_frequency    = output_frequency,
                        unitMHz             = unitMHz) 

    # Get the number of carrier waves for this oscillator
    n_carrier_freq_tmp : list[int] = [];
    for iosc in range(len(quandary.carrier_frequency)):
        n_carrier_freq_tmp.append(len(quandary.carrier_frequency[iosc]));
    n_carrier_freq.append(n_carrier_freq_tmp);

    if do_datageneration:

        # First simulate the unperturbed controls to get the params of the unperturbed controls
        datadir_orig        : str       = cwd + "/" + dirprefix + "_ctrlMHz" + str(ctrlMHz) + "_origpulse"
        times, p_orig, q_orig, _, _, _ = quandary.simulate( maxcores    = 8,
                                                            datadir     = datadir_orig)

        # Now perturb the controls: Scale each carrier wave component. 
        # Here: two carrier waves, but choose scaling g1 = g2 = const. TODO. 
        pcof_org            : np.ndarray        = np.loadtxt(datadir_orig + "/params.dat")
        pcof_pert           : list[np.ndarray]  = [pcofi * pert_alpha_scale + pert_sigma_offset for pcofi in pcof_org]

        # Generate training data: Simulate perturbed controls
        times, p_pert, q_pert, _, _, _ = quandary.simulate( pcof0     = pcof_pert,
                                                            maxcores  = 8, 
                                                            datadir   = trainingdatadir[-1]);

        # there is only one oscillator
        p_orig = p_orig[0]
        q_orig = q_orig[0]
        p_pert = p_pert[0]
        q_pert = q_pert[0]
        

        print("-> Generated trajectory for pulse amplitude ",       ctrlMHz, " with perturbation scale ", pert_alpha_scale, " and offset ", pert_sigma_offset)
        print("->   Unperturbed pulse trajectory directory:",       datadir_orig)
        print("->   Perturbed pulse (training data) directory:",    trainingdatadir[-1])
 

# Plot settings 
import matplotlib as mpl;
import matplotlib.pyplot as plt;
mpl.rcParams['lines.linewidth'] = 2;
mpl.rcParams['axes.linewidth']  = 1.5;
mpl.rcParams['axes.edgecolor']  = "black";
mpl.rcParams['grid.color']      = "gray";
mpl.rcParams['grid.linestyle']  = "dotted";
mpl.rcParams['grid.linewidth']  = .67;
mpl.rcParams['xtick.labelsize'] = 10;
mpl.rcParams['ytick.labelsize'] = 10;
mpl.rcParams['axes.labelsize']  = 11;
mpl.rcParams['axes.titlesize']  = 11;
mpl.rcParams['xtick.direction'] = 'in';
mpl.rcParams['ytick.direction'] = 'in';

# Plot the control trajectory
plt.figure(figsize=(10, 5))
plt.plot(times, p_orig, label = 'p_orig')
#plt.plot(times, q_orig, label = 'q_orig')
plt.plot(times, p_pert, label = 'p_pert')
#plt.plot(times, q_pert, label = 'q_pert')
plt.xlabel('Time [us]')
plt.ylabel('Control [MHz]')
plt.title('Control Trajectory')
plt.legend()
plt.show()



# -------------------------------------------------------------------------------------------------
# Train
# -------------------------------------------------------------------------------------------------

# Make sure to use the same above assumed base model Hamiltonian

# Set the UDE model: List of learnable terms, containing "hamiltonian" and/or "lindblad" and/or "transferLinear"
UDEmodel                : str       = "transferLinear"

# Set the training time domain
T_train                 : float     = T      

# Add data type specifier to the first element of the data list
trainingdatadir[0]      : str       = "synthetic, " + trainingdatadir[0]

# Switch between tikhonov regularization norms (L1 or L2 norm)
tik0_onenorm            : bool      = True             #  Use L1 for sparsification

# Factor to scale the loss objective function value
loss_scaling_factor     : float     = 1e3

# Output directory for training
UDEdatadir              : str       = cwd + "/" + dirprefix + "_UDE"

# Set training optimization parameters
quandary.gamma_tik0                 = 1e-9
quandary.gamma_tik0_onenorm         = tik0_onenorm
quandary.loss_scaling_factor        = loss_scaling_factor
quandary.tol_grad_abs               = 1e-7
quandary.tol_grad_rel               = 1e-7
quandary.tol_costfunc               = 1e-8
quandary.tol_infidelity             = 1e-7
quandary.gamma_leakage              = 0.0
quandary.gamma_energy               = 0.0
quandary.gamma_dpdm                 = 0.0
quandary.maxiter                    = 500



### TEST: Simulate with transfer functions set to the identity -> Loss should be large!
learnparams_identity = [1.0, 1.0] # one for each carrier wave
# quandary.UDEsimulate( pcof0           = pcof_org, 
#                       trainingdatadir = trainingdatadir, 
#                       UDEmodel        = UDEmodel, 
#                       learn_params    = learnparams_identity, 
#                       maxcores        = 8, 
#                       datadir         = UDEdatadir + "_identitytransfer")
# print(" CHECK: Loss should be large!\n")
### TEST: Simulate with transfer functions set to the exact perturbation from above -> Loss should be zero!
learnparams_perturb : list[float] = []
for ipulse in range(len(n_carrier_freq)):
    for iosc in range(len(n_carrier_freq[ipulse])):
        for icw in range(n_carrier_freq[ipulse][iosc]):
            learnparams_perturb.append(pert_alpha_scale)
            learnparams_perturb.append(pert_sigma_offset)
print("n_carrier_freq: ", n_carrier_freq);
print("learnparams_perturb: ", learnparams_perturb);

quandary.UDEsimulate(  pcof0           = pcof_org, 
                       trainingdatadir = trainingdatadir, 
                       UDEmodel        = UDEmodel, 
                       learn_params    = learnparams_perturb, 
                       maxcores        = 1, 
                       datadir         = UDEdatadir + "_scaledtransfer")
# print(" CHECK: Loss should be zero!\n")
exit()

if do_training:
    print("\n Starting UDE training for UDE model = ", UDEmodel, ")...")

    # Start training, use the unperturbed control parameters in pcof_org
    quandary.training(  pcof0               = pcof_org, 
                        trainingdatadir     = trainingdatadir, 
                        UDEmodel            = UDEmodel, 
                        datadir             = UDEdatadir, 
                        T_train             = T_train)
    
    filename        : str           = UDEdatadir + "/params.dat"
    learnparams_opt : np.ndarray    = np.loadtxt(filename)
    print("Training finished. Learned transfer function parameters: ", learnparams_opt, "\n")

    # Simulate forward with optimized parameters to write out the Training data evolutions and the 
    # learned evolution
    print("\n -> Eval loss of optimized UDE model.")
    quandary.UDEsimulate(   trainingdatadir = trainingdatadir, 
                            UDEmodel        = UDEmodel, 
                            datadir         = UDEdatadir + "/FWD_opt", 
                            T_train         = quandary.T, 
                            learn_params    = learnparams_opt)

    # Simulate forward the baseline model using identity transfer function
    identityinit = np.ones(len(learnparams_opt))
    print("\n -> Eval loss of initial guess UDE model.")
    quandary.UDEsimulate(   trainingdatadir = trainingdatadir, 
                            UDEmodel        = UDEmodel, 
                            datadir         = UDEdatadir + "/FWD_identityinit", 
                            T_train         = quandary.T, 
                            learn_params    = identityinit)
