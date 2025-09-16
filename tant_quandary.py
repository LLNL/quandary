from quandary import *
from OC_load_libraries import *

## Helper routine for simplified setup on Tant/QuDIT
def setup_quandary_1d(f01=3.416634567, f12=3.2074712470, Tduration=220.0, targetgate=np.eye(3), max_amp=5e-3, dt_knot=44.0):
    # NOTE: Tduration [ns] must be divisible by 4 [ns]
    
    Ne = [3]  # Number of essential energy levels
    Ng = [1]  # Number of extra guard levels
    
    # Frequency scaling factor relative to GHz and ns (1e-6 sec)
    freq_scale = 1.0 # 
    time_scale = 1/freq_scale
    
    # Scaled transition frequencies [GHz] 
    f01 = f01*freq_scale
    f12 = f12*freq_scale
    
    # quandary wants transition frequency as a per oscillator list
    freq01 = [f01] 
    # Anharmonicities [GHz] per oscillator
    selfkerr = [f01-f12]
    
    # Frequency of frame rotation 
    rotfreq = [f01] # [0.5*(f01+f12)]
    
    # Bounds on the control pulse (in rotational frame, p and q) per oscillator
    max_amp = max_amp*freq_scale
    init_amp = 3.0e-3*freq_scale
    
    rand_seed = 1237 # 1235
    tol_infidelity = 1e-6
    
    dt_knot = 44.0 # 22.0 # 18.0 # 6.0
    tikhonov_coeff = 1e-4 # general regularization term
    leakage_coeff = 1e-1
    dpdm_coeff = 10.0 # 0.1 # 10.0
    energy_coeff = 0.1 # 10.0
    control_enforce_BC = False
    # Prepare Quandary with the above options. This sets default options for all member variables and overwrites those that are passed through the constructor here. Use help(Quandary) to see all options.

    quandary = Quandary(Ne=Ne, Ng=Ng, freq01=freq01, selfkerr=selfkerr, rotfreq=rotfreq, initctrl=init_amp, maxctrl=max_amp, targetgate=targetgate, T=Tduration, control_enforce_BC=control_enforce_BC, rand_seed=rand_seed, cw_prox_thres=0.5*abs(selfkerr[0]), spline_knot_spacing=dt_knot, gamma_leakage=leakage_coeff, gamma_tik0=tikhonov_coeff, gamma_dpdm=dpdm_coeff, gamma_energy=energy_coeff, tol_infidelity=tol_infidelity, verbose=True)
    # Turn off verbosity after the carrier frequencies have been reported
    quandary.verbose = False
    return quandary

####################################
def scaleQuandaryCtrlVec(quandary, pcof, freq_weights):
    # allocate the scaled ctrl vector (pcof_opt)
    pcof_scaled = np.zeros(len(pcof))
    
    start = 0
    nparams = 2*quandary.nsplines
    
    for cf in range(2): # assumes 2 carrier freqs, 1 oscillator
    	scale_fact = freq_weights[cf]
    	print("cf = ", cf, " scale_fact = ", scale_fact)
    	pcof_scaled[start:start+nparams] = [x * scale_fact for x in pcof[start:start+nparams]]
    	start += nparams
 
    return pcof_scaled

####################################
def invScaleQuandaryCtrlVec(quandary, pcof, freq_weights):
    # allocate the scaled ctrl vector (pcof_opt)
    pcof_scaled = np.zeros(len(pcof))
    
    start = 0
    nparams = 2*quandary.nsplines
    
    for cf in range(2): # assumes 2 carrier freqs, 1 oscillator
    	scale_fact = 1/freq_weights[cf]
    	print("cf = ", cf, " scale_fact = ", scale_fact)
    	pcof_scaled[start:start+nparams] = [x * scale_fact for x in pcof[start:start+nparams]]
    	start += nparams
 
    return pcof_scaled

####################################
def evalControlPulse(quandary, pcof_scaled, *, quandary_exec="", samplerate=64, MHz_scalefact = 1e3):
    # Evaluate the control pulses on a fine grid in time using a specific sampling rate
    t1, p1_list, q1_list = quandary.evalControls(pcof0=pcof_scaled, points_per_ns=samplerate, quandary_exec=quandary_exec) # , datadir=eval_datadir
    
    # Remove last time point (just to be consistent with other Tant/QuDIT scripts)
    # can we make it cell-centered instead?
    
    t1 = t1[0:-1]
    p1 = p1_list[0][0:-1]
    q1 = q1_list[0][0:-1]

    # convert optimized pulse amplitudes to [MHz]    
    pp = MHz_scalefact*np.array(p1, dtype=np.float64)
    qq = MHz_scalefact*np.array(q1, dtype=np.float64)
    
    return t1, pp, qq

####################################
def extract_pop_data(new_path, clf=None, correction=[[1,0,0],[0,1,0],[0,0,1]]):
    '''    
    Parameters
    ----------
    new_path : .hdf5
        .hdf5 file from your measurement.
    clf : classifier
        Classifier for state classification
    correction : Matrix, optional
        Input needs to be inv(cmat). Default is identity -- no correction.

    Returns
    -------
    prob_avg, prob_corr_avg

    '''
    dev = find_dev_name(new_path)

    with h5py.File(new_path,'r') as f:
        tsteps = f['OPX_data/ind_arrays/array_1'][:]
        Navg = len(f['OPX_data/ind_arrays/array_0'][:])
    
    data = open_file(new_path, Navg)
    IData = data['I']
    QData = data['Q']
    base0 = data['base0']
    base1 = data['base1']

    if dev == 'Tant':
        if clf == None:
            clf, _ = get_classifier()  
        p_avg, p_avg_c = find_prob_Tant(Navg, IData, QData, clf, correction)
    else:
        p_avg = find_prob_Contralto(IData, QData, base0, base1)

    try:
        return tsteps, p_avg, p_avg_c
    except:
        return tsteps, p_avg

####################################
def my_plot_time_evolution(tsteps, p_avg, p_avg_c, time, population, iinit, *, figfile=""):
    '''    
    Parameters
    ----------
    tsteps: array of data time-levels
    p_avg:  array of shot-averaged populations
    p_avg_c: array of shot-averaged corrected populations

    time: array of simulated time-levels
    population: array of simulated populations
    iinit: initial state (0, 1, 2)
    figfile: (optional), filename for saving the figure

    Returns
    -------
    [diff, diff_c]: L2-norm differences between population data and simulations

    '''
    fig, ax = plt.subplots(1,2, figsize=(10, 4))
    #fig.suptitle(new_path)

    for i in range(3):
        try:
            ax[0].plot(tsteps, p_avg[:,i], label=str(i))
        except Exception as e:
            pass
       
    ax[0].grid()
    ax[0].set_xlabel('Time (ns)')
    ax[0].set_ylabel('Population')
    ax[0].set_title('As measured')
    ax[0].legend()
        
    for i in range(3):
        try:
            ax[1].plot(tsteps, p_avg_c[:,i], label=str(i))
        except:
            pass
    
    ax[1].set_xlabel('Time (ns)')
    ax[1].set_ylabel('Population')
    ax[1].set_title('After correction')
    ax[1].grid()
    ax[1].legend()

    # Add simulated populations
    ninit = len(population[0])
    iosc = 0 # only one oscillator here
    for istate in range(3):
        label = 'Qubit ' + str(iosc)
        label = label + " |"+str(istate)+">"
        ax[0].plot(time, population[iosc][iinit][istate], '--', label=label)
        ax[1].plot(time, population[iosc][iinit][istate], '--', label=label)

    # iinit = initialState
    # iosc = 0 for a single oscillator
    # istate = [0, 1, 2] Populations to plot

    # Evaluate L2-differences
    parr = np.array(population[0][iinit]) # get rid of redundant indices

    cost = []
    cost_c = []
    for i in range(3):
        pop_int = np.interp(tsteps, time, parr[i,:])
        l2_diff = np.linalg.norm(pop_int - p_avg[:,i])
        l2_diff_c = np.linalg.norm(pop_int - p_avg_c[:,i])
        cost_c.append(l2_diff_c)
        cost.append(l2_diff)

    l2diff = np.linalg.norm(cost)
    l2diff_c = np.linalg.norm(cost_c)
    #print("Measured L2-diff: ", l2diff, "Corrected L2-diff:", l2diff_c )
    ax[0].set_title(f'As measured, L2-diff = {l2diff:.3f}')
    ax[1].set_title(f'After correction, L2-diff = {l2diff_c:.3f}')

    # Save the plot as a PNG file with high resolution
    if len(figfile)>0:
        plt.savefig(figfile, dpi=200, bbox_inches='tight')

    plt.show()
    return [l2diff, l2diff_c]

    ##############################################
def learnTransferFunction(quandary, pcof_opt, *, quandary_exec="", maxcores=1, datadir="./Data", data_filenames=["init_0_pop_cor.dat", "init_1_pop_cor.dat", "init_2_pop_cor.dat"], UDEmodel="transferLinear"):
    # Modify quandary options for data generation & training (Use Lindblad's eqn)
    initialcondition =  "diagonal"  # "pure, 0" "diagonal" "basis" # Initial condition at t=0: Groundstate
    T1 = [100.0] # Decoherence times [us]
    T2 = [40.0]
    output_frequency = 1  # write every x-th timestep
    dirprefix = "population_training_" + initialcondition # add a prefix for run directories
    
    # Set the training time domain
    T_train = quandary.T # can be shorter than the full duration  
    
    # Set training data identifier and all filenames (all initial conditions)
    data_identifier = "Tant3Pop" # 3-level population data from "Tant"
    
    # Switch between tikhonov regularization norms (L1 or L2 norm)
    tik0_onenorm = True 			#  Use L1 for sparsification property
    loss_scaling_factor = 1e3 # Factor to scale the loss objective function value
    
    # Output directory for training
    cwd = os.getcwd()
    UDEdatadir = cwd + "/" + dirprefix+ "_UDE"
    
    trainingdata = [data_identifier + ", " + datadir]
    for filename in data_filenames:
    	trainingdata[0] +=  ", " + filename
    
    # Set training optimization parameters
    quandary.gamma_tik0 = 1e-1 # 1e-9
    quandary.gamma_tik0_onenorm = tik0_onenorm
    quandary.loss_scaling_factor = loss_scaling_factor
    quandary.tol_grad_abs = 1e-7
    quandary.tol_grad_rel = 1e-7
    quandary.tol_costfunc = 1e-8
    quandary.tol_infidelity = 1e-7
    quandary.gamma_leakage = 0.0
    quandary.gamma_energy = 0.0
    quandary.gamma_dpdm = 0.0
    quandary.maxiter = 50 # 500
    
    if UDEmodel == "transferLinear":
        learnparams_identity = np.zeros(2) 
        learnparams_identity[0] = 1.0
        learnparams_identity[1] = 1.0 
    elif UDEmodel == "hamiltonian":
        learnparams_identity = np.zeros(3) # Only 3 adjustable diagonal elements in a 4x4 Hamiltonian matrix
    else:
        # Hamiltonian (3) + TF (2)
        learnparams_identity = np.zeros(5) 
        learnparams_identity[3] = 1.0 
        learnparams_identity[4] = 1.0
    
    print("\nStarting training for UDE model = ", UDEmodel, " initial_params: ", learnparams_identity, " result-directory = ", UDEdatadir, "...")
    
    # Start training
    # NOTE: the optimized control vector 'pcof_opt' is saved from initial control optimization 
    # The result of the training end up in quandary.popt 
    quandary.training(pcof0=pcof_opt, trainingdata=trainingdata, UDEmodel=UDEmodel, datadir=UDEdatadir, T_train=T_train, learn_params=learnparams_identity, maxcores=maxcores, quandary_exec=quandary_exec) 
    
    # read training result from file, save in learnparams_opt
    filename = UDEdatadir + "/params.dat"
    learnparams_opt = np.loadtxt(filename)
    print("\n *** Training finished. Learned parameters: ", learnparams_opt, "\n")
    
    # Simulate forward with optimized paramters to write out the Training data evolutions and the learned evolution
    fwd_dir = UDEdatadir+"/FWD_sim"
    print("\n -> Eval loss of optimized UDE model. Results (populations) in dir: ", fwd_dir)
    # NOTE: the initial control vector is read from pcof0_filename, set above 
    time, pt, qt, cost, energy, pop = quandary.UDEsimulate(pcof0=pcof_opt, trainingdata=trainingdata, UDEmodel=UDEmodel, datadir=fwd_dir, T_train=quandary.T, learn_params=learnparams_opt, maxcores=maxcores, quandary_exec=quandary_exec)

    return learnparams_opt, time, pop
