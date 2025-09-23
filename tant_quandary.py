from quandary import *
from OC_load_libraries import *

## Helper routine for simplified setup on Tant/QuDIT
def setup_quandary_1d(f01=3.416634567, f12=3.2074712470, frot=3.416634567, Tduration=220.0, targetgate=np.eye(3), max_amp=5e-3, dt_knot=44.0, nsteps=-1):
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
    rotfreq = [frot] # [0.5*(f01+f12)]
    
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
    verbose = False
    # Prepare Quandary with the above options. This sets default options for all member variables and overwrites those that are passed through the constructor here. Use help(Quandary) to see all options.

    quandary = Quandary(Ne=Ne, Ng=Ng, freq01=freq01, selfkerr=selfkerr, rotfreq=rotfreq, initctrl=init_amp, maxctrl=max_amp, targetgate=targetgate, T=Tduration, control_enforce_BC=control_enforce_BC, rand_seed=rand_seed, cw_prox_thres=0.5*abs(selfkerr[0]), spline_knot_spacing=dt_knot, gamma_leakage=leakage_coeff, gamma_tik0=tikhonov_coeff, gamma_dpdm=dpdm_coeff, gamma_energy=energy_coeff, tol_infidelity=tol_infidelity, nsteps=nsteps, verbose=verbose)
    # Turn off verbosity after the carrier frequencies have been reported
    quandary.verbose = False
    return quandary

####################################
def scaleQuandaryCtrlVec(nsplines, pcof, freq_weights):
    # allocate the scaled ctrl vector (pcof_opt)
    pcof_scaled = np.zeros(len(pcof))
    
    start = 0
    nparams = 2*nsplines
    
    for cf in range(2): # assumes 2 carrier freqs, 1 oscillator
    	scale_fact = freq_weights[cf]
    	print("cf = ", cf, " scale_fact = ", scale_fact)
    	pcof_scaled[start:start+nparams] = [x * scale_fact for x in pcof[start:start+nparams]]
    	start += nparams
 
    return pcof_scaled

####################################
def invScaleQuandaryCtrlVec(nsplines, pcof, freq_weights):
    # allocate the scaled ctrl vector (pcof_opt)
    pcof_scaled = np.zeros(len(pcof))
    
    start = 0
    nparams = 2*nsplines
    
    for cf in range(2): # assumes 2 carrier freqs, 1 oscillator
    	scale_fact = 1/freq_weights[cf]
    	#print("cf = ", cf, " scale_fact = ", scale_fact)
    	pcof_scaled[start:start+nparams] = [x * scale_fact for x in pcof[start:start+nparams]]
    	start += nparams
 
    return pcof_scaled

####################################
def evalControlPulse(quandary, pcof, *, quandary_exec="", samplerate=64, MHz_scalefact = 1e3, freq_weights = [1.0, 1.0], ampFactor = 1.0):
    # scale the control pulse with frequency weights
    pcof_scaled = invScaleQuandaryCtrlVec(quandary.nsplines, pcof, freq_weights)
    # Evaluate the control pulses on a fine grid in time using a specific sampling rate
    t1, p1_list, q1_list = quandary.evalControls(pcof0=pcof_scaled, points_per_ns=samplerate, quandary_exec=quandary_exec) # , datadir=eval_datadir
    
    # Remove last time point (just to be consistent with other Tant/QuDIT scripts)
    # can we make it cell-centered instead?
    
    t1 = t1[0:-1]
    p1 = p1_list[0][0:-1]
    q1 = q1_list[0][0:-1]

    # convert optimized pulse amplitudes to [MHz] and scale by ampFactor  
    pp = ampFactor*MHz_scalefact*np.array(p1, dtype=np.float64)
    qq = ampFactor*MHz_scalefact*np.array(q1, dtype=np.float64)
    
    return t1, pp, qq

####################################
def accumulateFreqWeights(freq_weights0=[1.0, 1.0], learnparams_opt=[1.0, 1.0]):
    # accumulate the frequency weights from the initial guess
    freq_weights1 = freq_weights0.copy()

    # find the freq_weights in the learned parameters
    offs = len(learnparams_opt)-2
    #print(f"offset: {offs}")
    learned_weights = learnparams_opt[offs:]

    for q in range(2):
        freq_weights1[q] = freq_weights1[q] * learned_weights[q]

    print(f"Initial freq_weights0: {freq_weights0}")
    print(f"Learned weights: {learned_weights}")
    print(f"Updated freq_weights1: {freq_weights1}")
    return freq_weights1

####################################
def loadPulseConfig(config, pp, qq, *, name='my_pulse', samplerate=64):
    name_op = name + '_op'
    gate_op = gate_op=[name_op, 'andrew01']

    # NOTE: pulses will be downsampled to ONE point per ns for OPX
    pulse_info = [pp, qq, samplerate] 

    # no spectral filter
    config=upload_pulse(config, 'andrew01', name, pulse_info, scale=None, spectral_filter = False)
    # For the spectral filter approach
    #config=upload_pulse(config, 'andrew01', name_short, pulse_info, scale=None, spectral_filter = True, sf_weight=1.6)
    return gate_op


######################################
def runPopulationExperiments(user, config, gate_op, classifier, *, corrMat=np.eye(3), Nshots=1000):
    p_avg = [] # list for raw population data
    p_avg_c = [] # list for corrected population data

    for initialState in range(3):
        print(f"Starting from initial state: {initialState}")
        path = time_evolution(config, user, gate_op=gate_op, starting=initialState, Nshots=Nshots, tstep=4)
        # clear_output(wait=True)
        tsteps, avg, avg_c = extract_pop_data(path, clf=classifier, correction=corrMat)
        p_avg.append(avg)
        p_avg_c.append(avg_c)

    pop_cap = cap01_populations(p_avg_c) # cap the corrected populations to [0,1]
    return tsteps, p_avg, pop_cap

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
def cap01_populations(pop):
    pop_cap = copy.copy(pop)
    Ninit = len(pop)
    for iInit in range(Ninit):
        shape = pop[iInit].shape
        #print(f"iInit = {iInit}, shape of pop[iInit]: {shape}")
        pop_cap[iInit] = np.copy(pop[iInit])

        Ndata = shape[0]
        Nstate = shape[1]
        #print(f"Ndata = {Ndata}, Nstate = {Nstate}")
        for iData in range(Ndata):
            for iState in range(Nstate):
                if pop[iInit][iData, iState] < 0.0:
                    pop_cap[iInit][iData, iState] = 0.0
                elif pop[iInit][iData, iState] > 1.0:
                    pop_cap[iInit][iData, iState] = 1.0
    return pop_cap


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

########################################
def savePopulationData(pop_cap, *, iExp=0, c_name="pop_cap"):
    # Make sure we are in the home directory
    home_directory = os.path.expanduser("~")
    os.chdir(home_directory)
    cwd = os.getcwd()
    # save population data on file for running quandary in learning mode
    datadir = cwd + "/Data" # Data directory for saving experimental populations
    print(f"Data directory: {datadir}")

    data_filenames = [] # list for storing the file names
    for initialState in range(3):
        fname = "exp_" + str(iExp) + "_init_" + str(initialState) + "_" + c_name + ".dat"
        data_filenames.append(fname)
        file_path = datadir + "/" + fname
        np.savetxt(file_path, pop_cap[initialState], fmt='%.18e', delimiter=' ')
        print(f"Saved population data on file: {file_path}")
    return datadir, data_filenames

##############################################
def learnTransferFunction(quandary, pcof_opt, *, quandary_exec="", maxcores=1, datadir="./Data", data_filenames=["pop0.dat", "pop1.dat", "pop2.dat"], UDEmodel="transferLinear", UDErundir = "run_dir_UDE"):

    output_frequency = 1  # write every x-th timestep

    # NOTE: Initial condition at t=0: Groundstate

    # Set training data identifier and all filenames (all initial conditions)
    data_identifier = "Tant3Pop" # 3-level population data from "Tant"
    trainingdata = [data_identifier + ", " + datadir]
    for filename in data_filenames:
    	trainingdata[0] +=  ", " + filename
    
    # Set the training time domain
    T_train = quandary.T # can be shorter than the full duration  
    
    # Switch between tikhonov regularization norms (L1 or L2 norm)
    tik0_onenorm = False # True 			#  Use L1 for sparsification property
    loss_scaling_factor = 1e1 # Factor to scale the loss objective function value
    

    # Set training optimization parameters
    quandary.gamma_tik0 = 1e-3 # 1e-9
    quandary.gamma_tik0_onenorm = tik0_onenorm
    quandary.loss_scaling_factor = loss_scaling_factor
    quandary.tol_grad_abs = 1e-5 # Do these tolerance need to be this small
    quandary.tol_grad_rel = 1e-5
    quandary.tol_costfunc = 1e-5
    quandary.tol_infidelity = 1e-4
    quandary.gamma_leakage = 0.0
    quandary.gamma_energy = 0.0
    quandary.gamma_dpdm = 0.0
    quandary.maxiter = 15 # 500
    
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
    
    print("\nStarting training for UDE model = ", UDEmodel, " initial_params: ", learnparams_identity, " result-directory = ", UDErundir, "...")
    
    # Start training
    # NOTE: the optimized control vector 'pcof_opt' is saved from initial control optimization 
    # The result of the training end up in quandary.popt 
    quandary.training(pcof0=pcof_opt, trainingdata=trainingdata, UDEmodel=UDEmodel, datadir=UDErundir, T_train=T_train, learn_params=learnparams_identity, maxcores=maxcores, quandary_exec=quandary_exec) 
    
    # read training result from file, save in learnparams_opt
    filename = UDErundir + "/params.dat"
    learnparams_opt = np.loadtxt(filename)
    print("\n *** Training finished. Learned parameters: ", learnparams_opt, "\n")
    
    # Simulate forward with optimized paramters to write out the Training data evolutions and the learned evolution
    fwd_dir = UDErundir+"/FWD_sim"
    print("\n -> Eval loss of optimized UDE model. Results (populations) in dir: ", fwd_dir)
    # NOTE: the initial control vector is read from pcof0_filename, set above 
    time, pt, qt, cost, energy, pop = quandary.UDEsimulate(pcof0=pcof_opt, trainingdata=trainingdata, UDEmodel=UDEmodel, datadir=fwd_dir, T_train=quandary.T, learn_params=learnparams_opt, maxcores=maxcores, quandary_exec=quandary_exec)

    return learnparams_opt, time, pop

#######################
def getDiagHam(filename="HamFile.dat"):
    HamDiag = np.zeros(4)
    with open(filename, "r") as f:
        next(f)  # Skips the 1st line (header)
        next(f)  # Skips the 2nd line (header)
        q = 0
        for line in f:
            # Process the remaining lines (data)
            oneline = line.strip() # strips off leading and trailing white spaces
            # print(f"oneline: {oneline}, type: {type(oneline)}")
            words = oneline.split(' ')
            HamDiag[q] = float(words[q]) # only get the diagonal elements
            q=q+1
    return HamDiag

import h5py
from scipy.optimize import curve_fit
##############################
# this function in the library isn't working properly. Use this one instead.
def analyze_amp_calib(path, if_q = False, find_min=True):
    with h5py.File(path, 'r') as f:
        I = f['OPX_data/output_arrays/I/value'][:]
        Q = f['OPX_data/output_arrays/Q/value'][:]
    
        shots = f['OPX_data/ind_arrays/array_0'][:]
        amps = f['OPX_data/ind_arrays/array_1'][:]
    
        mag = np.sqrt(I**2+Q**2).reshape(len(shots),-1).mean(axis=0)
    
        I_rs = I.reshape(len(shots),-1).mean(axis=0)
        Q_rs = Q.reshape(len(shots),-1).mean(axis=0)
        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        fig.subplots_adjust(hspace=0.3)

        ax[0].plot(amps, I_rs, label="I")
        ax[0].plot(amps, Q_rs, label="Q")
    
        def sine(x, f, a, b,c):
            return a*np.sin(f*x+b)+c            

        try:
            if if_q:
                p0=[8, 0.02, 0.1, 0]
                popt, pcov = curve_fit(sine, amps, Q_rs, p0=p0)
            else:
                p0=[8, -0.02, -0.1, 0]
                popt, pcov = curve_fit(sine, amps, I_rs, p0=p0)
        except:
            popt = p0
            print("Warning: NO convergence in curve_fit")
        ax[0].plot(amps, sine(amps, *p0), label="Init")
        ax[0].plot(amps, sine(amps, *popt), "--", label="Fitted")
        amps_fine = np.linspace(amps.min(), amps.max(), 1000)
        fit_fine = sine(amps_fine, *popt)
        if find_min:
            amp_opt = amps_fine[fit_fine.argmin()]
        else:
            amp_opt = amps_fine[fit_fine.argmax()]
        print(f"Optimal amplitude = {amp_opt:.4f}")
        ax[0].scatter(amp_opt, sine(amp_opt, *popt), c='blue', label="Opt")
        ax[0].set_title(f"Optimal amplitude = {amp_opt:.4f}")
        ax[0].legend()

        ax[1].hist2d(I,Q, bins=160)
        ax[1].set_xlabel("In-phase")
        ax[1].set_ylabel("Quadrature")
        ax[1].set_title("IQ blob")