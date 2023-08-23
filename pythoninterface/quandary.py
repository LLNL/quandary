import os
import numpy as np
from subprocess import run, PIPE, Popen
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

@dataclass
class QuandaryConfig:

    # Quantum system specifications
    Ne        : list[int]   = field(default_factory=lambda: [3])        # Number of essential energy levels per qubit
    Ng        : list[int]   = field(default_factory=lambda: [0])        # Number of extra guard levels per qubit
    freq01    : list[float] = field(default_factory=lambda: [4.10595])  # 01-transition frequencies [GHz] per qubit
    selfkerr  : list[float] = field(default_factory=lambda: [0.2198])   # Anharmonicities [GHz] per qubit
    rotfreq   : list[float] = field(default_factory=list)               # Frequency of rotations for computational frame [GHz] per qubit (default =freq01)
    Jkl       : list[float] = field(default_factory=list)               # Jaynes-Cummings coupling strength [GHz]. Format [J01, J02, ..., J12, J13, ...]
    crosskerr : list[float] = field(default_factory=list)               # ZZ coupling strength [GHz]. Format [g01, g02, ..., g12, g13, ...]
    T1        : list[float] = field(default_factory=list)               # Optional: T1-Decay time per qubit (invokes Lindblad solver)
    T2        : list[float] = field(default_factory=list)               # Optional: T2-Dephasing time per qubit (invokes Lindlbad solver)

    # Time duration and discretization options
    T                   : float       = 100.0             # Final time duration
    Pmin                : int         = 40                # Number of discretization points to resolve the shortest period of the dynamics (determines <nsteps>)
    nsteps              : int         = -1                # Number of time-discretization points (will be computed internally based on Pmin, or can be set here)
    timestepper         : str         = "IMR"             # Time-discretization scheme

    # Hamiltonian model
    standardmodel       : bool              = True                          # Switch to use standard Hamiltonian model for superconduction qubits
    Hsys                : list[float]       = field(default_factory=list)   # Optional: User specified system Hamiltonian model
    Hc_re               : list[list[float]] = field(default_factory=list)   # Optional: User specified control Hamiltonian operators for each qubit (real-parts)
    Hc_im               : list[list[float]] = field(default_factory=list)   # Optional: User specified control Hamiltonian operators for each qubit (real-parts)

    # Control parameterization options
    maxctrl_MHz         : list[float] = field(default_factory=list)   # Amplitude bounds for the control pulses [MHz]
    control_enforce_BC  : bool        = True                          # Enforce that control pulses start and end at zero.
    dtau                : float       = 3.33                          # Spacing [ns] of Bspline basis functions. (The number of Bspline basis functions will be T/dtau + 2)
    nsplines            : int         = -1                            # Number of Bspline basis functions, will be computed from T and dtau. 
    # Control pulse initialization options
    pcof0               : list[float] = field(default_factory=list)   # Optional: Pass an initial control parameter vector
    pcof0_filename      : str         = ""                            # Optional: Load initial control parameter vector from a file
    randomize_init_ctrl : bool        = True                          # Randomize the initial control parameters (will be ignored if pcof0 or pcof0_filename are given)
    initctrl_MHz        : list[float] = field(default_factory=list)   # Amplitude [MHz] of initial control parameters (will be ignored if pcof0 or pcof0_filename are given)
    # Carrier frequency options
    carrier_frequency   : list[list[float]] = field(default_factory=list) # will be set in __post_init
    cw_amp_thres        : float             = 1e-2                        # Threshold to ignore carrier wave frequencies whose growth rate is below this value
    cw_prox_thres       : float             = 1e-3                        # Threshold to distinguish different carrier wave frequencies from each other

    # Optimization options
    costfunction        : str               = "Jtrace"                      # Cost function measure: "Jtrace" or "Jfrobenius"
    targetgate          : list[list[complex]] = field(default_factory=list) # Complex target unitary in the essential level dimensions for gate optimization
    optim_target        : str               = "gate"                        # Optional: Set optimization targets, if not specified through the targetgate
    initialcondition    : str               = "basis"                       # Initial states at time t=0.0: "basis", "diagonal", "pure, 0,0,1,...", "file, /path/to/file" 
    gamma_tik0          : float             = 1e-4 	                        # Parameter for Tikhonov regularization term
    gamma_energy        : float             = 0.01                          # Parameter for integral penality term on the control pulse energy
    gamma_dpdm          : float             = 0.01                          # Parameter for integral penality term on second state derivative
    tol_infidelity      : float             = 1e-3                          # Optimization stopping criterion based on the infidelity
    tol_costfunc        : float             = 1e-3                          # Optimization stopping criterion based on the objective function value
    maxiter             : int               = 100                           # Maximum number of optimization iterations

    # Quandary run options
    print_frequency_iter: int         = 1                   # Output frequency for optimization iterations. (Print every <x> iterations)
    usematfree          : bool        = True                # Switch between matrix-free vs. sparse-matrix solver

    # General options
    verbose             : bool        = False               # Switch to shut down printing to screen
    rand_seed           : int         = 1234                # Seed for random number generator


    # Internal configuration. Should not be changed by user.
    _hamiltonian_filename: str         = ""
    _gatefilename        : str         = ""

    ##
    # This function will be called during initialization of a QuandaryConfig instance.
    # It sets default options that are nor specified by the user and not by the above defaults.
    # It further sets
    #   - <nsteps>            : the number of time steps based on Hamiltonian eigenvalues and Pmin
    #   - <carrier_frequency> : carrier wave frequencies bases on system resonances
    ##
    def __post_init__(self):
        # Set default rotational frequency (=freq01), unless specified by user
        if len(self.rotfreq) == 0:
            self.rotfreq = self.freq01
        # Set default number of splines for control parameterization, unless specified by user
        if self.nsplines < 0:
            self.nsplines = int(np.max([np.ceil(self.T/self.dtau + 2), 5])) # 10
        # Set default bounds on control pulse amplitudes (default = no bounds), unless specified by user
        if len(self.maxctrl_MHz) == 0:
            self.maxctrl_MHz = [1e12 for _ in range(len(self.Ne))]
        # Set default amplitude of initial control parameters [MHz] (default = 9 MHz)
        if len(self.initctrl_MHz) == 0:
            self.initctrl_MHz = [9.0 for _ in range(len(self.Ne))]
        # Set default Hamiltonian operators, unless specified by user
        if len(self.Hsys) == 0:  # Using standard Hamiltonian model
            Ntot = [sum(x) for x in zip(self.Ne, self.Ng)]
            self.Hsys, self.Hc_re, self.Hc_im = hamiltonians(Ntot, self.freq01, self.selfkerr, self.crosskerr, self.Jkl, rotfreq=self.rotfreq, verbose=self.verbose)
            self.standardmodel=True
        else: # Using provided Hamiltonians, write them to hamiltonian.dat
            self.standardmodel=False   

        # Estimate number of time steps
        self.nsteps = estimate_timesteps(T=self.T, Hsys=self.Hsys, Hc_re=self.Hc_re, Hc_im=self.Hc_im, maxctrl_MHz=self.maxctrl_MHz, Pmin=self.Pmin)
        if self.verbose:
            print("Final time: ",self.T,"ns, Number of timesteps: ", self.nsteps,", dt=", self.T/self.nsteps, "ns")
            print("Maximum control amplitudes: ", self.maxctrl_MHz, "MHz")

        # Estimate carrier wave frequencies
        self.carrier_frequency, _ = get_resonances(Ne=self.Ne, Ng=self.Ng, Hsys=self.Hsys, Hc_re=self.Hc_re, Hc_im=self.Hc_im, verbose=self.verbose, cw_amp_thres=self.cw_amp_thres, cw_prox_thres=self.cw_prox_thres, stdmodel=self.standardmodel)

        if self.verbose:
            print("Carrier frequencies: ", self.carrier_frequency,"\n")


    ##
    # Dumps all required configuration options (and target gate, pcof0, Hamiltonian operators) into files for Quandary use.
    # Returns the name of the configuration file needed for executing Quandary
    ##
    def dump(self, runtype="simulation", datadir="./run_dir"):

        # If given, write the target gate to file
        if len(self.targetgate) > 0:
            gate_vectorized = np.concatenate((np.real(self.targetgate).ravel(), np.imag(self.targetgate).ravel()))
            self._gatefilename = "./targetgate.dat"
            with open(datadir+"/"+self._gatefilename, "w") as f:
                for value in gate_vectorized:
                    f.write("{:20.13e}\n".format(value))
            if self.verbose:
                print("Target gate written to ", datadir+"/"+self._gatefilename)

        # If not standard Hamiltonian model, write provided Hamiltonians to a file
        if not self.standardmodel:
            # Write non-standard Hamiltonians to file  
            self._hamiltonian_filename= "./hamiltonian.dat"
            with open(datadir+"/" + self._hamiltonian_filename, "w") as f:
                f.write("# Hsys \n")
                Hsyslist = list(np.array(self.Hsys).flatten(order='F'))
                for value in Hsyslist:
                    f.write("{:20.13e}\n".format(value))

            # Write control Hamiltonians to file, if given (append to file)
            for iosc in range(len(self.Ne)):
                # Real part, if given
                if len(self.Hc_re)>iosc and len(self.Hc_re[iosc])>0:
                    with open(datadir+"/" + self._hamiltonian_filename, "a") as f:
                        Hcrelist = list(np.array(self.Hc_re[iosc]).flatten(order='F'))
                        f.write("# Oscillator {:d} Hc_real \n".format(iosc))
                        for value in Hcrelist:
                            f.write("{:20.13e}\n".format(value))
                # Imaginary part, if given
                if len(self.Hc_im)>iosc and len(self.Hc_im[iosc])>0:
                    with open(datadir+"/" + self._hamiltonian_filename, "a") as f:
                        Hcimlist = list(np.array(self.Hc_im[iosc]).flatten(order='F'))
                        f.write("# Oscillator {:d} Hc_imag \n".format(iosc))
                        for value in Hcimlist:
                            f.write("{:20.13e}\n".format(value))
            if self.verbose:
                print("Hamiltonian operators written to ", self._hamiltonian_filename)
        
        # If pcof0 is given, write it to a file 
        if len(self.pcof0) > 0:
            self.pcof0_filename = datadir+"/pcof0.dat"
            with open(self.pcof0_filename, "w") as f:
                for value in self.pcof0:
                    f.write("{:20.13e}\n".format(value))
            if self.verbose:
                print("Initial control parameters written to ", self.pcof0_filename)

        # Set up string for Quandaries config file
        Nt = [self.Ne[i] + self.Ng[i] for i in range(len(self.Ng))]
        mystring = "nlevels = " + str(list(Nt))[1:-1] + "\n"
        mystring += "nessential= " + str(list(self.Ne))[1:-1] + "\n"
        mystring += "ntime = " + str(self.nsteps) + "\n"
        mystring += "dt = " + str(self.T / self.nsteps) + "\n"
        mystring += "transfreq = " + str(list(self.freq01))[1:-1] + "\n"
        mystring += "rotfreq= " + str(list(self.rotfreq))[1:-1] + "\n"
        mystring += "selfkerr = " + str(list(self.selfkerr))[1:-1] + "\n"
        if len(self.crosskerr)>0:
            mystring += "crosskerr= " + str(list(self.crosskerr))[1:-1] + "\n"
        else:
            mystring += "crosskerr= 0.0\n"
        if len(self.Jkl)>0:
            mystring += "Jkl= " + str(list(self.Jkl))[1:-1] + "\n"
        else:
            mystring += "Jkl= 0.0\n"
        decay = dephase = False
        if len(self.T1) > 0: 
            decay = True
            mystring += "decay_time = " + str(list(self.T1))[1:-1] + "\n"
        if len(self.T2) > 0:
            dephase = True
            mystring += "dephase_time = " + str(list(self.T2))[1:-1] + "\n"
        if decay and dephase:
            mystring += "collapse_type = both\n"
        elif decay:
            mystring += "collapse_type = decay\n"
        elif dephase:
            mystring += "collapse_type = dephase\n"
        else:
            mystring += "collapse_type = none\n"
        mystring += "initialcondition = " + str(self.initialcondition) + "\n"
        for iosc in range(len(self.Ne)):
            mystring += "control_segments" + str(iosc) + " = spline, " + str(self.nsplines) + "\n"
            if len(self.pcof0_filename)>0:
                initstring = "file, "+str(self.pcof0_filename) + "\n"
            else:
                # Scale initial control amplitudes by the number of carrier waves and convert to rad/ns
                initamp = self.initctrl_MHz[iosc] *2.0*np.pi/1000.0 / np.sqrt(2) / len(self.carrier_frequency[iosc])
                initstring = ("random, " if self.randomize_init_ctrl else "constant, ") + str(initamp) + "\n"
            mystring += "control_initialization" + str(iosc) + " = " + initstring 
            if len(self.maxctrl_MHz) == 0: # Disable bounds, if not specified
                initval = 1e+12*np.ones(len(self.Ne))
            else:
                initval = self.maxctrl_MHz[iosc]*2.0*np.pi/1000.0  # Scale to rad/ns
            mystring += "control_bounds" + str(iosc) + " = " + str(initval) + "\n"
            mystring += "carrier_frequency" + str(iosc) + " = "
            omi = self.carrier_frequency[iosc]
            for j in range(len(omi)):
                mystring += str(omi[j]) + ", " 
            mystring += "\n"
        mystring += "control_enforceBC = " + str(self.control_enforce_BC)+ "\n"
        if len(self._gatefilename) > 0:
            mystring += "optim_target = gate, fromfile, " + self._gatefilename + "\n"
        else: 
            mystring += "optim_target = " + str(self.optim_target) + "\n"
        mystring += "optim_objective = " + str(self.costfunction) + "\n"
        mystring += "gate_rot_freq = 0.0\n"
        mystring += "optim_weights= 1.0\n"
        mystring += "optim_atol= 1e-5\n"
        mystring += "optim_rtol= 1e-4\n"
        mystring += "optim_dxtol = 1e-8\n"
        mystring += "optim_ftol= " + str(self.tol_costfunc) + "\n"
        mystring += "optim_inftol= " + str(self.tol_infidelity) + "\n"
        mystring += "optim_maxiter= " + str(self.maxiter) + "\n"
        mystring += "optim_regul= " + str(self.gamma_tik0) + "\n"
        mystring += "optim_penalty= 0.0\n"
        mystring += "optim_penalty_param= 0.0\n"
        mystring += "optim_penalty_dpdm= " + str(self.gamma_dpdm) + "\n"
        mystring += "optim_penalty_energy= " + str(self.gamma_energy) + "\n"
        mystring += "datadir= ./\n"
        for iosc in range(len(self.Ne)):
            mystring += "output" + str(iosc) + "=expectedEnergy, population\n"
        mystring += "output_frequency = 1\n"
        mystring += "optim_monitor_frequency = " + str(self.print_frequency_iter) + "\n"
        mystring += "runtype = " + runtype + "\n"
        if len(self.Ne) < 6:
            mystring += "usematfree = " + str(self.usematfree) + "\n"
        else:
            mystring += "usematfree = false\n"
        mystring += "linearsolver_type = gmres\n"
        mystring += "linearsolver_maxiter = 20\n"
        if not self.standardmodel:
            mystring += "hamiltonian_file= "+str(self._hamiltonian_filename)+"\n"
        mystring += "timestepper = "+str(self.timestepper)+ "\n"

        # Write the file
        outpath = datadir+"/config.cfg"
        with open(outpath, "w") as file:
            file.write(mystring)

        if self.verbose:
            print("Quandary config file written to:", outpath)

        return "./config.cfg"

##
# Main interface function to run Quandary: 
#   1. Writes config files
#   2. Envokes subprocess to run quandary (on multiple cores)
#   3. Gathers results from Quandays output directory
##
def quandary_run(config: QuandaryConfig, *, runtype="optimization", ncores=-1, datadir="./run_dir", quandary_exec="/absolute/path/to/quandary/main"):

    # Create quandary data directory
    os.makedirs(datadir, exist_ok=True)

    # Write the configuration to file
    config_filename = config.dump(runtype=runtype, datadir=datadir)

    # Set default number of cores to dim(H), unless otherwise specified
    if ncores == -1:
        ncores = np.prod(config.Ne) 

    # Execute subprocess to run Quandary
    err = execute(runtype=runtype, ncores=ncores, config_filename=config_filename, datadir=datadir, quandary_exec=quandary_exec, verbose=config.verbose)

    if config.verbose:
        print("Quandary data dir: ", datadir, "\n")

    # Get results and return
    time, pt, qt, ft, expectedEnergy , popt, infidelity, optim_hist= get_results(Ne=config.Ne, datadir=datadir)

    return time, pt, qt, ft, expectedEnergy, popt, infidelity, optim_hist


##
# Helper function to evoke a subprossess that executes Quandary
##
def execute(*, runtype="simulation", ncores=1, config_filename="config.cfg", datadir=".", quandary_exec="/absolute/path/to/quandary/main", verbose=False, cygwin=False):

    
    # result = run(["pwd"], shell=True, capture_output=True, text=True)
    # print("Current location: ", result.stdout, "\n")
    # print("data dir: ", datadir, "\n")

    # Enter the directory where Quandary will be run
    dir_org = os.getcwd() 
    os.chdir(datadir)
    
    # Set up the run command
    runcommand = f"{quandary_exec} {config_filename}"
    if not verbose:
        runcommand += " --quiet"
    if ncores > 1:
        # prefix = f"mpirun -np {ncores}"
        runcommand = f"mpirun -np {ncores} " + runcommand

    # if verbose:
        # result = run(["pwd"], shell=True, capture_output=True, text=True)
    if verbose:
        print("Running Quandary ... ")

    # Execute Quandary
    if not cygwin: # NOT on Windows through Cygwin. Should work on Mac, Linux.
        # Pipe std output to file rather than screen
        # with open(os.path.join(datadir, "out.log"), "w") as stdout_file, \
        #      open(os.path.join(datadir, "err.log"), "w") as stderr_file:
        #         exec = run(runcommand, shell=True, stdout=stdout_file, stderr=stderr_file)
        print("Executing '", runcommand, "' ...")
        exec = run(runcommand, shell=True)
        # Check return code
        err = exec.check_returncode()
    else:
        # Execute Quandary on Windows through Cygwin
        p = Popen(r"C:/cygwin64/bin/bash.exe", stdin=PIPE, stdout=PIPE, stderr=PIPE)  
        p.stdin.write(runcommand.encode('ASCII')) 
        p.stdin.close()
        std_out, std_err = p.communicate()
        # Print stdout and stderr
        if verbose:
            print(std_out.strip().decode('ascii'))
            print(std_err.strip().decode('ascii'))

    # Return to previous directory
    os.chdir(dir_org)
    
    if verbose: 
        print("DONE. \n")

    return 1


##
# Helper function to gather results from Quandaries output directory
##
def get_results(*, Ne=[], datadir="./"):
    dataout_dir = datadir + "/"
    
    # Get control parameters
    try:
        pcof = np.loadtxt(dataout_dir + "/params.dat").astype(float)
    except:
        pcof=[]

    # Get optimization history information
    try:
        optim_hist = np.loadtxt(dataout_dir + "/optim_history.dat")
    except:
        optim_hist = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    if optim_hist.ndim == 2:
        optim_last = optim_hist[-1]
    else:
        optim_last = optim_hist
    infid_last = 1.0 - optim_last[4]
    tikhonov_last = optim_last[6]
    dpdm_penalty_last = optim_last[8] 

    # Get the time-evolution of the expected energy for each qubit, for each initial condition
    expectedEnergy = [[] for _ in range(len(Ne))]
    for iosc in range(len(Ne)):
        for iinit in range(np.prod(Ne)):
            try:
                x = np.loadtxt(dataout_dir + "./expected"+str(iosc)+".iinit"+str(iinit).zfill(4)+".dat")
                expectedEnergy[iosc].append(x[:,1])    # first column is time, second column is expected energy
            except:
                continue

    # Get the control pulses for each qubit
    pt = []
    qt = []
    ft = []
    for iosc in range(len(Ne)):
        # Read the control pulse file
        try:
            x = np.loadtxt(dataout_dir + "./control"+str(iosc)+".dat")
        except:
            x = np.zeros((1,4))
        # Extract the pulses 
        time = x[:,0]   # Time domain
        pt.append([x[n,1]/(2*np.pi)*1e+3 for n in range(len(x[:,0]))])     # Rot frame p(t), MHz
        qt.append([x[n,2]/(2*np.pi)*1e+3 for n in range(len(x[:,0]))])     # Rot frame q(t), MHz
        ft.append([x[n,3]/(2*np.pi)*1e+3 for n in range(len(x[:,0]))])     # Lab frame f(t)

    return time, pt, qt, ft, expectedEnergy, pcof, infid_last, optim_hist

##
# Estimates the number of time steps based on eigenvalues of the system Hamiltonian and maximum control Hamiltonians.
# Note: The estimate does not account for quickly varying signals or a large number of splines. Double check that at least 2-3 points per spline are present to resolve control function. #TODO: Automate this
##
def estimate_timesteps(*, T=1.0, Hsys=[], Hc_re=[], Hc_im=[], maxctrl_MHz=[], Pmin=40):
    assert len(maxctrl_MHz) >= len(Hc_re)

    # Set up Hsys +  maxctrl*Hcontrol
    K1 = np.copy(Hsys) 
    for i in range(len(Hc_re)):
        max_radns = maxctrl_MHz[i]*2.0*np.pi/1e+3
        if len(Hc_re[i])>0:
            K1 += max_radns * Hc_re[i] 
    for i in range(len(Hc_im)):
        max_radns = maxctrl_MHz[i]*2.0*np.pi/1e+3
        if len(Hc_im[i])>0:
            K1 = K1 + 1j * max_radns * Hc_im[i] # can't use += due to type!
    
    # Estimate time step
    eigenvalues = np.linalg.eigvals(K1)
    maxeig = np.max(np.abs(eigenvalues))
    # ctrl_fac = 1.2  # Heuristic, assuming that the total Hamiltonian is dominated by the system part.
    ctrl_fac = 1.0
    samplerate = ctrl_fac * maxeig * Pmin / (2 * np.pi)
#     print(f"{samplerate=}")
    nsteps = int(np.ceil(T * samplerate))

    return nsteps

##
# Computes system resonance frequencies. Those will be used as carrier waves.
# Note: If non-standard Hamiltonian model is used, those resonances might be wrong... Double check and change the carrier waves if needed.
# Returns resonance frequencies in GHz, and corresponding growth rates.
##
def get_resonances(*, Ne, Ng, Hsys, Hc_re=[], Hc_im=[], cw_amp_thres=6e-2, cw_prox_thres=1e-3,verbose=True, stdmodel=True):
    if verbose:
        print("\nComputing carrier frequencies, ignoring growth rate slower than:", cw_amp_thres, "and frequencies closer than:", cw_prox_thres, "[GHz])")

    nqubits = len(Ne)
    n = Hsys.shape[0]

    # Get eigenvalues of system Hamiltonian (GHz)
    Hsys_evals, Utrans = np.linalg.eig(Hsys)
    Hsys_evals = Hsys_evals.real  # Eigenvalues may have a small imaginary part due to numerical precision
    Hsys_evals = Hsys_evals / (2 * np.pi)

    resonances = []
    speed = []

    for q in range(nqubits):
        if stdmodel: # If standard model, get resonances from non-zeros in U'a'U TODO: WHY?
            Hctrl_ad = Hc_re[q] - Hc_im[q]  
            # Hctrl_ad = Hc_re[q]
            # Hctrl_ad = Hc_im[q]
        else: # Get resonances from non-zeros in U'(Hc_re - Hc_im)U TODO: WHY?
            Hctrl_ad =np.zeros(Hsys.shape)
            if len(Hc_re) > q:
                if len(Hc_re[q])> 0:
                    Hctrl_ad += Hc_re[q]
            if len(Hc_im) > q:
                if len(Hc_im[q])> 0:
                    Hctrl_ad -= Hc_im[q]
        Hctrl_ad_trans = Utrans.T @ Hctrl_ad @ Utrans

        resonances_a = []
        speed_a = []
        if verbose:
            print("  Resonances in oscillator #", q)
        # Iterate over non-zero elements in transformed control
        for i in range(n):
            for j in range(n):
                if abs(Hctrl_ad_trans[i,j])< 1e-14:
                    continue
                
                # Get the resonance
                delta_f = Hsys_evals[i] - Hsys_evals[j]
                if abs(delta_f) < 1e-10:
                    delta_f = 0.0

                # Get involved oscillator levels
                ids_j = map_to_oscillators(j, Ne, Ng)
                ids_i = map_to_oscillators(i, Ne, Ng)

                # Ignore resonance to non-essential levels
                if any(ids_i[k] > Ne[k]-1 for k in range(len(Ne))) or \
                   any(ids_j[k] > Ne[k]-1 for k in range(len(Ne))):
                   if verbose:
                       print("    Skipping non-essential resonance from ", ids_j, "to ", ids_i)
                # Ignore resonance with small growth rate
                elif abs(Hctrl_ad_trans[i, j]) < cw_amp_thres:
                    if verbose:
                        print("    Ignoring resonance from ", ids_j, "to ", ids_i, "due to small growth rate=", Hctrl_ad_trans[i,j])
                # Ignore resonance too close to each other
                elif any(abs(delta_f - f) < cw_prox_thres for f in resonances_a):
                    #  any(abs(delta_f + f) < cw_prox_thres for f in resonances_a):
                    if verbose:
                        print("    Ignoring resonance from ", ids_j, "to ", ids_i, "being too close to one that already exists.")
                # Otherwise, add resonance to the list
                else:
                    resonances_a.append(delta_f)
                    speed_a.append(abs(Hctrl_ad_trans[i, j]))
                    if verbose:
                        print("    Resonance from ", ids_j, "to ", ids_i, ", frequency", delta_f, ", growth rate=", abs(Hctrl_ad_trans[i, j]))
        
        resonances.append(resonances_a)
        speed.append(speed_a)

    # print("nqubits ", nqubits)
    Nfreq = np.zeros(nqubits, dtype=int)
    om = [[0.0] for _ in range(nqubits)]
    growth_rate = [[] for _ in range(nqubits)]
    
    # print("om = ", om)
    # print("Nfreq = ", Nfreq )

    for q in range(len(resonances)):
        Nfreq[q] = max(1, len(resonances[q]))  # at least one being 0.0
        om[q] = np.zeros(Nfreq[q])
        if len(resonances[q]) > 0:
            om[q] = np.array(resonances[q])
        growth_rate[q] = np.ones(Nfreq[q])
        if len(speed[q]) > 0:
            growth_rate[q] = np.array(speed[q])

    # return om, growth_rate, Utrans
    return om, growth_rate


# THE BELOW IS COPIED FROM Juqbox setup_std_model. It sorts and culls the carrier waves. SG: Not sure what to do with it. Needed? TODO: Anders?
    # # allocate and sort the vectors (ascending order)
    # om_p = [[]] * Nosc
    # growth_rate_p = [[]] * Nosc
    # use_p = [[]] * Nosc
    # for q in range(Nosc):
    #     om_p[q] = np.zeros(Nfreq[q])
    #     growth_rate_p[q] = np.zeros(Nfreq[q])
    #     use_p[q] = np.zeros(Nfreq[q], dtype=int)  # By default, don't use any freq's
    #     p = np.argsort(om[q])  # sort indices based on om[q]
    #     om_p[q][:] = om[q][p]
    #     growth_rate_p[q][:] = growth_rate[q][p]

    # print("Rotfreq =", rot_freq)
    # print("omp =", om_p)
    # print("growth_rate =", growth_rate_p)

    # print("Sorted CW freq's:")
    # for q in range(Nosc):
    #     print("Ctrl Hamiltonian #", q, ", lab frame carrier frequencies:", rot_freq[q] + om_p[q] / (2 * np.pi), "[GHz]")
    #     print("Ctrl Hamiltonian #", q, ",                   growth rate:", growth_rate_p[q], "[1/ns]")

    # # Try to identify groups of almost equal frequencies
    # for q in range(Nosc):
    #     seg = 0
    #     rge_q = np.max(om_p[q]) - np.min(om_p[q])  # this is the range of frequencies
    #     k0 = 0
    #     for k in range(1, Nfreq[q]):
    #         delta_k = om_p[q][k] - om_p[q][k0]
    #         if delta_k > 0.1 * rge_q:
    #             seg += 1
    #             # find the highest rate within the range [k0,k-1]
    #             rge = range(k0, k)
    #             om_avg = np.sum(om_p[q][rge]) / len(rge)
    #             print("Osc #", q, "segment #", seg, "Freq-range:", (np.max(om_p[q][rge]) - np.min(om_p[q][rge])) / (2 * np.pi), "Freq-avg:", om_avg / (2 * np.pi) + rot_freq[q])
    #             use_p[q][k0] = 1
    #             # average the cw frequency over the segment
    #             om_p[q][k0] = om_avg
    #             k0 = k  # start a new group
    #     # find the highest rate within the last range [k0,Nfreq[q]]
    #     seg += 1
    #     rge = range(k0, Nfreq[q])
    #     om_avg = np.sum(om_p[q][rge]) / len(rge)
    #     print("Osc #", q, "segment #", seg, "Freq-range:", (np.max(om_p[q][rge]) - np.min(om_p[q][rge])) / (2 * np.pi), "Freq-avg:", om_avg / (2 * np.pi) + rot_freq[q])
    #     use_p[q][k0] = 1
    #     om_p[q][k0] = om_avg

    #     # cull out unused frequencies
    #     om[q] = np.zeros(np.sum(use_p[q]))
    #     growth_rate[q] = np.zeros(np.sum(use_p[q]))
    #     j = 0
    #     for k in range(Nfreq[q]):
    #         if use_p[q][k] == 1:
    #             j += 1
    #             om[q][j] = om_p[q][k]
    #             growth_rate[q][j] = growth_rate_p[q][k]
    #     Nfreq[q] = j  # correct the number of CW frequencies for oscillator 'q'

    # print("\nSorted and culled CW freq's:")
    # for q in range(Nosc):
    #     print("Ctrl Hamiltonian #", q, ", lab frame carrier frequencies:", rot_freq[q] + om[q] / (2 * np.pi), "[GHz]")
    #     print("Ctrl Hamiltonian #", q, ",                   growth rate:", growth_rate[q], "[1/ns]")

# Lowering operator of dimension n
def lowering(n):
    return np.diag(np.sqrt(np.arange(1, n)), k=1)
# Number operator of dimension n
def number(n):
    return np.diag(np.arange(n))
# Return the local energy level of each oscillator for a given global index id
def map_to_oscillators(id, Ne, Ng):

    nlevels = [Ne[i]+Ng[i] for i in range(len(Ne))]
    localIDs = []

    index = int(id)
    for iosc in range(len(Ne)):
        postdim = np.prod(nlevels[iosc+1:])
        localIDs.append(int(index / postdim))
        index = index % postdim 

    return localIDs 

##
# Create standard Hamiltonian operators to model superconducting qubits. 
# Returns 
#   - Hsys  : System Hamiltonian (time-independent), units rad/ns
#   - Hc_re : Real parts of control Hamiltonian operators for each qubit (Hc = [ [Hc_qubit1], [Hc_qubit2],... ]). Unit-less.
#   - Hc_im : Imag parts of control Hamiltonian operators for each qubit (Hc = [ [Hc_qubit1], [Hc_qubit2],... ]). Unit-less.
##
def hamiltonians(N, freq01, selfkerr, crosskerr=[], Jkl = [], *, rotfreq=[], verbose=True):
    if len(rotfreq) == 0:
        rotfreq=np.zeros(len(N))

    nqubits = len(N)
    assert len(selfkerr) == nqubits
    assert len(freq01) == nqubits
    assert len(rotfreq) == nqubits

    n = np.prod(N)     # System size 

    # Set up lowering operators in full dimension
    Amat = []
    for i in range(len(N)):
        ai = lowering(N[i])
        for j in range(i):
            ai = np.kron(np.identity(N[j]), ai) 
        for j in range(i+1,len(N)):
            ai = np.kron(ai, np.identity(N[j])) 
        Amat.append(ai)

    # Set up system Hamiltonian: Duffing oscillators
    Hsys = np.zeros((n, n))
    for q in range(nqubits):
        domega_radns =  2.0*np.pi * (freq01[q] - rotfreq[q])
        selfkerr_radns = 2.0*np.pi * selfkerr[q]
        Hsys +=  domega_radns * Amat[q].T @ Amat[q]
        Hsys -= selfkerr_radns/2.0 * Amat[q].T @ Amat[q].T @ Amat[q] @ Amat[q]

    # Add cross cerr coupling, if given
    if len(crosskerr)>0:
        idkl = 0 
        for q in range(nqubits):
            for p in range(q + 1, nqubits):
                if abs(crosskerr[idkl]) > 1e-14:
                    crosskerr_radns = 2.0*np.pi * crosskerr[idkl]
                    Hsys -= crosskerr_radns * Amat[q].T @ Amat[q] @ Amat[p].T @ Amat[p]
                    idkl += 1
    
    # Add Jkl coupling term. 
    # Note that if the rotating frame frequencies are different amongst oscillators, then this is contributes to a *time-dependent* system Hamiltonian. Here, we treat this as time-independent, because this Hamiltonian here is *ONLY* used to compute the time-step size and resonances, and it is NOT passed to the quandary code. Quandary sets up the standard model with a time-dependent system Hamiltonian if the frequencies of rotation differ amongst oscillators.  
    if len(Jkl)>0:
        idkl = 0 
        for q in range(nqubits):
            for p in range(q + 1, nqubits):
                if abs(Jkl[idkl]) > 1e-14:
                    Jkl_radns  = 2.0*np.pi*Jkl[idkl]
                    Hsys += Jkl_radns * (Amat[q].T @ Amat[p] + Amat[q] @ Amat[p].T)
                    idkl += 1

    # Set up control Hamiltonians
    Hc_re = [Amat[q] + Amat[q].T for q in range(nqubits)]
    Hc_im = [Amat[q] - Amat[q].T for q in range(nqubits)]

    if verbose:
        print(f"*** {nqubits} coupled quantum systems setup ***")
        print("System Hamiltonian frequencies [GHz]: f01 =", freq01, "rot. freq =", rotfreq)
        print("Selfkerr=", selfkerr)
        print("Coupling: X-Kerr=", crosskerr, ", J-C=", Jkl)

    return Hsys, Hc_re, Hc_im

##
# Plot the control pulse for all qubits
##
def plot_pulse(Ne, time, pt, qt):
    fig = plt.figure()
    nrows = len(Ne)
    ncols = 1
    for iosc in range(len(Ne)):
        plt.subplot(nrows, ncols, iosc+1)
        plt.plot(time, pt[iosc], "r", label="p(t)")
        plt.plot(time, qt[iosc], "b", label="q(t)")
        plt.xlabel('time (ns)')
        plt.ylabel('Drive strength [MHz]')
        plt.title('Qubit '+str(iosc))
        plt.legend(loc='lower right')
        plt.xlim([0.0, time[-1]])
    # plt.grid()
    plt.subplots_adjust(hspace=0.6)
    plt.draw()
    print("\nPlotting control pulses.")
    print("-> Press <enter> to proceed.")
    plt.waitforbuttonpress(1); 
    input(); 
    plt.close(fig)

##
# Plot evolution of expected energy levels
##
def plot_expectedEnergy(Ne, time, expectedEnergy, densitymatrix_form=False):
    nplots = np.prod(Ne)
    ncols = 2 if nplots >= 4 else 1     # 2 rows if more than 3 plots
    nrows = int(np.ceil(np.prod(Ne)/ncols))
    figsizex = 6.4*nrows*0.75 
    figsizey = 4.8*nrows*0.75 
    fig = plt.figure(figsize=(figsizex,figsizey))
    for iplot in range(nplots):
        iinit = iplot if not densitymatrix_form else iplot*np.prod(Ne) + iplot
        plt.subplot(nrows, ncols, iplot+1)
        plt.figsize=(15, 15)
        for iosc in range(len(Ne)):
            label = 'Qubit '+str(iosc) if len(Ne)>0 else ''
            plt.plot(time, expectedEnergy[iosc][iinit], label=label)
        plt.xlabel('time (ns)')
        plt.ylabel('expected energy')
        plt.ylim([0.0-1e-2, Ne[0]-1.0 + 1e-2])
        plt.xlim([0.0, time[-1]])
        binary_ID = iinit if len(Ne) == 1 else bin(iinit).replace("0b", "").zfill(len(Ne))
        plt.title("init |"+str(binary_ID)+">")
        plt.legend(loc='lower right')
    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(wspace=0.5)
    plt.draw()
    print("\nPlotting expected energy of qubit ", iosc)
    print("-> Press <enter> to proceed.")
    plt.waitforbuttonpress(1); 
    input(); 
    plt.close(fig)
