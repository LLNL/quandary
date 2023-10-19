import os
import numpy as np
from subprocess import run, PIPE, Popen
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict
## For some Matplotlib installations to work, you might need the below...
# import PyQt6.QtCore

## 
# This class collects configuration options to run quandary. The default values are set to optimize for the swap02 gate. Fields in this configuration file are set through the constructor
#   > myconfig = QuandaryConfig(fieldname=mynewsetting, anotherfieldname=anothernewsetting, ...)
# which sets the default values for those variables that are not set through those arguments. 
# 
# In addition to setting defaults, the constructor also computes the number of time-steps required to resolve the time-domain, as well as the resonant carrier wave frequencies. If you attempt to change options of the configuration *after* construction, e.g. through
#   > myconfig.fieldname = mynewsetting
# it is advised to call
#  > myconfig.update()
# which recomputes the number of time-steps and carrier waves given the new settings. 
@dataclass
class QuandaryConfig:

    # Quantum system specifications
    Ne        : List[int]   = field(default_factory=lambda: [3])        # Number of essential energy levels per qubit
    Ng        : List[int]   = field(default_factory=lambda: [1])        # Number of extra guard levels per qubit
    freq01    : List[float] = field(default_factory=lambda: [4.10595])  # 01-transition frequencies [GHz] per qubit
    selfkerr  : List[float] = field(default_factory=lambda: [0.2198])   # Anharmonicities [GHz] per qubit
    rotfreq   : List[float] = field(default_factory=list)               # Frequency of rotations for computational frame [GHz] per qubit (default =freq01)
    Jkl       : List[float] = field(default_factory=list)               # Dipole-dipole coupling strength [GHz]. Format [J01, J02, ..., J12, J13, ...]
    crosskerr : List[float] = field(default_factory=list)               # ZZ coupling strength [GHz]. Format [g01, g02, ..., g12, g13, ...]
    T1        : List[float] = field(default_factory=list)               # Optional: T1-Decay time per qubit (invokes Lindblad solver)
    T2        : List[float] = field(default_factory=list)               # Optional: T2-Dephasing time per qubit (invokes Lindlbad solver)

    # Time duration and discretization options
    T                   : float       = 100.0             # Final time duration
    Pmin                : int         = 150               # Number of discretization points to resolve the shortest period of the dynamics (determines <nsteps>)
    nsteps              : int         = -1                # Number of time-discretization points (will be computed internally based on Pmin, or can be set here)
    timestepper         : str         = "IMR"             # Time-discretization scheme

    # Hamiltonian model
    standardmodel       : bool              = True                          # Switch to use standard Hamiltonian model for superconduction qubits
    Hsys                : List[float]       = field(default_factory=list)   # Optional: User specified system Hamiltonian model
    Hc_re               : List[List[float]] = field(default_factory=list)   # Optional: User specified control Hamiltonian operators for each qubit (real-parts)
    Hc_im               : List[List[float]] = field(default_factory=list)   # Optional: User specified control Hamiltonian operators for each qubit (real-parts)

    # Control parameterization options
    maxctrl_MHz         : List[float] = field(default_factory=list)   # Amplitude bounds for the control pulses [MHz]
    control_enforce_BC  : bool        = False                         # Enforce that control pulses start and end at zero.
    dtau                : float       = 10.0                          # Spacing [ns] of Bspline basis functions. (The number of Bspline basis functions will be T/dtau + 2)
    nsplines            : int         = -1                            # Number of Bspline basis functions, will be computed from T and dtau. 
    # Control pulse initialization options
    pcof0               : List[float] = field(default_factory=list)   # Optional: Pass an initial control parameter vector
    pcof0_filename      : str         = ""                            # Optional: Load initial control parameter vector from a file
    randomize_init_ctrl : bool        = True                          # Randomize the initial control parameters (will be ignored if pcof0 or pcof0_filename are given)
    initctrl_MHz        : List[float] = field(default_factory=list)   # Amplitude [MHz] of initial control parameters (will be ignored if pcof0 or pcof0_filename are given)
    # Carrier frequency options
    carrier_frequency   : List[List[float]] = field(default_factory=list) # will be set in __post_init
    cw_amp_thres        : float             = 1e-7                        # Threshold to ignore carrier wave frequencies whose growth rate is below this value
    cw_prox_thres       : float             = 1e-2                        # Threshold to distinguish different carrier wave frequencies from each other

    # Optimization options
    costfunction        : str               = "Jtrace"                      # Cost function measure: "Jtrace" or "Jfrobenius"
    targetgate          : List[List[complex]] = field(default_factory=list) # Complex target unitary in the essential level dimensions for gate optimization
    targetstate         : List[complex]     = field(default_factory=list) # Complex target state vector for state-to-state optimization
    optim_target        : str               = ""                          # Optional: Set optimization targets, if not specified through the targetgate or targetstate
    initialcondition    : str               = "basis"                         # Initial states at time t=0.0: "basis" (default), "diagonal", "pure, 0,0,1,...", "file, /path/to/file"
    gamma_tik0          : float             = 1e-4 	                        # Parameter for Tikhonov regularization ||alpha||^2
    gamma_tik0_interpolate : bool           = False                         # Switch to use ||alpha-alpha_0||^2 instead, where alpha_0 is the initial guess.
    gamma_leakage       : float             = 0.1 	                        # Parameter for leakage prevention
    gamma_energy        : float             = 0.1                           # Parameter for integral penality term on the control pulse energy
    gamma_dpdm          : float             = 0.01                          # Parameter for integral penality term on second state derivative
    tol_infidelity      : float             = 1e-5                          # Optimization stopping criterion based on the infidelity
    tol_costfunc        : float             = 1e-4                          # Optimization stopping criterion based on the objective function value
    maxiter             : int               = 100                           # Maximum number of optimization iterations

    # Quandary run options
    print_frequency_iter: int         = 1                   # Output frequency for optimization iterations. (Print every <x> iterations)
    usematfree          : bool        = True                # Switch between matrix-free vs. sparse-matrix solver

    # General options
    verbose             : bool        = False               # Switch to shut down printing to screen
    rand_seed           : int         = None                # Default: use system time(0) random seed


    # Internal configuration. Should not be changed by user.
    _hamiltonian_filename : str         = ""
    _gatefilename         : str         = ""

    # Storage for some optimization results, in case they are needed afterwards.
    popt        : List[float]   = field(default_factory=list)   # Optimized control paramters, could be useful to run quandary again after optimization
    time        : List[float]   = field(default_factory=list)   # Vector of discretized time points, could be useful for plotting the control pulses etc.
    optim_hist  : Dict          = field(default_factory=dict)   # Optimization history: all fields as in Quandary's output file <data>/optim_history.dat
    uT          : List[float]   = field(default_factory=list)   # Evolved states at final time T. This is the (unitary) solution operator, if the initial conditions span the full basis. 


    ##
    # This function will be called during initialization of a QuandaryConfig instance.
    # It sets default options that are nor specified by the user and not by the above defaults.
    # It further sets
    #   - <nsteps>            : the number of time steps based on Hamiltonian eigenvalues and Pmin
    #   - <carrier_frequency> : carrier wave frequencies bases on system resonances
    ##
    def __post_init__(self):

        # Set default two-level system, if Ne is not specified by user
        if len(self.freq01) != len(self.Ne):
            self.Ne = [2 for _ in range(len(self.freq01))]

        # Set default NO guard levels, if Ng is not specified by user
        if len(self.Ng) != len(self.Ne):
            self.Ng = [0 for _ in range(len(self.Ne))]
        
        # Set zero selfkerr, if not specified by user
        if len(self.selfkerr) != len(self.Ne):
            self.selfkerr= np.zeros(len(self.Ne))

        # Set default rotational frequency (default=freq01), unless specified by user
        if len(self.rotfreq) == 0:
            self.rotfreq = self.freq01

        # Set default number of splines for control parameterization, unless specified by user
        if self.nsplines < 0:
            minspline = 5 if self.control_enforce_BC else 3
            self.nsplines = int(np.max([np.ceil(self.T/self.dtau + 2), minspline]))
            
        # Set default amplitude of initial control parameters [MHz] (default = 1 MHz)
        if isinstance(self.initctrl_MHz, float) or isinstance(self.initctrl_MHz, int):
            max_alloscillators = self.initctrl_MHz
            self.initctrl_MHz = [max_alloscillators for _ in range(len(self.Ne))]
        if len(self.initctrl_MHz) == 0:
            self.initctrl_MHz = [1.0 for _ in range(len(self.Ne))]

        # Set default Hamiltonian operators, unless specified by user
        if len(self.Hsys) > 0 and not self.standardmodel: # User-provided Hamiltonian operators 
            self.standardmodel=False   
        else: # Using standard Hamiltonian model
            Ntot = [sum(x) for x in zip(self.Ne, self.Ng)]
            self.Hsys, self.Hc_re, self.Hc_im = hamiltonians(N=Ntot, freq01=self.freq01, selfkerr=self.selfkerr, crosskerr=self.crosskerr, Jkl=self.Jkl, rotfreq=self.rotfreq, verbose=self.verbose)
            self.standardmodel=True

        # Set the optimization target 
        if len(self.targetstate) > 0:
            self.optim_target = "file"
        if len(self.targetgate) > 0:
            self.optim_target = "gate, fromfile"
        
        # Change default initial condition to pure state if target is state-to-state optimization
        if len(self.targetstate) > 0:
            self.initialcondition = "pure, " 
            for i in range(len(self.Ne)):
                self.initialcondition += "0,"

        # Convert maxctrl_MHz to a list for each oscillator, if not so already
        if isinstance(self.maxctrl_MHz, float) or isinstance(self.maxctrl_MHz, int):
            max_alloscillators = self.maxctrl_MHz
            self.maxctrl_MHz = [max_alloscillators for _ in range(len(self.Ne))]

        # Estimate number of time steps
        self.nsteps = estimate_timesteps(T=self.T, Hsys=self.Hsys, Hc_re=self.Hc_re, Hc_im=self.Hc_im, maxctrl_MHz=self.maxctrl_MHz, Pmin=self.Pmin)
        if self.verbose:
            print("Final time: ",self.T,"ns, Number of timesteps: ", self.nsteps,", dt=", self.T/self.nsteps, "ns")
            print("Maximum control amplitudes: ", self.maxctrl_MHz, "MHz")

        # Estimate carrier wave frequencies
        if len(self.carrier_frequency) == 0: 
            self.carrier_frequency, _ = get_resonances(Ne=self.Ne, Ng=self.Ng, Hsys=self.Hsys, Hc_re=self.Hc_re, Hc_im=self.Hc_im, rotfreq=self.rotfreq, verbose=self.verbose, cw_amp_thres=self.cw_amp_thres, cw_prox_thres=self.cw_prox_thres, stdmodel=self.standardmodel)

        if self.verbose: 
            print("\n")
            for q in range(len(self.Ne)):
                print("System #", q, "Carrier frequencies (lab frame): ", self.rotfreq[q]+self.carrier_frequency[q])
                print("                               (rot frame): ", self.carrier_frequency[q])
            print("\n")

    ##
    # Call this function if you have changed a config option outside of the constructor, e.g. with "myconfig.variablename = new_variable". This will ensure that the number of time steps and carrier waves are re-computed, given the new setting. 
    ## 
    def update(self):
        self.__post_init__()


    ##
    # Dumps all required configuration options (and target gate, pcof0, Hamiltonian operators) into files for Quandary use.
    # Returns the name of the configuration file needed for executing Quandary
    ##
    def dump(self, *, runtype="simulation", datadir="./run_dir"):

        # If given, write the target gate to file
        if len(self.targetgate) > 0:
            gate_vectorized = np.concatenate((np.real(self.targetgate).ravel(order='F'), np.imag(self.targetgate).ravel(order='F')))
            self._gatefilename = "./targetgate.dat"
            with open(datadir+"/"+self._gatefilename, "w") as f:
                for value in gate_vectorized:
                    f.write("{:20.13e}\n".format(value))
            if self.verbose:
                print("Target gate written to ", datadir+"/"+self._gatefilename)

        # If given, write the target state to file
        if len(self.targetstate) > 0:
            gate_vectorized = np.concatenate((np.real(self.targetstate).ravel(order='F'), np.imag(self.targetstate).ravel(order='F')))
            self._gatefilename = "./targetstate.dat"
            with open(datadir+"/"+self._gatefilename, "w") as f:
                for value in gate_vectorized:
                    f.write("{:20.13e}\n".format(value))
            if self.verbose:
                print("Target state written to ", datadir+"/"+self._gatefilename)



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
                print("Hamiltonian operators written to ", datadir+"/"+self._hamiltonian_filename)
        
        # If pcof0 is given, write it to a file 
        if len(self.pcof0) > 0:
            self.pcof0_filename = "./pcof0.dat"
            with open(datadir+"/"+self.pcof0_filename, "w") as f:
                for value in self.pcof0:
                    f.write("{:20.13e}\n".format(value))
            if self.verbose:
                print("Initial control parameters written to ", datadir+"/"+self.pcof0_filename)

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
                boundval = 1e+12
            else:
                boundval = self.maxctrl_MHz[iosc]*2.0*np.pi/1000.0  # Scale to rad/ns
            mystring += "control_bounds" + str(iosc) + " = " + str(boundval) + "\n"
            mystring += "carrier_frequency" + str(iosc) + " = "
            omi = self.carrier_frequency[iosc]
            for j in range(len(omi)):
                mystring += str(omi[j]) + ", " 
            mystring += "\n"
        mystring += "control_enforceBC = " + str(self.control_enforce_BC)+ "\n"
        if len(self._gatefilename) > 0:
            mystring += "optim_target = " + self.optim_target + ", " + self._gatefilename + "\n"
        else: 
            mystring += "optim_target = " + str(self.optim_target) + "\n"
        mystring += "optim_objective = " + str(self.costfunction) + "\n"
        mystring += "gate_rot_freq = 0.0\n"
        mystring += "optim_weights= 1.0\n"
        mystring += "optim_atol= 1e-4\n"
        mystring += "optim_rtol= 1e-4\n"
        mystring += "optim_dxtol = 1e-8\n"
        mystring += "optim_ftol= " + str(self.tol_costfunc) + "\n"
        mystring += "optim_inftol= " + str(self.tol_infidelity) + "\n"
        mystring += "optim_maxiter= " + str(self.maxiter) + "\n"
        mystring += "optim_regul= " + str(self.gamma_tik0) + "\n"
        mystring += "optim_regul_interpolate= " + str(self.gamma_tik0_interpolate) + "\n"
        mystring += "optim_penalty= " + str(self.gamma_leakage) + "\n"
        mystring += "optim_penalty_param= 0.0\n"
        mystring += "optim_penalty_dpdm= " + str(self.gamma_dpdm) + "\n"
        mystring += "optim_penalty_energy= " + str(self.gamma_energy) + "\n"
        mystring += "datadir= ./\n"
        for iosc in range(len(self.Ne)):
            mystring += "output" + str(iosc) + "=expectedEnergy, population, fullstate\n"
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
        if self.rand_seed is not None and self.rand_seed >= 0:
            mystring += "rand_seed = "+str(int(self.rand_seed))+ "\n"

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
def quandary_run(config: QuandaryConfig, *, runtype="optimization", ncores=-1, datadir="./run_dir", quandary_exec="/absolute/path/to/quandary/main", cygwin=False):

    # Create quandary data directory
    os.makedirs(datadir, exist_ok=True)

    # Write the configuration to file
    config_filename = config.dump(runtype=runtype, datadir=datadir)

    # Set default number of cores to dim(H), unless otherwise specified
    if ncores == -1:
        ncores = np.prod(config.Ne) 

    # Execute subprocess to run Quandary
    err = execute(runtype=runtype, ncores=ncores, config_filename=config_filename, datadir=datadir, quandary_exec=quandary_exec, verbose=config.verbose, cygwin=cygwin)

    if config.verbose:
        print("Quandary data dir: ", datadir, "\n")

    # Get results from quandary output files
    lindblad_solver = True if (len(config.T1) > 0 or len(config.T2) > 0) else False
    time, pt, qt, uT, expectedEnergy, population, popt, infidelity, optim_hist = get_results(Ne=config.Ne, Ng=config.Ng, datadir=datadir, lindblad_solver=lindblad_solver)

    # Store some results in the config file
    config.optim_hist = optim_hist
    config.popt = popt[:]
    config.time = time[:]
    config.uT   = uT.copy()

    return config.time, pt, qt, infidelity, expectedEnergy, population


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
        if not runtype == "evalcontrols":
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
def get_results(*, Ne=[], Ng=[], datadir="./", lindblad_solver=False):
    dataout_dir = datadir + "/"
    
    # Get control parameters
    filename = dataout_dir + "/params.dat"
    try:
        pcof = np.loadtxt(filename).astype(float)
    except:
        print("Can't read control coefficients from $filename !\n")
        pcof=[]

    # Get optimization history information
    filename = dataout_dir + "/optim_history.dat"
    try:
        optim_hist_tmp = np.loadtxt(filename)
    except:
        print("Can't read optimization history from $filename")
        optim_hist_tmp = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    if optim_hist_tmp.ndim == 2:
        optim_last = optim_hist_tmp[-1]
    else:
        optim_last = optim_hist_tmp
        optim_hist_tmp = np.array([optim_hist_tmp])
    infid_last = 1.0 - optim_last[4]
    tikhonov_last = optim_last[6]
    dpdm_penalty_last = optim_last[8] 

    # Put optimization history into a dictionary: 
    optim_hist = {
        "Iters"                  : optim_hist_tmp[:,0],
        "Gradient"               : optim_hist_tmp[:,2],
        "Fidelity"               : optim_hist_tmp[:,4],
        "Cost"                   : optim_hist_tmp[:,5],
        "Tikhonov"               : optim_hist_tmp[:,6],
        "Penalty-Leakage"        : optim_hist_tmp[:,7],
        "Penalty-StateVariation" : optim_hist_tmp[:,8],
        "Penalty-TotalEnergy"    : optim_hist_tmp[:,9],
    }

    # Get the time-evolution of the expected energy for each qubit, for each initial condition
    expectedEnergy = [[] for _ in range(len(Ne))]
    for iosc in range(len(Ne)):
        ninit = np.prod(Ne) if not lindblad_solver else np.prod(Ne)**2
        for iinit in range(ninit):
            filename = dataout_dir + "./expected"+str(iosc)+".iinit"+str(iinit).zfill(4)+".dat"
            try:
                x = np.loadtxt(filename)
                expectedEnergy[iosc].append(x[:,1])    # 0th column is time, second column is expected energy
            except:
                print("Can't read expected energy from $filename !")

    # Get population for each qubit, for each initial condition
    population = [[] for _ in range(len(Ne))]
    for iosc in range(len(Ne)):
        ninit = np.prod(Ne) if not lindblad_solver else np.prod(Ne)**2
        for iinit in range(ninit):
            filename = dataout_dir + "./population"+str(iosc)+".iinit"+str(iinit).zfill(4)+".dat"
            try:
                x = np.loadtxt(filename)
                population[iosc].append(x[:,1:].transpose())    # first column is time
            except:
                print("Can't read population from $filename !")

    # Get last time-step unitary
    ninit = np.prod(Ne) if not lindblad_solver else np.prod(Ne)**2
    Ntot = [i+j for i,j in zip(Ne,Ng)]
    ndim = np.prod(Ntot) if not lindblad_solver else np.prod(Ntot)**2
    uT = np.zeros((ndim, ninit), dtype=complex)
    for iinit in range(ninit):
        file_index = str(iinit).zfill(4)
        xre = np.loadtxt(f"{dataout_dir}/rho_Re.iinit{file_index}.dat", skiprows=1, usecols=range(1, ndim+1))[-1]
        xim = np.loadtxt(f"{dataout_dir}/rho_Im.iinit{file_index}.dat", skiprows=1, usecols=range(1, ndim+1))[-1]
        uT[:, iinit] = xre + 1j * xim

    # Get the control pulses for each qubit
    pt = []
    qt = []
    ft = []
    for iosc in range(len(Ne)):
        # Read the control pulse file
        filename = dataout_dir + "./control"+str(iosc)+".dat"
        try:
            x = np.loadtxt(filename)
        except:
            print("Can't read control pulses from $filename !")
            x = np.zeros((1,4))
        # Extract the pulses 
        time = x[:,0]   # Time domain
        pt.append([x[n,1]/(2*np.pi)*1e+3 for n in range(len(x[:,0]))])     # Rot frame p(t), MHz
        qt.append([x[n,2]/(2*np.pi)*1e+3 for n in range(len(x[:,0]))])     # Rot frame q(t), MHz
        ft.append([x[n,3]/(2*np.pi)*1e+3 for n in range(len(x[:,0]))])     # Lab frame f(t)

    return time, pt, qt, uT, expectedEnergy, population, pcof, infid_last, optim_hist


##
# Helper function to re-evaluate the controls on a different time grid for a specific sample rate
#
def evalControls(config, *, pcof=[], points_per_ns=1, quandary_exec="/absolute/path/to/quandary/main", datadir="./data_controls", cygwin=False):

    # Copy original setting and overwrite number of time steps for simulation
    nsteps_org = config.nsteps
    config.nsteps = int(np.floor(config.T * points_per_ns))
    
    # Pass pcof to the configuration, if given
    if len(pcof) > 0:
        config.pcof0 = pcof[:]

    # Execute quandary in 'evalcontrols' mode
    runtype = 'evalcontrols'
    os.makedirs(datadir, exist_ok=True)
    configfile_eval= config.dump(runtype=runtype, datadir=datadir)
    err = execute(runtype=runtype, ncores=1, config_filename=configfile_eval, datadir=datadir, quandary_exec=quandary_exec, verbose=False, cygwin=cygwin)
    time, pt, qt, _, _, _, pcof, _, _ = get_results(Ne=config.Ne, Ng=config.Ng, datadir=datadir)

    # Save pcof to config.popt
    config.popt = pcof[:]
    
    # Restore original setting
    config.nsteps = nsteps_org

    return time, pt, qt

##
# Estimates the number of time steps based on eigenvalues of the system Hamiltonian and maximum control Hamiltonians.
# Note: The estimate does not account for quickly varying signals or a large number of splines. Double check that at least 2-3 points per spline are present to resolve control function. #TODO: Automate this
##
def estimate_timesteps(*, T=1.0, Hsys=[], Hc_re=[], Hc_im=[], maxctrl_MHz=[], Pmin=40):

    # Get estimated control pulse amplitude
    est_ctrl_MHz = maxctrl_MHz[:]
    if len(maxctrl_MHz) == 0:
        est_ctrl_MHz = [10.0 for _ in range(max(len(Hc_re), len(Hc_im)))] 

    # Set up Hsys +  maxctrl*Hcontrol
    K1 = np.copy(Hsys) 

    for i in range(len(Hc_re)):
        est_radns = est_ctrl_MHz[i]*2.0*np.pi/1e+3
        if len(Hc_re[i])>0:
            K1 += est_radns * Hc_re[i] 
    for i in range(len(Hc_im)):
        est_radns = est_ctrl_MHz[i]*2.0*np.pi/1e+3
        if len(Hc_im[i])>0:
            K1 = K1 + 1j * est_radns * Hc_im[i] # can't use += due to type!
    
    # Estimate time step
    eigenvalues = np.linalg.eigvals(K1)
    maxeig = np.max(np.abs(eigenvalues))
    # ctrl_fac = 1.2  # Heuristic, assuming that the total Hamiltonian is dominated by the system part.
    ctrl_fac = 1.0
    samplerate = ctrl_fac * maxeig * Pmin / (2 * np.pi)
#     print(f"{samplerate=}")
    nsteps = int(np.ceil(T * samplerate))

    return nsteps


# computes eigen decomposition and re-orders it to make the eigenvector matrix as close to the identity as posiible
def eigen_and_reorder(H0, verbose=False):

    # Get eigenvalues and vectors and sort them in ascending order
    Ntot = H0.shape[0]
    evals, evects = np.linalg.eig(H0)
    evects = np.asmatrix(evects) # convert ndarray to matrix ?
    reord = np.argsort(evals)
    evals = evals[reord]
    evects = evects[:,reord]

    # Find the column index corresponding to the largest element in each row of evects 
    max_col = np.zeros(Ntot, dtype=np.int32)
    for row in range(Ntot):
        max_col[row] = np.argmax(np.abs(evects[row,:]))

    # test the error detection
    # max_col[1] = max_col[0]

    # loop over all columns and check max_col for duplicates
    Ndup_col = 0 
    for row in range(Ntot-1): 
        for k in range(row+1, Ntot):
            if max_col[row] == max_col[k]:
                Ndup_col += 1
                print("Error: detected identical max_col =", max_col[row], "for rows", row, "and", k)


    if Ndup_col > 0:
        print("Found", Ndup_col, "duplicate column indices in max_col array")
        raise ValueError('Permutation of eigen-vector matrix failed')

    evects = evects[:,max_col]
    evals = evals[max_col]
    
    # Make sure all diagonal elements are positive
    for j in range(Ntot):
        if evects[j,j]<0.0:
            evects[:,j] = - evects[:,j]

    return evals, evects


# Computes system resonances, to be used as carrier wave frequencies
# Returns resonance frequencies in GHz and corresponding growth rates.
def get_resonances(*, Ne, Ng, Hsys, Hc_re=[], Hc_im=[], rotfreq=[], cw_amp_thres=1e-7, cw_prox_thres=1e-2,verbose=True, stdmodel=True):
    if verbose:
        print("\nComputing carrier frequencies, ignoring growth rate slower than:", cw_amp_thres, "and frequencies closer than:", cw_prox_thres, "[GHz])")

    nqubits = len(Ne)
    n = Hsys.shape[0]
    
    # Get eigenvalues of system Hamiltonian (GHz)
    Hsys_evals, Utrans = eigen_and_reorder(Hsys, verbose)
    Hsys_evals = Hsys_evals.real  # Eigenvalues may have a small imaginary part due to numerical imprecision
    Hsys_evals = Hsys_evals / (2 * np.pi) 
            
    # Look for resonances in the symmetric and anti-symmetric control Hamiltonians for each qubit
    resonances = []
    speed = []
    for q in range(nqubits):
       
        # Transform symmetric and anti-symmetric control Hamiltonians using eigenvectors (reordered)
        Hsym_trans = Utrans.H @ Hc_re[q] @ Utrans
        Hanti_trans = Utrans.H @ Hc_im[q] @ Utrans

        resonances_a = []
        speed_a = []
        if verbose:
            print("  Resonances in oscillator #", q)
        
        for Hc_trans in (Hsym_trans, Hanti_trans):

            # Iterate over non-zero elements in transformed control
            for i in range(n):
                # Only consider transitions from lower to higher levels
                for j in range(i):

                    # Look for non-zero elements (skip otherwise)
                    if abs(Hc_trans[i,j]) < 1e-14: 
                        continue 

                    # Get the resonance frequency
                    delta_f = Hsys_evals[i] - Hsys_evals[j]
                    if abs(delta_f) < 1e-10:
                        delta_f = 0.0

                    # Get involved oscillator levels
                    ids_i = map_to_oscillators(i, Ne, Ng)
                    ids_j = map_to_oscillators(j, Ne, Ng)

                    # make sure both indices correspond to essential energy levels
                    is_ess_i = all(ids_i[k] < Ne[k] for k in range(len(Ne)))
                    is_ess_j = all(ids_j[k] < Ne[k] for k in range(len(Ne)))

                    if (is_ess_i and is_ess_j):
                        # Ignore resonances that are too close by comparing to all previous resonances
                        if any(abs(delta_f - f) < cw_prox_thres for f in resonances_a):
                            if verbose:
                                print("    Ignoring resonance from ", ids_j, "to ", ids_i, ", lab-freq", rotfreq[q] + delta_f, ", growth rate=", abs(Hc_trans[i, j]), "being too close to one that already exists.")
                        # Ignore resonances with growth rate smaller than user-defined threshold
                        elif abs(Hc_trans[i,j]) < cw_amp_thres:
                            if verbose:
                                print("    Ignoring resonance from ", ids_j, "to ", ids_i, ", lab-freq", rotfreq[q] + delta_f, ", growth rate=", abs(Hc_trans[i, j]), "growth rate is too slow.")
                        # Otherwise, add resonance to the list
                        else:
                            resonances_a.append(delta_f)
                            speed_a.append(abs(Hc_trans[i, j]))
                            if verbose:
                                print("    Resonance from ", ids_j, "to ", ids_i, ", lab-freq", rotfreq[q] + delta_f, ", growth rate=", abs(Hc_trans[i, j]))

        # Append resonances for this qubit to overall list        
        resonances.append(resonances_a)
        speed.append(speed_a)

    # Prepare output for carrier frequencies (om) and growth_rate
    Nfreq = np.zeros(nqubits, dtype=int)
    om = [[0.0] for _ in range(nqubits)]
    growth_rate = [[] for _ in range(nqubits)]
    for q in range(len(resonances)):
        Nfreq[q] = max(1, len(resonances[q]))  # at least one being 0.0
        om[q] = np.zeros(Nfreq[q])
        if len(resonances[q]) > 0:
            om[q] = np.array(resonances[q])
        growth_rate[q] = np.ones(Nfreq[q])
        if len(speed[q]) > 0:
            growth_rate[q] = np.array(speed[q])

    return om, growth_rate


# Lowering operator of dimension n
def lowering(n):
    return np.diag(np.sqrt(np.arange(1, n)), k=1)
# Number operator of dimension n
def number(n):
    return np.diag(np.arange(n))
# Return the local energy level of each oscillator for a given global index id
def map_to_oscillators(id, Ne, Ng):
    # len(Ne) = number of subsystems
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
def hamiltonians(*, N, freq01, selfkerr, crosskerr=[], Jkl = [], rotfreq=[], verbose=True):
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
        #print("Amat i =", i)
        #print(ai) 
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
    # Note that if the rotating frame frequencies are different amongst oscillators, then this contributes to a *time-dependent* system Hamiltonian. Here, we treat this as time-independent, because this Hamiltonian here is *ONLY* used to compute the time-step size and resonances, and it is NOT passed to the quandary code. Quandary sets up the standard model with a time-dependent system Hamiltonian if the frequencies of rotation differ amongst oscillators.  
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
        maxp = max(np.abs(pt[iosc]))
        maxq = max(np.abs(qt[iosc]))
        plt.title('Qubit '+str(iosc)+'\n max. drive '+str(round(maxp,1))+", "+str(round(maxq,1))+" MHz")
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
def plot_expectedEnergy(Ne, time, expectedEnergy, *, lindblad_solver=False):
    ninit = len(expectedEnergy[0])
    nplots = ninit                    # one plot for each initial state
    # nplots = np.prod(Ne)                # one plot for each initial state
    ncols = 2 if nplots >= 4 else 1     # 2 rows if more than 3 plots
    nrows = int(np.ceil(nplots/ncols))
    figsizex = 6.4*nrows*0.75 
    figsizey = 4.8*nrows*0.75 
    fig = plt.figure(figsize=(figsizex,figsizey))
    for iplot in range(nplots):
        iinit = iplot if not lindblad_solver else iplot*nplots + iplot
        plt.subplot(nrows, ncols, iplot+1)
        plt.figsize=(15, 15)
        for iosc in range(len(Ne)):
            label = 'Qubit '+str(iosc) if len(Ne)>1 else ''
            plt.plot(time, expectedEnergy[iosc][iinit], label=label)
        plt.xlabel('time (ns)')
        plt.ylabel('expected energy')
        plt.ylim([0.0-1e-2, Ne[0]-1.0 + 1e-2])
        plt.xlim([0.0, time[-1]])
        binary_ID = iplot if len(Ne) == 1 else bin(iplot).replace("0b", "").zfill(len(Ne))
        plt.title("from |"+str(binary_ID)+">")
        plt.legend(loc='lower right')
    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(wspace=0.5)
    plt.draw()
    print("\nPlotting expected energy dynamics")
    print("-> Press <enter> to proceed.")
    plt.waitforbuttonpress(1); 
    input(); 
    plt.close(fig)


##
# Plot evolution of population
##
def plot_population(Ne, time, population, *, lindblad_solver=False):
    ninit = len(population[0])
    # nplots = np.prod(Ne)                # one plot for each initial state
    nplots = ninit                      # one plot for each initial state
    ncols = 2 if nplots >= 4 else 1     # 2 rows if more than 3 plots
    nrows = int(np.ceil(nplots/ncols))
    figsizex = 6.4*nrows*0.75 
    figsizey = 4.8*nrows*0.75 
    fig = plt.figure(figsize=(figsizex,figsizey))

    # Iterate over initial conditions (one plot for each)
    for iplot in range(nplots):
        iinit = iplot if not lindblad_solver else iplot*nplots + iplot
        plt.subplot(nrows, ncols, iplot+1)
        plt.figsize=(15, 15)
        for iosc in range(len(Ne)):
            for istate in range(Ne[iosc]):
                label = 'Qubit '+str(iosc) if len(Ne)>1 else ''
                label = label + " |"+str(istate)+">"
                plt.plot(time, population[iosc][iinit][istate], label=label)
        plt.xlabel('time (ns)')
        plt.ylabel('population')
        plt.ylim([0.0-1e-4, 1.0 + 1e-2])
        plt.xlim([0.0, time[-1]])
        binary_ID = iplot if len(Ne) == 1 else bin(iplot).replace("0b", "").zfill(len(Ne))
        plt.title("from |"+str(binary_ID)+">")
        plt.legend(loc='lower right')
    plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(wspace=0.5)
    plt.draw()
    print("\nPlotting population dynamics")
    print("-> Press <enter> to proceed.")
    plt.waitforbuttonpress(1); 
    input(); 
    plt.close(fig)


def plot_results_1osc(myconfig, p, q, expectedEnergy, population):

    fig, ax = plt.subplots(2, 3, figsize=(20,8))
    fig.subplots_adjust(hspace=0.3)

    t = myconfig.time

    # Plot pulses
    ax[0,0].plot(t, p, label='I') # y label: MHz
    ax[0,0].plot(t, q, label='Q') # y label: MHz
    ax[0,0].set_ylabel('Pulse amplitude (MHz)')
    ax[0,0].set_xlabel('Time (ns)')
    ax[0,0].legend()
    ax[0,0].grid()


    # Compute and plot FFT
    zlist = np.array(p)*1e-3 + 1j*np.array(q)*1e-3
    fft = np.fft.fft(zlist)
    dt = myconfig.T / myconfig.nsteps
    fftfr = np.fft.fftfreq(len(zlist), d=dt)

    ax[0,1].scatter(fftfr*1e3, np.abs(fft)**2)
    ax[0,1].set_ylabel('FFT')
    ax[0,1].set_xlabel('Frequency (MHz)')
    ax[0,1].grid()
    ax[0,1].set_title('FFT')
    ax[0,1].set_yscale('log')
    ax[0,1].set_xlim(-500, 500)
    ax[0,1].set_ylim(1e-8, 1e5)

    # Plot Populations for each initial condition 
    for iinit in range(len(population)):  # for each of the 3 initial states
        row = 1
        col = iinit
            
        for istate in range(myconfig.Ne[0]): # for each essential level
            label = "|"+str(istate)+">"
            ax[row, col].plot(t, population[iinit][istate], label=label)
            # ax[row, col+1].plot(np.arange(0, numgate), prob_me_gate[i].real, label=str(i))
            
        ax[row, col].set_xlabel('Time (ns)')
        ax[row, col].set_ylabel('Population')
        ax[row, col].legend()
        ax[row, col].set_title('Populations from |%d>' % iinit)
        ax[row, col].grid()
        
        # ax[row, col+1].set_xlabel('Gate repetition')
        # ax[row, col+1].set_ylabel('Population')
        # ax[row, col+1].legend()
        # ax[row, col+1].set_title('ME solve, starting from %d' % state)

    # Plot expected Energy
    row, col = 0, 2
    for iinit in range(len(expectedEnergy)):
        label = 'from |'+str(iinit)+'>' 
        ax[row, col].plot(t, expectedEnergy[iinit], label=label)
    ax[row, col].set_xlabel('Time (ns)')
    ax[row, col].set_ylabel('Expected Energy Level')
    ax[row, col].legend()
    ax[row, col].set_title('Expected Energy Level')
    ax[row, col].grid()

    plt.draw()
    print("\nPlotting results...")
    print("-> Press <enter> to proceed.")
    plt.waitforbuttonpress(1); 
    input(); 
    plt.close(fig)