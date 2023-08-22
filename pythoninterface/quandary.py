import os
import numpy as np
from subprocess import run, PIPE, Popen
import matplotlib.pyplot as plt


# Main interface function to create a pulse with Quandary. 
def quandary_run(Ne, Ng, freq01, selfkerr, crosskerr, Jkl, rotfreq, maxctrl_MHz, T, initctrl_MHz, rand_seed, randomize_init_ctrl, targetgate, *, dtau=3.33, Pmin=40, cw_amp_thres=1e-2, cw_prox_thres=1e-3, datadir=".", tol_infidelity=1e-3, tol_costfunc=1e-3, maxiter=100, gamma_tik0=1e-4, gamma_energy=1e-2, gamma_dpdm=1e-2, costfunction="Jtrace", initialcondition="basis", T1=None, T2=None, runtype="simulation", quandary_exec="/absolute/path/to/quandary/main", ncores=1, print_frequency_iter=1, verbose=False, pcof0=[], Hsys=[], Hc_re=[], Hc_im=[]):

    # Create quandary data directory
    os.makedirs(datadir, exist_ok=True)

    # Hamiltonian operators: Either use the provided ones, or set up standard model. 
    if len(Hsys) == 0:  # Using standard Hamiltonian model
        standardmodel=True
        Ntot = [sum(x) for x in zip(Ne, Ng)]
        Hsys, Hc_re, Hc_im = hamiltonians(Ntot, freq01, selfkerr, crosskerr, Jkl, rotfreq=rotfreq, verbose=verbose)
 
    else: # Using provided Hamiltonians, write them to hamiltonian.dat
        standardmodel=False   
        hamiltonianfilename = datadir + "/hamiltonian.dat"

        # Write system Hamiltonian to file  
        with open(hamiltonianfilename, "w") as f:
            f.write("# Hsys \n")
            Hsyslist = list(np.array(Hsys).flatten(order='F'))
            for value in Hsyslist:
                f.write("{:20.13e}\n".format(value))
        
        # Write control Hamiltonians to file, if given (append to file)
        for iosc in range(len(Ne)):
            # Real part, if given
            if len(Hc_re)>iosc and len(Hc_re[iosc])>0:
                with open(hamiltonianfilename, "a") as f:
                    Hcrelist = list(np.array(Hc_re[iosc]).flatten(order='F'))
                    f.write("# Oscillator {:d} Hc_real \n".format(iosc))
                    for value in Hcrelist:
                        f.write("{:20.13e}\n".format(value))
            # Imaginary part, if given
            if len(Hc_im)>iosc and len(Hc_im[iosc])>0:
                with open(hamiltonianfilename, "a") as f:
                    Hcimlist = list(np.array(Hc_im[iosc]).flatten(order='F'))
                    f.write("# Oscillator {:d} Hc_imag \n".format(iosc))
                    for value in Hcimlist:
                        f.write("{:20.13e}\n".format(value))
   
    # print("Hsys=", Hsys)
    # print("Hc_re=", Hc_re)
    # print("Hc_im=", Hc_im)

    # Estimate number of time steps
    nsteps = estimate_timesteps(T=T, Hsys=Hsys, Hc_re=Hc_re, Hc_im=Hc_im, maxctrl_MHz=maxctrl_MHz, Pmin=Pmin)
    if verbose:
        print("Final time: ",T,"ns, Number of timesteps: ", nsteps,", dt=", T/nsteps, "ns")
        print("Maximum control amplitudes: ", maxctrl_MHz, "MHz")
        # print("Hsys = ", Hsys)
        # print("Hc_real = ", Hc_re)
        # print("Hc_im = ", Hc_im)


    # Estimate carrier wave frequencies
    carrierfreq, _ = get_resonances(Ne=Ne, Ng=Ng, Hsys=Hsys, Hc_re=Hc_re, Hc_im=Hc_im, verbose=verbose, cw_amp_thres=cw_amp_thres, cw_prox_thres=cw_prox_thres, stdmodel=standardmodel) 

    if verbose:
        print("Carrier frequencies: ", carrierfreq)

    # Write target gate to file
    gatefilename = datadir + "/targetgate.dat"
    gate_vectorized = np.concatenate((np.real(targetgate).ravel(), np.imag(targetgate).ravel()))
    with open(gatefilename, "w") as f:
        for value in gate_vectorized:
            f.write("{:20.13e}\n".format(value))
    if verbose:
        print("Target gate written to ", gatefilename)

    # Write initial pcof0 to file, if given
    if len(pcof0) > 0:
        pcof0filename = datadir + "/pcof0.dat"
        with open(pcof0filename, "w") as f:
            for value in pcof0:
                f.write("{:20.13e}\n".format(value))
        if verbose:
            print("Initial control parameters written to ", pcof0filename)

    # Write Quandary configuration file
    nsplines = int(np.max([np.ceil(T/dtau + 2), 5])) # 10
    config_filename = write_config(Ne=Ne, Ng=Ng, T=T, nsteps=nsteps, freq01=freq01, rotfreq=rotfreq, selfkerr=selfkerr, crosskerr=crosskerr, Jkl=Jkl, nsplines=nsplines, carrierfreq=carrierfreq, tol_infidelity=tol_infidelity, tol_costfunc=tol_costfunc, maxiter=maxiter, maxctrl_MHz=maxctrl_MHz, initctrl_MHz=initctrl_MHz, randomize_init_ctrl=randomize_init_ctrl, gamma_tik0=gamma_tik0, gamma_energy=gamma_energy, gamma_dpdm=gamma_dpdm, costfunction=costfunction, initialcondition=initialcondition, T1=T1, T2=T2, runtype=runtype, gatefilename="./targetgate.dat", print_frequency_iter=print_frequency_iter, datadir=datadir, verbose=verbose, pcof0=pcof0, standardmodel=standardmodel)


    # Call Quandary
    err = execute(runtype=runtype, ncores=ncores, config_filename=config_filename, datadir=datadir, quandary_exec=quandary_exec, verbose=verbose)

    # Get results and return
    time, pt, qt, ft, expectedEnergy , popt, infidelity, optim_hist= get_results(Ne=Ne, datadir=datadir)

    return time, pt, qt, ft, expectedEnergy, popt, infidelity, optim_hist



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


def write_config(*, Ne, Ng, T, nsteps, freq01, rotfreq, selfkerr, crosskerr=[], Jkl=[], nsplines=5, carrierfreq, T1=[], T2=[], gatefilename="./gatefile.dat", runtype="optimization",maxctrl_MHz=None, initctrl_MHz=None, randomize_init_ctrl=True, maxiter=1000,tol_infidelity=1e-3, tol_costfunc=1e-3, gamma_tik0=1e-4, gamma_dpdm=0.0, gamma_energy=0.0, costfunction="Jtrace", initialcondition="basis", datadir=".", configfilename="config.cfg", print_frequency_iter=1, pcof0=[], control_enforce_BC=True, verbose=False, standardmodel=True, usematfree=True):

    if maxctrl_MHz is None:
        maxctrl_MHz = 1e+12*np.ones(len(Ne))

    # Scale initial control amplitudes by the number of carrier waves
    initamp = np.zeros(len(Ne))
    if initctrl_MHz is not None:
        for q in range(len(Ne)):
            initamp[q] = initctrl_MHz[q] *2.0*np.pi/1000.0 / np.sqrt(2) / len(carrierfreq[q])

    Nt = [Ne[i] + Ng[i] for i in range(len(Ng))]
    mystring = "nlevels = " + str(list(Nt))[1:-1] + "\n"
    mystring += "nessential= " + str(list(Ne))[1:-1] + "\n"
    mystring += "ntime = " + str(nsteps) + "\n"
    mystring += "dt = " + str(T / nsteps) + "\n"
    mystring += "transfreq = " + str(list(freq01))[1:-1] + "\n"
    mystring += "rotfreq= " + str(list(rotfreq))[1:-1] + "\n"
    mystring += "selfkerr = " + str(list(selfkerr))[1:-1] + "\n"
    if len(crosskerr)>0:
        mystring += "crosskerr= " + str(list(crosskerr))[1:-1] + "\n"
    else:
        mystring += "crosskerr= 0.0\n"
    if len(Jkl)>0:
        mystring += "Jkl= " + str(list(Jkl))[1:-1] + "\n"
    else:
        mystring += "Jkl= 0.0\n"
    decay = dephase = False
    if len(T1) > 0: 
        decay = True
        mystring += "decay_time = " + str(list(T1))[1:-1] + "\n"
    if len(T2) > 0:
        dephase = True
        mystring += "dephase_time = " + str(list(T2))[1:-1] + "\n"
    if decay and dephase:
        mystring += "collapse_type = both\n"
    elif decay:
        mystring += "collapse_type = decay\n"
    elif dephase:
        mystring += "collapse_type = dephase\n"
    else:
        mystring += "collapse_type = none\n"
    mystring += "initialcondition = " + str(initialcondition) + "\n"
    for iosc in range(len(Ne)):
        mystring += "control_segments" + str(iosc) + " = spline, " + str(nsplines) + "\n"
        if len(pcof0)>0:
            initstring = "file, ./pcof0.dat\n"
        else:
            initstring = ("random, " if randomize_init_ctrl else "constant, ") + str(initamp[iosc]) + "\n"
        mystring += "control_initialization" + str(iosc) + " = " + initstring
        mystring += "control_bounds" + str(iosc) + " = " + str(maxctrl_MHz[iosc]*2.0*np.pi/1000.0) + "\n"
        mystring += "carrier_frequency" + str(iosc) + " = "
        omi = carrierfreq[iosc]
        for j in range(len(omi)):
            mystring += str(omi[j]) + ", " 
        mystring += "\n"
    mystring += "control_enforceBC = " + str(control_enforce_BC)+ "\n"
    mystring += "optim_target = gate, fromfile, " + gatefilename + "\n"
    mystring += "optim_objective = " + str(costfunction) + "\n"
    mystring += "gate_rot_freq = 0.0\n"
    mystring += "optim_weights= 1.0\n"
    mystring += "optim_atol= 1e-5\n"
    mystring += "optim_rtol= 1e-4\n"
    mystring += "optim_dxtol = 1e-8\n"
    mystring += "optim_ftol= " + str(tol_costfunc) + "\n"
    mystring += "optim_inftol= " + str(tol_infidelity) + "\n"
    mystring += "optim_maxiter= " + str(maxiter) + "\n"
    mystring += "optim_regul= " + str(gamma_tik0) + "\n"
    mystring += "optim_penalty= 0.0\n"
    mystring += "optim_penalty_param= 0.1\n"
    ninitscale = np.prod(Ne)
    mystring += "optim_penalty_dpdm= " + str(gamma_dpdm) + "\n"
    mystring += "optim_penalty_energy= " + str(gamma_energy) + "\n"
    mystring += "datadir= ./\n"
    for iosc in range(len(Ne)):
        mystring += "output" + str(iosc) + "=expectedEnergy, population\n"
    mystring += "output_frequency = 1\n"
    mystring += "optim_monitor_frequency = " + str(print_frequency_iter) + "\n"
    mystring += "runtype = " + runtype + "\n"
    if len(Ne) < 6:
        mystring += "usematfree = " + str(usematfree) + "\n"
    else:
        mystring += "usematfree = false\n"
    mystring += "linearsolver_type = gmres\n"
    mystring += "linearsolver_maxiter = 20\n"
    if not standardmodel:
        mystring += "hamiltonian_file= ./hamiltonian.dat\n"
    mystring += "timestepper = IMR\n"


    # Write the file
    outpath = datadir+"/"+configfilename
    with open(outpath, "w") as file:
        file.write(mystring)

    if verbose:
        print("Quandary config written to:", outpath)

    return configfilename


def get_results(*, Ne=[], datadir="./"):
    dataout_dir = datadir + "/"
    
    # Get control parameters
    pcof = np.loadtxt(dataout_dir + "/params.dat").astype(float)

    # Get optimization history information
    optim_hist = np.loadtxt(dataout_dir + "/optim_history.dat")

    if optim_hist.ndim == 2:
        optim_last = optim_hist[-1]
    else:
        optim_last = optim_hist
    infid_last = 1.0 - optim_last[4]
    tikhonov_last = optim_last[6]
    dpdm_penalty_last = optim_last[8] 

    # # Get last time-step unitary
    # uT = np.zeros((np.prod(Nt), np.prod(Ne)), dtype=np.complex128)
    # for i in range(np.prod(Ne)):
    #     # Read from file
    #     xre = np.loadtxt(dataout_dir + "/rho_Re.iinit" + str(i).zfill(4) + ".dat")[-1, 1:]
    #     xim = np.loadtxt(dataout_dir + "/rho_Im.iinit" + str(i).zfill(4) + ".dat")[-1, 1:]
    #     uT[:, i] = xre + 1j * xim

    # Get the time-evolution of the expected energy for each qubit, for each initial condition
    expectedEnergy = [[] for _ in range(len(Ne))]
    for iosc in range(len(Ne)):
        for iinit in range(np.prod(Ne)):
            x = np.loadtxt(dataout_dir + "./expected"+str(iosc)+".iinit"+str(iinit).zfill(4)+".dat")
            expectedEnergy[iosc].append(x[:,1])    # first column is time, second column is expected energy

    # Get the control pulses for each qubit
    pt = []
    qt = []
    ft = []
    for iosc in range(len(Ne)):
        x = np.loadtxt(dataout_dir + "./control"+str(iosc)+".dat")
        time = x[:,0]   # Time domain
        pt.append([x[n,1]/(2*np.pi)*1e+3 for n in range(len(x[:,0]))])     # Rot frame p(t), MHz
        qt.append([x[n,2]/(2*np.pi)*1e+3 for n in range(len(x[:,0]))])     # Rot frame q(t), MHz
        ft.append([x[n,3]/(2*np.pi)*1e+3 for n in range(len(x[:,0]))])     # Lab frame f(t)

    return time, pt, qt, ft, expectedEnergy, pcof, infid_last, optim_hist


# Estimates the number of time steps based on eigenvalues of the system Hamiltonian and maximum control Hamiltonians.
# NOTE: The estimate does not account for quickly varying signals or a large number of splines. Double check that at least 2-3 points per spline are present to resolve control function. #TODO: Automate this
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



# Computes system resonances, to be used as carrier wave frequencies
# Returns resonance frequencies in GHz and corresponding growth rates.
def get_resonances(*, Ne, Ng, Hsys, Hc_re=[], Hc_im=[], cw_amp_thres=6e-2, cw_prox_thres=1e-3,verbose=True, stdmodel=True):
    if verbose:
        print("\nget_resonances: Ignoring growth rate slower than:", cw_amp_thres, "and frequencies closer than:", cw_prox_thres, "[GHz]")

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

# Create Hamiltonian operators.
def hamiltonians(N, freq01, selfkerr, crosskerr=[], Jkl = [], *, rotfreq=None, verbose=True):
    if rotfreq is None:
        rotfreq=np.zeros(len(N))

    nqubits = len(N)
    assert len(selfkerr) == nqubits
    assert len(freq01) == nqubits
    assert len(rotfreq) == nqubits
    assert len(selfkerr) == nqubits

    n = np.prod(N)     # System size 

    # Set up lowering operators in full dimension
    Amat = []
    for i in range(len(N)):
        # predim = 1
        # for j in range(i):
                # predim*=N[j]
        ai = lowering(N[i])
        for j in range(i):
            ai = np.kron(np.identity(N[j]), ai) 
        # postdim = 1
        for j in range(i+1,len(N)):
                # postdim*=N[j]
            ai = np.kron(ai, np.identity(N[j])) 
        # a.append(tensor(qeye(predim), lowering(N[i]), ident(postdim)))
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


# Plot the control pulse for all qubits
def plot_pulse(Ne, time, pt, qt):
    fig = plt.figure()
    # # Increase figure size if more than one oscillator will be plotted. 
    # # Default size for one figure is [6.4, 4.8] -> Multiply by number of figures times 75%
    # if len(Ne) > 1:
    #     size_org = plt.rcParams['figure.figsize']
    #     plt.rcParams['figure.figsize'] = [size*len(Ne)*0.75 for size in size_org] 
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

# Plot evolution of expected energy levels
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
