import os
import shutil
import numpy as np
from subprocess import run, PIPE, Popen


# Main interface function to create a pulse with Quandary. 
def pulse_gen(Ne, Ng, freq01, selfkerr, crosskerr, Jkl, rotfreq, maxctrl_MHz, T, initctrl_MHz, rand_seed, randomize_init_ctrl, targetgate, *, dtau=3.33, Pmin=40, cw_amp_thres=6e-2, cw_prox_thres=1e-3, datadir=".", tol_infidelity=1e-3, tol_costfunc=1e-3, maxiter=100, gamma_tik0=1e-4, gamma_energy=1e-2, costfunction="Jtrace", initialcondition="basis", T1=None, T2=None, runtype="simulation", quandary_exec="/absolute/path/to/quandary/main", ncores=1, print_frequency_iter=1, verbose=False):

    # Create quandary data directory
    os.makedirs(datadir, exist_ok=True)

    # Set up Hamiltonians in essential levels only
    Hsys, Hc_re, Hc_im = hamiltonians(Ne, freq01, selfkerr, crosskerr, Jkl, rotfreq=rotfreq)

    # Estimate number of time steps
    nsteps = estimate_timesteps(T, Hsys, Hc_re, Hc_im, maxctrl_MHz, Pmin=Pmin)
    if verbose:
        print("Final time: ",T,"ns, Number of timesteps: ", nsteps,", dt=", T/nsteps, "ns")
        print("Maximum control amplitudes: ", maxctrl_MHz, "MHz")


    # Estimate carrier wave frequencies
    carrierfreq, growth_rate = get_resonances(Ne, Hsys, Hc_re, Hc_im, verbose=verbose) 
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


    # Write Quandary configuration file
    nsplines = int(np.max([np.ceil(T/dtau + 2), 5])) # 10
    config_filename = write_config(Ne=Ne, Ng=Ng, T=T, nsteps=nsteps, freq01=freq01, rotfreq=rotfreq, selfkerr=selfkerr, crosskerr=crosskerr, Jkl=Jkl, nsplines=nsplines, carrierfreq=carrierfreq, tol_infidelity=tol_infidelity, tol_costfunc=tol_costfunc, maxiter=maxiter, maxctrl_MHz=maxctrl_MHz, initctrl_MHz=initctrl_MHz, randomize_init_ctrl=randomize_init_ctrl, gamma_tik0=gamma_tik0, gamma_energy=gamma_energy, costfunction=costfunction, initialcondition=initialcondition, T1=T1, T2=T2, runtype=runtype, gatefilename="./targetgate.dat", print_frequency_iter=print_frequency_iter, datadir=datadir, verbose=verbose)


    # Call Quandary
    err = execute(runtype=runtype, ncores=ncores, config_filename=config_filename, datadir=datadir, quandary_exec=quandary_exec, verbose=verbose)

    # Get results and return
    popt, infidelity, optim_hist = get_results(datadir)

    return popt, infidelity, optim_hist



def execute(*, runtype="simulation", ncores=1, config_filename="config.cfg", datadir=".", quandary_exec="/absolute/path/to/quandary/main", verbose=False):

    
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

    if verbose:
        result = run(["pwd"], shell=True, capture_output=True, text=True)
        print("Running Quandary in directory ", result.stdout)
        print("Executing '", runcommand, "'")
        print("...\n")


    # Execute Quandary
    # Pipe std output to file rather than screen
    # with open(os.path.join(datadir, "out.log"), "w") as stdout_file, \
    #      open(os.path.join(datadir, "err.log"), "w") as stderr_file:
    #         exec = run(runcommand, shell=True, stdout=stdout_file, stderr=stderr_file)
    exec = run(runcommand, shell=True)

    # Check return code
    err = exec.check_returncode()

    # # Execute Quandary on Windows through Cygwin
    # p = Popen(r"C:/cygwin64/bin/bash.exe", stdin=PIPE, stdout=PIPE, stderr=PIPE)  
    # p.stdin.write(b"/cygdrive/c/Users/scada-125/quandary/main.exe config.cfg") 
    # p.stdin.close()
    # # # Print stdout and stderr
    # if verbose:
    #     print(p.stdout.read())
    #     print(p.stderr.read())

    # Return to previous directory
    os.chdir(dir_org)
    
    if verbose: 
        print("DONE. \n")

    return 1


def write_config(*, Ne, Ng, T, nsteps, freq01, rotfreq, selfkerr, crosskerr=[], Jkl=[], nsplines=5, carrierfreq, T1=[], T2=[], gatefilename="./gatefile.dat", runtype="optimization",maxctrl_MHz=None, initctrl_MHz=None, randomize_init_ctrl=True, maxiter=1000,tol_infidelity=1e-3, tol_costfunc=1e-3, gamma_tik0=1e-4, gamma_dpdm=0.0, gamma_energy=0.0, costfunction="Jtrace", initialcondition="basis", datadir=".", configfilename="config.cfg", print_frequency_iter=1, verbose=False):

    if maxctrl_MHz is None:
        maxctrl_MHz = 1e+12*np.ones(len(Ne))

    maxamp = np.zeros(len(Ne))
    if initctrl_MHz is not None:
        for q in range(len(Ne)):
            maxamp[q] = initctrl_MHz[q] / np.sqrt(2) / len(carrierfreq[q])

    Nt = [Ne[i] + Ng[i] for i in range(len(Ng))]
    mystring = "nlevels = " + str(Nt)[1:-1] + "\n"
    mystring += "nessential= " + str(Ne)[1:-1] + "\n"
    mystring += "ntime = " + str(nsteps) + "\n"
    mystring += "dt = " + str(T / nsteps) + "\n"
    mystring += "transfreq = " + str(freq01)[1:-1] + "\n"
    mystring += "rotfreq= " + str(rotfreq)[1:-1] + "\n"
    mystring += "selfkerr = " + str(selfkerr)[1:-1] + "\n"
    if len(crosskerr)>0:
        mystring += "crosskerr= " + str(crosskerr)[1:-1] + "\n"
    else:
        mystring += "crosskerr= 0.0\n"
    if len(Jkl)>0:
        mystring += "Jkl= " + str(Jkl)[1:-1] + "\n"
    else:
        mystring += "Jkl= 0.0\n"
    decay = dephase = False
    if len(T1) > 0: 
        decay = True
        mystring += "decay_time = " + str(T1)[1:-1] + "\n"
    if len(T2) > 0:
        dephase = True
        mystring += "dephase_time = " + str(T2)[1:-1] + "\n"
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
        mystring += "control_initialization" + str(iosc) + " = " + ("random, " if randomize_init_ctrl else "constant, ") + str(maxamp[iosc]) + "\n"
        mystring += "control_bounds" + str(iosc) + " = " + str(maxctrl_MHz[iosc]*2.0*np.pi/1000.0) + "\n"
        mystring += "carrier_frequency" + str(iosc) + " = "
        omi = carrierfreq[iosc]
        for j in range(len(omi)):
            mystring += str(omi[j]) + ", " 
        mystring += "\n"
    mystring += "optim_target = gate, fromfile, " + gatefilename + "\n"
    mystring += "optim_objective = " + str(costfunction) + "\n"
    mystring += "gate_rot_freq = 0.0\n"
    mystring += "optim_weights= 1.0\n"
    mystring += "optim_atol= 1e-5\n"
    mystring += "optim_rtol= 1e-4\n"
    mystring += "optim_ftol= " + str(tol_costfunc) + "\n"
    mystring += "optim_inftol= " + str(tol_infidelity) + "\n"
    mystring += "optim_maxiter= " + str(maxiter) + "\n"
    mystring += "optim_regul= " + str(gamma_tik0) + "\n"
    mystring += "optim_penalty= 1.0\n"
    mystring += "optim_penalty_param= 0.0\n"
    ninitscale = np.prod(Ne)
    mystring += "optim_regul_dpdm= " + str(gamma_dpdm) + "\n"
    mystring += "optim_penalty_energy= " + str(gamma_energy) + "\n"
    mystring += "datadir= ./\n"
    for iosc in range(len(Ne)):
        mystring += "output" + str(iosc) + "=expectedEnergy, population, fullstate\n"
    mystring += "output_frequency = " + str(nsteps) + "\n"
    mystring += "optim_monitor_frequency = " + str(print_frequency_iter) + "\n"
    mystring += "runtype = " + runtype + "\n"
    if len(Ne) < 6:
        mystring += "usematfree = true\n"
    else:
        mystring += "usematfree = false\n"
    mystring += "linearsolver_type = gmres\n"
    mystring += "linearsolver_maxiter = 20\n"

    # Write the file
    outpath = datadir+"/"+configfilename
    with open(outpath, "w") as file:
        file.write(mystring)

    if verbose:
        print("Quandary config written to:", outpath)

    return configfilename


def get_results(datadir="./"):
    # TODO: Output directory of quandary run. 
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
    # tikhonov_last = optim_last[6]
#     dpdm_penalty_last = optim_last[8]  # TODO: add dpdm penalty

    # # Get last time-step unitary
    # uT = np.zeros((np.prod(Nt), np.prod(Ne)), dtype=np.complex128)

    # for i in range(np.prod(Ne)):
    #     # Read from file
    #     xre = np.loadtxt(dataout_dir + "/rho_Re.iinit" + str(i).zfill(4) + ".dat")[-1, 1:]
    #     xim = np.loadtxt(dataout_dir + "/rho_Im.iinit" + str(i).zfill(4) + ".dat")[-1, 1:]
    #     uT[:, i] = xre + 1j * xim

    # # grad = np.zeros(len(pcof))
    # if runtype == "gradient":  # the grad.dat file is not created by the optimization mode
    #     # chop up the long vector into individual column vectors for the result
    #     grad = np.loadtxt(dataout_dir + "/grad.dat")[:, 0]

    # TODO:
    return pcof, infid_last, optim_hist


# Estimates the number of time steps based on eigenvalues of the system Hamiltonian and maximum control Hamiltonians.
# NOTE: The estimate does not account for quickly varying signals or a large number of splines. Double check that at least 2-3 points per spline are present to resolve control function.
# TODO: Automate this!
def estimate_timesteps(T, Hsys, Hc_re=[], Hc_im=[], maxctrl_MHz=[], *, Pmin=40):
    assert len(maxctrl_MHz) >= len(Hc_re)


    # Set up Hsys + maxctrl*Hcontrol
    K1 = np.copy(Hsys) 
    for i in range(len(Hc_re)):
        max_radns = maxctrl_MHz[i]*2.0*np.pi/1e+3
        K1 += max_radns * Hc_re[i] 
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
# Returns resonance frequencies in GHz and corresponding growth rates
def get_resonances(Ne, Hsys, Hc_re, Hc_im, *, cw_amp_thres=6e-2, cw_prox_thres=1e-3, verbose=True):
    if verbose:
        print("\nget_resonances: Ignoring growth rate slower than:", cw_amp_thres, "and frequencies closer than:", cw_prox_thres, "[GHz]")

    nqubits = len(Hc_re)
    n = Hsys.shape[0]

    # Get eigenvalues of system Hamiltonian (GHz)
    Hsys_evals, Utrans = np.linalg.eig(Hsys)
    Hsys_evals = Hsys_evals.real  # Eigenvalues may have a small imaginary part due to numerical precision
    Hsys_evals = Hsys_evals / (2 * np.pi)

    resonances = []
    speed = []
    for q in range(nqubits):
        Hctrl_ad = Hc_re[q] - Hc_im[q]   # divide by 2. Adjust the cw_amp_thres. 
        Hctrl_ad_trans = Utrans.T @ Hctrl_ad @ Utrans

        resonances_a = []
        speed_a = []
        if verbose:
            print("  Resonances in oscillator #", q)
        for i in range(n):
            for j in range(i):
                if abs(Hctrl_ad_trans[i, j]) >= cw_amp_thres:
                    delta_f = Hsys_evals[i] - Hsys_evals[j]
                    if abs(delta_f) < 1e-10:
                        delta_f = 0.0
                    if not any(abs(delta_f - f) < cw_prox_thres for f in resonances_a):
                        resonances_a.append(delta_f)
                        speed_a.append(abs(Hctrl_ad_trans[i, j]))

                        if verbose:
                            print("    Resonance from =", j, "to =", i, ", frequency", delta_f, ", growth rate=", abs(Hctrl_ad_trans[i, j]))
        
        resonances.append(resonances_a)
        speed.append(speed_a)

    Nfreq = np.zeros(nqubits, dtype=int)
    om = [[] for _ in range(nqubits)]
    growth_rate = [[] for _ in range(nqubits)]
    
    for q in range(nqubits):
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


# Identity operator of dimension n
def ident(n):
    return np.diag(np.ones(n))

# Lowering operator of dimension n
def lowering(n):
    return np.diag(np.sqrt(np.arange(1, n)), k=1)

# Number operator of dimension n
def number(n):
    return np.diag(np.arange(n))


# Create Hamiltonian operators. Essential levels ONLY!
def hamiltonians(Ne, freq01, selfkerr, crosskerr=[], Jkl = [], *, rotfreq=None, verbose=True):
    if rotfreq==None:
        rotfreq=np.zeros(len(Ne))

    nqubits = len(Ne)
    assert len(selfkerr) == nqubits
    assert len(freq01) == nqubits
    assert len(rotfreq) == nqubits
    assert len(selfkerr) == nqubits

    n = np.prod(Ne)     # System size 

    # Set up lowering operators in full dimension
    Amat = []
    for i in range(len(Ne)):
        # predim = 1
        # for j in range(i):
                # predim*=N[j]
        ai = lowering(Ne[i])
        for j in range(i):
            ai = np.kron(ident(Ne[j]), ai) 
        # postdim = 1
        for j in range(i+1,len(Ne)):
                # postdim*=N[j]
            ai = np.kron(ai, ident(Ne[j])) 
        # a.append(tensor(qeye(predim), lowering(N[i]), ident(postdim)))
        Amat.append(ai)

    # Set up system Hamiltonian: Duffing oscillators
    Hsys = np.zeros((n, n))
    for q in range(nqubits):
        domega_radns =  2.0*np.pi * (freq01[q] - rotfreq[q])
        selfkerr_radns = 2.0*np.pi * selfkerr[q]
        Hsys +=  domega_radns * Amat[q].T @ Amat[q]
        Hsys -= selfkerr_radns/2 * Amat[q].T @ Amat[q].T @ Amat[q] * Amat[q]

    # Add system Hamiltonian coupling terms
    if len(crosskerr)>0:
        idkl = 0 
        for q in range(nqubits):
            for p in range(q + 1, nqubits):
                crosskerr_radns = 2.0*np.pi * crosskerr[idkl]
                Hsys -= crosskerr_radns * Amat[q].T @ Amat[q] @ Amat[p].T @ Amat[p]
                idkl += 1
    if len(Jkl)>0:
        idkl = 0 
        for q in range(nqubits):
            for p in range(q + 1, nqubits):
                Jkl_radns  = 2.0*np.pi*Jkl[idkl]
                Hsys += Jkl_radns * Amat[q].T @ Amat[p] + Amat[q] @ Amat[p].T
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
