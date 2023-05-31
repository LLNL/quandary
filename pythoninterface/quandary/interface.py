from subprocess import run, PIPE
import numpy as np


def write_config(*, Ne, Ng, T, nsteps, freq01, rotfreq, selfkerr, crosskerr=[], Jkl=[], nsplines=5, carrierfreq, T1=[], T2=[], gatefilename="gatefile.dat", runtype="optimization",maxctrl_MHz=None, initctrl_MHz=None, randomize_init_ctrl=True, maxiter=1000,tol_infidelity=1e-3, tol_costfunc=1e-3, gamma_tik0=1e-4, gamma_dpdm=0.0, gamma_energy=0.0, costfunction="Jtrace", initialcondition="basis", datadir=".", configfilename="config.cfg", print_frequency_iter=1):

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
    mystring += "datadir= " + datadir + "/data_out\n"
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

    outpath = datadir+"/"+configfilename
    with open(outpath, "w") as file:
        file.write(mystring)

    print("Quandary config file:", outpath)

    return configfilename

def execute(*, runtype="simulation", ncores=1, quandary_exec="./main", config_filename="config.cfg"):

    # Set up the run command
    if ncores > 1:
        runcommand = f"mpirun -np {ncores} {quandary_exec} {config_filename} --quiet"
    else:
        runcommand = f"{quandary_exec} {config_filename} --quiet"

    # # If not optimizing: Pipe std output to file rather than screen
    # if runtype == "simulation" or runtype == "gradient":
    #     with open(os.path.join(datadir, "out.log"), "w") as stdout_file, \
    #          open(os.path.join(datadir, "err.log"), "w") as stderr_file:
    #         exec = run(runcommand, shell=True, stdout=stdout_file, stderr=stderr_file)
    # else:
    with open("./err.log", "w") as stderr_file:
        exec = run(runcommand, shell=True, stderr=stderr_file)

    # Run Quandary
    exec.check_returncode()

def get_results(datadir="./"):
    # TODO: Output directory of quandary run. 
    dataout_dir = datadir + "/data_out"
    
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