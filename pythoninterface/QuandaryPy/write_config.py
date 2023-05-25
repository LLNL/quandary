import numpy as np

def write_Quandary_config_file(Ne, Ng, T, nsteps, freq01, rotfreq, selfkerr, crosskerr, Jkl, T1, T2, D1, carrierfreq, gatefilename, initialpcof_filename, optim_bounds, inftol, maxiter, tik0, leakage_weights, print_frequency_iter, runtype, gamma_dpdm, final_objective, gamma_energy, *, datadir=".", configfilename="config.cfg"):
    # final_objective = 1 uses the trace infidelity;
    # final_objective = 2 uses the Frobenius norm squared
    # final_objective = 3 for the Jmeasure 
    # gamma_dpdm > 0 to suppress 2nd time-derivative of the population


    # TODO: 
    #  * defaults
    #  * Create data dir if does not exists!

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
    decay = False
    if len(T1) > 0: 
        decay = True
        mystring += "decay_time = " + str(T1)[1:-1] + "\n"
    dephase = False
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
    mystring += "initialcondition=basis\n"

    # choose between having splines for both the real & imaginary parts, or only for the amplitude with a fixed phase
    for iosc in range(1, len(Ne) + 1):
        mystring += "control_segments" + str(iosc - 1) + " = spline, " + str(D1) + "\n"
        mystring += "control_initialization" + str(iosc - 1) + " = file, ./" + initialpcof_filename + "\n"
        mystring += "control_bounds" + str(iosc - 1) + " = " + str(optim_bounds[iosc - 1]) + "\n"
        mystring += "carrier_frequency" + str(iosc - 1) + " = "
        omi = carrierfreq[iosc - 1]
        for j in range(len(omi)):
            mystring += str(omi[j]) + ", " 
        mystring += "\n"
    mystring += "optim_target = gate, fromfile, " + gatefilename + "\n"
    if final_objective == 1:
        mystring += "optim_objective = Jtrace\n"
    elif final_objective == 2:
        mystring += "optim_objective = Jfrobenius\n"
    elif final_objective == 3:
        mystring += "optim_objective = Jmeasure\n"
    else:
        print("ERROR: Objective type not defined\n")
        stop
    mystring += "gate_rot_freq = 0.0\n"
    mystring += "optim_weights= 1.0\n"
    mystring += "optim_atol= 1e-5\n"
    mystring += "optim_rtol= 1e-4\n"
    mystring += "optim_ftol= " + str(inftol) + "\n"
    mystring += "optim_inftol= " + str(inftol) + "\n"
    mystring += "optim_maxiter= " + str(maxiter) + "\n"
    mystring += "optim_regul= " + str(tik0) + "\n"
    mystring += "optim_penalty= 1.0\n"
    mystring += "optim_penalty_param= 0.0\n"
    ninitscale = np.prod(Ne)
    mystring += "optim_leakage_weights= " + str([w * ninitscale for w in leakage_weights])[1:-1] + "\n"
    mystring += "optim_regul_dpdm= " + str(gamma_dpdm) + "\n"
    mystring += "optim_penalty_energy= " + str(gamma_energy) + "\n"
    mystring += "datadir= " + datadir + "/data_out\n"
    for iosc in range(1, len(Ne) + 1):
        mystring += "output" + str(iosc - 1) + "=expectedEnergy, population, fullstate\n"
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