def write_Quandary_config_file(configfilename, Nt, Ne, T, nsteps, freq01, rotfreq, selfkerr, couple_coeff, couple_type, D1, carrierfreq, gatefilename, initialpcof_filename, optim_bounds, inftol, maxiter, tik0, leakage_weights, print_frequency_iter, runtype="optimization", gamma_dpdm=0.0, final_objective=1, gamma_energy=0.0, splines_real_imag=True, phase_scaling_factor=1.0, datadir="./"):
    # final_objective = 1 uses the trace infidelity;
    # final_objective = 2 uses the Frobenius norm squared
    # gamma_dpdm > 0 to suppress 2nd time-derivative of the population
    assert final_objective == 1 or final_objective == 2

    mystring = "nlevels = " + str(Nt)[1:-1] + "\n"
    mystring += "nessential= " + str(Ne)[1:-1] + "\n"
    mystring += "ntime = " + str(nsteps) + "\n"
    mystring += "dt = " + str(T / nsteps) + "\n"
    mystring += "transfreq = " + str(freq01)[1:-1] + "\n"
    mystring += "rotfreq= " + str(rotfreq)[1:-1] + "\n"
    mystring += "selfkerr = " + str(selfkerr)[1:-1] + "\n"
    if couple_type == 1:
        mystring += "crosskerr= " + str(couple_coeff)[1:-1] + "\n"
        mystring += "Jkl= 0.0\n"
    else:
        mystring += "crosskerr= 0.0\n"
        mystring += "Jkl= " + str(couple_coeff)[1:-1] + "\n"
    mystring += "collapse_type=none\n"
    mystring += "initialcondition=basis\n"

    # choose between having splines for both the real & imaginary parts, or only for the amplitude with a fixed phase
    for iosc in range(1, len(Ne) + 1):
        if splines_real_imag:
            mystring += "control_segments" + str(iosc - 1) + " = spline, " + str(D1) + "\n"
        else:
            mystring += "control_segments" + str(iosc - 1) + " = spline_amplitude, " + str(D1) + ", " + str(
                phase_scaling_factor) + "\n"
        mystring += "control_initialization" + str(iosc - 1) + " = file, ./" + initialpcof_filename + "\n"
        mystring += "control_bounds" + str(iosc - 1) + " = " + str(optim_bounds[iosc - 1]) + "\n"
        mystring += "carrier_frequency" + str(iosc - 1) + " = "
        omi = carrierfreq[iosc - 1]
        mystring += ", ".join(str(omi[j] / (2 * np.pi)) for j in range(len(omi))) + "\n"

    mystring += "optim_target = gate, fromfile, " + gatefilename + "\n"
    if final_objective == 1:
        mystring += "optim_objective = Jtrace\n"
    elif final_objective == 2:
        mystring += "optim_objective = Jfrobenius\n"
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
    mystring += "usematfree = true\n"
    mystring += "linearsolver_type = gmres\n"
    mystring += "linearsolver_maxiter = 20\n"
    mystring += "np_init = " + str(np.prod(Ne)) + "\n"

    with open(configfilename, "w") as file:
        file.write(mystring)

    #print("Quandary config file:", configfilename)