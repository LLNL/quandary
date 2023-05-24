import os
import numpy as np
from subprocess import run, PIPE

def run_Quandary(params, pcof0, optim_bounds, runIdx=3, maxIter=100, ncores=1, quandary_exec="./main", print_frequency_iter=1, gamma_dpdm=0.0, gamma_energy=0.0, final_objective=1, splines_real_imag=True, phase_scaling_factor=1.0, datadir="./"):
    # gamma_dpdm > 0.0 to penalize the 2nd time derivative of the population
    # final_objective = 1 corresponds to the trace infidelity

    if runIdx == 1:
        runtype = "simulation"
    elif runIdx == 2:
        runtype = "gradient"
    else:
        runtype = "optimization"

    Nt = params.Nt
    Ne = params.Ne
    T = params.T
    nsteps = params.nsteps
    couple_type = params.couple_type

    rotfreq = params.Rfreq
    carrierfreq = params.Cfreq

    tikQ = 2 * params.tik0 / params.nCoeff  # Scale the tikhonov coefficient

    leakage_weights = np.diag(params.wmat)  # Scaling of leakage contribution

    if splines_real_imag:
        D1 = params.nCoeff // (2 * params.NfreqTot)
        assert 2 * D1 * params.NfreqTot == params.nCoeff  # no remainder is allowed
    else:
        D1p1 = params.nCoeff // params.NfreqTot
        assert D1p1 * params.NfreqTot == params.nCoeff  # no remainder is allowed
        D1 = D1p1 - 1

    freq01 = params.freq01
    selfkerr = -params.self_kerr  # self-Kerr coefficient (Quandary reverses the sign)
    if couple_type == 1:
        couple_coeff = -params.couple_coeff  # cross-Kerr coefficient (Quandary reverses the sign)
    else:
        couple_coeff = params.couple_coeff  # cross-Kerr coefficient (Quandary reverses the sign)

    inftol = params.traceInfidelityThreshold

    isEss, it2in = Juqbox.identify_essential_levels(Ne, Nt, False)  # Quandary uses LSB ordering
    Ness = params.N
    Ntot = np.prod(params.Nt)
    targetgate = np.zeros((Ness, Ness), dtype=np.complex128)
    i0 = 0
    for i in range(Ntot):
        if isEss[i]:
            i0 += 1
            targetgate[i0-1, :] = params.Utarget_r[i, :] + 1j * params.Utarget_i[i, :]

    # Create working directory
    os.makedirs(datadir, exist_ok=True)

    # Write gate to file
    gatefilename = os.path.join(datadir, "targetgate.dat")
    assert targetgate.shape[0] == np.prod(Ne)
    assert targetgate.shape[1] == np.prod(Ne)
    gate_1d = np.concatenate((np.real(targetgate).ravel(), np.imag(targetgate).ravel()))
    with open(gatefilename, "w") as f:
        for value in gate_1d:
            f.write("{:20.13e}\n".format(value))

    # Write initial pcof to file
    initialpcof_filename = os.path.join(datadir, "pcof_init.dat")
    with open(initialpcof_filename, "w") as f:
        for value in pcof0:
            f.write("{:20.13e}\n".format(value))

    # Write Quandary's configuration file
    config_filename = os.path.join(datadir, "config.cfg")
    write_Quandary_config_file(config_filename, Nt, Ne, T, nsteps, freq01, rotfreq, selfkerr, couple_coeff, couple_type,
                               D1, carrierfreq, gatefilename, initialpcof_filename, optim_bounds, inftol, maxIter,
                               tikQ, leakage_weights, print_frequency_iter, runtype=runtype, gamma_dpdm=gamma_dpdm,
                               final_objective=final_objective, gamma_energy=gamma_energy,
                               splines_real_imag=splines_real_imag, phase_scaling_factor=phase_scaling_factor,
                               datadir=datadir)

    # Set up the run command
    if ncores > 1:
        runcommand = f"mpirun -np {ncores} {quandary_exec} {config_filename} --quiet"
    else:
        runcommand = f"{quandary_exec} {config_filename} --quiet"

    # If not optimizing: Pipe std output to file rather than screen
    if runtype == "simulation" or runtype == "gradient":
        with open(os.path.join(datadir, "out.log"), "w") as stdout_file, \
             open(os.path.join(datadir, "err.log"), "w") as stderr_file:
            exec = run(runcommand, shell=True, stdout=stdout_file, stderr=stderr_file)
    else:
        with open(os.path.join(datadir, "err.log"), "w") as stderr_file:
            exec = run(runcommand, shell=True, stderr=stderr_file)

    # Run Quandary
    exec.check_returncode()

    return get_Quandary_results(params, os.path.join(datadir, "data_out"), params.Nt, params.Ne, runtype=runtype)