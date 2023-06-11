import os
import numpy as np
from quandary import preprocess
from quandary import interface


def run(Ne, Ng, freq01, selfkerr, crosskerr, Jkl, rotfreq, maxctrl_MHz, T, initctrl_MHz, rand_seed, randomize_init_ctrl, targetgate, *, dtau=3.33, Pmin=80, cw_amp_thres=6e-2, cw_prox_thres=1e-3, datadir=".", tol_infidelity=1e-3, tol_costfunc=1e-3, maxiter=100, gamma_tik0=1e-4, gamma_energy=1e-2, costfunction="Jtrace", initialcondition="basis", T1=None, T2=None, runtype="simulation", ncores=0, quandary_exec="./main", print_frequency_iter=1, verbose=False):

    # Create quandary data directory
    os.makedirs(datadir, exist_ok=True)

    # Set up Hamiltonians in essential levels only
    Hsys, Hc_re, Hc_im = preprocess.hamiltonians(Ne, freq01, selfkerr, crosskerr, Jkl, rotfreq=rotfreq)

    # Estimate number of time steps
    nsteps = preprocess.estimate_timesteps(T, Hsys, Hc_re, Hc_im, maxctrl_MHz, Pmin=Pmin)
    if verbose:
        print("Final time: ",T,"ns, Number of timesteps: ", nsteps,", dt=", T/nsteps, "ns")
        print("Maximum control amplitudes: ", maxctrl_MHz, "MHz")


    # Estimate carrier wave frequencies
    carrierfreq, growth_rate = preprocess.get_resonances(Ne, Hsys, Hc_re, Hc_im, verbose=verbose) 
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
    config_filename = interface.write_config(Ne=Ne, Ng=Ng, T=T, nsteps=nsteps, freq01=freq01, rotfreq=rotfreq, selfkerr=selfkerr, crosskerr=crosskerr, Jkl=Jkl, nsplines=nsplines, carrierfreq=carrierfreq, tol_infidelity=tol_infidelity, tol_costfunc=tol_costfunc, maxiter=maxiter, maxctrl_MHz=maxctrl_MHz, initctrl_MHz=initctrl_MHz, randomize_init_ctrl=randomize_init_ctrl, gamma_tik0=gamma_tik0, gamma_energy=gamma_energy, costfunction=costfunction, initialcondition=initialcondition, T1=T1, T2=T2, runtype=runtype, gatefilename="./targetgate.dat", print_frequency_iter=print_frequency_iter, datadir=datadir)


    # Call Quandary
    err = interface.execute(runtype=runtype, ncores=ncores, quandary_exec=quandary_exec, config_filename=config_filename, datadir=datadir)

    # Get results and return
    popt, infidelity, optim_hist = interface.get_results(datadir)

    return popt, infidelity, optim_hist