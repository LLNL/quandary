import numpy as np

def get_Quandary_results(params, datadir, Nt, Ne, runtype="simulation"):
    # Get pcof
    pcof = np.loadtxt(datadir + "/params.dat")[:, 0].astype(float)

    # Get Infidelity, norm of gradient and iterations taken
    optim_hist = np.loadtxt(datadir + "/optim_history.dat")
    nOptimIter = optim_hist.shape[0] - 1

    objective_last = optim_hist[-1, 1]
    infid_last = optim_hist[-1, 5]
    tikhonov_last = optim_hist[-1, 6]
    penalty_last = optim_hist[-1, 8]  # dpdm penalty

    # copy history arrays to the params structure for post processing
    params.saveConvHist = True
    params.objHist = optim_hist[1:, 1].copy()
    params.dualInfidelityHist = optim_hist[1:, 2].copy()
    params.primaryHist = optim_hist[1:, 5].copy()
    params.secondaryHist = optim_hist[1:, 7]  # penalty = leakage

    # print("Quandary result: Infidelity=", infid_last)

    # Get last time-step unitary
    uT = np.zeros((np.prod(Nt), np.prod(Ne)), dtype=np.complex128)

    for i in range(np.prod(Ne)):
        # Read from file
        xre = np.loadtxt(datadir + "/rho_Re.iinit" + str(i).zfill(4) + ".dat")[-1, 1:]
        xim = np.loadtxt(datadir + "/rho_Im.iinit" + str(i).zfill(4) + ".dat")[-1, 1:]
        uT[:, i] = xre + 1j * xim

    grad = np.zeros(len(pcof))
    if runtype == "gradient":  # the grad.dat file is not created by the optimization mode
        # chop up the long vector into individual column vectors for the result
        grad = np.loadtxt(datadir + "/grad.dat")[:, 0]

    # make the return args similar to traceobjgrad()
    if runtype == "simulation":
        return infid_last + tikhonov_last + penalty_last, infid_last, penalty_last, uT
    elif runtype == "gradient":
        gradnorm = np.linalg.norm(grad) / np.sqrt(len(grad))
        return infid_last + tikhonov_last + penalty_last, grad, infid_last, penalty_last, 1.0 - infid_last
    else:  # "optimization"
        # similar to run_optimizer()
        return pcof, infid_last, penalty_last, tikhonov_last, params.objHist, nOptimIter