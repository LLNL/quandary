import numpy as np
import random

def init_control(*, initctrl_MHz=None, nsplines=None, carrierfreq=None, startFile="", rand_seed=-1, randomize=True, verbose=True):
    if verbose:
        print("*** Starting from ", "RANDOM" if randomize else "CONSTANT", "control vector with max. amplitude=", initctrl_MHz, "MHz" if len(startFile)==0 else "B-spline coefficients in file:", startFile)
 
    # Load the initial parameters from a file?
    if len(startFile) > 0:
        # use if you want to read the initial coefficients from file
        pcof0 = np.loadtxt(startFile).flatten()  # TODO: Check if this is working!
        assert nparams == len(pcof0)
    else: 
        # Compute the total number of control parameters
        nqubits = len(carrierfreq)
        nparams = sum([2*nsplines*len(carrierfreq[q]) for q in range(nqubits)])

        # Set controls for each oscillator
        offc = 0
        pcof0 = np.zeros(nparams)
        for q in range(nqubits):

            nparams_q = 2 * nsplines * len(carrierfreq[q]) 
            maxamp = 2.0*np.pi/1e+3 * initctrl_MHz[q] / np.sqrt(2) / len(carrierfreq[q])

            if randomize:
                # Set random seed, if given
                if rand_seed >= 0:
                    random.seed(rand_seed)
                myparams = maxamp * 2.0 * (np.random.rand(nparams_q) - 0.5)
            else:
                myparams = maxamp * np.ones(nparams_q)

            pcof0[offc:offc + nparams_q] = myparams
            offc += nparams_q

    if verbose:
        print("Number of control parameters: ", len(pcof0))

    return pcof0