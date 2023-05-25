import numpy as np
import random
from scipy.linalg import eigvals


def assign_thresholds(params, D1, maxAmp):
    Nfreq = params.Nfreq
    Ncoupled = params.Ncoupled
    Nunc = params.Nunc
    assert Nunc == 0

    NfreqTot = params.NfreqTot
    nCoeff = 2 * D1 * NfreqTot
    minCoeff = np.zeros(nCoeff)  # Initialize storage
    maxCoeff = np.zeros(nCoeff)

    baseOffset = 0
    for c in range(1, Ncoupled + 1):  # We assume that either Nunc = 0 or Ncoupled = 0
        for f in range(1, Nfreq[c] + 1):
            offset1 = baseOffset + (f - 1) * 2 * D1
            bound = maxAmp[c] / (np.sqrt(2) * Nfreq[c])  # Divide bounds equally between the carrier frequencies for each control
            minCoeff[offset1:offset1 + 2 * D1] = -bound  # same for p(t) and q(t)
            maxCoeff[offset1:offset1 + 2 * D1] = bound
        baseOffset += 2 * D1 * Nfreq[c]

    return minCoeff, maxCoeff    


def init_control(amp_frac, maxAmp, D1, Nfreq, startFile="", seed=-1, randomize=True, growth_rate=None, splines_real_imag=True):
    Nosc = len(Nfreq)
    if splines_real_imag:
        nCoeff = 2 * D1 * sum(Nfreq)
    else:
        nCoeff = (D1 + 1) * sum(Nfreq)

    # initial parameter guess: from file?
    if len(startFile) > 0:
        # use if you want to read the initial coefficients from file
        pcof0 = np.loadtxt(startFile).flatten()  # change to jld2?
        print("*** Starting from B-spline coefficients in file:", startFile)
        assert nCoeff == len(pcof0)
    else:
        if seed >= 0:
            random.seed(seed)

        pcof0 = np.zeros(nCoeff)
        offc = 0

        if randomize:
            if splines_real_imag:
                for q in range(Nosc):
                    if Nfreq[q] > 0:
                        maxrand = amp_frac * maxAmp[q] / np.sqrt(2) / Nfreq[q]
                        Nq = 2 * D1 * Nfreq[q]
                        pcof0[offc:offc + Nq] = maxrand * 2 * (np.random.rand(Nq) - 0.5)
                        offc += Nq
            else:
                for q in range(Nosc):
                    for k in range(Nfreq[q]):
                        maxrand = amp_frac * maxAmp[q] / np.sqrt(2) / Nfreq[q]
                        pcof0[offc:offc + D1] = maxrand * 2 * (np.random.rand(D1) - 0.5)
                        pcof0[offc + D1] = 2 * np.pi * (np.random.rand(1)[0] - 0.5)  # [-pi, pi]
                        offc += (D1 + 1)
            print("*** Starting from RANDOM control vector with amp_frac =", amp_frac)
        else:  # picewise constant with amplitude depending on scaled growth rate
            max_rate = 0.0
            for q in range(Nosc):
                max_rate = max(max_rate, np.max(growth_rate[q]))
            print("max_rate =", max_rate)
            if splines_real_imag:
                for q in range(Nosc):
                    for k in range(Nfreq[q]):
                        const_amp = amp_frac * maxAmp[q] / np.sqrt(2) / Nfreq[q] * max_rate / (growth_rate[q][k])
                        pcof0[offc:offc + 2 * D1] = const_amp
                        offc += 2 * D1
            else:
                for q in range(Nosc):
                    for k in range(Nfreq[q]):
                        const_amp = amp_frac * maxAmp[q] / np.sqrt(2) / Nfreq[q] * max_rate / (growth_rate[q][k])
                        pcof0[offc:offc + D1] = const_amp
                        pcof0[offc + D1] = 0.0  # zero phase ok?
                        offc += (D1 + 1)
            print("*** Starting from PIECEWISE CONSTANT control vector with amp_frac =", amp_frac)

    return pcof0


def calculate_timestep(T, H0, Hsym_ops=[], Hanti_ops=[], Hunc_ops=[], maxCoupled=[], maxUnc=[], Pmin=40):
    Ncoupled = len(Hsym_ops)
    Nunc = len(Hunc_ops)
    assert len(maxCoupled) >= Ncoupled
    assert len(maxUnc) >= Nunc

    K1 = np.copy(H0)  # system Hamiltonian

    # Test: Only compute eigenvalues of the system Hamiltonian. Then increase the sample rate by a factor > 1 to compensate for the control Hamiltonian: Tends to underestimate the number of time steps

    # Coupled control Hamiltonians
    for i in range(Ncoupled):
        K1 += maxCoupled[i] * Hsym_ops[i] + 1j * maxCoupled[i] * Hanti_ops[i]

    # Uncoupled control Hamiltonians
    for i in range(Nunc):
        if np.allclose(Hunc_ops[i], Hunc_ops[i].T):
            K1 += maxUnc[i] * Hunc_ops[i]
        elif np.linalg.norm(Hunc_ops[i] + Hunc_ops[i].T) < 1e-14:
            K1 += 1j * maxUnc[i] * Hunc_ops[i]
        else:
            raise ValueError("Uncoupled Hamiltonians must currently be either symmetric or anti-symmetric.\n")

    # Estimate time step
    lamb = eigvals(K1)
    maxeig = np.max(np.abs(lamb))
    mineig = np.min(np.abs(lamb))

    ctrlFactor = 1.2  # Heuristic, assuming that the total Hamiltonian is dominated by the system part.
    samplerate1 = ctrlFactor * maxeig * Pmin / (2 * np.pi)
    nsteps = int(np.ceil(T * samplerate1))

    # NOTE: The above estimate does not account for quickly varying signals or a large number of splines.
    # Double check at least 2-3 points per spline to resolve control function.

    return nsteps