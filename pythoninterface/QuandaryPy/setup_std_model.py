import numpy as np

def setup_std_model(Ne, Ng, f01, xi, couple_coeff, couple_type, rot_freq, T, D1, gate_final, maxctrl_MHz=10.0, msb_order=False, Pmin=40, init_amp_frac=0.0, randomize_init_ctrl=True, rand_seed=-1, pcofFileName="", zeroCtrlBC=True, use_eigenbasis=False, cw_amp_thres=5e-2, cw_prox_thres=2e-3, splines_real_imag=True):

    # enforce inequality constraint on the leakage?
    useLeakIneq = False # true
    leakThreshold = 1e-3

    # convert maxctrl_MHz to rad/ns per frequency
    # This is (approximately) the max amplitude of each control function (p & q)
    maxctrl_radns = 2 * np.pi * maxctrl_MHz * 1e-3

    pdim = len(Ne)

    # General case
    Hsys, Hsym_ops, Hanti_ops = hamiltonians(Nsys=pdim, Ness=Ne, Nguard=Ng, freq01=f01, anharm=xi, rot_freq=rot_freq, couple_coeff=couple_coeff, couple_type=couple_type, msb_order=msb_order)

    is_ess, it2in = identify_essential_levels(Ne, Ne+Ng, msb_order)

    om, growth_rate, Utrans = get_resonances(is_ess, it2in, Ness=Ne, Nguard=Ng, Hsys=Hsys, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, msb_order=msb_order, cw_amp_thres=cw_amp_thres, cw_prox_thres=cw_prox_thres, rot_freq=rot_freq)

    Ness = np.prod(Ne)
    Nosc = len(om)
    assert Nosc == pdim  # Nosc must equal pdim
    Nfreq = np.zeros(Nosc, dtype=int)  # Number of frequencies per control Hamiltonian
    for q in range(Nosc):
        Nfreq[q] = len(om[q])

    print("D1 =", D1, " Ness =", Ness, " Nosc =", Nosc, " Nfreq =", Nfreq)

    # Amplitude bounds to be imposed during optimization
    maxAmp = np.full(Nosc, maxctrl_radns)  # internally scaled by 1/(sqrt(2)*Nfreq[q]) in setup_ipopt() and Quandary

    # allocate and sort the vectors (ascending order)
    om_p = [[]] * Nosc
    growth_rate_p = [[]] * Nosc
    use_p = [[]] * Nosc
    for q in range(Nosc):
        om_p[q] = np.zeros(Nfreq[q])
        growth_rate_p[q] = np.zeros(Nfreq[q])
        use_p[q] = np.zeros(Nfreq[q], dtype=int)  # By default, don't use any freq's
        p = np.argsort(om[q])  # sort indices based on om[q]
        om_p[q][:] = om[q][p]
        growth_rate_p[q][:] = growth_rate[q][p]

    print("Rotfreq =", rot_freq)
    print("omp =", om_p)
    print("growth_rate =", growth_rate_p)

    print("Sorted CW freq's:")
    for q in range(Nosc):
        print("Ctrl Hamiltonian #", q, ", lab frame carrier frequencies:", rot_freq[q] + om_p[q] / (2 * np.pi), "[GHz]")
        print("Ctrl Hamiltonian #", q, ",                   growth rate:", growth_rate_p[q], "[1/ns]")

    # Try to identify groups of almost equal frequencies
    for q in range(Nosc):
        seg = 0
        rge_q = np.max(om_p[q]) - np.min(om_p[q])  # this is the range of frequencies
        k0 = 0
        for k in range(1, Nfreq[q]):
            delta_k = om_p[q][k] - om_p[q][k0]
            if delta_k > 0.1 * rge_q:
                seg += 1
                # find the highest rate within the range [k0,k-1]
                rge = range(k0, k)
                om_avg = np.sum(om_p[q][rge]) / len(rge)
                print("Osc #", q, "segment #", seg, "Freq-range:", (np.max(om_p[q][rge]) - np.min(om_p[q][rge])) / (2 * np.pi), "Freq-avg:", om_avg / (2 * np.pi) + rot_freq[q])
                use_p[q][k0] = 1
                # average the cw frequency over the segment
                om_p[q][k0] = om_avg
                k0 = k  # start a new group
        # find the highest rate within the last range [k0,Nfreq[q]]
        seg += 1
        rge = range(k0, Nfreq[q])
        om_avg = np.sum(om_p[q][rge]) / len(rge)
        print("Osc #", q, "segment #", seg, "Freq-range:", (np.max(om_p[q][rge]) - np.min(om_p[q][rge])) / (2 * np.pi), "Freq-avg:", om_avg / (2 * np.pi) + rot_freq[q])
        use_p[q][k0] = 1
        om_p[q][k0] = om_avg

        # cull out unused frequencies
        om[q] = np.zeros(np.sum(use_p[q]))
        growth_rate[q] = np.zeros(np.sum(use_p[q]))
        j = 0
        for k in range(Nfreq[q]):
            if use_p[q][k] == 1:
                j += 1
                om[q][j] = om_p[q][k]
                growth_rate[q][j] = growth_rate_p[q][k]
        Nfreq[q] = j  # correct the number of CW frequencies for oscillator 'q'

    print("\nSorted and culled CW freq's:")
    for q in range(Nosc):
        print("Ctrl Hamiltonian #", q, ", lab frame carrier frequencies:", rot_freq[q] + om[q] / (2 * np.pi), "[GHz]")
        print("Ctrl Hamiltonian #", q, ",                   growth rate:", growth_rate[q], "[1/ns]")

    # Set the initial condition: Basis with guard levels
Ubasis = initial_cond_general(is_ess, Ne, Ng)

# NOTE:
# To impose the target transformation in the eigenbasis, keep the Hamiltonians the same
# but change the target to be Utrans*Ubasis*gate_final

if use_eigenbasis:
    Utarget = Utrans * Ubasis * gate_final
else:
    Utarget = Ubasis * gate_final

# use_diagonal_H0 = False  # For comparisson with Quandary: use original Hamiltonian
# if use_diagonal_H0 # transformation to diagonalize the system Hamiltonian
#   transformHamiltonians!(Hsys, Hsym_ops, Hanti_ops, Utrans) 
# end

if splines_real_imag:
    nCoeff = 2 * D1 * sum(Nfreq)  # factor '2' is for Re/Im parts of ctrl vector
else:
    nCoeff = (D1 + 1) * sum(Nfreq)  # Use a variable amplitude with a fixed phase

# Set up the initial control parameter
pcof0 = init_control(amp_frac=init_amp_frac, maxAmp=maxAmp, D1=D1, Nfreq=Nfreq, startFile=pcofFileName, seed=rand_seed, randomize=randomize_init_ctrl, growth_rate=growth_rate, splines_real_imag=splines_real_imag)

# Estimate time step based on the number of time steps per shortest period

# Note: calculate_timestep expects maxCoupled to have Nosc elements
nsteps = calculate_timestep(T, Hsys, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, maxCoupled=maxAmp, Pmin=Pmin)
print("Starting point: nsteps =", nsteps, " maxAmp =", maxAmp, "[rad/ns]")

# create a linear solver object
linear_solver = Juqbox.lsolver_object(solver=Juqbox.JACOBI_SOLVER, max_iter=100, tol=1e-12, nrhs=np.prod(Ne))

# create diagonal W-matrix with weights for suppressing leakage
wmatScale = 1.0
# w_diag_mat = wmatScale * wmatsetup_old(Ne, Ng, msb_order)
w_diag_mat = wmatsetup(is_ess, it2in, Ne, Ng)

# println("norm(wmat1 - wmat2): ", norm(w_diag_mat-w_diag_2))
# println("w_diag_1: ", diag(w_diag_mat))
# println("w_diag_2: ", diag(w_diag_2))
# println("differen: ", diag(w_diag_mat-w_diag_2))

# Set up parameter struct using the free evolution target
if useLeakIneq:
    params = Juqbox.objparams(Ne, Ng, T, nsteps, Uinit=Ubasis, Utarget=Utarget, Cfreq=om, Rfreq=rot_freq, Hconst=Hsys, w_diag_mat=w_diag_mat, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, linear_solver=linear_solver, objFuncType=3, leak_ubound=leakThreshold, nCoeff=nCoeff, msb_order=msb_order)
else:
    params = Juqbox.objparams(Ne, Ng, T, nsteps, Uinit=Ubasis, Utarget=Utarget, Cfreq=om, Rfreq=rot_freq, Hconst=Hsys, w_diag_mat=w_diag_mat, Hsym_ops=Hsym_ops, Hanti_ops=Hanti_ops, linear_solver=linear_solver, nCoeff=nCoeff, freq01=f01, self_kerr=xi, couple_coeff=couple_coeff, couple_type=couple_type, msb_order=msb_order, zeroCtrlBC=zeroCtrlBC)

print("*** Settings ***")
print("Number of coefficients per spline =", D1, "Total number of control parameters =", len(pcof0))
print()
print("Returning problem setup as a tuple (params, pcof0, maxAmp)")
print("params::objparams: object holding the Hamiltonians, carrier freq's, time-stepper, etc")
print("pcof0:: Vector{Float64}: Initial coefficient vector is stored in 'pcof0' vector")
print("maxAmp:: Vector{Float64}: Approximate max control amplitude for the p(t) and q(t) control function for each control Hamiltonian")
print("")


return params, pcof0, maxAmp
