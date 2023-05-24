import numpy as np

def get_resonances(is_ess, it2in, Ness, Nguard, Hsys, Hsym_ops, Hanti_ops, msb_order=True, cw_amp_thres, cw_prox_thres, rot_freq):
    Nosc = len(Hsym_ops)

    nrows = Hsys.shape[0]
    ncols = Hsys.shape[1]

    Nt = Ness + Nguard
    Ntot = np.prod(Nt)

    Hsys_evals, Utrans = np.linalg.eig(Hsys)
    Hsys_evals = Hsys_evals.real  # Eigenvalues may have a small imaginary part due to numerical precision

    ka_delta = Hsys_evals / (2 * np.pi)

    print("\nget_resonances: Ignoring couplings slower than (ad_coeff):", cw_amp_thres, "and frequencies closer than:", cw_prox_thres, "[GHz]")

    resonances = []
    speed = []
    for q in range(Nosc):
        Hctrl_ad = Hsym_ops[q] - Hanti_ops[q]
        Hctrl_ad_trans = np.dot(np.dot(Utrans.T, Hctrl_ad), Utrans)

        resonances_a = []
        speed_a = []
        
        print("\nResonances in oscillator #", q, "Ignoring transitions with ad_coeff <:", cw_amp_thres)
        for i in range(nrows):
            for j in range(i):
                if abs(Hctrl_ad_trans[i, j]) >= cw_amp_thres:
                    if is_ess[i] and is_ess[j]:
                        delta_f = ka_delta[i] - ka_delta[j]
                        if abs(delta_f) < 1e-10:
                            delta_f = 0.0
                        if not any(abs(delta_f - f) < cw_prox_thres for f in resonances_a):
                            resonances_a.append(delta_f)
                            speed_a.append(abs(Hctrl_ad_trans[i, j]))
                            print("resonance from (j-idx) =", it2in[j, :], "to (i-idx) =", it2in[i, :], ", lab-freq =", rot_freq[q] + delta_f, "= l_", i, "- l_", j, ", ad_coeff =", abs(Hctrl_ad_trans[i, j]))
        
        resonances.append(resonances_a)
        speed.append(speed_a)

    Nfreq = np.zeros(Nosc, dtype=int)
    om = [[] for _ in range(Nosc)]
    growth_rate = [[] for _ in range(Nosc)]
    
    for q in range(Nosc):
        Nfreq[q] = max(1, len(resonances[q]))  # at least one being 0.0
        om[q] = np.zeros(Nfreq[q])
        if len(resonances[q]) > 0:
            om[q] = np.array(resonances[q])
        growth_rate[q] = np.ones(Nfreq[q])
        if len(speed[q]) > 0:
            growth_rate[q] = np.array(speed[q])

    om = 2 * np.pi * om

    print()
    return om, growth_rate, Utrans