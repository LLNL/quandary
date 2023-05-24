import numpy as np

def identify_essential_levels(Ness, Nt, msb_order):
    assert len(Ness) == len(Nt)
    Nosc = len(Nt)

    Ntot = np.prod(Nt)
    it2in = np.zeros((Ntot, Nosc), dtype=np.int64)
    is_ess = np.zeros(Ntot, dtype=bool)

    if msb_order:
        t = tuple(range(1, x+1) for x in Nt)
    else:
        t = tuple(range(1, x+1)[::-1] for x in Nt)
    R = np.indices(t).reshape(Nosc, -1).T
    itot = 0

    if msb_order:
        for ind in R:
            itot += 1
            for j in range(Nosc):
                it2in[itot-1, j] = ind[j] - 1
    else:
        for ind in R:
            itot += 1
            ess = True
            for j in range(Nosc):
                it2in[itot-1, Nosc-j-1] = ind[j] - 1
                ess = ess and ind[j] <= Ness[Nosc-j-1]
            is_ess[itot-1] = ess

    return is_ess, it2in