import os
import pytest
import numpy as np
from quandary import Quandary
from utils import assert_results_equal

# Mark all tests in this file as regression tests
pytestmark = pytest.mark.regression

EXPECTED_LENGTH = 1652
EXPECTED_INFIDELITY = 8.695620910992297e-06

EXPECTED_PT = [
    [
        -0.351017944518694, -0.9111853674706331, 1.56337057296288, 1.32102312178454, -2.17680392763229,
        -1.91465552269025, -2.40030147272628, 2.44449966044797, 2.09663827839874, -1.0145362400273399
    ],
]

EXPECTED_QT = [
    [
        -0.35489683713910297, -2.8226401868623903, -2.64838539616493, -1.78300097541017, -2.75117332619166,
        -2.7605889552578198, -2.5975353420470504, -2.74733785636068, -2.76505669888879, -2.09765274564999
    ],
]

EXPECTED_ENERGY = [
    [
        [
            0.0, 0.00692816781265531, 0.0328301908913508, 0.0695204206507133, 0.115813024750246,
            0.172087630256885, 0.251613716534252, 0.336745657239306, 0.426441526468596, 0.502139791825633
        ],
    ],
]

EXPECTED_POPULATION = [
    [
        [
            1.0, 0.993078206283365, 0.967173742450001, 0.930495435459176, 0.884259384596979,
            0.828002837889172, 0.748574947700958, 0.663432310188568, 0.573644729851791, 0.497863551562967
        ],
    ],
]

# Compare output to expected result for 10 points
NUM_SAMPLES = 10
SAMPLE_INDICES = [int(i * (EXPECTED_LENGTH - 1) / (NUM_SAMPLES - 1)) for i in range(NUM_SAMPLES)]


def test_example_state_to_state(mpi_exec, tmp_path, request):
    """Test state-to-state preparation using Python interface."""
    datadir_path = os.path.join(tmp_path, request.node.name)

    Ne = [2]
    Ng = [1]
    freq01 = [4.10595]
    selfkerr = [0.2198]
    T = 50.0
    maxctrl_MHz = 4.0
    initialcondition = [1.0, 0.0]
    targetstate = [1.0/np.sqrt(2), 1.0/np.sqrt(2)]
    n_osc = 1
    n_levels = 1

    quandary = Quandary(
        Ne=Ne,
        Ng=Ng,
        freq01=freq01,
        selfkerr=selfkerr,
        maxctrl_MHz=maxctrl_MHz,
        initialcondition=initialcondition,
        targetstate=targetstate,
        T=T,
        tol_infidelity=1e-5,
        rand_seed=4321,
        verbose=False
    )

    t, pt, qt, infidelity, energy, population = quandary.optimize(
        mpi_exec=mpi_exec,
        maxcores=1,
        datadir=datadir_path,
    )

    assert_results_equal(
        t=t,
        pt=pt,
        qt=qt,
        infidelity=infidelity,
        energy=energy,
        population=population,
        T=T,
        n_osc=n_osc,
        n_levels=n_levels,
        expected_length=EXPECTED_LENGTH,
        expected_infidelity=EXPECTED_INFIDELITY,
        expected_pt=EXPECTED_PT,
        expected_qt=EXPECTED_QT,
        expected_energy=EXPECTED_ENERGY,
        expected_population=EXPECTED_POPULATION,
        sample_indices=SAMPLE_INDICES
    )
