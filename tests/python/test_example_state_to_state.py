import os
import pytest
import numpy as np
from quandary import Quandary
from utils import assert_results_equal

# Mark all tests in this file as regression tests
pytestmark = pytest.mark.regression

EXPECTED_LENGTH = 1652
EXPECTED_INFIDELITY = 9.002940985003427e-05

EXPECTED_PT = [
    [
        -0.430435057899144, 0.518899252233085, 0.0553961310496238, -2.73778883011829, 0.836879517793378,
        1.68236101443972, 1.2272994947870701, -1.13889765832405, 2.7441415074662903, 1.68020127607366
    ],
]

EXPECTED_QT = [
    [
        -1.21262883183835, -2.77343358982429, -2.7674896542845198, -2.81712074264336, -2.71698129305364,
        -2.7908352239901797, -2.65458260105392, -2.57461802357684, -2.0247101898358, -1.26152275887394
    ],
]

EXPECTED_ENERGY = [
    [
        [
            0.0, 0.00694252103065461, 0.0331419031652123, 0.0744089523543541, 0.137405174039168,
            0.205169499219536, 0.282367422349298, 0.368813810556003, 0.455572996577133, 0.496383344017663
        ],
    ],
]

EXPECTED_POPULATION = [
    [
        [
            1.0, 0.99305963258305, 0.966873105642978, 0.925633731947042, 0.862637301677907,
            0.794942546965401, 0.717715083633589, 0.631227300284134, 0.54475917511335, 0.503663559455785
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
