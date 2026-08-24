import os
import pytest
import numpy as np
from quandary import Quandary
from utils import assert_results_equal
from utils import print_expected_values

# Mark all tests in this file as regression tests
pytestmark = pytest.mark.regression

EXPECTED_LENGTH = 401
EXPECTED_INFIDELITY = 0.9630415175272145

EXPECTED_PT = [
    [
        10.0, -10.8398480569392, -8.095412755539291, 12.5190612632368, 5.45298111312658,
        -13.6291109621782, -2.56263648501672, 14.1195301285843, -0.44421521492720695, -13.9680224666742
    ],
]

EXPECTED_QT = [
    [
        10.0, 9.0828241259243, -11.5958739350448, -6.57822963169608, 13.0485630235626,
        3.7745641312123497, -13.9080154675518, -0.799292779899325, 14.135157333501, -2.21231742082474
    ],
]

EXPECTED_ENERGY = [
    [
        [
            0.0, 0.0376013589789126, 0.144137894175883, 0.303807239454152, 0.483767304122812,
            0.659906942899169, 0.759734611084046, 0.75719321030616, 0.684487663596382, 0.524921683025195
        ],
        [
            1.0, 0.962398641021087, 0.855862105824117, 0.696192760545848, 0.516232695877188,
            0.34009305710083, 0.240265388915953, 0.24280678969384, 0.315512336403618, 0.475078316974805
        ],
    ],
]

EXPECTED_POPULATION = [
    [
        [
            1.0, 0.962398641021087, 0.855862105824116, 0.696192760545847, 0.516232695877188,
            0.34009305710083, 0.240265388915952, 0.242806789693839, 0.315512336403617, 0.475078316974804
        ],
        [
            0.0, 0.0376013589789126, 0.144137894175883, 0.303807239454152, 0.483767304122812,
            0.65990694289917, 0.759734611084047, 0.757193210306161, 0.684487663596382, 0.524921683025195
        ],
    ],
]

# Compare output to expected result for 10 points
NUM_SAMPLES = 10
SAMPLE_INDICES = [int(i * (EXPECTED_LENGTH - 1) / (NUM_SAMPLES - 1)) for i in range(NUM_SAMPLES)]



def test_example_flux(mpi_exec, tmp_path, request):
    """Test state-to-state preparation using Python interface."""
    datadir_path = os.path.join(tmp_path, request.node.name)

    Ne = [2]
    Ng = [0]
    freq01 = [4.10]
    rotfreq= [4.0]
    T = 20.0
    dT = 0.05

    targetstate = [1.0/np.sqrt(2), 1.0/np.sqrt(2)]
    n_osc = 1
    n_levels = 1

    quandary = Quandary(
        Ne=Ne,
        Ng=Ng,
        freq01=freq01,
        rotfreq=rotfreq,
        T = T, 
        dT = dT,
        spline_order=0,
        nsplines=40,
        randomize_init_ctrl=False,
        carrier_frequency = [[0.12]],
        initctrl_MHz = 0.01 *1000.0*np.sqrt(2),
        maxctrl_MHz = 0.05*1000.0,
        flux_enabled = True,
        flux_randomize_init_ctrl = True,
        flux_initctrl_MHz = 0.02*1000.0,
        flux_maxctrl_MHz = 0.1*1000.0,
        flux_enforce_BC = False,
        flux_nsplines = 10,
        flux_spline_order = 0,
        targetstate=targetstate,
        rand_seed=1234,
        verbose=False
    )

    t, pt, qt, infidelity, energy, population = quandary.simulate(
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
