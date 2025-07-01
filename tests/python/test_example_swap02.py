import os
import pytest
from quandary import Quandary
from utils import assert_results_equal

# Mark all tests in this file as regression tests
pytestmark = pytest.mark.regression

EXPECTED_LENGTH = 9906
EXPECTED_INFIDELITY = 7.743371585999803e-05

EXPECTED_PT = [
    [
        3.49878461498772, 0.9372034359063061, 3.82522391628158, 4.94417898650528, 1.6970695838818899,
        3.4150750131677996, 1.64742105679164, 6.27372291394662, -0.158178972429773, 3.33064952087047
    ],
]

EXPECTED_QT = [
    [
        -0.737198940278714, -2.5404099566412097, 1.9430855372276599, -3.29007822559714, 0.202359122117534,
        -4.06151032304104, -0.727084835198226, -0.338926104324614, -4.19020409943546, 2.14870117067776
    ],
]

EXPECTED_ENERGY = [
    [
        [
            0.0, 0.0617825490723183, 0.176079313745057, 0.439566533429592, 0.717418204516129,
            1.01689731884274, 1.33955354416972, 1.62218904519213, 1.91000947440407, 2.00001741934672
        ],
        [
            1.0, 1.02614035923091, 1.13754267526885, 1.24603141974462, 1.2695443541876,
            1.29270091463921, 1.26762689553064, 1.21374312361065, 1.06169311172988, 1.00012844594568
        ],
        [
            2.0, 1.91263442680844, 1.68709072718142, 1.31556532558582, 1.01316292157842,
            0.691604243676856, 0.39326419761164, 0.166193685451392, 0.0295179209802044, 0.000174705079131304
        ],
    ],
]

EXPECTED_POPULATION = [
    [
        [
            1.0, 0.939711012272369, 0.839442071980425, 0.638260402817787, 0.447629939443993,
            0.235560324657557, 0.0934912404073985, 0.0185680932614118, 0.000551089845975009, 2.31447279607491e-05
        ],
        [
            0.0, 0.059062097817141, 0.148421106108336, 0.286722174550912, 0.352246338671119,
            0.340677230088839, 0.256925388814789, 0.133855761134253, 0.0279850195965806, 2.43268076119764e-06
        ],
        [
            0.0, 0.00122660793399808, 0.0121354622064046, 0.0750116963555328, 0.200113854353825,
            0.423748503355938, 0.649569571317751, 0.847561660359807, 0.971435980115172, 0.999932452759026
        ],
    ],
]

# Compare output to expected result for 10 points
NUM_SAMPLES = 10
SAMPLE_INDICES = [int(i * (EXPECTED_LENGTH - 1) / (NUM_SAMPLES - 1)) for i in range(NUM_SAMPLES)]


def test_example_swap02(mpi_exec, tmp_path, request):
    """Test SWAP 0-2 gate optimization using Python interface."""
    datadir_path = os.path.join(tmp_path, request.node.name)

    Ne = [3]
    Ng = [1]
    freq01 = [4.10595]
    selfkerr = [0.2198]
    T = 100.0
    maxctrl_MHz = 8.0
    unitary = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    n_osc = 1
    n_levels = 3

    quandary = Quandary(
        Ne=Ne,
        Ng=Ng,
        freq01=freq01,
        selfkerr=selfkerr,
        maxctrl_MHz=maxctrl_MHz,
        targetgate=unitary,
        T=T,
        rand_seed=1234,
        verbose=False
    )

    t, pt, qt, infidelity, energy, population = quandary.optimize(
        mpi_exec=mpi_exec,
        maxcores=4,
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
