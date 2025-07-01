import pytest
from quandary import Quandary
from utils import assert_results_equal

# Mark all tests in this file as regression tests
pytestmark = pytest.mark.regression

EXPECTED_LENGTH = 9906
EXPECTED_INFIDELITY = 4.30051844739765e-05

EXPECTED_PT = [
    [
        1.5223974313145099, 0.192367631943629, 5.2322350273611695, 0.270716585386651, 4.38454614717518,
        1.64522208663637, 4.27356481827349, 3.56918803082488, 0.809746914647563, 2.49847938610855
    ],
]

EXPECTED_QT = [
    [
        1.1486887461253, 3.65174025044863, 3.38068715198705, 1.69637577485486, 3.89987855203689,
        1.02123892969852, 4.5603555728564, -0.6527678876090169, 4.1927845343931, 0.519653744346407
    ],
]

EXPECTED_ENERGY = [
    [
        [
            0.0, 0.0234550545874589, 0.151322202170957, 0.38091326425715, 0.719219922127726,
            1.07914866155792, 1.38417447451341, 1.73449380418287, 1.90878994427285, 2.00000694611498
        ],
        [
            1.0, 1.03449485473264, 1.15972091768289, 1.18383141548819, 1.21165277321355,
            1.20900227532045, 1.21269089383777, 1.11510843776812, 1.0552883588209, 1.00005015890853
        ],
        [
            2.0, 1.94250327364116, 1.68998722508855, 1.43543971159315, 1.070068198256,
            0.711977722635159, 0.404183040483815, 0.151128785805321, 0.0373575963912251, 8.02625600351541e-05
        ],
    ],
]

EXPECTED_POPULATION = [
    [
        [
            1.0, 0.97686084850513, 0.859760974702525, 0.661516606458905, 0.413421449887264,
            0.205924265997096, 0.0705988862580945, 0.01147031596463, 0.00057333624469326, 3.75770574331089e-06
        ],
        [
            0.0, 0.0226342148878521, 0.127263830043423, 0.268727310512903, 0.379204026955525,
            0.377295442001178, 0.277750731738844, 0.13081630109352, 0.0355226617637922, 1.27899738623382e-05
        ],
        [
            0.0, 0.000504783498382912, 0.0129728034877987, 0.0697471926129869, 0.207353891744709,
            0.41674941229836, 0.651616413277044, 0.857685579183397, 0.96388427223235, 0.9999635620352
        ],
    ],
]

# Compare output to expected result for 10 points
NUM_SAMPLES = 10
SAMPLE_INDICES = [int(i * (EXPECTED_LENGTH - 1) / (NUM_SAMPLES - 1)) for i in range(NUM_SAMPLES)]


def test_example_swap02(mpi_exec):
    """Test SWAP 0-2 gate optimization using Python interface."""

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
        maxcores=4
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
