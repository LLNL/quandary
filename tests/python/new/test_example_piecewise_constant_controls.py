import os
import pytest
import numpy as np
from quandary.new import create_config, optimize
from utils import assert_results_equal


# Mark all tests in this file as regression tests
pytestmark = pytest.mark.regression

EXPECTED_LENGTH = 1222
EXPECTED_INFIDELITY = 8.972282935304499e-05

EXPECTED_PT = [
    [
        2.3240212289146402, -6.02741442812791, 4.271414297263941, -1.6893550288325399, -4.195931405082329,
        -3.3264144188882496, 2.72645314994069, 1.72093818240711, -6.32707179290129, 1.5157846892070401
    ],
    [
        -0.943696996738463, -1.54008956041859, 0.974092331855258, -2.11725961973716, 7.23084135035911,
        -3.1750515857345603, 0.0973783023189745, 0.740021282630812, 0.224629091879865, -0.38396655714211003
    ],
]

EXPECTED_QT = [
    [
        -0.283811542906026, -4.8701859115599895, 8.59671132827831, -4.93214646562814, 0.477911223869226,
        -3.1148996029979603, 4.12633502186139, -8.7119625977595, 5.5314651532369705, -0.625574956673312
    ],
    [
        -0.046697131454043, -1.52515337633141, 5.679892341493789, -1.50359934971612, -0.582665615741822,
        4.230557199744791, -2.99582215425441, -1.1935353454714102, 2.07237924275629, 0.13749148048099802
    ],
]

EXPECTED_ENERGY = [
    [
        [
            0.0, 0.064018398811131, 0.0255074890379477, 0.173927949744261, 0.353741217394772,
            0.333549464588712, 0.0380707852734636, 0.0978615998308722, 0.0618885484468087, 6.94795847509597e-05
        ],
        [
            0.0, 0.0750126715029975, 0.0387021812369554, 0.220410492760229, 0.371592727474775,
            0.343070198738427, 0.0473104104549414, 0.117393335085893, 0.0897222369151197, 8.06388758826829e-05
        ],
        [
            1.0, 0.909562029576399, 0.965027097412589, 0.849049560581231, 0.722364276021594,
            0.659433342057967, 0.946974537591837, 0.916811332807015, 0.931008800470933, 0.999927503842888
        ],
        [
            1.0, 0.951406900090192, 0.970763232452821, 0.756611997583106, 0.552301780066315,
            0.663946995450787, 0.967644266892654, 0.867933732713146, 0.917380414668408, 0.999922377839154
        ],
    ],
    [
        [
            0.0, 0.0154519046340033, 0.272719890877112, 0.669983983382366, 0.626662668535732,
            0.199164427176118, 0.112770100796748, 0.00923055315622832, 0.00450364216119423, 5.28865549854226e-05
        ],
        [
            1.0, 0.968099645561614, 0.708445003836376, 0.329798181366376, 0.392735918297498,
            0.780285311703389, 0.88903936651078, 0.979999480543979, 0.971653054632594, 0.999938954380437
        ],
        [
            0.0, 0.0451449782994161, 0.0505984687229202, 0.121614356690835, 0.277861896059578,
            0.588976554630449, 0.663768893088262, 0.829908977810533, 0.942764763301231, 0.999934439834562
        ],
        [
            1.0, 0.971303471542686, 0.96823663612046, 0.878603478091698, 0.70273951707397,
            0.431573707037999, 0.334421640363678, 0.180860988819672, 0.08107854004672, 7.37193894270867e-05
        ],
    ],
]

EXPECTED_POPULATION = [
    [
        [
            1.0, 0.935981601199934, 0.974492510980937, 0.826072050278979, 0.646258782633309,
            0.66645053546358, 0.961929214793174, 0.902138400239093, 0.938111451624294, 0.999930520488841
        ],
        [
            1.0, 0.924987328497812, 0.961297818766506, 0.779589507247078, 0.628407272539148,
            0.656929801300661, 0.952689589629836, 0.882606665005681, 0.910277763177695, 0.999919361218406
        ],
        [
            0.0, 0.0904379704248041, 0.0349729025906582, 0.150950439424077, 0.277635723986669,
            0.340566657969653, 0.0530254624435817, 0.0831886672364308, 0.0689911995803502, 7.24962159420225e-05
        ],
        [
            0.0, 0.0485930999180725, 0.0292367675682773, 0.243388002447181, 0.447698219970424,
            0.336053004606078, 0.0323557331692003, 0.132066267352639, 0.0826195853995922, 7.76222302322741e-05
        ],
    ],
    [
        [
            1.0, 0.984548095377062, 0.727280109141772, 0.330016016640874, 0.373337331492349,
            0.800835572876174, 0.887229899269889, 0.990769446913737, 0.995496357909908, 0.999947113518607
        ],
        [
            0.0, 0.0319003544391963, 0.291554996167086, 0.670201818640931, 0.607264081716425,
            0.219714688335699, 0.110960633573997, 0.020000519547595, 0.0283469454602211, 6.1045713851674e-05
        ],
        [
            1.0, 0.954855021701787, 0.949401531280327, 0.878385643314473, 0.722138103948685,
            0.41102344539717, 0.336231106947157, 0.170091022232912, 0.0572352367500526, 6.55602242680236e-05
        ],
        [
            0.0, 0.0286965284655776, 0.031763363900639, 0.121396521938589, 0.297260482962769,
            0.568426293018865, 0.665578359698176, 0.819139011246113, 0.91892146002128, 0.999926280679959
        ],
    ],
]

# Compare output to expected result for 10 points
NUM_SAMPLES = 10
SAMPLE_INDICES = [int(i * (EXPECTED_LENGTH - 1) / (NUM_SAMPLES - 1)) for i in range(NUM_SAMPLES)]


def test_example_piecewise_constant_controls(tmp_path, request):
    """Test CNOT gate optimization with piecewise constant controls using new Python interface."""
    datadir_path = os.path.join(tmp_path, request.node.name)

    freq01 = [4.80595, 4.8601]
    Jkl = [0.005]
    favg = sum(freq01)/len(freq01)
    rotfreq = favg*np.ones(len(freq01))
    T = 200.0

    unitary = np.identity(4)
    unitary[2, 2] = 0.0
    unitary[3, 3] = 0.0
    unitary[2, 3] = 1.0
    unitary[3, 2] = 1.0

    spline_order = 0
    nsplines = 1000
    gamma_variation = 1.0
    control_enforce_BC = True

    n_osc = 2
    n_levels = 4

    setup = create_config(
        nessential=[2, 2],
        transition_frequency=freq01,
        total_time=T,
        dipole_coupling=Jkl,
        rotation_frequency=list(rotfreq),
        spline_order=spline_order,
        nspline=nsplines,
        control_zero_boundary_condition=control_enforce_BC,
        output_directory=datadir_path,
    )

    # Match old interface defaults
    setup.rand_seed = 1234
    setup.optim_tol_final_cost = 1e-4
    setup.optim_penalty_leakage = 0.1
    setup.optim_penalty_energy = 0.1
    setup.optim_penalty_dpdm = 0.01
    setup.output_optimization_stride = 1
    setup.linearsolver_maxiter = 20
    setup.optim_penalty_variation = gamma_variation

    # Match old initialization amplitude
    # With spline_order=0, carrier_frequencies are [[0.0], ...] so N_carriers=1
    num_carrier_freqs = len(setup.carrier_frequencies[0])
    init_amplitude = 10.0 / 1000.0 / np.sqrt(2) / num_carrier_freqs

    results = optimize(
        setup,
        target=unitary,
        control_amplitude=init_amplitude,
        quiet=True,
    )

    assert_results_equal(
        t=results.time,
        p_samples=results.p_samples,
        q_samples=results.q_samples,
        infidelity=results.infidelity,
        energy=results.expected_energy,
        population=results.population,
        T=T,
        n_osc=n_osc,
        n_levels=n_levels,
        expected_length=EXPECTED_LENGTH,
        expected_infidelity=EXPECTED_INFIDELITY,
        expected_pt=EXPECTED_PT,
        expected_qt=EXPECTED_QT,
        expected_energy=EXPECTED_ENERGY,
        expected_population=EXPECTED_POPULATION,
        sample_indices=SAMPLE_INDICES,
    )
