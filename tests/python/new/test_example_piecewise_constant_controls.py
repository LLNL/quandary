import os
import pytest
import numpy as np
from quandary.new import create_config, optimize
from utils import assert_results_equal


# Mark all tests in this file as regression tests
pytestmark = pytest.mark.regression

EXPECTED_LENGTH = 1222
EXPECTED_INFIDELITY = 0.9983418376969986

EXPECTED_PT = [
    [
        0.0, -4.8518197500656095, -0.919207929271984, -3.2249957482986003, -0.92114482623898,
        1.93315890839137, 0.658140222874109, 5.07940133470365, -2.4168149481258197, 0.0
    ],
    [
        0.0, -3.13304232909638, -0.7967336551230491, -2.376076501544, -5.85570122095727,
        -2.3292783252579, -4.77076715773557, 3.20102268375658, -6.285837832781939, 0.0
    ],
]

EXPECTED_QT = [
    [
        0.0, 5.90758134884679, -7.04592613637372, -3.25894928737577, -0.800817084628982,
        -2.3976535386134, 3.9315378509346695, -3.28183258383752, -6.565512515967479, 0.0
    ],
    [
        0.0, 2.41721255118214, 3.64915818836219, -0.5216394312446021, 5.25921254533198,
        4.88341950629601, -3.9822455064974203, 6.369366721451731, 1.27668851544572, 0.0
    ],
]

EXPECTED_ENERGY = [
    [
        [
            0.0, 0.00243967056990113, 0.00227467632544818, 0.0022306900254119, 0.00142665442570709,
            0.00602627516242494, 0.00372960880190912, 0.0226938315897022, 0.00126797052234643, 0.0127964254418217
        ],
        [
            0.0, 0.0149270791539703, 0.0352421059044232, 0.0276731126329321, 0.00650148755370536,
            0.00463842634377513, 0.0356315774485707, 0.0667260115605683, 0.0215239781477449, 0.0324789168489543
        ],
        [
            1.0, 0.984669710299609, 0.966216595031293, 0.97194005720683, 0.993715539413359,
            0.990621772804508, 0.969384043338436, 0.942621288023607, 0.985165793609021, 0.987033480946644
        ],
        [
            1.0, 0.997963539991471, 0.996266622882233, 0.99815614031788, 0.998356318772616,
            0.998713525811507, 0.99125477057213, 0.967958869072029, 0.992042257970517, 0.967691176949385
        ],
    ],
    [
        [
            0.0, 0.000338147889305681, 0.0232181054866423, 0.00590027393238777, 0.0190491906373522,
            0.0174267712656614, 0.00828373261156806, 0.015687943547395, 0.0339978294585551, 0.0221748123708485
        ],
        [
            1.0, 0.986745552617178, 0.947455571116607, 0.970166224441799, 0.976374603055368,
            0.979099094024306, 0.967254080420733, 0.956886444587524, 0.956906510783005, 0.977817745036422
        ],
        [
            0.0, 0.0132242933461922, 0.0588146082463399, 0.0378552079759778, 0.0393285198616124,
            0.0411018163045288, 0.0498294309787528, 0.0705164979822579, 0.0702780551298505, 0.0352474069723287
        ],
        [
            1.0, 0.999692006207016, 0.97051171527442, 0.986078293766796, 0.965247686629246,
            0.962372318668043, 0.974632756251195, 0.956909114165098, 0.938817604983754, 0.964760036013681
        ],
    ],
]

EXPECTED_POPULATION = [
    [
        [
            1.0, 0.997560329461219, 0.997725323780871, 0.997769310086819, 0.998573345706141,
            0.993973724973072, 0.996270391353532, 0.977306168570687, 0.998732029643265, 0.987203574793765
        ],
        [
            1.0, 0.985072920851846, 0.964757894103558, 0.97232688737857, 0.993498512466342,
            0.995361573676897, 0.964368422574248, 0.933273988474539, 0.978476021897399, 0.967521083199914
        ],
        [
            0.0, 0.015330289706674, 0.0337834049793186, 0.028059942808521, 0.00628446060435686,
            0.00937822721401021, 0.0306159566854054, 0.0573787120073113, 0.0148342064328325, 0.0129665190995316
        ],
        [
            0.0, 0.00203646003860252, 0.00373337722501898, 0.00184385979921405, 0.00164368135559459,
            0.00128647432098616, 0.00874522958071298, 0.0320411310853615, 0.00795774219105144, 0.0323088232387448
        ],
    ],
    [
        [
            1.0, 0.999661852141814, 0.976781894619677, 0.994099726179844, 0.980950809494496,
            0.982573228869836, 0.991716267543873, 0.984312056612995, 0.966002170707056, 0.977825187864738
        ],
        [
            0.0, 0.013254447388638, 0.052544428891374, 0.0298337755697025, 0.0236253969646786,
            0.020900905996366, 0.0327459196020861, 0.0431135554475834, 0.0430934892621391, 0.0221822550124464
        ],
        [
            1.0, 0.986775706660091, 0.941185391764271, 0.962144792039374, 0.960671480156103,
            0.95889818371399, 0.950170569045089, 0.929483502048661, 0.929721944912003, 0.964752593073847
        ],
        [
            0.0, 0.000307993823057012, 0.0294882848328322, 0.0139217063502976, 0.0347523134989655,
            0.0376276814644502, 0.0253672439016486, 0.0430908859922925, 0.0611823951778144, 0.035239964174448
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
