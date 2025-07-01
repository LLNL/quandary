import numpy as np
from pytest import approx
from quandary import Quandary


def test_evalControls_updates_timestep():
    """
    Test that evalControls properly updates timestep (dT) to match sampling rate.
    """

    T = 5.0
    quandary = Quandary(
        Ne=[2],
        freq01=[4.0],
        T=T,
        verbose=False
    )

    original_dT = quandary.dT
    original_nsteps = quandary.nsteps

    # Test evalControls with different sampling rate
    points_per_ns = 2
    time, _, _ = quandary.evalControls(points_per_ns=points_per_ns)

    expected_nsteps = int(np.floor(T * points_per_ns))
    expected_dT = T / expected_nsteps

    assert time[0] == approx(0.0)
    assert time[-1] == approx(T)
    assert time[1] - time[0] == approx(expected_dT)

    # Verify original settings are restored
    assert quandary.dT == approx(original_dT)
    assert quandary.nsteps == original_nsteps
