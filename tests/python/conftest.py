"""Shared pytest fixtures for Python tests."""

import pytest


@pytest.fixture
def mpi_exec(request):
    """Get MPI executor and options from pytest options."""
    executor = request.config.getoption("--mpi-exec")
    options = request.config.getoption("--mpi-opt")
    if executor != "mpirun":
        return f"{executor} {options} -n"
    return f"mpirun {options} -np"
