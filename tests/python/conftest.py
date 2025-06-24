"""Shared pytest fixtures for Python tests."""

import pytest


@pytest.fixture
def mpi_exec(request):
    """Get MPI executor from pytest option."""
    executor = request.config.getoption("--mpi-exec")
    if executor != "mpirun":
        return f"{executor} -n "
    return "mpirun -np "
