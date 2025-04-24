"""Common pytest configuration."""


def add_common_options(parser):
    """Add common command line options to pytest."""
    try:
        parser.addoption(
            "--mpi-exec",
            action="store",
            default="mpirun",
            help="Path to the MPI executable (e.g., mpirun or srun)"
        )
    except ValueError:
        # Option already exists, skip it
        pass

    try:
        parser.addoption(
            "--mpi-opt",
            action="store",
            default="",
            help="Extra options to pass to mpi exec command)"
        )
    except ValueError:
        # Option already exists, skip it
        pass
