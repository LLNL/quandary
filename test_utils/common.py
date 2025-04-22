"""Common utility functions for quandary tests."""

def build_mpi_command(mpi_exec: str, num_processes: int, mpi_opt: str, quandary_path: str, config_file: str):
    """Build MPI command."""
    command = mpi_exec.split()
    command.extend(["-n", str(num_processes)])
    if mpi_opt:
        command.extend([mpi_opt])
    command.extend([quandary_path, config_file])
    return command
