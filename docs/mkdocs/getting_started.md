# Installation
Read the `README.md`! In short:

1. Install PETSc (https://petsc.org/).
2. Compile the quandary executable and install with:
``` console
  > mkdir build && cd build
  > cmake ..
  > make
  > sudo cmake --install .
```
3. To use the python interface, create a virtual environment and do:
``` console
  > pip install -e .
```


## Quick start
The C++ Quandary executable takes a configuration input file. As a quick start, test it with
``` console
> ./quandary config_template.cfg  # (serial execution)
```
``` console
> mpirun -np 4 ./quandary config_template.cfg  # (on 4 cores)
```

You can silence Quandary by adding the `--quiet` command line argument.

Results are written as column-based text files in the output directory. Gnuplot is an excellent plotting tool to visualize the written output files, see below. The `config_template.cfg` is currently set to run a CNOT optimization test case. It lists all available options and configurations, and is filled with comments that should help users to set up new simulation and optimization runs, and match the input options to the equations found in this document.

Test the python interface by running one of the examples in `examples/`, e.g.
``` console
> python3 example_swap02.py
```
The python interpreter will start background processes on the C++ executable using a config file written by the python interpreter, and gathers quandary's output results back into the python shell for plotting.
