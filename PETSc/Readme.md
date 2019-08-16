# Parallel-in-time quantum control (PinT-QC)

## Requirements:
To build this project, you need to have the following packages installed:
* Petsc [https://www.mcs.anl.gov/petsc/]
* Xbraid [https://github.com/XBraid/xbraid]

## Installation
* Build XBraid library with `make braid` inside the XBraid directory.
* Install Petsc (see Petsc manual for installation). Set the `PETSC_DIR` and `PETSC_ARCH` variables.
* In the main directory of this project, adapt the beginning of the Makefile to set your XBraid and Petsc locations. 
* Type `make cleanup` to clean the build directory.
* Type `make main` to build the code, giving the executable `./main`.

## Running
Show available command line arguments with `./main --help`. 
* For serial time-stepping, use `./main -ml 1`, which executes XBraid on one time-grid level, which is equivalent to serial time-stepping. 
* For multigrid time-stepping, use higher `-ml` argument, e.g. the default `./main -ml 5`. 
* For parallel computations, run with `mpirun -np <nprocessors> ./main <args>`


