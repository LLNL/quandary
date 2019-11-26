# Quac - Quantum control for HPC
This project implements a parallel-in-time optimization solver for quantum control based on the density matrix formulation. The underlying dynamics model open quantum systems, using the Lindblad master equation to evolve the density matrix in time. The software library XBraid is used to distribute the simulation time domain onto multiple cores and apply the time-parallel multigrid iterations.

## Requirements:
To build this project, you need to have the following packages installed:
* Petsc [https://www.mcs.anl.gov/petsc/]
* Xbraid [https://github.com/XBraid/xbraid], on branch 'solveadjointwithxbraid'

## Installation
* Download XBraid, switch to the 'solveadjointwithxbraid' branch and build the shared library:
    - git clone https://github.com/XBraid/xbraid.git
    - cd xbraid
    - git checkout solveadjointwithxbraid
    - make braid
* Install Petsc (see Petsc manual for installation guide). Set the `PETSC_DIR` and `PETSC_ARCH` variables.
* In the main directory of this project, adapt the beginning of the Makefile to set the path to the XBraid and Petsc locations. 
* Type `make cleanup` to clean the build directory.
* Type `make main` to build the code. 

## Running
The code builds into the executable `./main`.
* Show available command line arguments, type `./main --help`. This however also shows PETSC's command line arguments. So you might want to have a look into 'src/main.c' to see the command line arguments specific for this project.
* For serial time-stepping, use `./main -ml 1`, which executes XBraid on one time-grid level, which is equivalent to serial time-stepping. 
* For multigrid time-stepping, use higher `-ml` argument, e.g. the default `./main -ml 5`. 
* For parallel computations, run with `mpirun -np <nprocessors> ./main <args>`


