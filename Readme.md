# Quandary - Quantum control on HPC clusters
This project implements a parallel-in-time optimization solver for quantum control. The underlying quantum dynamics model open quantum systems, using the Lindblad master equation to evolve the density matrix in time. The control problem aims to find control pulses that realize a certain gate, i.e. drive the system to a desired target state

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
* In the main directory of this project, adapt the beginning of the Makefile to set the path to XBraid and Petsc. 
* Type `make cleanup` to clean the build directory.
* Type `make -j main` to build the code. 

### Notes for installing petsc
* Clone Petsc from github
* By default, Petsc will compile in debug mode. To configure petsc with compiler optimization, run
  `./configure --with-debugging=0 --with-fc=0 --with-cxx=mpicxx --with-cc=mpicc COPTFLAGS='-O3' CXXOPTFLAGS='-O3'`
* The output of `./configure` reports on how to set the `PETSC_DIR` and `PETSC_ARCH` variables
* Compile petsc with `make all test`

 

## Running
The code builds into the executable `./main`. It takes one argument being the name of the config file. The config file `AxC.cfg`, lists all possible config options. It is currently set to run the Alice-Cavity testcase (3x20 levels).
* `./main AxC.cfg` for serial run
* `srun -n36 ./main AxC.cfg` for parallel run using 36 cores

