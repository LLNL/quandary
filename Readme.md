# Quandary - Quantum control on HPC clusters
This project implements a parallel-in-time optimization solver for quantum control. The underlying quantum dynamics model open quantum systems, using the Lindblad master equation to evolve the density matrix in time. The control problem aims to find control pulses that drive the system to a desired target state.

## Installation
This project relies on the Petsc package [https://www.mcs.anl.gov/petsc/] to handle linear algebra. Optionally, XBraid is used for realizing time-parallelization [https://github.com/XBraid/xbraid].
* Required: Install Petsc
    * `git clone -b maint https://gitlab.com/petsc/petsc.git petsc`
    * `cd petsc`
    * Configure petsc with `./configure`, check [https://www.mcs.anl.gov/petsc/download/index.html] for optional arguments
    * Note: Petsc compiles in debug mode by default. To configure petsc with compiler optimization, run
        `./configure --with-debugging=0 --with-fc=0 --with-cxx=mpicxx --with-cc=mpicc COPTFLAGS='-O3' CXXOPTFLAGS='-O3'`
    * The output of `./configure` reports on how to set the `PETSC_DIR` and `PETSC_ARCH` variables
    * Compile petsc with `make all test`
    * Append Petsc directory to the `LD_LIBRARY_PATH`:
        `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PETSC_DIR/$PETSC_ARCH`

* Optional: Install XBraid, using the branch 'solveadjointwithxbraid': 
    - git clone https://github.com/XBraid/xbraid.git
    - cd xbraid
    - git checkout solveadjointwithxbraid
    - make braid
* In the main directory of this project, adapt the beginning of the Makefile to set the path to Petsc, and possibly to XBraid
* Type `make cleanup` to clean the build directory.
* Type `make -j main` to build the code. 

### Petsc on LC 
* Petc is already installed on LC machines, in the directory
`/usr/tce/packages/petsc/petsc-3.12.4-mvapich2-2.3-gcc-4.8-redhat`
* To use it, load the following modules
`module load gcc/8.1.0`
`module load mvapich2/2.3`
* Set the `PETSC_DIR` variable to point to the Petsc folder and add it to the `LD_LIBRARY_PATH`:
`export PETSC_DIR=/usr/tce/packages/petsc/petsc-3.12.4-mvapich2-2.3-gcc-4.8-redhat`
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PETSC_DIR`
 

## Running
The code builds into the executable `main`. It takes one argument being the name of the config file. The config file `AxC.cfg`, lists all possible config options. It is currently set to run the Alice-Cavity testcase (3x20 levels).
* `./main AxC.cfg` for serial run
* `srun -n36 ./main AxC.cfg` for parallel run using 36 cores

