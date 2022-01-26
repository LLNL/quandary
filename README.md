# Quandary - Optimal control for open and closed quantum systems
Quandary implements an optimization solver for open and closed optimal quantum control. The underlying quantum dynamics model open or closed quantum systems, using either Schroedinger's equation for a state vector (closed), or Lindblad master equation for a density matrix (open). The control problem aims to find control pulses that drive the system to a desired target state. Quandary targets deployment on High-Performance Computing platforms, offering various levels for parallelization using the message passing paradigm. 

A documentation is under development. In the meantime, refer to the user guide in the `doc/` folder for information on the underlying mathematical models as well as details on their implementation in Quandary, and parallel distribution.

For questions, feel free to reach out to Stefanie Guenther [guenther5@llnl.gov].

## Dependencies
This project relies on Petsc [https://petsc.org/release/] to handle (parallel) linear algebra. Optionally, XBraid [https://github.com/XBraid/xbraid] can be linked to realize time-parallelization, and Slepsc [https://slepc.upv.es] can be used to solve some eigenvalue problems if desired (e.g. for the Hessian...)
* **Required:** Install Petsc:
    You can try the below, but make sure to also check [https://petsc.org/release/] for the newest installation guide. (On MacOS, you can also `brew install petsc`.)
    * `git clone -b release https://gitlab.com/petsc/petsc.git petsc`
    * `cd petsc`
    * Configure Petsc with `./configure`, check [https://petsc.org/release/install/install_tutorial] for optional arguments. Note that Petsc compiles in debug mode by default. To configure petsc with compiler optimization, consider configuration such as
        `./configure --prefix=/YOUR/INSTALL/DIR --with-debugging=0 --with-fc=0 --with-cxx=mpicxx --with-cc=mpicc COPTFLAGS='-O3' CXXOPTFLAGS='-O3'`
    * The output of `./configure` reports on how to set the `PETSC_DIR` and `PETSC_ARCH` variables
        * `export PETSC_DIR=/YOUR/INSTALL/DIR`
        * `export PETSC_ARCH=/YOUR/ARCH/PREFIX`
    * Compile petsc with `make all check'
    * Append Petsc directory to the `LD_LIBRARY_PATH`:
        * `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PETSC_DIR/$PETSC_ARCH`

* **Optional:** Install XBraid, using the branch 'solveadjointwithxbraid': 
    - git clone https://github.com/XBraid/xbraid.git
    - cd xbraid
    - git checkout solveadjointwithxbraid
    - make braid

* **Optional:** Install Slepsc
    * Read the docs here: [https://slepc.upv.es/documentation/slepc.pdf]
 
###  Petsc on LLNL's LC
Petc is already installed on LLNL LC machines, see here [https://hpc.llnl.gov/software/mathematical-software/petsc]. To use it, first load the following modules:
* `module load gcc/8.1.0`
* `module load mvapich2/2.3`

Then set the `PETSC_DIR` variable to point to the Petsc folder and add it to the `LD_LIBRARY_PATH`:
* `export PETSC_DIR=/usr/tce/packages/petsc/petsc-3.12.4-mvapich2-2.3-gcc-4.8-redhat`
* `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PETSC_DIR`

## Installation
Adapt the beginning of the 'Makefile' to set the path to your Petsc (and possibly XBraid and/or Slepsc) installation. Then,
* `make cleanup` to clean the build directory. (Note the *up* in *cleanup*.)
* `make -j main` to build the code (using 'j' threads)


## Running
The code builds into the executable `main`. It takes one argument being the name of the test-case's configuration file. The file `config_template.cfg`, lists all possible configuration options. It is currently set to simulate a bipartite system with 3x20 levels (Alice - cavity testcase "AxC"). The configuration file is filled with comments that should help users set up their test case and match the options to the description in the user guide.
* `./main config_template.cfg`


## Community and Contributing

Quandary is an open source project that is under heavy development. Contributions in all forms are very welcome, and can be anything from new features to bugfixes, documentation, or even discussions. Contributing is easy, work on your branch, create a pull request to master when you're good to go and the regression tests in 'tests/' pass.

## Publications
* S. Guenther, N.A. Petersson, J.L. DuBois: "Quantum Optimal Control for Pure-State Preparation Using One Initial State", AVS Quantum Science, vol. 3, arXiv preprint <https://arxiv.org/abs/2106.09148> (2021)
* S. Guenther, N.A. Petersson, J.L. DuBois: "Quandary: An open-source C++ package for High-Performance Optimal Control of Open Quantum Systems", submitted to IEEE Supercomputing 2021, arXiv preprint <https://arxiv.org/abs/2110.10310> (2021)

## License

Quandary is distributed under the terms of the MIT license. All new contributions must be made under this license. See LICENSE, and NOTICE, for details. 

SPDX-License-Identifier: MIT
