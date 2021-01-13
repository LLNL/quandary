# Quandary - Optimal Open Quantum Control
Quandary implements an optimization solver for quantum control. The underlying quantum dynamics model open quantum systems, using the Lindblad master equation to evolve a density matrix in time. The control problem aims to find control pulses that drive the system to a desired target state.

A documentation is under development. In the meantime, refer to the user guide in the `doc/` folder for information on the underlying mathematical models as well as details on their implementation in Quandary. 

For questions, feel free to reach out to Stefanie Guenther [guenther5@llnl.gov].

## Dependencies
This project relies on Petsc [https://www.mcs.anl.gov/petsc/] to handle linear algebra. Optionally, XBraid [https://github.com/XBraid/xbraid] can be linked to realize time-parallelization, and Slepsc [https://slepc.upv.es] can be used to solve some eigenvalue problems (e.g. for the Hessian...)
* **Required:** Install Petsc:
    You can try the below, but make sure to also check [https://www.mcs.anl.gov/petsc/] for the newest installation guide. 
    * `git clone -b release https://gitlab.com/petsc/petsc.git petsc`
    * `cd petsc`
    * Configure Petsc with `./configure`, check [https://www.mcs.anl.gov/petsc/download/index.html] for optional arguments. Petsc compiles in debug mode by default. To configure petsc with compiler optimization, consider configuration such as
        `./configure --prefix=/YOUR/INSTALL/DIR --with-debugging=0 --with-fc=0 --with-cxx=mpicxx --with-cc=mpicc COPTFLAGS='-O3' CXXOPTFLAGS='-O3'`
    * The output of `./configure` reports on how to set the `PETSC_DIR` and `PETSC_ARCH` variables
        * `export PETSC_DIR=/YOUR/INSTALL/DIR`
        * `export PETSC_ARCH=/YOUR/ARCH/PREFIX`
    * Compile petsc with `make all test`
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
Petc is already installed on LLNL LC machines, see here [https://hpc.llnl.gov/software/mathematical-software/petsc]

## Installation
In the main directory of this project, adapt the beginning of the Makefile to set the path to your Petsc (and possibly XBraid and/or Slepsc) installation.
* `make cleanup` to clean the build directory
* `make -j main` to build the code


## Running
The code builds into the executable `main`. It takes one argument being the name of the test-case's configuration file. The file `config_template.cfg`, lists all possible configuration options. It is currently set to simulate a bipartite system with 3x20 levels (Alice - cavity testcase "AxC"). 
* `./main config_template.cfg`


## Community and Contributing

Quandary is an open source project that is under heavy development. Contributions in all forms are very welcome, and can be anything from new features to bugfixes, documentation, or even discussions. Contributing is easy, work on your branch, create a pull request to master when you're good to go and the regression tests in 'tests/' pass.

## License

Quandary is distributed under the terms of the MIT license. All new contributions must be made under this license. See LICENSE, and NOTICE, for details. 

SPDX-License-Identifier: MIT
