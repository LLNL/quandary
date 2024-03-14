# Quandary - Optimal control for open and closed quantum systems
Quandary implements an optimization solver for open and closed optimal quantum control. The underlying quantum dynamics model open or closed quantum systems, using either Schroedinger's equation for a state vector (closed), or Lindblad master equation for a density matrix (open). The control problem aims to find control pulses that drive the system to a desired target, such as a target unitary solution operator or to a predefined target state. Quandary targets deployment on High-Performance Computing platforms, offering various levels for parallelization using the message passing paradigm. 

It is advised to look at the user guide in `doc/`, describing the underlying mathematical models, their implementation and usage in Quandary. The `config_template.cfg` gathers all available options for Quandary's execution. 

Feel free to reach out to Stefanie Guenther [guenther5@llnl.gov] for any question you may have. 

## Dependencies
This project relies on Petsc [https://petsc.org/release/] to handle (parallel) linear algebra. Optionally Slepsc [https://slepc.upv.es] can be used to solve some eigenvalue problems if desired (e.g. for the Hessian...)
* **Required:** Install Petsc:

    Check out [https://petsc.org/release/] for the latest installation guide. On MacOS, you can also `brew install petsc`. As a quick start, you can also try the below:
    * Download tarball for Petsc here [https://ftp.mcs.anl.gov/pub/petsc/release-snapshots/].   
    * `tar -xf petsc-<version>.tar.gz`
    * `cd petsc-<version>`
    * Configure Petsc with `./configure`. Please check [https://petsc.org/release/install/install_tutorial] for optional arguments. For example, 
        `./configure --prefix=/YOUR/INSTALL/DIR --with-debugging=0 --with-fc=0 --with-cxx=mpicxx --with-cc=mpicc COPTFLAGS='-O3' CXXOPTFLAGS='-O3'`
    * The output of `./configure` reports on how to set the `PETSC_DIR` and `PETSC_ARCH` variables
        * `export PETSC_DIR=/YOUR/INSTALL/DIR`
        * `export PETSC_ARCH=/YOUR/ARCH/PREFIX`
    * Compile petsc with `make all check'
    * Append Petsc directory to the `LD_LIBRARY_PATH`:
        * `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PETSC_DIR/$PETSC_ARCH/lib`

* **Optional:** Install Slepsc
    * Read the docs here: [https://slepc.upv.es/documentation/slepc.pdf]

###  Petsc on LLNL's LC
Petc is already installed on LLNL LC machines, see here [https://hpc.llnl.gov/software/mathematical-software/petsc]. It is located at '/usr/tce/packages/petsc/<version>'. To use it, export the 'PETSC_DIR' variable to point to the Petsc folder, and add the 'lib' subfolder to the 'LD_LIBRARY_PATH` variable: 
* `export PETSC_DIR=/usr/tce/packages/petsc/<version>` (check the folder name for version number)
* `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PETSC_DIR/lib`

The 'PETSC_ARCH' variable is not needed in this case. 

Depending on your setup, you might need to load some additional modules, such as openmpi, e.g. as so:
* `module load openmpi`

## Installation
Adapt the beginning of the 'Makefile' to set the path to your Petsc (and possibly Slepsc, python path, and fitpackpp) installation, if not exported. Then,
* `make cleanup` to clean the build directory. (Note the *up* in *cleanup*.)
* `make quandary` to build the code (or 'make -j quandary' for faster build using multiple threads)
It is advised to add Quandary to your `PATH`, e.g.
* `export PATH=$PATH:/path/to/quandary/`

* **Optional:** To run Quandary from within a Python environment, you should have a working python interpreter with numpy and matplotlib installed. Then, append Quandary's location to your `PYTHONPATH`, e.g. with  
* `export PYTHONPATH=$PYTHONPATH:/path/to/quandary/`
 

## Running
The code builds into the executable `quandary`. It takes one argument being the name of the test-case's configuration file. The file `config_template.cfg`, lists all possible configuration options. The configuration file is filled with comments that should help users set up their test case and match the options to the description in the user guide. Also compare the examples folder.
* `./quandary config_template.cfg`
* `mpirun -np 4 ./quandary config_template.cfg --quiet`


## Community and Contributing

Quandary is an open source project that is under heavy development. Contributions in all forms are very welcome, and can be anything from new features to bugfixes, documentation, or even discussions. Contributing is easy, work on your branch, create a pull request to master when you're good to go and the regression tests in 'tests/' pass.

## Publications
* S. Guenther, N.A. Petersson, J.L. DuBois: "Quantum Optimal Control for Pure-State Preparation Using One Initial State", AVS Quantum Science, vol. 3, arXiv preprint <https://arxiv.org/abs/2106.09148> (2021)
* S. Guenther, N.A. Petersson, J.L. DuBois: "Quandary: An open-source C++ package for High-Performance Optimal Control of Open Quantum Systems", submitted to IEEE Supercomputing 2021, arXiv preprint <https://arxiv.org/abs/2110.10310> (2021)

## License

Quandary is distributed under the terms of the MIT license. All new contributions must be made under this license. See LICENSE, and NOTICE, for details. 

SPDX-License-Identifier: MIT
