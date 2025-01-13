# Quandary - Optimal control for open and closed quantum systems
Quandary implements an optimization solver for open and closed optimal quantum control. The underlying quantum dynamics model open or closed quantum systems, using either Schroedinger's equation for a state vector (closed), or Lindblad master equation for a density matrix (open). The control problem aims to find control pulses that drive the system to a desired target, such as a target unitary solution operator or to a predefined target state. Quandary targets deployment on High-Performance Computing platforms, offering various levels for parallelization using the message passing paradigm. 

It is advised to look at the user guide in `doc/`, describing the underlying mathematical models, their implementation and usage in Quandary. 

Feel free to reach out to Stefanie Guenther [guenther5@llnl.gov] for any question you may have. 

## Building
Quandary uses CMake and BLT to handle builds. Since BLT is included as a
submodule, first make sure you run:
```
git submodule init && git submodule update
```

This project relies on Petsc [https://petsc.org/release/] to handle (parallel) linear algebra. Optionally Slepsc [https://slepc.upv.es] can be used to solve some eigenvalue problems if desired (e.g. for the Hessian...).

### Spack
Petc, Slepc, and other dependencies such as Python packages can be managed and installed using Spack. Additionally, Spack can build Quandary itself from your local source code.

1. To install Spack, clone the repo and add to your shell following the steps [here](https://spack.readthedocs.io/en/latest/getting_started.html#installation).

2. To setup your compilers in your local Spack configuration:
   ```
   spack compiler find
   ```

3. To activate Quandary's spack environment, run:
    ```
    spack env activate .spack_env/
    ```

4. Trust Spack's binary mirror so we can speed up the installation process:
    ```
    spack buildcache keys --install --trust
    ```
5. Finally, to install the necessary dependencies and build Quandary run:
    ```
    spack install
    ```
    Note: This step could take a while the first time.

Note that `spack install` will build Quandary using CMake from your local source code. The second time you run this is should be much faster, only looking for changes in the environment or local code. By default the specification in the Spack environment `.spack_env/spack.yaml` includes Slepc and Petsc in debug mode (`quandary+slepc+debug`).

## Manually installing dependencies
If you don't want to use Spack to install dependencies as explained above, you can follow these steps to install Petsc and optionally Slepc.

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


## Building without Spack
If you don't want to use Spack to build Quandary, as explained above, you can build directly with CMake, using:
```
mkdir build && cd build
cmake ..
make
```
To use SLEPc, you can pass a flag to `cmake`:
```
cmake -DWITH_SLEPC=ON ..
```

**Optional:** To run Quandary from within a Python environment, you should have a working python interpreter with numpy and matplotlib installed. Then, append Quandary's location to your `PYTHONPATH`, e.g. with  
* `export PYTHONPATH=$PYTHONPATH:/path/to/quandary/`
and have a look into the examples.
 

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
