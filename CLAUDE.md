# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Quandary is a C++ quantum optimal control solver for simulating and optimizing the time evolution of closed and open quantum systems. The project targets high-performance computing platforms, offering various levels of parallelization using MPI. A Python interface (`quandary.py`) provides an easier way to configure and run simulations from Python.

The underlying dynamics are modeled by:
- **Schroedinger's equation** (closed systems, state vector)
- **Lindblad's master equation** (open systems, density matrix)

## Build System

Quandary uses CMake with BLT (Build-Layout-Tool) and depends on PETSc for parallel linear algebra.

### Building with Spack (recommended)

```bash
# Initialize BLT submodule
git submodule init && git submodule update

# Activate Spack environment
spack env activate .spack_env/

# Install dependencies and build
spack install

# Install Python interface
pip install -e .
```

### Building with CMake (without Spack)

Requires a pre-existing PETSc installation:

```bash
# Set PETSc location
export PKG_CONFIG_PATH=$PETSC_DIR/$PETSC_ARCH/lib/pkgconfig/:$PKG_CONFIG_PATH

# Build
mkdir build && cd build
cmake ..
make

# Install Python dependencies
pip install -e .
```

**Debug builds**: Use `cmake -DCMAKE_BUILD_TYPE=Debug ..` or with Spack use `quandary@develop+test build_type=Debug`

## Running Quandary

### C++ executable

```bash
# Serial execution
quandary config_template.toml

# Parallel execution
mpirun -np 4 quandary config_template.toml

# Suppress output
quandary config_template.toml --quiet
```

### Python interface

```bash
python examples/example_cnot.py
```

See `examples/` for usage patterns of the Python interface.

## Testing

### Regression tests

```bash
# Run all regression tests
pytest -m regression

# Run tests in specific directories
pytest tests/regression
pytest tests/python

# Run a single test by name
pytest -k "AxC_detuning"

# Verbose output
pytest -v

# Show print statements
pytest -s
```

See [tests/regression/README.md](tests/regression/README.md) for how to add new tests and rebase test outputs.

### Unit tests

```bash
# From build directory
make test
```

### Performance tests

Performance regression tests are in `tests/performance/`. Results from `main` are tracked on the [performance dashboard](https://software.llnl.gov/quandary/dev/bench/).

## Code Architecture

### Core Components

The codebase is organized into modular components that work together to solve quantum optimal control problems:

**MasterEq** ([mastereq.hpp](include/mastereq.hpp), [mastereq.cpp](src/mastereq.cpp))
- Implements the real-valued right-hand-side (RHS) system matrix for both Lindblad master equation (open systems) and Schroedinger equation (closed systems)
- Supports matrix-free and sparse-matrix implementations for RHS application
- Handles gradient computation for optimization (`compute_dRHS_dParams`)
- Contains array of Oscillator objects for each subsystem
- Manages time-dependent control pulses and coupling terms

**Oscillator** ([oscillator.hpp](include/oscillator.hpp), [oscillator.cpp](src/oscillator.cpp))
- Represents individual quantum subsystems (qubits)
- Stores Hamiltonian operators and energy level structure
- Manages control pulse evaluation via B-spline basis functions
- Handles both drive control (p(t), q(t)) and flux control

**OptimProblem** ([optimproblem.hpp](include/optimproblem.hpp), [optimproblem.cpp](src/optimproblem.cpp))
- Top-level optimization problem formulation
- Interfaces with PETSc's TAO optimization solvers
- Manages objective function and gradient evaluation
- Coordinates forward time-stepping and adjoint backward sweeps
- Handles regularization penalties (Tikhonov, leakage, energy, etc.)

**TimeStepper** ([timestepper.hpp](include/timestepper.hpp), [timestepper.cpp](src/timestepper.cpp))
- Implements time integration schemes for ODE solving
- Supports implicit midpoint rule (IMR), explicit methods, and PETSc's adaptive timestepper
- Handles both forward (primal) and adjoint (backward) time evolution

**Gate** ([gate.hpp](include/gate.hpp), [gate.cpp](src/gate.cpp))
- Defines target quantum gates for optimization
- Implements fidelity measures (trace fidelity, Frobenius norm)
- Supports gate rotation in computational frames

**ControlBasis** ([controlbasis.hpp](include/controlbasis.hpp), [controlbasis.cpp](src/controlbasis.cpp))
- Manages B-spline parameterization of control pulses
- Handles both order-0 (piecewise constant) and order-2 (quadratic) splines
- Evaluates basis functions and derivatives for optimization

**Config** ([config.hpp](include/config.hpp), [config.cpp](src/config.cpp))
- Parses TOML configuration files
- Validates input parameters
- Uses tomlplusplus library for parsing

### Python Interface

The Python interface in [quandary.py](quandary.py) provides:
- `Quandary` dataclass that mirrors C++ configuration options
- `simulate()` and `optimize()` methods that write TOML configs and launch C++ executable
- Helper functions for Hamiltonian construction, carrier frequency computation, and result visualization
- Support for both standard superconducting qubit models and user-defined Hamiltonians

### Key Data Flow

1. **Configuration**: User provides TOML config (C++) or creates `Quandary` object (Python)
2. **Setup**: `OptimProblem` initializes `MasterEq` with array of `Oscillator` objects based on system specification
3. **Optimization**: TAO optimizer calls objective/gradient evaluation
4. **Time Evolution**: `TimeStepper` integrates forward dynamics using `MasterEq::assemble_RHS` at each timestep
5. **Gradient**: Adjoint equations are solved backward in time, accumulating gradient contributions
6. **Control Update**: Optimizer updates B-spline coefficients based on gradient
7. **Output**: Final control pulses, state evolution, and optimization history written to files

### Matrix-Free vs Sparse Matrix

Quandary offers two modes for applying the RHS system matrix:
- **Matrix-free** (default for ≤5 oscillators): Applies Hamiltonian action on-the-fly without storing full matrix
- **Sparse matrix**: Pre-assembles and stores RHS as PETSc sparse matrix

The choice is controlled by `usematfree` config option. Matrix-free is more memory-efficient for smaller systems.

### Parallelization

MPI parallelization occurs at two levels:
1. **Initial condition level**: Different initial states distributed across MPI ranks
2. **PETSc level**: Linear algebra operations parallelized within PETSc

The code automatically determines optimal MPI task distribution based on number of initial conditions.

## Configuration Files

Configuration files use TOML format. Key sections:
- `[system]`: Quantum system parameters (energy levels, frequencies, coupling strengths, decoherence)
- `[control]`: Control pulse parameterization (B-splines, carrier frequencies, amplitude bounds)
- `[control.flux]`: Optional flux control settings
- `[optimization]`: Target gates/states, objective function, tolerances, penalty weights
- `[output]`: Output directory and observables to save
- `[solver]`: Solver settings (time integrator, linear solver, random seed)

See [config_template.toml](config_template.toml) for a comprehensive template with all options.

## Documentation

- **User guide**: Build with `mkdocs build` and view with `mkdocs serve` at http://127.0.0.1:8000/
- **Doxygen**: Build with `make quandary_doxygen` from build directory, view at `build/docs/doxygen/html/index.html`
- **Online docs**: https://software.llnl.gov/quandary/

## Contributing

Contributions are welcome. Create a pull request to `main` after regression tests pass. The project is under active development.
