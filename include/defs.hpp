
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define SQR(x) (x)*(x)

#pragma once

/**
 * @brief Available Lindblad operator types for open quantum systems.
 *
 * Defines the types of dissipation and decoherence operators that can be
 * applied to the quantum system in the Lindblad master equation.
 */
enum class LindbladType {
  NONE,    ///< No Lindblad operators (closed system)
  DECAY,   ///< Decay operators only
  DEPHASE, ///< Dephasing operators only
  BOTH     ///< Both decay and dephasing operators
};

/**
 * @brief Available types of initial conditions for quantum systems.
 *
 * Defines how the initial quantum state is specified and prepared
 * for simulation or optimization.
 */
enum class InitialConditionType {
  FROMFILE,    ///< Read initial condition from file
  PURE,        ///< Pure state initial condition
  ENSEMBLE,    ///< Ensemble of states
  DIAGONAL,    ///< Diagonal density matrix
  BASIS,       ///< Basis state
  THREESTATES, ///< Three-state system
  NPLUSONE,    ///< N+1 state system
  PERFORMANCE  ///< Performance test configuration
};

/**
 * @brief Types of optimization targets for quantum control.
 *
 * Defines the target quantum state or operation for optimization.
 */
enum class TargetType {
  GATE,      ///< Gate optimization: \f$\rho_{\text{target}} = V\rho(0) V^\dagger\f$
  PURE,      ///< Pure state preparation: \f$\rho_{\text{target}} = e_m e_m^\dagger\f$ for some integer \f$m\f$
  FROMFILE   ///< Target read from file with vectorized density matrix format
};

/**
 * @brief Types of objective functions for quantum control optimization.
 *
 * Defines different metrics for measuring the quality of quantum control.
 */
enum class ObjectiveType {
  JFROBENIUS, ///< Weighted Frobenius norm: \f$\frac{1}{2} \frac{\|\rho_{\text{target}} - \rho(T)\|_F^2}{w}\f$, where \f$w\f$ = purity of \f$\rho_{\text{target}}\f$
  JTRACE,     ///< Weighted Hilbert-Schmidt overlap: \f$1 - \frac{\text{Tr}(\rho_{\text{target}}^\dagger \rho(T))}{w}\f$, where \f$w\f$ = purity of \f$\rho_{\text{target}}\f$
  JMEASURE    ///< Pure state measurement: \f$\text{Tr}(O_m \rho(T))\f$ for observable \f$O_m\f$
};

/**
 * @brief Available linear solver types for quantum dynamics.
 *
 * Defines the numerical methods used to solve linear systems
 * arising in quantum dynamics simulations.
 */
enum class LinearSolverType{
  GMRES,   ///< Uses Petsc's GMRES solver
  NEUMANN  ///< Uses Neuman power iterations
};

/**
 * @brief Types of solver execution modes.
 *
 * Defines what type of computation the solver should perform.
 */
enum class RunType {
  SIMULATION,   ///< Runs one simulation to compute the objective function (forward)
  GRADIENT,     ///< Runs a simulation followed by the adjoint for gradient computation (forward & backward)
  OPTIMIZATION, ///< Runs optimization iterations
  EVALCONTROLS, ///< Runs optimization iterations
  NONE          ///< Don't run anything
};

/**
 * @brief Types of control parameterizations for quantum control pulses.
 *
 * Defines how control pulses are mathematically represented and parameterized
 * for optimization and simulation.
 */
enum class ControlType {
  NONE,       ///< Non-controllable
  BSPLINE,    ///< Control parameters are the amplitudes of 2nd order BSpline basis functions
  BSPLINEAMP, ///< Control parameters are the amplitudes of BSpline basis functions (amplitude only)
  STEP,       ///< Control parameter is the width of a step envelope function for a given amplitude
  BSPLINE0    ///< Zeroth order Bspline (piece-wise constant)
};
