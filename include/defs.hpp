
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define SQR(x) (x)*(x)

#pragma once

/* Available lindblad types */
enum class LindbladType {NONE, DECAY, DEPHASE, BOTH};

/* Available types of initial conditions */
enum class InitialConditionType {FROMFILE, PURE, ENSEMBLE, DIAGONAL, BASIS, THREESTATES, NPLUSONE, PERFORMANCE};

/* Types of optimization targets: Either gate optimization or pure state preparation */
enum class TargetType {GATE,      // \rho_target = V\rho(0) V^\dagger
                       PURE,      // \rho_target = e_m e_m^\dagger for some integer m
                       FROMFILE}; // \rho_target will be read from file. Format same as initial condition (one column with vectorized density matrix, fist all real then all imaginary parts)

/* Typye of objective functions */
enum class ObjectiveType {JFROBENIUS,    // weighted Frobenius norm: 1/2 * ||\rho_target - rho(T)||^2_F / w, where w = purity of \rho_target
                          JTRACE,        // weighted Hilber-Schmidt overlap: 1 - Tr(\rho_target^\dagger rho(T)) / w, where w = purity of \rho_target
                          JMEASURE};     // Measure a pure state: Tr(O_m \rho(T)) for observable O_m

/* Linear solver */
enum class LinearSolverType{
  GMRES,   // uses Petsc's GMRES solver
  NEUMANN   // uses Neuman power iterations 
};

/* Solver run type */
enum class RunType {
  SIMULATION,        // Runs one simulation to compute the objective function (forward)
  GRADIENT,          // Runs a simulation followed by the adjoint for gradient computation (forward & backward)
  OPTIMIZATION,      // Runs optimization iterations
  EVALCONTROLS,      // Runs optimization iterations
  NONE               // Don't run anything.
};

enum class ControlType {
  NONE,       // Non-controllable
  BSPLINE,    // Control paremters are the amplitudes of 2nd order BSpline basis functions
  BSPLINEAMP, // Control paremters are the amplitudes of BSpline basis functions. ONLY FOR AMPLITUDE
  STEP,       // Control parameter is the width of a step envelop function for a given amplitude
  BSPLINE0,   // Zeroth order Bspline (piece-wise constant)
};
