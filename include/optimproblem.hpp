#include "math.h"
#include <assert.h>
#include <petsctao.h>
#include "defs.hpp"
#include "timestepper.hpp"
#ifdef WITH_BRAID
  #include "braid_wrapper.hpp"
#endif

#pragma once



class OptimProblem {

  public: 

  /* ODE stuff */
#ifdef WITH_BRAID
  myBraidApp* primalbraidapp;         /* Primal BraidApp to carry out PinT forward sim.*/
  myAdjointBraidApp* adjointbraidapp; /* Adjoint BraidApp to carry out PinT backward sim. */
#endif
  int ninit;                            /* Number of initial conditions to be considered (N^2, N, or 1) */
  int ninit_local;                      /* Local number of initial conditions on this processor */
  Vec rho_t0;                            /* Storage for initial condition of the ODE */
  Vec rho_t0_bar;                        /* Adjoint of ODE initial condition */
  InitialConditionType initcond_type;    /* Type of initial conditions */
  std::vector<int> initcond_IDs;         /* IDs of subsystem oscillators considered for initial conditions */

  TimeStepper* timestepper;

  /* MPI stuff */
  MPI_Comm comm_hiop, comm_init;
  int mpirank_braid, mpisize_braid;
  int mpirank_space, mpisize_space;
  int mpirank_optim, mpisize_optim;
  int mpirank_world, mpisize_world;
  int mpirank_init, mpisize_init;

  /* Optimization stuff */
  ObjectiveType objective_type;    /* Type of objective function (Gate, <N>, Groundstate, ... ) */
  std::vector<int> obj_oscilIDs;   /* List of oscillator IDs that are considered for the optimizer */
  std::vector<double> obj_weights; /* List of weights for averaging expected value objective */
  Gate  *targetgate;               /* Target gate */
  int ndesign;                     /* Number of global design parameters */
  double objective;                /* Holds current objective function value */
  double obj_cost;                 /* Cost function term in objective */
  double obj_regul;                /* Regularization term in objective */
  double obj_penal;                /* Penalty term in objective */
  double gnorm;                    /* Holds current norm of gradient */
  double gamma_tik;                /* Parameter for tikhonov regularization */
  double penalty_coeff;            /* Parameter multiplying integral penalty term */
  double penalty_exp;              /* Exponent inside integral penalty term (p) */
  double gatol;                    /* Stopping criterion based on absolute gradient norm */
  double grtol;                    /* Stopping criterion based on relative gradient norm */
  int maxiter;                     /* Stopping criterion based on maximum number of iterations */
  Tao tao;                        /* Petsc's Optimization solver */
  Vec xinit;                       /* Initial guess */
  Vec xlower, xupper;              /* Optimization bounds */
  std::string initguess_type;      /* Type of initial guess */
  std::vector<double> initguess_amplitudes; /* Initial amplitudes of controles, or NULL */
  double* mygrad;  /* Auxiliary */
  
  /* Output */
  Output* output;

  /* Constructor */
  OptimProblem(MapParam config, TimeStepper* timestepper_, MPI_Comm comm_hiop_, MPI_Comm comm_init_, int ninit_, Output* output_);
#ifdef WITH_BRAID
  OptimProblem(MapParam config, TimeStepper* timestepper_, myBraidApp* primalbraidapp_, myAdjointBraidApp* adjointbraidapp_, MPI_Comm comm_hiop_, MPI_Comm comm_init_, int ninit_, Output* output_);
#endif
  ~OptimProblem();

  /* Evaluate the objective function F(x) */
  double evalF(const Vec x);

  /* Evaluate gradient \nabla F(x) */
  void evalGradF(const Vec x, Vec G);

  /* Run optimization solver */
  void solve();

  /* Compute initial guess for optimization variables */
  void getStartingPoint(Vec x);

  /* Call this after TaoSolve() has finished to print out some information */
  void getSolution(Vec* opt);

  static void integral_penalty();
};

/* Monitor the optimization progress. This routine is called in each iteration of TaoSolve() */
PetscErrorCode TaoMonitor(Tao tao,void*ptr);

/* Petsc's Tao interface routine for evaluating the objective function f = f(x) */
PetscErrorCode TaoEvalObjective(Tao tao, Vec x, PetscReal *f, void*ptr);

/* Petsc's Tao interface routine for evaluating the gradient g = \nabla f(x) */
PetscErrorCode TaoEvalGradient(Tao tao, Vec x, Vec G, void*ptr);

/* Petsc's Tao interface routine for evaluating the gradient g = \nabla f(x) */
PetscErrorCode TaoEvalObjectiveAndGradient(Tao tao, Vec x, PetscReal *f, Vec G, void*ptr);


/* Compute local objective function J(rho(t)) */
double objectiveT(MasterEq* mastereq, ObjectiveType objective_type, const std::vector<int>& obj_oscilIDs, const std::vector<double>& obj_weights, const Vec state, const Vec rho_t0, Gate* targetgate);

/* Derivative of local objective function times obj_bar */
void objectiveT_diff(MasterEq* mastereq, ObjectiveType objective_type, const std::vector<int>& obj_oscilIDs, const std::vector<double>& obj_weights, Vec state, Vec state_bar, const Vec rho_t0, const double obj_bar, Gate* targetgate);
