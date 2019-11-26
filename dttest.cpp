
static char help[] ="Solves a simple time-dependent linear PDE (the heat equation).\n\
Input parameters include:\n\
  -m <points>, where <points> = number of grid points\n\
  -time_dependent_rhs : Treat the problem as having a time-dependent right-hand side\n\
  -use_ifunc          : Use IFunction/IJacobian interface\n\
  -debug              : Activate debugging printouts\n\
  -nox                : Deactivate x-window graphics\n\n";

/*
   Concepts: TS^time-dependent linear problems
   Concepts: TS^heat equation
   Concepts: TS^diffusion equation
   Processors: 1
*/

/* ------------------------------------------------------------------------

   This program solves the one-dimensional heat equation (also called the
   diffusion equation),
       u_t = u_xx,
   on the domain 0 <= x <= 1, with the boundary conditions
       u(t,0) = 0, u(t,1) = 0,
   and the initial condition
       u(0,x) = sin(6*pi*x) + 3*sin(2*pi*x).
   This is a linear, second-order, parabolic equation.

   We discretize the right-hand side using finite differences with
   uniform grid spacing h:
       u_xx = (u_{i+1} - 2u_{i} + u_{i-1})/(h^2)
   We then demonstrate time evolution using the various TS methods by
   running the program via
       ex3 -ts_type <timestepping solver>

   We compare the approximate solution with the exact solution, given by
       u_exact(x,t) = exp(-36*pi*pi*t) * sin(6*pi*x) +
                      3*exp(-4*pi*pi*t) * sin(2*pi*x)

   Notes:
   This code demonstrates the TS solver interface to two variants of
   linear problems, u_t = f(u,t), namely
     - time-dependent f:   f(u,t) is a function of t
     - time-independent f: f(u,t) is simply f(u)

    The parallel version of this code is ts/examples/tutorials/ex4.c

  ------------------------------------------------------------------------- */

/*
   Include "petscts.h" so that we can use TS solvers.  Note that this file
   automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h  - vectors
     petscmat.h  - matrices
     petscis.h     - index sets            petscksp.h  - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h   - preconditioners
     petscksp.h   - linear solvers        petscsnes.h - nonlinear solvers
*/

#include <petscts.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/
typedef struct {
  Vec         solution;          /* global exact solution vector */  /* error norms */
  Mat         A;                 /* RHS mat, used with IFunction interface */
  PetscReal    lambda;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode InitialConditions(Vec,AppCtx*);
extern PetscErrorCode ExactSolution(PetscReal,Vec,AppCtx*);
extern PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);


int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  FILE *file;
  file = fopen("dttestout.txt", "w");
  for(PetscReal dt = 1.0; dt > 0.0000000001; dt = dt / 10) {
  AppCtx         appctx;                 /* user-defined application context */
  TS             ts;                     /* timestepping context */
  Mat            A;                      /* matrix data structure */
  Vec            u, e;                      /* approximate solution vector */
  PetscReal      time_total_max = 3.0; /* default max total time */
  PetscInt       time_steps_max = (int)(time_total_max / dt);   /* default max timesteps */


  PetscMPIInt    size;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");


  appctx.lambda = -1;

  ierr = PetscPrintf(PETSC_COMM_SELF,"Solving a linear TS problem on 1 processor\n");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create vector data structures for approximate and exact solutions
  */
  ierr = VecCreateSeq(PETSC_COMM_SELF,1,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.solution);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&e);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set up displays to show graphs of the solution and error
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set optional user-defined monitoring routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*ierr = TSMonitorSet(ts,Monitor,&appctx,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

     Create matrix data structure; set matrix evaluation routine.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,1,1);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);



  appctx.A = A;
      /*
         For linear problems with a time-independent f(u) in the equation
         u_t = f(u), the user provides the discretized right-hand-side
         as a matrix only once, and then sets the special Jacobian evaluation
         routine TSComputeRHSJacobianConstant() which will NOT recompute the Jacobian.
      */
      PetscInt i = 0;
      PetscScalar v[1];
      v[0] = appctx.lambda;
      ierr = MatSetValues(A,1,&i,1,&i,v,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      MatView(A, PETSC_VIEWER_STDOUT_WORLD);

  /*
     Set and option to indicate that we will never add a new nonzero location
     to the matrix. If we do, it will generate an error.
  */
    ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);

    ierr = TSSetRHSFunction(ts,NULL,RHSFunction,&appctx);CHKERRQ(ierr);
    ierr = TSSetRHSJacobian(ts,A,A,RHSJacobian,&appctx);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solution vector and initial timestep
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,u);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize timestepping solver:
       - Set the solution method to be the Backward Euler method.
       - Set timestepping duration info
     Then set runtime options, which can override these defaults.
     For example,
          -ts_max_steps <maxsteps> -ts_max_time <maxtime>
     to override the defaults set by TSSetMaxSteps()/TSSetMaxTime().
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSSetMaxSteps(ts,time_steps_max);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,time_total_max);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the problem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Evaluate initial conditions
  */
  ierr = InitialConditions(u,&appctx);CHKERRQ(ierr);
  VecView(u, PETSC_VIEWER_STDOUT_WORLD);


  /*
     Run the timestepping solver
  */

  PetscReal time, s_norm, e_norm;
  double uval;
  for(PetscInt step = 1; step <= time_steps_max; step++) {
    TSSetPreStep(ts, NULL);
    TSStep(ts);
    TSGetTime(ts, &time);

    ierr = ExactSolution(time,appctx.solution,&appctx);CHKERRQ(ierr);
    const PetscScalar *a, *b;
    ierr = VecGetArrayRead(u, &a); CHKERRQ(ierr);
    ierr = VecGetArrayRead(appctx.solution, &b); CHKERRQ(ierr);
    uval = (double)PetscRealPart(a[0]);
    double sval = (double)PetscRealPart(b[0]);
    ierr = VecRestoreArrayRead(u,&a);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(appctx.solution,&b);CHKERRQ(ierr);
    ierr = VecWAXPY(e,-1.0,u,appctx.solution);CHKERRQ(ierr);
    ierr = VecNorm(appctx.solution,NORM_2,&s_norm);CHKERRQ(ierr);
    ierr = VecScale(e,(1 / s_norm));
    ierr = VecNorm(e,NORM_2,&e_norm);CHKERRQ(ierr);
  }

  fprintf(file,"%1.14e  %1.14e\n",dt,uval);
  /*ierr = TSSolve(ts,u);CHKERRQ(ierr);
  /*ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     View timestepping solver info
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */



  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.solution);CHKERRQ(ierr);
  ierr = VecDestroy(&e);CHKERRQ(ierr);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_view).
  */
}
  fclose(file);
  ierr = PetscFinalize();
  return ierr;
}
/* --------------------------------------------------------------------- */
/*
   InitialConditions - Computes the solution at the initial time.

   Input Parameter:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
*/
PetscErrorCode InitialConditions(Vec u,AppCtx *appctx)
{
  PetscScalar    *u_localptr;
  PetscErrorCode ierr;

  /*
    Get a pointer to vector data.
    - For default PETSc vectors, VecGetArray() returns a pointer to
      the data array.  Otherwise, the routine is implementation dependent.
    - You MUST call VecRestoreArray() when you no longer need access to
      the array.
    - Note that the Fortran interface to VecGetArray() differs from the
      C version.  See the users manual for details.
  */
  ierr = VecGetArray(u,&u_localptr);CHKERRQ(ierr);

  /*
     We initialize the solution array by simply writing the solution
     directly into the array locations.  Alternatively, we could use
     VecSetValues() or VecSetValuesLocal().
  */
  u_localptr[0] = 1;

  /*
     Restore vector
  */
  ierr = VecRestoreArray(u,&u_localptr);CHKERRQ(ierr);

  /*
     Print debugging information if desired
  */

  return 0;
}
/* --------------------------------------------------------------------- */
/*
   ExactSolution - Computes the exact solution at a given time.

   Input Parameters:
   t - current time
   solution - vector in which exact solution will be computed
   appctx - user-defined application context

   Output Parameter:
   solution - vector with the newly computed exact solution
*/
PetscErrorCode ExactSolution(PetscReal t,Vec solution,AppCtx *appctx)
{
  PetscScalar    *s_localptr,tc = t;
  PetscErrorCode ierr;

  /*
     Get a pointer to vector data.
  */
  ierr = VecGetArray(solution,&s_localptr);CHKERRQ(ierr);

  /*
     Simply write the solution directly into the array locations.
     Alternatively, we culd use VecSetValues() or VecSetValuesLocal().
  */
  s_localptr[0] = PetscExpScalar(appctx->lambda*tc);

  /*
     Restore vector
  */
  ierr = VecRestoreArray(solution,&s_localptr);CHKERRQ(ierr);
  return 0;
}
/* --------------------------------------------------------------------- */
/*
   Monitor - User-provided routine to monitor the solution computed at
   each timestep.  This example plots the solution and computes the
   error in two different norms.

   This example also demonstrates changing the timestep via TSSetTimeStep().

   Input Parameters:
   ts     - the timestep context
   step   - the count of the current step (with 0 meaning the
             initial condition)
   time   - the current time
   u      - the solution at this timestep
   ctx    - the user-provided context for this monitoring routine.
            In this case we use the application context which contains
            information about the problem size, workspace and the exact
            solution.
*/

// PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal time,Vec u,void *ctx)
// {
//   AppCtx         *appctx = (AppCtx*) ctx;   /* user-defined application context */
//   PetscErrorCode ierr;
//   PetscReal      norm_max,dt;
//
//   /*
//      View a graph of the current iterate
//   */
//
//
//   /*
//      Compute the exact solution
//   */
//   ierr = ExactSolution(time,appctx->solution,appctx);CHKERRQ(ierr);
//
//   /*
//      Print debugging information if desired
//   */
//
//   /*
//      Compute the 2-norm and max-norm of the error
//   */
//   const PetscScalar *a, *b;
//   ierr = VecGetArrayRead(u, &a); CHKERRQ(ierr);
//   ierr = VecGetArrayRead(appctx->solution, &b); CHKERRQ(ierr);
//   double aval = (double)PetscRealPart(a[0]);
//   double bval = (double)PetscRealPart(b[0]);
//   ierr = VecRestoreArrayRead(u,&a);CHKERRQ(ierr);
//   ierr = VecRestoreArrayRead(appctx->solution,&b);CHKERRQ(ierr);
//   ierr   = VecAXPY(appctx->solution,-1.0,u);CHKERRQ(ierr);
//   ierr   = VecNorm(appctx->solution,NORM_MAX,&norm_max);CHKERRQ(ierr);
//   if (norm_max < 1e-14) norm_max = 0;
//
//   ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
//
//
//
//
//   ierr = PetscPrintf(PETSC_COMM_WORLD,"Timestep %3D: step size = %g, time = %g, u value = %1.6e, exact value = %g, max norm error = %g\n",step,(double)dt,(double)time, aval, bval,(double)norm_max);CHKERRQ(ierr);
//
//
//   appctx->norm_max += norm_max;
//
//   /*
//      View a graph of the error
//   */
//
//   /*
//      Print debugging information if desired
//   */
//
//   return 0;
// }

PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec u,Vec f,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;
  PetscErrorCode ierr;
  ierr = MatMult(appctx->A, u, f);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec u,Mat A,Mat B,void *ctx)
{
  return 0;
}
