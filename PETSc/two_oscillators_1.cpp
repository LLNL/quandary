
static char help[] ="Solves a simple time-dependent linear PDE (the heat equation).\n\
Input parameters include:\n\
  -m <points>, where <points> = number of grid points\n\
  -time_dependent_rhs : Treat the problem as having a time-dependent right-hand side\n\
  -use_ifunc          : Use IFunction/IJacobian interface\n\
  -debug              : Activate debugging printouts\n\
  -nox                : Deactivate x-window graphics\n\n";

#include <petscts.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/
typedef struct {
  Vec         solution, p0;          /* global exact solution vector */
  PetscInt    n, N;

} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode InitialConditions(Vec,AppCtx*);
extern PetscErrorCode ExactSolution(PetscReal,Vec,AppCtx*);
extern PetscErrorCode F1(PetscReal,PetscScalar*,AppCtx*);
extern PetscErrorCode F2(PetscReal,PetscScalar*,AppCtx*);
extern PetscErrorCode G1(PetscReal,PetscScalar*,AppCtx*);
extern PetscErrorCode G2(PetscReal,PetscScalar*,AppCtx*);
extern PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);


int main(int argc,char **argv)
{
  PetscInt       n = 2; //number of levels
  PetscInt       N = n*n; //dimension of vectors and scalars
  AppCtx         appctx;                 /* user-defined application context */
  TS             ts;                     /* timestepping context */
  Mat            Hs, IKaMad, IKbMbd, aMadTKI, bMbdTKI, HsTKI, aPadTKI, bPbdTKI, IKHs, IaPad, IKbPbd, A, B;                      /* matrix data structure */
  Vec            x, e;                      /* approximate solution vector */
  PetscScalar f1, f2, g1, g2;
  PetscReal      time_total_max = 100.0; /* default max total time */
  PetscInt       time_steps_max = 100;   /* default max timesteps */
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscReal      dt;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

  appctx.n = n;
  appctx.N = N;

  ierr = PetscPrintf(PETSC_COMM_SELF,"Solving a linear TS problem on 1 processor\n");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create vector data structures for approximate and exact solutions
  */
  ierr = VecCreateSeq(PETSC_COMM_SELF,2*N,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&appctx.solution);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&e);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

     Create matrix data structure; set matrix evaluation routine.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
//
  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,1,1);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

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
  dt   = 0.0001;
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);

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

  /*
     Evaluate initial conditions
  */
  ierr = InitialConditions(u,&appctx);CHKERRQ(ierr);
  VecView(u, PETSC_VIEWER_STDOUT_WORLD);

  /*
     Run the timestepping solver
  */
  FILE *file;
  file = fopen("output.txt", "w");
  PetscReal t, s_norm, e_norm;
  for(PetscInt step = 1; step <= t_steps_max; step++) {
    TSSetPreStep(ts, NULL);
    TSStep(ts);
    TSGetTime(ts, &t);



    ierr = F1(t,&f1,&appctx);CHKERRQ(ierr);
    ierr = F2(t,&f2,&appctx);CHKERRQ(ierr);
    ierr = G1(t,&g1,&appctx);CHKERRQ(ierr);
    ierr = G2(t,&g2,&appctx);CHKERRQ(ierr);
    ierr = ExactSolution(t,appctx.solution,&appctx);CHKERRQ(ierr);

    ierr = MatAXPY(A,f2,IKaMad,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(A,g2,IKbMbd,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(A,-1*f2,aMadTKI,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(A,-1*g2,bMbdTKI,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

    ierr = MatAXPY(B,1,HsTKI,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(B,f1,aPadTKI,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(B,g1,bPbdTKI,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(B,-1,IKHs,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(B,-1*f1,IKaPad,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatAXPY(B,-1*g1,IKbPbd,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

    const PetscScalar *a, *b;
    ierr = VecGetArrayRead(x, &a); CHKERRQ(ierr);
    ierr = VecGetArrayRead(appctx.solution, &b); CHKERRQ(ierr);

    ierr = VecWAXPY(e,-1.0,x,appctx.solution);CHKERRQ(ierr);
    ierr = VecNorm(appctx.solution,NORM_2,&s_norm);CHKERRQ(ierr);
    ierr = VecScale(e,(1 / s_norm));
    ierr = VecNorm(e,NORM_2,&e_norm);CHKERRQ(ierr);

    fprintf(file,"%3d  %1.14e  %1.14e  %1.14e  %1.14e\n",step,(double)t, uval, sval,(double)e_norm);

    ierr = VecRestoreArrayRead(u,&a);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(appctx.solution,&b);CHKERRQ(ierr);
  }
  fclose(file);

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
PetscErrorCode InitialConditions(Vec x,AppCtx *appctx)
{
  PetscScalar    *x_localptr;
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
  ierr = VecGetArray(x,&x_localptr);CHKERRQ(ierr);

  /*
     We initialize the solution array by simply writing the solution
     directly into the array locations.  Alternatively, we could use
     VecSetValues() or VecSetValuesLocal().
  */
  for(int i = 0; i < appctx->N; i++) {
    x_localptr[i] = PetscRealPart(appctx->p[i]);  //u
    x_localptr[(appctx->N + i)] = PetscImaginaryPart(appctx->p[i]); //v
  }


  /*
     Restore vector
  */
  ierr = VecRestoreArray(x,&x_localptr);CHKERRQ(ierr);

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

PetscErrorCode F1(PetscReal t,PetscScalar *f1,AppCtx *appctx)
{
  PetscScalar    tc = t;
  PetscErrorCode ierr;

  *f1 = PetscExpScalar(tc);

  return 0;
}

PetscErrorCode F2(PetscReal t,PetscScalar *f2,AppCtx *appctx)
{
  PetscScalar    tc = t;
  PetscErrorCode ierr;

  *f2 = PetscExpScalar(tc);

  return 0;
}

PetscErrorCode G1(PetscReal t,PetscScalar *g1,AppCtx *appctx)
{
  PetscScalar    tc = t;
  PetscErrorCode ierr;

  *g1 = PetscExpScalar(tc);

  return 0;
}

PetscErrorCode G2(PetscReal t,PetscScalar *g2,AppCtx *appctx)
{
  PetscScalar    tc = t;
  PetscErrorCode ierr;

  *g2 = PetscExpScalar(tc);

  return 0;
}

PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec x,Vec f,void *ctx)
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
/* --------------------------------------------------------------------- */
