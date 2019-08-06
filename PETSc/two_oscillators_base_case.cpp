
static char help[] ="Solves the Liouville-von-Neumann equations, two oscillators.\n\
Input parameters:\n\
  -nlevels <int>      : Set the number of levels (default: 2) \n\
  -noscillators <int> : Set the number of oscillators (default: 2) \n\
  -ntime <int>        : Set the number of time steps \n\
  -dt <double>        : Set the time step size \n\
  -w  <double>        : Set the oscillator frequency\n\n";

#include <petscts.h>


/*
   Application context contains data needed to perform a time step.
*/
typedef struct {
  Vec         s;       /* global exact solution vector */
  PetscInt    n;       /* State space dimension */
  PetscInt    N;       /* Dimension of vectorized system */
  Mat         IKbMbd, bMbdTKI, aPadTKI, IKaPad, A, B;
  PetscReal   w;       /* Oscillator frequencies */
} AppCtx;


/*  Declare external routines */
extern PetscErrorCode SetUpMatrices(AppCtx*);
extern PetscErrorCode InitialConditions(Vec,AppCtx*);
extern PetscErrorCode ExactSolution(PetscReal,Vec,AppCtx*);
extern PetscScalar F(PetscReal,AppCtx*);
extern PetscScalar G(PetscReal,AppCtx*);
extern PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);



int main(int argc,char **argv)
{
  PetscInt       nlevels;      // Number of system levels (currently 2)
  PetscInt       noscillators; // Number of oscillators (currently 2)
  PetscInt       ntime;        // Number of time steps
  PetscReal      dt;           // Time step size
  PetscReal      total_time;   // Total end time T
  PetscInt       n;            // State space dimension (nlevels^noscillators)
  PetscInt       N;            // Dimension of vectorized system
  Vec            x;            // Solution vector
  Vec            e;            // Error to analytical solution
  Mat            M;            // ??
  AppCtx         appctx;       // Application context 
  TS             ts;           // timestepping context 
  PetscReal      w;            // Oscillator frequency

  PetscReal t, x_norm, s_norm, e_norm;

  PetscErrorCode ierr;
  PetscMPIInt    mpisize;

  /* Initialize Petsc */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&mpisize);CHKERRQ(ierr);
  if (mpisize != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

  /* Set default constants */
  nlevels      = 2;
  noscillators = 2;
  ntime        = 100;
  dt           = 0.01;
  w            = 1.0;

  /* Parse command line arguments to overwrite default constants */
  PetscOptionsGetInt(NULL,NULL,"-nlevels",&nlevels,NULL);
  PetscOptionsGetInt(NULL,NULL,"-noscillators",&nlevels,NULL);
  PetscOptionsGetInt(NULL,NULL,"-ntime",&ntime,NULL);
  PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);

  /* Sanity check */
  if (noscillators != 2 || nlevels != 2)
  {
    printf("\nERROR: Current only 2 levels and 2 oscillators are supported.\n You chose %d levels, %d oscillators.\n\n", nlevels, noscillators);
    exit(0);
  }
 
  /* Initialize parameters */
  n          = (PetscInt) pow(nlevels,noscillators); 
  N          = n * n;
  total_time = ntime * dt;

  printf("System with %d noscillators %d nlevels. \n", noscillators, nlevels);
  printf("Time horizon:   [0,%f]\n", total_time);
  printf("Number of time steps: %d\n", ntime);
  printf("Time step size: %f\n", dt );


  /* Initialize the App coefficients */
  appctx.n = n;
  appctx.N = N;
  appctx.w = w;


  /*
     Create vectors for approximate (x) and exact (s) solution, and error (e)
  */
  ierr = VecCreateSeq(PETSC_COMM_SELF,2*N,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&appctx.s);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&e);CHKERRQ(ierr);

  /* Create Petsc's timestepping context */
  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);


  /* Set up the system Hamiltonian matrices */
  SetUpMatrices(&appctx);

  /* Set up M (What is M?) */
  ierr = MatCreate(PETSC_COMM_SELF,&M);CHKERRQ(ierr);
  ierr = MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE, N*2,N*2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(M);CHKERRQ(ierr);
  ierr = MatSetUp(M);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Prepare Petsc's Time-stepper */
  ierr = TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,&appctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,M,M,RHSJacobian,&appctx);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);

  /* Set Time-stepping options */
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,ntime);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,total_time);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* Set the initial conditions */
  ierr = InitialConditions(x,&appctx);CHKERRQ(ierr);
  // VecView(x, PETSC_VIEWER_STDOUT_WORLD);

  /* Prepare output */
  FILE *logfile, *sfile, *xfile;
  logfile = fopen("out_log.dat", "w");
  sfile = fopen("out_analytic.dat", "w");
  xfile = fopen("out_x.dat", "w");
  fprintf(logfile, "# istep  time    ||x||                 ||analytic||           rel. error\n");
  printf("# istep  time    ||x||                 ||analytic||           rel. error\n");


  /* Run the timestepping loop */
  for(PetscInt istep = 1; istep <= ntime; istep++) {

    /* Step forward one time step */
    TSStep(ts);

    /* Get the exact solution at current time step */
    TSGetTime(ts, &t);
    ierr = ExactSolution(t,appctx.s,&appctx);CHKERRQ(ierr);

    /* Compute the relative error */
    ierr = VecWAXPY(e,-1.0,x,appctx.s);CHKERRQ(ierr);
    ierr = VecNorm(e,NORM_2,&e_norm);CHKERRQ(ierr);
    ierr = VecNorm(appctx.s,NORM_2,&s_norm);CHKERRQ(ierr);
    e_norm = e_norm / s_norm;

    /* Output */
    ierr = VecNorm(x,NORM_2,&x_norm);CHKERRQ(ierr);
    fprintf(logfile,"%3d  %1.5f  %1.14e  %1.14e  %1.14e\n",istep,(double)t, x_norm, s_norm,(double)e_norm);
    printf("%3d  %1.5f  %1.14e  %1.14e  %1.14e\n",istep,(double)t, x_norm, s_norm,(double)e_norm);

    /* Write numeric and analytic solution to files */
    const PetscScalar *x_ptr, *s_ptr;
    VecGetArrayRead(x, &x_ptr);
    VecGetArrayRead(appctx.s, &s_ptr);
    for (int i = 0; i < 2*N; i++) 
    {
      fprintf(xfile, "%1.14e  ", x_ptr[i]);
      fprintf(sfile, "%1.14e  ", s_ptr[i]);
    }
    fprintf(xfile, "\n");
    fprintf(sfile, "\n");

  }


  /* Clean up */
  fclose(logfile);
  fclose(sfile);
  fclose(xfile);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = MatDestroy(&appctx.A);CHKERRQ(ierr);
  ierr = MatDestroy(&appctx.B);CHKERRQ(ierr);
  ierr = MatDestroy(&appctx.IKbMbd);CHKERRQ(ierr);
  ierr = MatDestroy(&appctx.bMbdTKI);CHKERRQ(ierr);
  ierr = MatDestroy(&appctx.aPadTKI);CHKERRQ(ierr);
  ierr = MatDestroy(&appctx.IKaPad);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.s);CHKERRQ(ierr);
  ierr = VecDestroy(&e);CHKERRQ(ierr);

  /* Finallize Petsc */
  ierr = PetscFinalize();
  return ierr;
}


/*
 *  Set the initial condition at time t_0
 *  Input:
 *     u - uninitialized solution vector (global)
 *     appctx - application context
 *  Output Parameter:
 *     u - vector with solution at initial time (global)
 */
PetscErrorCode InitialConditions(Vec x,AppCtx *appctx)
{
  ExactSolution(0,x,appctx);
  return 0;
}

/*
 *   Compute the exact solution at a given time.
 *   Input:
 *      t - current time
 *      s - vector in which exact solution will be computed
 *      appctx - application context
 *   Output:
 *      s - vector with the newly computed exact solution
 */
PetscErrorCode ExactSolution(PetscReal t,Vec s,AppCtx *appctx)
{
  PetscScalar    *s_localptr;
  PetscErrorCode ierr;

  /* Get a pointer to vector data. */
  ierr = VecGetArray(s,&s_localptr);CHKERRQ(ierr);

  /* Write the solution into the array locations. 
   *  Alternatively, we could use VecSetValues() or VecSetValuesLocal(). */
  PetscScalar phi = (1./4.) * (t - (1./appctx->w)*PetscSinScalar(appctx->w*t));
  PetscScalar theta = (1./4.) * (t + (1./appctx->w)*PetscCosScalar(appctx->w*t) - 1.);
  PetscScalar cosphi = PetscCosScalar(phi);
  PetscScalar costheta = PetscCosScalar(theta);
  PetscScalar sinphi = PetscSinScalar(phi);
  PetscScalar sintheta = PetscSinScalar(theta);

  s_localptr[0] = cosphi*costheta*cosphi*costheta;
  s_localptr[1] = -1.*cosphi*sintheta*cosphi*costheta;
  s_localptr[2] = 0.;
  s_localptr[3] = 0.;
  s_localptr[4] = -1.*cosphi*costheta*cosphi*sintheta;
  s_localptr[5] = cosphi*sintheta*cosphi*sintheta;
  s_localptr[6] = 0.;
  s_localptr[7] = 0.;
  s_localptr[8] = 0.;
  s_localptr[9] = 0.;
  s_localptr[10] = sinphi*costheta*sinphi*costheta;
  s_localptr[11] = -1.*sinphi*sintheta*sinphi*costheta;
  s_localptr[12] = 0.;
  s_localptr[13] = 0.;
  s_localptr[14] = -1.*sinphi*costheta*sinphi*sintheta;
  s_localptr[15] = sinphi*sintheta*sinphi*sintheta;
  s_localptr[16] = 0.;
  s_localptr[17] = 0.;
  s_localptr[18] = -1.*sinphi*costheta*cosphi*costheta;
  s_localptr[19] = sinphi*sintheta*cosphi*costheta;
  s_localptr[20] = 0.;
  s_localptr[21] = 0.;
  s_localptr[22] = sinphi*costheta*cosphi*sintheta;
  s_localptr[23] = -1.*sinphi*sintheta*cosphi*sintheta;
  s_localptr[24] = cosphi*costheta*sinphi*costheta;
  s_localptr[25] = -1.*cosphi*sintheta*sinphi*costheta;
  s_localptr[26] = 0.;
  s_localptr[27] = 0.;
  s_localptr[28] = -1.*cosphi*costheta*sinphi*sintheta;
  s_localptr[29] = cosphi*sintheta*sinphi*sintheta;
  s_localptr[30] = 0.;
  s_localptr[31] = 0.;

  /* Restore solution vector */
  ierr = VecRestoreArray(s,&s_localptr);CHKERRQ(ierr);
  return 0;
}


/*
 * Oscillator 1 (real part)
 */
PetscScalar F(PetscReal t,AppCtx *appctx)
{
  PetscScalar f = (1./4.) * (1. - PetscCosScalar(appctx->w*t));
  return f;
}

/* 
 * Oscillator 2 (imaginary part) 
 */
PetscScalar G(PetscReal t,AppCtx *appctx)
{
  PetscScalar g = (1./4.) * (1. - PetscSinScalar(appctx->w*t));
  return g;
}

/*
 * Evaluate the right-hand side Matrix (real, vectorized Hamiltonian system matrix)
 * TODO: Add comments inside this routine!
 */
PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec u,Mat M,Mat P,void *ctx)
{
  AppCtx         *appctx = (AppCtx*)ctx;
  PetscScalar f, g, q1[(appctx->N * appctx->N)], q2[(appctx->N * appctx->N)], q3[(appctx->N * appctx->N)];
  PetscInt idx[appctx->N], idxN[appctx->N];
  PetscErrorCode ierr;

  for(int i = 0; i < appctx->N; i++)
  {
    idx[i] = i;
    idxN[i] = i + appctx->N;
  }

  f = F(t, appctx);
  g = G(t, appctx);

  ierr = MatAXPY(appctx->A,g,appctx->IKbMbd,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(appctx->A,-1*g,appctx->bMbdTKI,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  ierr = MatAXPY(appctx->B,f,appctx->aPadTKI,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(appctx->B,-1*f,appctx->IKaPad,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  MatGetValues(appctx->A, appctx->N, idx, appctx->N, idx, q1);
  MatSetValues(M, appctx->N, idx, appctx->N, idx, q1, INSERT_VALUES);
  MatSetValues(M, appctx->N, idxN, appctx->N, idxN, q1, INSERT_VALUES);

  MatGetValues(appctx->B, appctx->N, idx, appctx->N, idx, q3);
  MatSetValues(M, appctx->N, idxN, appctx->N, idx, q3, INSERT_VALUES);
  for(int i = 0; i < appctx->N * appctx->N; i++)
  {
    q2[i] = -1 * q3[i];
  }
  MatSetValues(M, appctx->N, idx, appctx->N, idxN, q2, INSERT_VALUES);

  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  return 0;
}




/* 
 * Set up the system Hamiltonian matrices 
 * TODO: Add comments in this routine!
 */
PetscErrorCode SetUpMatrices(AppCtx *appctx)
{
  PetscInt       i, j;
  PetscScalar    v[1];
  PetscErrorCode ierr;

  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,appctx->N,appctx->N,1,NULL,&appctx->IKbMbd);CHKERRQ(ierr);
  ierr = MatSetFromOptions(appctx->IKbMbd);CHKERRQ(ierr);
  ierr = MatSetUp(appctx->IKbMbd);CHKERRQ(ierr);

  i = 1;
  j = 0;
  v[0] = -1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 0;
  j = 1;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 3;
  j = 2;
  v[0] = -1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 2;
  j = 3;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 5;
  j = 4;
  v[0] = -1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 4;
  j = 5;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 7;
  j = 6;
  v[0] = -1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 6;
  j = 7;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 9;
  j = 8;
  v[0] = -1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 8;
  j = 9;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 11;
  j = 10;
  v[0] = -1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 10;
  j = 11;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 13;
  j = 12;
  v[0] = -1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 12;
  j = 13;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 15;
  j = 14;
  v[0] = -1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 14;
  j = 15;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(appctx->IKbMbd,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(appctx->IKbMbd,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

///////////////////////////////////////////////////////////////////////////////

  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,appctx->N,appctx->N,1,NULL,&appctx->bMbdTKI);CHKERRQ(ierr);
  ierr = MatSetFromOptions(appctx->bMbdTKI);CHKERRQ(ierr);
  ierr = MatSetUp(appctx->bMbdTKI);CHKERRQ(ierr);

  i = 4;
  j = 0;
  v[0] = 1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 5;
  j = 1;
  v[0] = 1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 6;
  j = 2;
  v[0] = 1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 7;
  j = 3;
  v[0] = 1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 0;
  j = 4;
  v[0] = -1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 1;
  j = 5;
  v[0] = -1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 2;
  j = 6;
  v[0] = -1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 3;
  j = 7;
  v[0] = -1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 12;
  j = 8;
  v[0] = 1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 13;
  j = 9;
  v[0] = 1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 14;
  j = 10;
  v[0] = 1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 15;
  j = 11;
  v[0] = 1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 8;
  j = 12;
  v[0] = -1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 9;
  j = 13;
  v[0] = -1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 10;
  j = 14;
  v[0] = -1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 11;
  j = 15;
  v[0] = -1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(appctx->bMbdTKI,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(appctx->bMbdTKI,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

///////////////////////////////////////////////////////////////////////////////

  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,appctx->N,appctx->N,1,NULL,&appctx->aPadTKI);CHKERRQ(ierr);
  ierr = MatSetFromOptions(appctx->aPadTKI);CHKERRQ(ierr);
  ierr = MatSetUp(appctx->aPadTKI);CHKERRQ(ierr);

  i = 8;
  j = 0;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 9;
  j = 1;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 10;
  j = 2;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 11;
  j = 3;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 12;
  j = 4;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 13;
  j = 5;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 14;
  j = 6;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 15;
  j = 7;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 0;
  j = 8;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 1;
  j = 9;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 2;
  j = 10;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 3;
  j = 11;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 4;
  j = 12;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 5;
  j = 13;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 6;
  j = 14;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 7;
  j = 15;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(appctx->aPadTKI,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(appctx->aPadTKI,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

////////////////////////////////////////////////////////////////////////////////

  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,appctx->N,appctx->N,1,NULL,&appctx->IKaPad);CHKERRQ(ierr);
  ierr = MatSetFromOptions(appctx->IKaPad);CHKERRQ(ierr);
  ierr = MatSetUp(appctx->IKaPad);CHKERRQ(ierr);

  i = 2;
  j = 0;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 3;
  j = 1;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 0;
  j = 2;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 1;
  j = 3;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 6;
  j = 4;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 7;
  j = 5;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 4;
  j = 6;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 5;
  j = 7;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 10;
  j = 8;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 11;
  j = 9;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 8;
  j = 10;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 9;
  j = 11;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 14;
  j = 12;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 15;
  j = 13;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 12;
  j = 14;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 13;
  j = 15;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(appctx->IKaPad,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(appctx->IKaPad,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

//////////////////////////////////////////////////////////////////////////////

  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,appctx->N,appctx->N,0,NULL,&appctx->A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(appctx->A);CHKERRQ(ierr);
  ierr = MatSetUp(appctx->A);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(appctx->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(appctx->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,appctx->N,appctx->N,0,NULL,&appctx->B);CHKERRQ(ierr);
  ierr = MatSetFromOptions(appctx->B);CHKERRQ(ierr);
  ierr = MatSetUp(appctx->B);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(appctx->B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(appctx->B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  return 0;
}
