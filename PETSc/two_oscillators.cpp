#include "braid_test.h"
#include "braid_wrapper.c"


int main(int argc,char **argv)
{
  PetscInt       nlvl;         // Number of levels for each oscillator (currently 2)
  PetscInt       nosci;        // Number of oscillators (currently 2)
  PetscInt       nsys;         // Dimension of system state space (nlvl^nosci)
  PetscInt       nvec;         // Dimension of vectorized system (nsys^2)
  PetscInt       nreal;        // Dimension of real-valued system (2*nvec)
  PetscInt       ntime;        // Number of time steps
  PetscReal      dt;           // Time step size
  PetscReal      total_time;   // Total end time T
  Mat            M;            // System matrix for real-valued system
  TS             ts;           // Timestepping context
  PetscReal      w;            // Oscillator frequency
  braid_Core     braid_core;         // Core for XBraid simulation
  XB_App        *braid_app;          // XBraid's application context
  TS_App        *petsc_app;       // Petsc's application context

  FILE *sufile, *svfile, *ufile, *vfile;
  PetscErrorCode ierr;
  PetscMPIInt    mpisize;

  /* Initialize Petsc */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&mpisize);CHKERRQ(ierr);

  /* Set default constants */
  nlvl = 2;
  nosci = 2;
  ntime = 1000;
  dt = 0.0001;
  w = 1.0;

  /* Parse command line arguments to overwrite default constants */
  PetscOptionsGetInt(NULL,NULL,"-nlvl",&nlvl,NULL);
  PetscOptionsGetInt(NULL,NULL,"-nosci",&nlvl,NULL);
  PetscOptionsGetInt(NULL,NULL,"-ntime",&ntime,NULL);
  PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);

  /* Sanity check */
  if (nosci != 2 || nlvl != 2)
  {
    printf("\nERROR: Current only 2 levels and 2 oscillators are supported.\n You chose %d levels, %d oscillators.\n\n", nlvl, nosci);
    exit(0);
  }

  /* Initialize simulation parameters */
  nsys = (PetscInt) pow(nlvl,nosci);
  nvec = (PetscInt) pow(nsys,2);
  nreal = 2 * nvec;
  total_time = ntime * dt;
  printf("System with %d oscillators, %d levels. \n", nosci, nlvl);
  printf("Time horizon:   [0,%.1f]\n", total_time);
  printf("Number of time steps: %d\n", ntime);
  printf("Time step size: %f\n", dt );

  /* Open output files */
  sufile = fopen("out_u_exact.dat", "w");
  svfile = fopen("out_v_exact.dat", "w");
  ufile = fopen("out_u.dat", "w");
  vfile = fopen("out_v.dat", "w");


  /* Allocate right hand side matrix */
  ierr = MatCreate(PETSC_COMM_SELF,&M);CHKERRQ(ierr);
  ierr = MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE,nreal,nreal);CHKERRQ(ierr);
  ierr = MatSetFromOptions(M);CHKERRQ(ierr);
  ierr = MatSetUp(M);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Initialize Petsc's application context */
  petsc_app = (TS_App*) malloc(sizeof(TS_App));
  petsc_app->nvec = nvec;
  petsc_app->w = w;
  SetUpMatrices(petsc_app);

  /* Allocate and initialize Petsc's Time-stepper */
  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,petsc_app);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,M,M,RHSJacobian,petsc_app);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,ntime);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,total_time);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* Set up XBraid's applications structure */
  braid_app = (XB_App*) malloc(sizeof(XB_App));
  braid_app->petsc_app = petsc_app;
  braid_app->ts     = ts;
  braid_app->ufile  = ufile;
  braid_app->vfile  = vfile;
  braid_app->sufile = sufile;
  braid_app->svfile = svfile;

  /* Initialize Braid */
  braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, 0.0, total_time, ntime, braid_app, my_Step, my_Init, my_Clone, my_Free, my_Sum, my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack, &braid_core);
  
  /* Set Braid options */
  braid_SetPrintLevel( braid_core, 2);
  braid_SetAccessLevel( braid_core, 1);
  braid_SetMaxLevels(braid_core, 2);
  braid_SetNRelax(braid_core, -1, 1);
  braid_SetAbsTol(braid_core, 1e-6);
  braid_SetCFactor(braid_core, -1, 2);
  braid_SetMaxIter(braid_core, 10);
  braid_SetSeqSoln(braid_core, 0);
  int fmg = 0;
  if (fmg)
  {
     braid_SetFMG(braid_core);
  }
 
  /* Run braid */
  braid_Drive(braid_core);



#if 0
/* 
 * Testing time stepper convergence (dt-test) 
 */
  Vec x;      // numerical solution
  Vec exact;  // exact solution
  Vec error;  // error  
  double t;
  double error_norm, exact_norm;

  VecCreateSeq(PETSC_COMM_SELF,nreal,&x);
  VecCreateSeq(PETSC_COMM_SELF,nreal,&exact);
  VecCreateSeq(PETSC_COMM_SELF,nreal,&error);

  total_time = 10.0;
  printf("\n\n Running time-stepping convergence test... \n\n");
  printf(" Time horizon: [0, %.1f]\n\n", total_time);

  /* Decrease time step size */
  printf("   ntime      dt    error\n");
  for (int ntime = 10; ntime <= 1e+5; ntime = ntime * 10)
  {
    dt = total_time / ntime;

    /* Reset the time stepper */
    InitialConditions(x,petsc_app);
    TSSetTime(ts, 0.0); 
    TSSetTimeStep(ts,dt);
    TSSetMaxSteps(ts,ntime);
    TSSetSolution(ts, x);

    /* Run time-stepping loop */
    for(PetscInt istep = 0; istep <= ntime; istep++) 
    {
      TSStep(ts);
    }

    /* Compute the relative error at last time step (max-norm) */
    TSGetTime(ts, &t);
    ExactSolution(t,exact,petsc_app);
    VecWAXPY(error,-1.0,x, exact);
    VecNorm(error, NORM_INFINITY,&error_norm);
    VecNorm(exact, NORM_INFINITY,&exact_norm);
    error_norm = error_norm / exact_norm;

    /* Print error norm */
    printf("%8d   %1.e   %1.14e\n", ntime, dt, error_norm);

  }

  VecDestroy(&x);
  VecDestroy(&exact);
  VecDestroy(&error);

#endif

  /* Clean up */
  fclose(sufile);
  fclose(svfile);
  fclose(ufile);
  fclose(vfile);
  TSDestroy(&ts);CHKERRQ(ierr);
  MatDestroy(&M);CHKERRQ(ierr);
  MatDestroy(&petsc_app->A);CHKERRQ(ierr);
  MatDestroy(&petsc_app->B);CHKERRQ(ierr);
  MatDestroy(&petsc_app->IKbMbd);CHKERRQ(ierr);
  MatDestroy(&petsc_app->bMbdTKI);CHKERRQ(ierr);
  MatDestroy(&petsc_app->aPadTKI);CHKERRQ(ierr);
  MatDestroy(&petsc_app->IKaPad);CHKERRQ(ierr);
  free(petsc_app);

  /* Cleanup XBraid */
  braid_Destroy(braid_core);
  free(braid_app);

  /* Finallize Petsc */
  ierr = PetscFinalize();

  return ierr;
}


