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
  Vec            x;            // Solution vector
  Vec            e;            // Error to analytical solution
  Mat            M;            // System Matrix for real-valued system
  AppCtx         appctx;       // Application context
  TS             ts;           // Timestepping context
  PetscReal      w;            // Oscillator frequency


  PetscReal t, s_norm, e_norm;
  const PetscScalar *x_ptr, *s_ptr;

  PetscErrorCode ierr;
  PetscMPIInt    mpisize;

  /* Initialize Petsc */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&mpisize);CHKERRQ(ierr);
  if (mpisize != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

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

  /* Initialize parameters */
  nsys = (PetscInt) pow(nlvl,nosci);
  nvec = (PetscInt) pow(nsys,2);
  nreal = 2 * nvec;
  total_time = ntime * dt;

  printf("System with %d oscillators, %d levels. \n", nosci, nlvl);
  printf("Time horizon:   [0,%.1f]\n", total_time);
  printf("Number of time steps: %d\n", ntime);
  printf("Time step size: %f\n", dt );


  /* Initialize the App coefficients */
  appctx.nvec = nvec;
  appctx.w = w;


  /*
     Create vectors for approximate (x) and exact (s) solution, and error (e)
  */
  ierr = VecCreateSeq(PETSC_COMM_SELF,nreal,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&appctx.s);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&e);CHKERRQ(ierr);

  /* Create Petsc's timestepping context */
  ierr = TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);

  /* Allocate and initialize matrices for evaluating system Hamiltonian */
  SetUpMatrices(&appctx);

  /* Allocate system matrix */
  ierr = MatCreate(PETSC_COMM_SELF,&M);CHKERRQ(ierr);
  ierr = MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE,nreal,nreal);CHKERRQ(ierr);
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
  FILE *logfile, *sufile, *svfile, *ufile, *vfile;
  logfile = fopen("out_log.dat", "w");
  sufile = fopen("out_u_exact.dat", "w");
  svfile = fopen("out_v_exact.dat", "w");
  ufile = fopen("out_u.dat", "w");
  vfile = fopen("out_v.dat", "w");
  fprintf(logfile, "# istep  time    x[1]                 exact[1]           rel. error\n");
  printf("# istep  time    x[1]                 exact[1]           rel. error\n");


  /* Run the timestepping loop */
  for(PetscInt istep = 0; istep <= ntime; istep++) {

    /* Step forward one time step */
    TSStep(ts);

    /* Get the exact solution at current time step */
    TSGetTime(ts, &t);
    ierr = ExactSolution(t,appctx.s,&appctx);CHKERRQ(ierr);

    /* Compute the relative error (max-norm) */
    ierr = VecWAXPY(e,-1.0,x, appctx.s);CHKERRQ(ierr);
    ierr = VecNorm(e, NORM_INFINITY,&e_norm);CHKERRQ(ierr);
    ierr = VecNorm(appctx.s, NORM_INFINITY,&s_norm);CHKERRQ(ierr);
    e_norm = e_norm / s_norm;

    /* Output */
    VecGetArrayRead(x, &x_ptr);
    VecGetArrayRead(appctx.s, &s_ptr);


    fprintf(logfile, "%5d  %1.5f  %1.14e  %1.14e  %1.14e\n",istep,(double)t, x_ptr[1], s_ptr[1], (double)e_norm);
    printf("%5d  %1.5f  %1.14e  %1.14e  %1.14e\n",istep,(double)t, x_ptr[1], s_ptr[1], (double)e_norm);

    /* Write numeric and analytic solution to files */
    fprintf(ufile,  "%.2f  ", (double) t);
    fprintf(vfile,  "%.2f  ", (double) t);
    fprintf(sufile, "%.2f  ", (double) t);
    fprintf(svfile, "%.2f  ", (double) t);
    for (int i = 0; i < nreal; i++)
    {

      if (i < nvec) // real part
      {
        fprintf(ufile, "%1.14e  ", x_ptr[i]);  
        fprintf(sufile, "%1.14e  ", s_ptr[i]);
      }  
      else  // imaginary part
      {
        fprintf(vfile, "%1.14e  ", x_ptr[i]); 
        fprintf(svfile, "%1.14e  ", s_ptr[i]);
      }
      
    }
    fprintf(ufile, "\n");
    fprintf(vfile, "\n");
    fprintf(sufile, "\n");
    fprintf(svfile, "\n");

  }



#if 0
/* 
 * Testing time stepper convergence (dt-test) 
 */

  total_time = 10.0;
  printf("\n\n Running time-stepping convergence test... \n\n");
  printf(" Time horizon: [0, %.1f]\n\n", total_time);

  /* Decrease time step size */
  printf("   ntime      dt    error\n");
  for (int ntime = 10; ntime <= 1e+5; ntime = ntime * 10)
  {
    dt = total_time / ntime;

    /* Reset the time stepper */
    ierr = InitialConditions(x,&appctx);CHKERRQ(ierr);  
    TSSetTime(ts, 0.0); 
    TSSetTimeStep(ts,dt);
    TSSetMaxSteps(ts,ntime);

    /* Run time-stepping loop */
    for(PetscInt istep = 0; istep <= ntime; istep++) 
    {
      TSStep(ts);
      TSGetTime(ts, &t);
      VecGetArrayRead(x, &x_ptr);
      // printf("%5d  %1.5f  %1.14e \n",istep,(double)t, x_ptr[1]);
    }
    /* Compute the relative error at last time step (max-norm) */
    TSGetTime(ts, &t);
    ExactSolution(t,appctx.s,&appctx);CHKERRQ(ierr);
    VecWAXPY(e,-1.0,x, appctx.s);CHKERRQ(ierr);
    VecNorm(e, NORM_INFINITY,&e_norm);CHKERRQ(ierr);
    VecNorm(appctx.s, NORM_INFINITY,&s_norm);CHKERRQ(ierr);
    e_norm = e_norm / s_norm;

    /* Print error norm */
    printf("%8d   %1.e   %1.14e\n", ntime, dt, e_norm);

  }

#endif

  /* Clean up */
  fclose(logfile);
  fclose(sufile);
  fclose(svfile);
  fclose(ufile);
  fclose(vfile);
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


