#include "braid_wrapper.hpp"
#include "timestepper.hpp"
#include "braid.h"
#include "braid_test.h"
#include "bspline.hpp"
#include "vector.hpp"
#include "oscillator.hpp" 
#include "hamiltonian.hpp"

#define FD_TEST 1
#define DT_TEST 0

static char help[] ="Solves the Liouville-von-Neumann equations, two oscillators.\n\
Input parameters:\n\
  -nlvl <int>      : Set the number of levels     (default: 2) \n\
  -ntime <int>     : Set the number of time steps (default: 1000) \n\
  -dt <double>     : Set the time step size       (default: 0.01)\n\
  -nspline <int>   : Set the number of spline basis functions (default: 100) \n\
  -cf <int>        : Set XBraid's coarsening factor           (default: 5) \n\
  -ml <int>        : Set XBraid's max levels                  (default: 5)\n\
  -mi <int>        : Set XBraid's max number of iterations    (default: 50)\n\n\
  -analytic <0 or 1> : If 1: runs analytic testcase (2-level, 2-oscillator, pure state) (default: 0) \n\n";


int main(int argc,char **argv)
{
  PetscInt       nlvl;         // Number of levels for each oscillator (currently 2)
  PetscInt       nosci;        // Number of oscillators (currently 2)
  PetscInt       ntime;        // Number of time steps
  PetscReal      dt;           // Time step size
  PetscReal      total_time;   // Total end time T
  TS             ts;           // Timestepping context
  braid_Core     braid_core;   // Core for XBraid simulation
  XB_App        *braid_app;    // XBraid's application context
  PetscInt       cfactor;      // XBraid's coarsening factor
  PetscInt       maxlevels;    // XBraid's maximum number of levels
  PetscInt       maxiter;      // XBraid's maximum number of iterations
  PetscInt       nspline;      // Number of spline basis functions
  PetscInt       analytic;     // If 1: runs analytic test case
  Hamiltonian*   hamiltonian;  // Hamiltonian system


  FILE *ufile, *vfile;
  char filename[255];
  PetscErrorCode ierr;
  PetscMPIInt    mpisize, mpirank;
  double StartTime, StopTime;
  double UsedTime = 0.0;

  /* Initialize Petsc */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&mpisize);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&mpirank);CHKERRQ(ierr);

  /* Set default constants */
  nlvl = 2;
  nosci = 2;
  ntime = 1000;
  dt = 0.01;
  nspline = 100;
  cfactor = 5;
  maxlevels = 5;
  maxiter = 50;
  analytic = 0;

  /* Parse command line arguments to overwrite default constants */
  PetscOptionsGetInt(NULL,NULL,"-nlvl",&nlvl,NULL);
  PetscOptionsGetInt(NULL,NULL,"-ntime",&ntime,NULL);
  PetscOptionsGetReal(NULL,NULL,"-dt",&dt,NULL);
  PetscOptionsGetInt(NULL,NULL,"-nspline",&nspline,NULL);
  PetscOptionsGetInt(NULL,NULL,"-cf",&cfactor,NULL);
  PetscOptionsGetInt(NULL,NULL,"-ml",&maxlevels,NULL);
  PetscOptionsGetInt(NULL,NULL,"-mi",&maxiter,NULL);
  PetscOptionsGetInt(NULL,NULL,"-analytic",&analytic,NULL);

  /* Sanity check */
  if (nosci != 2)
  {
    printf("\nERROR: Current only 2 oscillators are supported.\n You chose %d oscillators.\n\n", nlvl, nosci);
    exit(0);
  }

  /* Initialize time horizon */
  total_time = ntime * dt;

  /* Initialize the Hamiltonian */
  Oscillator** oscil_vec = new Oscillator*[nosci];
  if (analytic == 1) {
    double omegaF1 = 1.0;
    double omegaG2 = 1.0;
    oscil_vec[0] = new FunctionOscillator(omegaF1, &F1_analytic, &dF1_analytic, 0.0, NULL, NULL );
    oscil_vec[1] = new FunctionOscillator(0.0, NULL, NULL, omegaG2, &G2_analytic, &dG2_analytic);
  } else {
    for (int i = 0; i < nosci; i++){
      oscil_vec[i] = new SplineOscillator(nspline, total_time);
    }
    // oscil_vec[0]->dumpControl(total_time, dt, "initcontrol.dat");
  }


  /* Set frequencies for drift hamiltonian Hd xi = [xi_1, xi_2, xi_12] */
  double* xi = new double[nlvl*nlvl];
  if (analytic == 1) {  // no drift Hamiltonian in analytic case
    xi[0] = 0.0;
    xi[1] = 0.0;
    xi[2] = 0.0;
  } else {
    xi[0] =  2. * (2.*M_PI*0.1099);  // from Anders
    xi[1] =  2. * (2.*M_PI*0.1126);  // from Anders
    xi[2] =  0.1;                    // from Anders, might be too big!
  }

  /* Initialize the Hamiltonian  */
  if (analytic == 1) {
    hamiltonian = new AnalyticHam(xi, oscil_vec);
  } else {
    hamiltonian = new TwoOscilHam(nlvl, xi, oscil_vec);
  }

  /* Screen output */
  if (mpirank == 0)
  {
    printf("System with %d oscillators, %d levels. \n", nosci, nlvl);
    printf("Time horizon:   [0,%.1f]\n", total_time);
    printf("Number of time steps: %d\n", ntime);
    printf("Time step size: %f\n", dt );
  }

  /* Open output files */
  sprintf(filename, "out_u.%04d.dat", mpirank);       ufile = fopen(filename, "w");
  sprintf(filename, "out_v.%04d.dat", mpirank);       vfile = fopen(filename, "w");

  /* Allocate and initialize Petsc's Time-stepper */
  BuildTimeStepper(&ts, hamiltonian, ntime, dt, total_time);

  /* Set up XBraid's applications structure */
  braid_app = (XB_App*) malloc(sizeof(XB_App));
  braid_app->ts     = ts;
  braid_app->hamiltonian = hamiltonian;
  braid_app->ntime  = ntime;
  braid_app->ufile  = ufile;
  braid_app->vfile  = vfile;

  /* Initialize Braid */
  braid_Init(MPI_COMM_WORLD, MPI_COMM_WORLD, 0.0, total_time, ntime, braid_app, my_Step, my_Init, my_Clone, my_Free, my_Sum, my_SpatialNorm, my_Access, my_BufSize, my_BufPack, my_BufUnpack, &braid_core);
  
  /* Set Braid options */
  braid_SetPrintLevel( braid_core, 2);
  braid_SetAccessLevel( braid_core, 1);
  braid_SetMaxLevels(braid_core, maxlevels);
  braid_SetNRelax(braid_core, -1, 1);
  braid_SetAbsTol(braid_core, 1e-6);
  braid_SetCFactor(braid_core, -1, cfactor);
  braid_SetMaxIter(braid_core, maxiter);
  braid_SetSkip(braid_core, 0);
  braid_SetSeqSoln(braid_core, 0);




   /* Measure wall time */
  StartTime = MPI_Wtime();
  StopTime = 0.0;
  UsedTime = 0.0;

  /* Run braid */
  braid_Drive(braid_core);


  /* Stop timer */
  StopTime = MPI_Wtime();
  UsedTime = StopTime - StartTime;


  /* Get and print convergence history */
  int niter;
  braid_GetNumIter(braid_core, &niter);
  double* norms = (double*) malloc(niter*sizeof(double));
  braid_GetRNorms(braid_core, &niter, norms);

  if (mpirank == 0)
  {
    FILE* braidlog;
    braidlog = fopen("braid.out.log", "w");
    fprintf(braidlog,"# ntime %d\n", (int) ntime);
    fprintf(braidlog,"# dt %f\n", (double) dt);
    fprintf(braidlog,"# cf %d\n", (int) cfactor);
    fprintf(braidlog,"# ml %d\n", (int) maxlevels);
    for (int i=0; i<niter; i++)
    {
      fprintf(braidlog, "%d  %1.14e\n", i, norms[i]);
    }
    fprintf(braidlog, "\n\n\n");
    fprintf(braidlog, "\n wall time\n %f", UsedTime);
  }
  

  free(norms);

#if FD_TEST
  printf("\n\n Running finite-differences test...\n\n");
  
  double EPS = 1e-8;

  double t = 0.3;
  double f, g;
  double f_pert, g_pert;

  int nparam = oscil_vec[0]->getNParam();
  double *dirRe = new double[nparam];
  double *dirIm = new double[nparam];
  for (int iparam = 0; iparam<nparam; iparam++){
    dirRe[iparam] = 0.0;
    dirIm[iparam] = 0.0;
  }

  /* Init derivative */
  Vector *dfdw = new Vector(nparam, 0.0);
  Vector *dgdw = new Vector(nparam, 0.0);

  for (int i=0; i<nosci; i++)
  {
    printf("FD for oscillator %d:\n", i);
    // oscil_vec[i]->dumpControl(total_time, dt);

    for (int iparam = 0; iparam < nparam; iparam++)
    {
      printf("  param %d:\n", iparam);

      /* Reset gradient */
      dfdw->Fill(0.0);
      dgdw->Fill(0.0);

      /* Eval original objectives and gradient */
      oscil_vec[i]->evalControl(t, &f, &g);
      oscil_vec[i]->evalDerivative(t, dfdw->GetData(), dgdw->GetData());

      /* Eval perturbed objectives */
      dirRe[iparam] = 1.0;
      dirIm[iparam] = 1.0;
      oscil_vec[i]->updateParams(EPS, dirRe, dirIm);
      oscil_vec[i]->evalControl(t, &f_pert, &g_pert);

      /* Eval FD and error */
      double f_fd = (f_pert - f) / EPS;
      double g_fd = (g_pert - g) / EPS;
      double f_err = 0.0;
      double g_err = 0.0;
      if (f_fd != 0.0) f_err = (dfdw->GetData()[iparam] - f_fd) / f_fd;
      if (g_fd != 0.0) g_err = (dgdw->GetData()[iparam] - g_fd) / g_fd;
      printf("    f_pert %1.12e, f %1.12e, f_fd %1.12e, dfdw %1.12e, f_err %2.4f\%\n", f_pert, f, f_fd, dfdw->GetData()[iparam],  f_err*100.0);
      printf("    g_pert %1.12e, g %1.12e, g_fd %1.12e, dgdw %1.12e, g_err %2.4f\%\n", g_pert, g, g_fd, dgdw->GetData()[iparam],  g_err*100.0);

      /* Restore parameter */
      oscil_vec[i]->updateParams(-EPS, dirRe, dirIm);
      dirRe[iparam] = 0.0;
      dirIm[iparam] = 0.0;

    }

  }
  exit(1);


  Vec x;
  double Tfinal;
  double objective_ref, gradient_ref;
  double objective_perturb;
  double finite_differences;
  // double err;



  int nreal = 2*hamiltonian->getDim();

  /* Set initial condition */
  VecCreateSeq(PETSC_COMM_SELF,nreal,&x);
  hamiltonian->initialCondition(x);

  /* Build a new time-stepper */
  TSDestroy(&ts);
  BuildTimeStepper(&ts, hamiltonian, ntime, dt, total_time);
  TSSetSolution(ts, x);

  /* Get unperturbed objective function */
  for(PetscInt istep = 0; istep <= ntime; istep++) {
    TSStep(ts);
  }
  TSGetTime(ts, &Tfinal);
  hamiltonian->evalObjective(Tfinal, x, &objective_ref);


  /* TODO: Perturb the design */
  // double dirRe = 1.0; 
  // double dirIm = 0.0; 
  oscil_vec[0]->updateParams(EPS, dirRe, dirIm);


  /* Evaluate perturbed objective function  AND STORE TRAJECTORY FOR ADJOINT */
  TSSetTime(ts, 0.0);
  TSSetSaveTrajectory(ts);
  hamiltonian->initialCondition(x);
  TSSetSolution(ts, x);
  for(PetscInt istep = 0; istep <= ntime; istep++) {
    TSStep(ts);
  }
  TSGetTime(ts, &Tfinal);
  hamiltonian->evalObjective(Tfinal, x, &objective_perturb);

  Mat dHdp;
 /* Allocate dHdp (derivative of RHS wrt controls) */
  ierr = MatCreate(PETSC_COMM_SELF,&dHdp);CHKERRQ(ierr);
  ierr = MatSetSizes(dHdp, PETSC_DECIDE, PETSC_DECIDE,nreal,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(dHdp);CHKERRQ(ierr);
  ierr = MatSetUp(dHdp);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(dHdp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(dHdp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  Vec mu[1];      // dfdp
  Vec lambda[1];  // dfdy
  PetscScalar *x_ptr;
  MatCreateVecs(hamiltonian->getRHS(), &lambda[0], NULL);
  MatCreateVecs(dHdp, &mu[0], NULL);
  VecZeroEntries(mu[0]);
  VecZeroEntries(lambda[0]);
  ierr = VecGetArray(lambda[0], &x_ptr);
  x_ptr[1] = 200.0;
  ierr = VecRestoreArray(lambda[0], &x_ptr);
  TSSetCostGradients(ts, 1, lambda, mu);

  TSSetRHSJacobianP(ts,dHdp, RHSJacobianP, hamiltonian);

  /* Run adjoint backwards in time */
  for (int k=ntime; k>0; k--)
  {
    TSAdjointStep(ts);
  }

  
  /* Evaluate finite difference */
  /* Compute finite differences and relative error */
   finite_differences = (objective_perturb - objective_ref) / EPS;
  //  err = (gradient_ref - finite_differences) / finite_differences;

   /* Output */
   printf("Objectives: %1.14e %1.14e\n", objective_ref, objective_perturb);
   printf("Finite Differences: %1.14e\n", finite_differences);
  //  printf(" Relative gradient error: %1.6f\n\n", err);



  VecDestroy(&x);
#endif

#if DT_TEST
  /* 
   * Testing time stepper convergence (dt-test) 
   */  
  if (analytic != 1){
    printf("\n WARNING: DT-test works for analytic test case only. Run with \"-analytic 1\" if you want to test the time-stepper convergence. \n");
    return 0;
  }

  Vec x;      // numerical solution
  Vec exact;  // exact solution
  Vec error;  // error  
  double t;
  double error_norm, exact_norm;

  int nreal = 2*hamiltonian->getDim();
  VecCreateSeq(PETSC_COMM_SELF,nreal,&x);
  VecCreateSeq(PETSC_COMM_SELF,nreal,&exact);
  VecCreateSeq(PETSC_COMM_SELF,nreal,&error);

  /* Destroy old time stepper */
  TSDestroy(&ts);

  /* Set time horizon */
  total_time = 10.0;
  printf("\n\n Running time-stepping convergence test... \n\n");
  printf(" Time horizon: [0, %.1f]\n\n", total_time);

  /* Decrease time step size */
  printf("   ntime      dt    error\n");
  for (int ntime = 10; ntime <= 1e+5; ntime = ntime * 10)
  {
    dt = total_time / ntime;

    /* Create and set up the time stepper */
    BuildTimeStepper(&ts, hamiltonian, ntime, dt, total_time);
    TSSetSolution(ts, x);

    // /* Set the initial condition */
    hamiltonian->initialCondition(x);

    /* Run time-stepping loop */
    for(PetscInt istep = 0; istep <= ntime; istep++) 
    {
      TSStep(ts);
    }

    /* Compute the relative error at last time step (max-norm) */
    TSGetTime(ts, &t);
    hamiltonian->ExactSolution(t,exact);
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
  fclose(ufile);
  fclose(vfile);
  TSDestroy(&ts);

  /* Clean up Oscillator */
  for (int i=0; i<nosci; i++){
    delete oscil_vec[i];
  }
  delete [] oscil_vec;

  delete [] xi;

  /* Clean up Hamiltonian */
  delete hamiltonian;

  /* Cleanup XBraid */
  braid_Destroy(braid_core);
  free(braid_app);

  /* Finallize Petsc */
  ierr = PetscFinalize();

  return ierr;
}


