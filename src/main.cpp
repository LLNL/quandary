#include "braid_wrapper.hpp"
#include "timestepper.hpp"
#include "braid.h"
#include "braid_test.h"
#include "bspline.hpp"
#include "oscillator.hpp" 
#include "hamiltonian.hpp"
#include "config.hpp"
#include "optimizer.hpp"
#include "_braid.h"
#include <stdlib.h>
#include "IpIpoptApplication.hpp"

using namespace Ipopt;

#define EPS 1e-8

#define TEST_FD_TS 0
#define TEST_FD_SPLINE 0
#define TEST_DT 0


int main(int argc,char **argv)
{
  PetscInt       nlvl;         // Number of levels for each oscillator (currently 2)
  PetscInt       nosci;        // Number of oscillators (currently 2)
  PetscInt       ntime;        // Number of time steps
  PetscReal      dt;           // Time step size
  PetscReal      total_time;   // Total end time T
  TS             ts;           // Timestepping context
  PetscInt       iolevel;      // Level of file output (0: no output)
  PetscInt       nspline;      // Number of spline basis functions
  Hamiltonian*   hamiltonian;  // Hamiltonian system
  PetscBool      analytic;     // If true: runs analytic test case
  PetscBool      primal_only;  // If true: runs only one primal simulation
  PetscBool      monitor;      // If true: Print out additional time-stepper information
  /* Braid */
  myBraidApp *primalbraidapp;
  myAdjointBraidApp *adjointbraidapp;

  Vec            x;          // solution vector
  // bool           tj_save;    // Determines wether trajectory should be stored in primal run
  PetscInt ndesign;           // Number of design variables 
  double *mygrad = NULL;      // Reduced gradient, local on this processor

  /* Optimization */
  double objective;        // Objective function value f
  Vec* lambda = new Vec;   // Adjoint solution in lambda[0]
  Vec* mu = new Vec;       // Reduced gradient in mu[0]


  char filename[255];
  FILE* ufile, *uadjfile;
  FILE* vfile, *vadjfile;
  PetscErrorCode ierr;
  PetscMPIInt    mpisize, mpirank;
  double StartTime, StopTime;
  double UsedTime = 0.0;


  /* Initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm comm_braid, comm_petsc;
  /* TODO:  FIX THE MPI SHIT ! */
  // braid_SplitCommworld(&comm, 1, &comm_petsc, &comm_braid);
  // PETSC_COMM_WORLD = comm_petsc;
  comm_petsc = MPI_COMM_WORLD;
  comm_braid = MPI_COMM_WORLD;

  /* Initialize Petsc */
  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;

  /* Read config file and set default */
  if (argc != 2) {
    if (mpirank == 0) {
      printf("\n");
      printf("USAGE: ./main </path/to/configfile> \n");
    }
    MPI_Finalize();
    return 0;
  }
  MapParam config(comm);
  config.ReadFile(argv[1]);
  nlvl  = config.GetIntParam("nlevels", 2);
  nosci = config.GetIntParam("noscillators", 2);
  ntime = config.GetIntParam("ntime", 1000);
  dt    = config.GetDoubleParam("dt", 0.01);
  nspline = config.GetIntParam("nspline", 10);
  analytic = (PetscBool) config.GetBoolParam("analytic", false);
  primal_only = (PetscBool) config.GetBoolParam("primal_only", false);
  monitor = (PetscBool) config.GetBoolParam("monitor", false);
  iolevel = (PetscInt) config.GetIntParam("iolevel", 1);
  
  /* Initialize time horizon */
  total_time = ntime * dt;

  /* Initialize the Oscillators */
  if (analytic) nosci = 2;
  Oscillator** oscil_vec = new Oscillator*[nosci];
  if (analytic) {
    double omegaF1 = 1.0;
    double omegaG2 = 1.0;
    oscil_vec[0] = new FunctionOscillator(nlvl, omegaF1, &F1_analytic, &dF1_analytic, 0.0, NULL, NULL );
    oscil_vec[1] = new FunctionOscillator(nlvl, 0.0, NULL, NULL, omegaG2, &G2_analytic, &dG2_analytic);
  } else {
    for (int i = 0; i < nosci; i++){
      oscil_vec[i] = new SplineOscillator(nlvl, nspline, total_time);
    }
  }

  /* Flush control functions */
  if (mpirank == 0 && iolevel > 0) {
    for (int i = 0; i < nosci; i++){
      sprintf(filename, "control_%04d.dat", i);
      oscil_vec[i]->flushControl(ntime, dt, filename);
    }
  }

  /* Set frequencies for drift hamiltonian Hd xi = [xi_1, xi_2, xi_12] */
  double* xi = new double[nlvl*nlvl];
  if (analytic) {  // no drift Hamiltonian in analytic case
    xi[0] = 0.0;
    xi[1] = 0.0;
    xi[2] = 0.0;
  } else {
    xi[0] =  2. * (2.*M_PI*0.1099);  // from Anders
    xi[1] =  2. * (2.*M_PI*0.1126);  // from Anders
    xi[2] =  0.1;                    // from Anders, might be too big!
  }

  /* Initialize the Hamiltonian  */
  if (analytic) {
    hamiltonian = new AnalyticHam(xi, oscil_vec); // always 2levels
  } else {
    // hamiltonian = new TwoOscilHam(nlvl, xi, oscil_vec);
    hamiltonian = new LiouvilleVN(xi, nosci, oscil_vec);
  }

  /* Initialize the target */
  Gate* targetgate = new CNOT();

  /* Create solution vector x */
  MatCreateVecs(hamiltonian->getRHS(), &x, NULL);

  /* Initialize reduced gradient and adjoints */
  MatCreateVecs(hamiltonian->getRHS(), lambda, NULL);  // adjoint 
  MatCreateVecs(hamiltonian->getdRHSdp(), mu, NULL);   // reduced gradient

  /* Screen output */
  if (mpirank == 0)
  {
    printf("# System with %d oscillators, %d levels. \n", nosci, nlvl);
    printf("# Time horizon:   [0,%.1f]\n", total_time);
    printf("# Number of time steps: %d\n", ntime);
    printf("# Time step size: %f\n", dt );
  }

  TimeStepper *mytimestepper = new ImplMidpoint(hamiltonian);

  /* Allocate and initialize Petsc's Time-stepper */
  TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  TSInit(ts, hamiltonian, ntime, dt, total_time, x, lambda, mu, monitor);

  /* Initialize Braid */
  primalbraidapp = new myBraidApp(comm_braid, comm_petsc, total_time, ntime, ts, mytimestepper, hamiltonian, targetgate, &config);
  adjointbraidapp = new myAdjointBraidApp(comm_braid, comm_petsc, total_time, ntime, ts, mytimestepper, hamiltonian, targetgate, *mu, &config, primalbraidapp->getCore());

  /* Prepare output */
  sprintf(filename, "out_u.%04d.dat", mpirank);
  if (iolevel > 0) ufile = fopen(filename, "w");
  sprintf(filename, "out_v.%04d.dat", mpirank);
  if (iolevel > 0) vfile = fopen(filename, "w");
  sprintf(filename, "out_uadj.%04d.dat", mpirank);
  if (iolevel > 0) uadjfile = fopen(filename, "w");
  sprintf(filename, "out_vadj.%04d.dat", mpirank);
  if (iolevel > 0) vadjfile = fopen(filename, "w");


  primalbraidapp->ufile = ufile;
  primalbraidapp->vfile = vfile;
  adjointbraidapp->ufile = uadjfile;
  adjointbraidapp->vfile = vadjfile;

  /* Print some information on the time-grid distribution */
  // int ilower, iupper;
  // _braid_GetDistribution(braid_core, &ilower, &iupper);
  // printf("ilower %d, iupper %d\n", ilower, iupper);


  /* Initialize the optimization */
  SmartPtr<TNLP> optimproblem = new OptimProblem(primalbraidapp, adjointbraidapp);
  SmartPtr<IpoptApplication> optimapp = IpoptApplicationFactory(); // why "factory"?
  /* Set options */
  optimapp->Options()->SetNumericValue("tol", 1e-7);
	optimapp->Options()->SetStringValue("output_file", "optim.out");
	optimapp->Options()->SetStringValue("hessian_approximation", "limited-memory");
	optimapp->Options()->SetStringValue("derivative_test", "first-order");
	optimapp->Options()->SetStringValue("derivative_test_print_all", "yes");
  optimapp->Options()->SetIntegerValue("max_iter", 0);
  /* Initialize optim status */
  ApplicationReturnStatus optimstatus;
  optimstatus = optimapp->Initialize();
  if (optimstatus != Solve_Succeeded) {
    printf("\n\n*** Error during optimization init!\n\n");
    return (int) optimstatus;
  }


   /* Measure wall time */
  StartTime = MPI_Wtime();
  StopTime = 0.0;
  UsedTime = 0.0;


  /* Solve the optimization  */
  // printf("Now solving the optim problem \n");
  // optimstatus = optimapp->OptimizeTNLP(optimproblem);


  int n, m, nnz_jac, nnz_h;
  OptimProblem::IndexStyleEnum index_style;
  optimproblem->get_nlp_info(n,m,nnz_jac, nnz_h, index_style);
  printf("n=%d, m=%d, nnz_jac=%d, nnz_h=%d\n", n,m,nnz_jac, nnz_h);

  /* Test optimproblem */
  double* myinit = new double[n];
  optimproblem->get_starting_point(n, true, myinit, false, NULL, NULL, m, false, NULL);


  double obj_orig, obj_perturb;


  /* --- Solve primal --- */
  printf("\nRunning optimizer eval_f... ");
  optimproblem->eval_f(n, myinit, true, obj_orig);
  printf(" Objective_orig %1.14e\n", obj_orig);

  // perturb design */
  int iperturb = 0;
  myinit[iperturb] += EPS;

  /* --- Solve primal --- */
  printf("Running optimizer eval_f... ");
  optimproblem->eval_f(n, myinit, true, obj_perturb);
  printf(" Objective_perturb %1.14e\n", obj_perturb);

  myinit[iperturb] -= EPS;

  /* --- Solve adjoint --- */

  printf("\nRunning optimizer eval_grad_f...\n");
  double* optimgrad = new double[n];
  optimproblem->eval_grad_f(n, myinit, true, optimgrad);
  if (mpirank == 0) {
    printf("\n %d: My awesome gradient:\n", mpirank);
    for (int i=0; i<n; i++) {
      printf("%1.14e\n", optimgrad[i]);
    }
  }

  /* compute FD */
  double findiff = (obj_perturb - obj_orig) / EPS;
  double error = 0.0;
  if (findiff != 0.0) error = ( optimgrad[iperturb] - findiff ) / findiff;
  printf("Findiff = %1.14e, grad=%1.14e, error=%1.14e\n", findiff, optimgrad[iperturb], error);


  /* Gradient output */
  // /* Stop timer */
  // StopTime = MPI_Wtime();
  // UsedTime = StopTime - StartTime;

  // /* Print convergence history */
  // braid_printConvHistory(braid_core_adj, "braid_adj.out.log");

exit:


#if TEST_FD_TS

  printf("\n\n#########################\n");
  printf(" FD Testing... \n");
  printf("#########################\n\n");

  double objective_ref;
  braid_SetMaxLevels(braid_core, 1);

  /* Solve */
  braid_Drive(braid_core);

  /* Evaluate the objective function */
  evalObjective(braid_core, braid_app, &objective_ref);
  printf("Objective_ref = %1.13e\n", objective_ref);

  /* Reset gradient */
  VecZeroEntries(mu[0]);

  /* Set the derivatives for TS */
  braid_Drive(braid_core_adj);

  /* Get the results */
  printf("Gradient:\n");
  VecView(mu[0], PETSC_VIEWER_STDOUT_WORLD);

  /* Perturb controls */
  int nparam = oscil_vec[0]->getNParam();
  double *dirRe = new double[nparam];
  double *dirIm = new double[nparam];
  for (int iparam = 0; iparam<nparam; iparam++){
    dirRe[iparam] = 0.0;
    dirIm[iparam] = 0.0;
  }


  /* Finite Differences */
  double fd, grad, err;
  double objective_pert;
  for (int i=0; i<nosci; i++){
    printf("FD for Oscillator %d\n", i);

    for (int iparam=0; iparam<nparam; iparam++){
        printf("  param %d: \n", iparam);

        /* Perturb Re*/
        dirRe[iparam] = 1.0;
        oscil_vec[i]->updateParams(EPS, dirRe, dirIm);

        /* Run the time stepper */
        braid_Drive(braid_core);
        evalObjective(braid_core, braid_app, &objective_pert);

        /* Eval FD and error */
        const PetscScalar *x_ptr;
        VecGetArrayRead(mu[0], &x_ptr);
        grad = x_ptr[i*2*nparam + iparam];
        fd = (objective_pert - objective_ref) / EPS;
        err = ( grad - fd) / fd;
        printf("     Re: obj_pert %1.14e, obj %1.14e, fd %1.14e, mu %1.14e, err %1.14e\n", objective_pert, objective_ref, fd, grad, err);

        // /* Restore parameter */
        oscil_vec[i]->updateParams(-EPS, dirRe, dirIm);
        dirRe[iparam] = 0.0;


        /* Perturb Im*/
        dirIm[iparam] = 1.0;
        oscil_vec[i]->updateParams(EPS, dirRe, dirIm);

        /* Run the time stepper */
        braid_Drive(braid_core);
        evalObjective(braid_core, braid_app, &objective_pert);

        /* Eval FD and error */
        fd = (objective_pert - objective_ref) / EPS;
        VecGetArrayRead(mu[0], &x_ptr);
        grad = x_ptr[i*2*nparam + iparam + nparam];
        err = ( grad - fd) / fd;
        printf("     Im: obj_pert %1.14e, obj %1.14e, fd %1.14e, mu %1.14e, err %1.12e\n", objective_pert, objective_ref, fd, grad, err);

        /* Restore parameter */
        oscil_vec[i]->updateParams(-EPS, dirRe, dirIm);
        dirIm[iparam] = 0.0;
    }
  }

  /* Evaluate finite difference */
  /* Compute finite differences and relative error */
  //  finite_differences = (objective_perturb - objective_ref) / EPS;
  //  err = (gradient_ref - finite_differences) / finite_differences;

   /* Output */
  //  printf("Objectives: %1.14e %1.14e\n", objective_ref, objective_perturb);
  //  printf("Finite Differences: %1.14e\n", finite_differences);
  //  printf(" Relative gradient error: %1.6f\n\n", err);



  VecDestroy(&x);
#endif

#if TEST_FD_SPLINE
  printf("\n\n Running finite-differences test...\n\n");
  

  double t = 1.0;
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
  double *dfdw = new double[nparam];
  double *dgdw = new double[nparam];

  for (int i=0; i<nosci; i++)
  {
    printf("FD for oscillator %d:\n", i);

    for (int iparam = 0; iparam < nparam; iparam++)
    {
      printf("  param %d:\n", iparam);

      /* Reset gradient */
      for (int i=0; i< nparam; i++) {
        dfdw[i] = 0.0;
        dgdw[i] = 0.0;
      }

      /* Eval original objectives and gradient */
      oscil_vec[i]->evalControl(t, &f, &g);
      oscil_vec[i]->evalDerivative(t, dfdw, dgdw);

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
      if (f_fd != 0.0) f_err = (dfdw[iparam] - f_fd) / f_fd;
      if (g_fd != 0.0) g_err = (dgdw[iparam] - g_fd) / g_fd;
      printf("    f_pert %1.12e, f %1.12e, f_fd %1.12e, dfdw %1.12e, f_err %2.4f\%\n", f_pert, f, f_fd, dfdw[iparam],  f_err*100.0);
      printf("    g_pert %1.12e, g %1.12e, g_fd %1.12e, dgdw %1.12e, g_err %2.4f\%\n", g_pert, g, g_fd, dgdw[iparam],  g_err*100.0);

      /* Restore parameter */
      oscil_vec[i]->updateParams(-EPS, dirRe, dirIm);
      dirRe[iparam] = 0.0;
      dirIm[iparam] = 0.0;

    }

  }
#endif



#if TEST_DT
  /* 
   * Testing time stepper convergence (dt-test) 
   */  
  if (!analytic){
    printf("\n WARNING: DT-test works for analytic test case only. Run with \"-analytic \" if you want to test the time-stepper convergence. \n");
    return 0;
  }

  Vec exact;  // exact solution
  Vec error;  // error  
  double t;
  double error_norm, exact_norm;

  int nreal = 2*hamiltonian->getDim();
  VecCreateSeq(PETSC_COMM_WORLD,nreal,&x);
  VecCreateSeq(PETSC_COMM_WORLD,nreal,&exact);
  VecCreateSeq(PETSC_COMM_WORLD,nreal,&error);

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
    TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
    TSInit(ts, hamiltonian, ntime, dt, total_time, x, lambda, mu, monitor);
    TSSetSolution(ts, x);

    // /* Set the initial condition */
    hamiltonian->initialCondition(0,x);

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

#ifdef SANITY_CHECK
  printf("\n\n Sanity checks have been performed. Check output for warnings and errors!\n\n");
#endif


  /* Close output files */
  if (ufile != NULL) fclose(ufile);
  if (vfile != NULL) fclose(vfile);
  if (uadjfile != NULL) fclose(uadjfile);
  if (vadjfile != NULL) fclose(vadjfile);

  /* Clean up */
  // TSDestroy(&ts);  /* TODO */

  /* Clean up Oscillator */
  for (int i=0; i<nosci; i++){
    delete oscil_vec[i];
  }
  delete [] oscil_vec;

  delete [] xi;


  delete lambda;
  delete mu;
  delete [] mygrad;

  /* Clean up Hamiltonian */
  delete hamiltonian;
  delete targetgate;

  /* Cleanup */
  // TSDestroy(&braid_app->ts);

  delete mytimestepper;
  
  delete primalbraidapp;
  delete adjointbraidapp;

  /* Finallize Petsc */
  ierr = PetscFinalize();

  MPI_Finalize();
  return ierr;
}


