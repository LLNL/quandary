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
#include "hiopAlgFilterIPM.hpp"

#define EPS 1e-7

#define TEST_DRHSDP 0
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

  /* Optimization */
  double objective;        // Objective function value f
  Vec* lambda = new Vec;   // Adjoint solution in lambda[0]
  Vec* mu = new Vec;       // Reduced gradient in mu[0]


  char filename[255];
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
  braid_SplitCommworld(&comm, 1, &comm_petsc, &comm_braid);

  /* Initialize Petsc using petsc's communicator */
  PETSC_COMM_WORLD = comm_petsc;
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
  // TimeStepper *mytimestepper = new ExplEuler(hamiltonian);

  /* Allocate and initialize Petsc's Time-stepper */
  TSCreate(PETSC_COMM_SELF,&ts);CHKERRQ(ierr);
  TSInit(ts, hamiltonian, ntime, dt, total_time, x, lambda, mu, monitor);

  /* Initialize Braid */
  primalbraidapp = new myBraidApp(comm_braid, comm_petsc, total_time, ntime, ts, mytimestepper, hamiltonian, targetgate, &config);
  adjointbraidapp = new myAdjointBraidApp(comm_braid, comm_petsc, total_time, ntime, ts, mytimestepper, hamiltonian, targetgate, *mu, &config, primalbraidapp->getCore());

  /* Prepare output */
  if (iolevel > 0) {
    sprintf(filename, "out_u.%04d.dat", mpirank);
    primalbraidapp->ufile = fopen(filename, "w");
    sprintf(filename, "out_v.%04d.dat", mpirank);
    primalbraidapp->vfile = fopen(filename, "w");
    sprintf(filename, "out_uadj.%04d.dat", mpirank);
    adjointbraidapp->ufile = fopen(filename, "w");
    sprintf(filename, "out_vadj.%04d.dat", mpirank);
    adjointbraidapp->vfile = fopen(filename, "w");
  }

  /* Print some information on the time-grid distribution */
  // int ilower, iupper;
  // _braid_GetDistribution(braid_core, &ilower, &iupper);
  // printf("ilower %d, iupper %d\n", ilower, iupper);


  /* Initialize the optimization */
  long long int ndesign,m;
  OptimProblem optimproblem(primalbraidapp, adjointbraidapp);
  hiop::hiopNlpDenseConstraints nlp(optimproblem);
  optimproblem.get_prob_sizes(ndesign, m);
  /* Set options */
  double optim_tol = config.GetDoubleParam("optim_tol", 1e-4);
  nlp.options->SetNumericValue("tolerance", optim_tol);
  double optim_maxiter = config.GetIntParam("optim_maxiter", 200);
  nlp.options->SetIntegerValue("max_iter", optim_maxiter);
  /* Create solver */
  hiop::hiopAlgFilterIPM optimsolver(&nlp);

   /* Measure wall time */
  StartTime = MPI_Wtime();
  StopTime = 0.0;
  UsedTime = 0.0;


  // /* --- Test optimproblem --- */
  // printf("ndesign=%d\n", ndesign);
  // double* myinit = new double[ndesign];
  // optimproblem.get_starting_point(ndesign, myinit);

  // // /* --- Solve primal --- */
  // printf("\nRunning optimizer eval_f... ");
  // optimproblem.eval_f(ndesign, myinit, true, objective);
  // printf(" Objective %1.14e\n", objective);

  // /* --- Solve adjoint --- */
  // printf("\nRunning optimizer eval_grad_f...\n");
  // double* optimgrad = new double[ndesign];
  // optimproblem.eval_grad_f(ndesign, myinit, true, optimgrad);
  // if (mpirank == 0) {
  //   printf("\n %d: My awesome gradient:\n", mpirank);
  //   for (int i=0; i<ndesign; i++) {
  //     printf("%1.14e\n", optimgrad[i]);
  //   }
  // }

  /* Solve the optimization  */
  printf("Now starting HiOp \n");
  hiop::hiopSolveStatus status = optimsolver.run();


exit:


#if TEST_DRHSDP
  printf("\n\n#########################\n");
  printf(" dRHSdp Testing... \n");
  printf("#########################\n\n");

  double t = 0.345;
  double err_i = 0.0;
  Vec Ax, Bx, Cx, fd, err, grad_col;
  Mat G;
  int size;
  VecGetSize(x, &size);
  VecDuplicate(x, &Ax);
  VecDuplicate(x, &Bx);
  VecDuplicate(x, &Cx);
  VecDuplicate(x, &fd);
  VecDuplicate(x, &err);
  VecDuplicate(x, &grad_col);

  /* Set x to last time step */
  // x = primalbraidapp->getStateVec(total_time);
  /* Set x to all ones */
  VecZeroEntries(x);
  VecShift(x, 1.0);

  /* Evaluate RHS(p)x */
  optimproblem->setDesign(ndesign, myinit);
  hamiltonian->assemble_RHS(t);
  hamiltonian->getRHS();
  MatMult(hamiltonian->getRHS(), x, Ax);

  /* Evaluate dRHSdp(t,p)x */
  hamiltonian->assemble_dRHSdp(t, x);
  G = hamiltonian->getdRHSdp();
  // MatView(G, PETSC_VIEWER_STDOUT_WORLD);

  /* Flush control functions */
  for (int i = 0; i < nosci; i++){
    sprintf(filename, "control_%04d.dat", i);
    oscil_vec[i]->flushControl(ntime, dt, filename);
  }

  /* FD loop */
  // for (int i=0; i<ndesign; i++) {
  {int i=3;

    /* Eval RHS(p+eps)x */
    myinit[i] -= EPS;
    optimproblem->setDesign(ndesign, myinit);
    hamiltonian->assemble_RHS(t);
    MatMult(hamiltonian->getRHS(), x, Bx);

    /* Eval RHS(p-eps)x */
    myinit[i] += 2.*EPS;
    optimproblem->setDesign(ndesign, myinit);
    hamiltonian->assemble_RHS(t);
    MatMult(hamiltonian->getRHS(), x, Cx);

    /* Eval central finite differences (Bx - Cx) / 2eps */
    VecAXPBYPCZ(fd, 1.0, -1.0, 0.0, Bx, Cx);
    VecScale(fd, 1./(2.*EPS));

    /* Compute error */
    MatGetColumnVector(G, grad_col, i);
    VecAXPBYPCZ(err, 1.0, -1.0, 0.0, grad_col, fd); 
    VecPointwiseDivide(err, err, fd);
    VecNorm(err, NORM_2, &err_i);

    /* Output */
    printf(" %d: || e_i|| = %1.4e\n", i, err_i);
    if (err_i > 1e-5) {
      const PetscScalar *err_ptr, *grad_ptr, *fd_ptr, *ax_ptr, *bx_ptr, *cx_ptr;
      VecGetArrayRead(err, &err_ptr);
      VecGetArrayRead(fd, &fd_ptr);
      VecGetArrayRead(Ax, &ax_ptr);
      VecGetArrayRead(Bx, &bx_ptr);
      VecGetArrayRead(Cx, &cx_ptr);
      VecGetArrayRead(grad_col, &grad_ptr);
      printf("ERR    Ax[i]     Bx[i]     Cx[i]    FD       GRADCOL\n");
      for (int j=0; j<size; j++){
        printf("%1.14e  %1.20e  %1.20e  %1.20e  %1.14e  %1.14e\n", err_ptr[j], ax_ptr[i], bx_ptr[i], cx_ptr[i], fd_ptr[j], grad_ptr[j]);
      }
    }


    // printf("FD column:\n");
    // VecView(fd, PETSC_VIEWER_STDOUT_WORLD);
    // printf("dRHSdp column:\n");
    // VecView(grad_col, PETSC_VIEWER_STDOUT_WORLD);

    /* Reset design */
    myinit[i] -= EPS;
  }


#endif

#if TEST_FD_TS

  printf("\n\n#########################\n");
  printf(" FD Testing... \n");
  printf("#########################\n\n");

  double obj_org;
  double obj_pert1, obj_pert2;

  long long int n,l;
  optimproblem.get_prob_sizes(n, m);

  double* myx = new double[n];
  optimproblem.get_starting_point(n, myx);


  // /* --- Solve primal --- */
  printf("\nRunning optimizer eval_f... ");
  optimproblem.eval_f(n, myx, true, obj_org);
  printf(" Obj_orig %1.14e\n", obj_org);

  /* --- Solve adjoint --- */
  printf("\nRunning optimizer eval_grad_f...\n");
  double* testgrad = new double[n];
  optimproblem.eval_grad_f(n, myx, true, testgrad);

  /* Finite Differences */
  printf("FD...\n");
  for (int i=0; i<n; i++){
  // {int i=0;

    /* Evaluate f(p+eps)*/
    myx[i] += EPS;
    optimproblem.eval_f(n, myx, true, obj_pert1);

    /* Evaluate f(p-eps)*/
    myx[i] -= 2.*EPS;
    optimproblem.eval_f(n, myx, true, obj_pert2);

    /* Eval FD and error */
    double fd = (obj_pert1 - obj_pert2) / (2.*EPS);
    double err = 0.0;
    if (fd != 0.0) err = (testgrad[i] - fd) / fd;
    printf(" %d: obj %1.14e, obj_pert1 %1.14e, obj_pert2 %1.14e, fd %1.14e, grad %1.14e, err %1.14e\n", i, obj_org, obj_pert1, obj_pert2, fd, testgrad[i], err);

    /* Restore parameter */
    myx[i] += EPS;
  }
  
  delete [] testgrad;
  delete [] myx;

#endif

#if TEST_FD_SPLINE
  printf("\n\n Finite-differences for Spline discretization...\n\n");
  
  // double t = 0.345;
  double t = 0.345;
  double f, g;
  double f_pert1, g_pert1, f_pert2, g_pert2;

  int nparam = oscil_vec[0]->getNParam();

  /* Init derivative */
  double *dfdw = new double[nparam];
  double *dgdw = new double[nparam];

  for (int iosc=0; iosc<nosci; iosc++)
  {
    printf("FD for oscillator %d:\n", iosc);

    /* Eval gradients */
    optimproblem->setDesign(ndesign, myinit);
    oscil_vec[iosc]->evalControl(t, &f, &g);
    for (int iparam=0; iparam< nparam; iparam++) {
      dfdw[iparam] = 0.0;
      dgdw[iparam] = 0.0;
    }
    oscil_vec[iosc]->evalDerivative(t, dfdw, dgdw);

    /* FD testing for all parameters of oscil 1 */
    for (int iparam = 0; iparam < nparam; iparam++)
    {
      int alpha_id = iosc * 2 * nparam + iparam;
      int beta_id  = iosc * 2 * nparam + nparam + iparam;
      printf("  param %d: \n", iparam);

      /* Eval perturbed objectives */
      myinit[alpha_id] += EPS;
      myinit[beta_id]  += EPS;
      optimproblem->setDesign(ndesign, myinit);
      oscil_vec[iosc]->evalControl(t, &f_pert1, &g_pert1);

      myinit[alpha_id] -= 2.*EPS;
      myinit[beta_id]  -= 2.*EPS;
      optimproblem->setDesign(ndesign, myinit);
      oscil_vec[iosc]->evalControl(t, &f_pert2, &g_pert2);

      /* Eval FD and error */
      double f_fd = f_pert1/(2.*EPS) - f_pert2 / (2.*EPS);
      double g_fd = g_pert1/(2.*EPS) - g_pert2 / (2.*EPS);
      double f_err = 0.0;
      double g_err = 0.0;
      if (f_fd != 0.0) f_err = (dfdw[iparam] - f_fd) / f_fd;
      if (g_fd != 0.0) g_err = (dgdw[iparam] - g_fd) / g_fd;
      printf("    f %1.12e  f1 %1.12e  f2 %1.12e  f_fd %1.12e, dfdw %1.12e, f_err %1.8e\n", f, f_pert1, f_pert2, f_fd, dfdw[iparam],  f_err);
      printf("    g %1.12e  g1 %1.12e  f2 %1.12e  g_fd %1.12e, dgdw %1.12e, g_err %1.8e\n", g, g_pert1, g_pert2, g_fd, dgdw[iparam],  g_err);

      /* Restore parameter */
      myinit[alpha_id] += EPS;
      myinit[beta_id]  += EPS;
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
  if (primalbraidapp->ufile != NULL) fclose(primalbraidapp->ufile);
  if (primalbraidapp->vfile != NULL) fclose(primalbraidapp->vfile);
  if (adjointbraidapp->ufile != NULL) fclose(adjointbraidapp->ufile);
  if (adjointbraidapp->vfile != NULL) fclose(adjointbraidapp->vfile);

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


