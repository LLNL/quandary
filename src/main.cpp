#include "braid_wrapper.hpp"
#include "timestepper.hpp"
#include "braid.h"
#include "braid_test.h"
#include "bspline.hpp"
#include "oscillator.hpp" 
#include "mastereq.hpp"
#include "config.hpp"
#include "_braid.h"
#include <stdlib.h>
#include <sys/resource.h>
#include "optimproblem.hpp"

#define EPS 1e-4

#define TEST_DRHSDP 0
#define TEST_FD_TS 0
#define TEST_FD_SPLINE 0
#define TEST_DT 0

enum RunType {
  primal,            // Runs one objective function evaluation (forward)
  adjoint,           // Runs one gradient computation (forward & backward)
  optimization,      // Run optimization 
  none               // Don't run anything.
};


int main(int argc,char **argv)
{
  char filename[255];
  PetscErrorCode ierr;

  /* Initialize MPI */
  MPI_Init(&argc, &argv);
  int mpisize_world, mpirank_world;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize_world);
  if (mpirank_world == 0) printf("Running on %d cores.\n", mpisize_world);

  /* Read config file */
  if (argc != 2) {
    if (mpirank_world == 0) {
      printf("\nUSAGE: ./main </path/to/configfile> \n");
    }
    MPI_Finalize();
    return 0;
  }
  std::stringstream log;
  MapParam config(MPI_COMM_WORLD, log);
  config.ReadFile(argv[1]);

  /* --- Get some options from the config file --- */
  std::vector<int> nlevels;
  config.GetVecIntParam("nlevels", nlevels, 0);
  int ntime = config.GetIntParam("ntime", 1000);
  double dt    = config.GetDoubleParam("dt", 0.01);
  int nspline = config.GetIntParam("nspline", 10);
  PetscBool monitor = (PetscBool) config.GetBoolParam("monitor", false);
  RunType runtype;
  std::string runtypestr = config.GetStrParam("runtype", "primal");
  if      (runtypestr.compare("primal")      == 0) runtype = primal;
  else if (runtypestr.compare("adjoint")     == 0) runtype = adjoint;
  else if (runtypestr.compare("optimization")== 0) runtype = optimization;
  else {
    printf("\n\n WARNING: Unknown runtype: %s.\n\n", runtypestr.c_str());
    runtype = none;
  }
  std::vector<double> f;
  config.GetVecDoubleParam("frequencies", f, 1e20); // These are actually never used in the code... 


  /* Get the IDs of oscillators that are concerned for optimization */
  std::vector<std::string> oscilIDstr;
  std::vector<int> obj_oscilIDs; 
  config.GetVecStrParam("optim_oscillators", oscilIDstr);
  if (oscilIDstr[0].compare("all") == 0) {
    for (int iosc = 0; iosc < nlevels.size(); iosc++) 
      obj_oscilIDs.push_back(iosc);
  } else {
    config.GetVecIntParam("optim_oscillators", obj_oscilIDs, 0);
  }
  /* Sanity check for oscillator IDs */
  bool err = false;
  assert(obj_oscilIDs.size() > 0);
  for (int i=0; i<obj_oscilIDs.size(); i++){
    if ( obj_oscilIDs[i] >= nlevels.size() )       err = true;
    if ( i>0 &&  ( obj_oscilIDs[i] != obj_oscilIDs[i-1] + 1 ) ) err = true;
  }
  if (err) {
    printf("ERROR: List of oscillator IDs for objective function invalid\n"); 
    exit(1);
  }

  /* Get type and the total number of initial conditions */
  int ninit = 1;
  std::vector<std::string> initcondstr;
  config.GetVecStrParam("optim_initialcondition", initcondstr, "basis");
  InitialConditionType initcond_type;
  assert (initcondstr.size() > 0);
  if      (initcondstr[0].compare("file") == 0 ) {
    initcond_type = FROMFILE;
    ninit = 1;
  }     
  else if (initcondstr[0].compare("pure") == 0 ) {
    initcond_type = PURE;
    ninit = 1;
  }     
  else if (initcondstr[0].compare("diagonal") == 0 ) {
    initcond_type = DIAGONAL;
    /* Compute ninit = dim(subsystem defined by obj_oscilIDs) */
    ninit = 1;
    for (int i=0; i<obj_oscilIDs.size(); i++) {
      ninit *= nlevels[obj_oscilIDs[i]];
    }
  }
  else if (initcondstr[0].compare("basis")    == 0 ) {
    initcond_type = BASIS;
    /* Compute ninit = dim(subsystem defined by obj_oscilIDs)^2 */
    ninit = 1;
    for (int i=0; i<obj_oscilIDs.size(); i++) {
      ninit *= nlevels[obj_oscilIDs[i]];
    }
    ninit = (int) pow(ninit, 2);
  }
  else {
    printf("\n\n ERROR: Wrong setting for initial condition.\n");
    exit(1);
  }

  /* --- Split communicators for distributed initial conditions, distributed linear algebra, time-parallel braid (and parallel optimizer, if HiOp) --- */
  int mpirank_init, mpisize_init;
  int mpirank_braid, mpisize_braid;
  int mpirank_petsc, mpisize_petsc;
  MPI_Comm comm_braid, comm_init, comm_petsc, comm_hiop;

  /* Split aside communicator for hiop. Size 1 for now */  
  MPI_Comm_split(MPI_COMM_WORLD, mpirank_world, mpirank_world, &comm_hiop);

  /* Get the size of communicators  */
  int np_braid = config.GetIntParam("np_braid", 1);
  int np_init  = min(ninit, config.GetIntParam("np_init", 1)); 
  np_braid = min(np_braid, mpisize_world); 
  np_init  = min(np_init,  mpisize_world); 
  int np_petsc = mpisize_world / (np_init * np_braid);

  /* Sanity check for communicator sizes */ 
  if (ninit % np_init != 0){
    printf("ERROR: Wrong processor distribution! \n Size of communicator for distributing initial conditions (%d) must be integer divisor of the total number of initial conditions (%d)!!\n", np_init, ninit);
    exit(1);
  }
  if (mpisize_world % (np_init * np_braid) != 0) {
    printf("ERROR: Wrong number of threads! \n Total number of threads (%d) must be integer multiple of the product of communicator sizes for initial conditions and braid (%d * %d)!\n", mpisize_world, np_init, np_braid);
    exit(1);
  }

  /* Split communicators */
  // Distributed initial conditions 
  int color_init = mpirank_world % (np_petsc * np_braid);
  MPI_Comm_split(MPI_COMM_WORLD, color_init, mpirank_world, &comm_init);
  MPI_Comm_rank(comm_init, &mpirank_init);
  MPI_Comm_size(comm_init, &mpisize_init);

  // Time-parallel Braid
  int color_braid = mpirank_world % np_petsc + mpirank_init * np_petsc;
  MPI_Comm_split(MPI_COMM_WORLD, color_braid, mpirank_world, &comm_braid);
  MPI_Comm_rank(comm_braid, &mpirank_braid);
  MPI_Comm_size(comm_braid, &mpisize_braid);

  // Distributed Linear algebra: Petsc
  int color_petsc = mpirank_world / np_petsc;
  MPI_Comm_split(MPI_COMM_WORLD, color_petsc, mpirank_world, &comm_petsc);
  MPI_Comm_rank(comm_petsc, &mpirank_petsc);
  MPI_Comm_size(comm_petsc, &mpisize_petsc);

  std::cout<< "Parallel distribution: " << "init " << mpirank_init << "/" << mpisize_init \
           << ", braid " << mpirank_braid << "/" << mpisize_braid  \
           << ", petsc " << mpirank_petsc << "/" << mpisize_petsc << ", " << std::endl;

  /* Initialize Petsc using petsc's communicator */
  PETSC_COMM_WORLD = comm_petsc;
  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;
  PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, 	PETSC_VIEWER_ASCII_MATLAB );


  double total_time = ntime * dt;

  /* Initialize the Oscillators */
  Oscillator** oscil_vec = new Oscillator*[nlevels.size()];
  for (int i = 0; i < nlevels.size(); i++){
    std::vector<double> carrier_freq;
    std::string key = "carrier_frequency" + std::to_string(i);
    config.GetVecDoubleParam(key, carrier_freq, 0.0);
    oscil_vec[i] = new Oscillator(i, nlevels, nspline, carrier_freq, total_time);
  }

  /* --- Initialize the Master Equation  --- */
  std::vector<double> xi, t_collapse;
  config.GetVecDoubleParam("xi", xi, 2.0);
  config.GetVecDoubleParam("lindblad_collapsetime", t_collapse, 0.0);
  std::string lindblad = config.GetStrParam("lindblad_type", "none");
  LindbladType lindbladtype;
  if      (lindblad.compare("none")      == 0 ) lindbladtype = NONE;
  else if (lindblad.compare("decay")     == 0 ) lindbladtype = DECAY;
  else if (lindblad.compare("dephase")   == 0 ) lindbladtype = DEPHASE;
  else if (lindblad.compare("both")      == 0 ) lindbladtype = BOTH;
  else {
    printf("\n\n ERROR: Unnown lindblad type: %s.\n", lindblad.c_str());
    printf(" Choose either 'none', 'decay', 'dephase', or 'both'\n");
    exit(1);
  }
  MasterEq* mastereq = new MasterEq(nlevels.size(), oscil_vec, xi, lindbladtype, initcond_type, t_collapse);


  /* Screen output */
  if (mpirank_world == 0) {
    std::cout << "Time: [0:" << total_time << "], ";
    std::cout << "N="<< ntime << ", dt=" << dt << std::endl;
    std::cout<< "System: ";
    for (int i=0; i<nlevels.size(); i++) {
      std::cout<< nlevels[i];
      if (i < nlevels.size()-1) std::cout<< "x";
    }
    std::cout << "\nT1/T2 times: ";
    for (int i=0; i<t_collapse.size(); i++) {
      std::cout << t_collapse[i];
      if ((i+1)%2 == 0 && i < t_collapse.size()-1) std::cout<< ",";
      std::cout<< " ";
    }
    std::cout << std::endl;
  }

  /* --- Initialize the time-stepper --- */
  /* My time stepper */
  TimeStepper *mytimestepper = new ImplMidpoint(mastereq);
  // TimeStepper *mytimestepper = new ExplEuler(mastereq);

  /* Petsc's Time-stepper */
  TS ts;
  Vec x;
  TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  MatCreateVecs(mastereq->getRHS(), &x, NULL);
  TSInit(ts, mastereq, ntime, dt, total_time, x, monitor);
   

  /* --- Create braid instances --- */
  myBraidApp* primalbraidapp = new myBraidApp(comm_braid, total_time, ntime, ts, mytimestepper, mastereq, &config);
  myAdjointBraidApp *adjointbraidapp = new myAdjointBraidApp(comm_braid, total_time, ntime, ts, mytimestepper, mastereq, &config, primalbraidapp->getCore());
  primalbraidapp->InitGrids();
  adjointbraidapp->InitGrids();

  /* --- Initialize optimization --- */
  OptimProblem* optimctx = new OptimProblem(config, primalbraidapp, adjointbraidapp, comm_hiop, comm_init, obj_oscilIDs, initcond_type, ninit);

  /* Set upt solution and gradient vector */
  Vec xinit;
  VecCreateSeq(PETSC_COMM_SELF, optimctx->ndesign, &xinit);
  VecSetFromOptions(xinit);
  Vec grad;
  VecCreateSeq(PETSC_COMM_SELF, optimctx->ndesign, &grad);
  VecSetUp(grad);
  VecZeroEntries(grad);
  Vec opt;

  /* Some output */
  if (mpirank_world == 0)
  {
    /* Print parameters to file */
    sprintf(filename, "%s/config_log.dat", primalbraidapp->datadir.c_str());
    ofstream logfile(filename);
    if (logfile.is_open()){
      logfile << log.str();
      logfile.close();
      printf("File written: %s\n", filename);
    }
    else std::cerr << "Unable to open " << filename;
  }

  /* Start timer */
  double StartTime = MPI_Wtime();

  /* --- Solve primal --- */
  double objective = 0.0;
  if (runtype == primal || runtype == adjoint) {
    optimctx->getStartingPoint(xinit);
    objective = optimctx->evalF(xinit);
    if (mpirank_world == 0) printf("%d: Tao primal: Objective %1.14e, \n", mpirank_world, objective);
    optimctx->getSolution(&opt);
  } 
  
  /* --- Solve adjoint --- */
  if (runtype == adjoint) {
    double gnorm = 0.0;

    optimctx->evalGradF(xinit, grad);
    VecNorm(grad, NORM_2, &gnorm);
    VecView(grad, PETSC_VIEWER_STDOUT_WORLD);
    if (mpirank_world == 0) {
      printf("Tao gradient norm: %1.14e\n", gnorm);
    }
  }

  /* --- Solve the optimization  --- */
  if (runtype == optimization) {
    if (mpirank_world == 0) printf("\nNow starting Optim solver ... \n");
    optimctx->solve();
    optimctx->getSolution(&opt);
  }


  /* --- Finalize --- */

  /* Get timings */
  double UsedTime = MPI_Wtime() - StartTime;
  /* Get memory usage */
  struct rusage r_usage;
  getrusage(RUSAGE_SELF, &r_usage);
  double myMB = (double)r_usage.ru_maxrss / 1024.0;
  double globalMB;
  MPI_Allreduce(&myMB, &globalMB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  /* Print statistics */
  if (mpirank_world == 0) {
    printf("\n");
    printf(" Used Time:        %.2f seconds\n", UsedTime);
    printf(" Global Memory:    %.2f MB\n", globalMB);
    printf(" Processors used:  %d\n", mpisize_world);
    printf("\n");
  }
  printf("Rank %d: %.2fMB\n", mpirank_world, myMB );

  /* Print timing to file */
  if (mpirank_world == 0) {
    sprintf(filename, "%s/timing.dat", primalbraidapp->datadir.c_str());
    FILE* timefile = fopen(filename, "w");
    fprintf(timefile, "%d  %1.8e\n", mpisize_world, UsedTime);
    fclose(timefile);
    printf("%s written.\n", filename);
  }


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
  mastereq->assemble_RHS(t);
  mastereq->getRHS();
  MatMult(mastereq->getRHS(), x, Ax);

  /* Evaluate dRHSdp(t,p)x */
  mastereq->assemble_dRHSdp(t, x);
  G = mastereq->getdRHSdp();
  // MatView(G, PETSC_VIEWER_STDOUT_WORLD);

  /* Flush control functions */
  for (int i = 0; i < nlevels.size(); i++){
    sprintf(filename, "control_%04d.dat", i);
    oscil_vec[i]->flushControl(ntime, dt, filename);
  }

  /* FD loop */
  // for (int i=0; i<ndesign; i++) {
  {int i=3;

    /* Eval RHS(p+eps)x */
    myinit[i] -= EPS;
    optimproblem->setDesign(ndesign, myinit);
    mastereq->assemble_RHS(t);
    MatMult(mastereq->getRHS(), x, Bx);

    /* Eval RHS(p-eps)x */
    myinit[i] += 2.*EPS;
    optimproblem->setDesign(ndesign, myinit);
    mastereq->assemble_RHS(t);
    MatMult(mastereq>getRHS(), x, Cx);

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
  if (mpirank_world == 0)  {
    printf("\n\n#########################\n");
    printf(" FD Testing... \n");
    printf("#########################\n\n");
  }

  double obj_org;
  double obj_pert1, obj_pert2;

  optimctx->getStartingPoint(xinit);

  /* --- Solve primal --- */
  if (mpirank_world == 0) printf("\nRunning optimizer eval_f... ");
  obj_org = optimctx->evalF(xinit);
  if (mpirank_world == 0) printf(" Obj_orig %1.14e\n", obj_org);

  /* --- Solve adjoint --- */
  if (mpirank_world == 0) printf("\nRunning optimizer eval_grad_f...\n");
  optimctx->evalGradF(xinit, grad);
  VecView(grad, PETSC_VIEWER_STDOUT_WORLD);
  

  /* --- Finite Differences --- */
  if (mpirank_world == 0) printf("\nFD...\n");
  for (int i=0; i<optimctx->ndesign; i++){
  // {int i=0;

    /* Evaluate f(p+eps)*/
    VecSetValue(xinit, i, EPS, ADD_VALUES);
    obj_pert1 = optimctx->evalF(xinit);

    /* Evaluate f(p-eps)*/
    VecSetValue(xinit, i, -2*EPS, ADD_VALUES);
    obj_pert2 = optimctx->evalF(xinit);

    /* Eval FD and error */
    double fd = (obj_pert1 - obj_pert2) / (2.*EPS);
    double err = 0.0;
    double gradi; 
    VecGetValues(grad, 1, &i, &gradi);
    if (fd != 0.0) err = (gradi - fd) / fd;
    if (mpirank_world == 0) printf(" %d: obj %1.14e, obj_pert1 %1.14e, obj_pert2 %1.14e, fd %1.14e, grad %1.14e, err %1.14e\n", i, obj_org, obj_pert1, obj_pert2, fd, gradi, err);

    /* Restore parameter */
    VecSetValue(xinit, i, EPS, ADD_VALUES);
  }
  
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

  for (int iosc=0; iosc<nlevels.size(); iosc++)
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

  Vec exact;  // exact solution
  Vec error;  // error  
  double t;
  double error_norm, exact_norm;

  int nreal = 2*mastereq->getDim();
  VecCreate(PETSC_COMM_WORLD,&x);
  VecSetSizes(x, PETSC_DECIDE, nreal);
  VecSetFromOptions(x);
  VecDuplicate(x, &exact);
  VecDuplicate(x, &error);

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
    TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
    TSInit(ts, mastereq, ntime, dt, total_time, x, lambda, mu, monitor);
    TSSetSolution(ts, x);

    // /* Set the initial condition */
    mastereq->initialCondition(0,x);

    /* Run time-stepping loop */
    for(PetscInt istep = 0; istep <= ntime; istep++) 
    {
      TSStep(ts);
    }

    /* Compute the relative error at last time step (max-norm) */
    TSGetTime(ts, &t);
    mastereq->ExactSolution(t,exact);
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

  /* Clean up */
  // TSDestroy(&ts);  /* TODO */

  /* Clean up Oscillator */
  for (int i=0; i<nlevels.size(); i++){
    delete oscil_vec[i];
  }
  delete [] oscil_vec;


  /* Clean up Master equation*/
  delete mastereq;

  /* Cleanup */
  // TSDestroy(&braid_app->ts);

  delete mytimestepper;
  
  delete primalbraidapp;
  delete adjointbraidapp;

  delete optimctx;



  /* Finallize Petsc */
  ierr = PetscFinalize();

  MPI_Finalize();
  return ierr;
}


