#ifdef WITH_BRAID
#include "braid_wrapper.hpp"
#endif
#include "timestepper.hpp"
#include "bspline.hpp"
#include "oscillator.hpp" 
#include "mastereq.hpp"
#include "config.hpp"
#include <stdlib.h>
#include <sys/resource.h>
#include "optimproblem.hpp"
#include "output.hpp"
#ifdef WITH_SLEPC
#include <slepceps.h>
#endif


#define TEST_FD_GRAD 0    // Run Finite Differences gradient test
#define TEST_FD_HESS 0    // Run Finite Differences Hessian test
#define HESSIAN_DECOMPOSITION 0 // Run eigenvalue analysis for Hessian
#define EPS 1e-5          // Epsilon for Finite Differences

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
  if (argc < 2) {
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
  RunType runtype;
  std::string runtypestr = config.GetStrParam("runtype", "simulation");
  if      (runtypestr.compare("simulation")      == 0) runtype = RunType::SIMULATION;
  else if (runtypestr.compare("gradient")     == 0)    runtype = RunType::GRADIENT;
  else if (runtypestr.compare("optimization")== 0)     runtype = RunType::OPTIMIZATION;
  else {
    printf("\n\n WARNING: Unknown runtype: %s.\n\n", runtypestr.c_str());
    runtype = RunType::NONE;
  }

  /* Get the number of essential levels per oscillator. 
   * Default: same as number of levels */  
  std::vector<int> nessential(nlevels.size());
  for (int iosc = 0; iosc<nlevels.size(); iosc++) nessential[iosc] = nlevels[iosc];
  /* Overwrite if config option is given */
  std::vector<int> read_nessential;
  config.GetVecIntParam("nessential", read_nessential, -1);
  if (read_nessential[0] > -1) {
    for (int iosc = 0; iosc<nlevels.size(); iosc++){
      if (iosc < read_nessential.size()) nessential[iosc] = read_nessential[iosc];
      else                               nessential[iosc] = read_nessential[read_nessential.size()-1];
      if (nessential[iosc] > nlevels[iosc]) nessential[iosc] = nlevels[iosc];
    }
  }


  /* Get type and the total number of initial conditions */
  int ninit = 1;
  std::vector<std::string> initcondstr;
  config.GetVecStrParam("initialcondition", initcondstr, "none");
  assert (initcondstr.size() >= 1);
  if      (initcondstr[0].compare("file") == 0 ) ninit = 1;
  else if (initcondstr[0].compare("pure") == 0 ) ninit = 1;
  else if (initcondstr[0].compare("ensemble") == 0 ) ninit = 1;
  else if (initcondstr[0].compare("3states") == 0 ) ninit = 3;
  else if (initcondstr[0].compare("Nplus1") == 0 )  {
    // compute system dimension N 
    ninit = 1;
    for (int i=0; i<nlevels.size(); i++){
      ninit *= nlevels[i];
    }
    ninit +=1;
  }
  else if ( initcondstr[0].compare("diagonal") == 0 ||
            initcondstr[0].compare("basis")    == 0  ) {
    /* Compute ninit = dim(subsystem defined by list of oscil IDs) */
    ninit = 1;
    if (initcondstr.size() < 2) {
      printf("ERROR for initial condition option: If choosing 'basis' or 'diagonal', specify the list oscillator IDs!\n");
      exit(1);
    }
    for (int i = 1; i<initcondstr.size(); i++){
      int oscilID = atoi(initcondstr[i].c_str());
      ninit *= nessential[oscilID];
    }
    if (initcondstr[0].compare("basis") == 0  ) {
      // if Schroedinger solver: ninit = N, do nothing.
      // else Lindblad solver: ninit = N^2
      std::string tmpstr = config.GetStrParam("collapse_type", "none", false);
      if (tmpstr.compare("none") != 0 ) ninit = (int) pow(ninit,2.0);
    }
  }
  else {
    printf("\n\n ERROR: Wrong setting for initial condition.\n");
    exit(1);
  }

  /* --- Split communicators for distributed initial conditions, distributed linear algebra, time-parallel braid --- */
  int mpirank_init, mpisize_init;
  int mpirank_braid, mpisize_braid;
  int mpirank_petsc, mpisize_petsc;
  MPI_Comm comm_braid, comm_init, comm_petsc;

  /* Get the size of communicators  */
#ifdef WITH_BRAID
  int np_braid = config.GetIntParam("np_braid", 1);
  np_braid = min(np_braid, mpisize_world); 
#else 
  int np_braid = 1; 
#endif
  int np_init  = min(ninit, config.GetIntParam("np_init", 1)); 
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

#ifdef WITH_BRAID
  // Time-parallel Braid
  int color_braid = mpirank_world % np_petsc + mpirank_init * np_petsc;
  MPI_Comm_split(MPI_COMM_WORLD, color_braid, mpirank_world, &comm_braid);
  MPI_Comm_rank(comm_braid, &mpirank_braid);
  MPI_Comm_size(comm_braid, &mpisize_braid);
#else 
  mpirank_braid = 0;
  mpisize_braid = 1;
#endif

  // Distributed Linear algebra: Petsc
  int color_petsc = mpirank_world / np_petsc;
  MPI_Comm_split(MPI_COMM_WORLD, color_petsc, mpirank_world, &comm_petsc);
  MPI_Comm_rank(comm_petsc, &mpirank_petsc);
  MPI_Comm_size(comm_petsc, &mpisize_petsc);

  if (mpirank_world == 0)  std::cout<< "Parallel distribution: " << mpisize_init << " np_init  X  " << mpisize_petsc<< " np_petsc";
#ifdef WITH_BRAID
  std::cout<< "  X  " << mpisize_braid  << "np_braid" << std::endl;
#endif
  if (mpirank_world == 0) std::cout<<std::endl;

  /* Initialize Petsc using petsc's communicator */
  PETSC_COMM_WORLD = comm_petsc;
#ifdef WITH_SLEPC
  ierr = SlepcInitialize(&argc, &argv, (char*)0, NULL);if (ierr) return ierr;
#else
  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;
#endif
  PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, 	PETSC_VIEWER_ASCII_MATLAB );



  double total_time = ntime * dt;

  /* --- Initialize the Oscillators --- */
  Oscillator** oscil_vec = new Oscillator*[nlevels.size()];
  // Get fundamental and rotation frequencies from config file 
  std::vector<double> trans_freq, rot_freq;
  config.GetVecDoubleParam("transfreq", trans_freq, 1e20);
  if (trans_freq.size() < nlevels.size()) {
    printf("Error: Number of given fundamental frequencies (%lu) is smaller than the the number of oscillators (%lu)\n", trans_freq.size(), nlevels.size());
    exit(1);
  } 
  config.GetVecDoubleParam("rotfreq", rot_freq, 1e20);
  if (rot_freq.size() < nlevels.size()) {
    printf("Error: Number of given rotation frequencies (%lu) is smaller than the the number of oscillators (%lu)\n", rot_freq.size(), nlevels.size());
    exit(1);
  } 
  // Get self kerr coefficient
  std::vector<double> selfkerr;
  config.GetVecDoubleParam("selfkerr", selfkerr, 0.0);   // self ker \xi_k 
  assert(selfkerr.size() >= nlevels.size());
  // Get lindblad type and collapse times
  std::string lindblad = config.GetStrParam("collapse_type", "none");
  std::vector<double> decay_time, dephase_time;
  config.GetVecDoubleParam("decay_time", decay_time, 0.0);
  config.GetVecDoubleParam("dephase_time", dephase_time, 0.0);
  LindbladType lindbladtype;
  if      (lindblad.compare("none")      == 0 ) lindbladtype = LindbladType::NONE;
  else if (lindblad.compare("decay")     == 0 ) lindbladtype = LindbladType::DECAY;
  else if (lindblad.compare("dephase")   == 0 ) lindbladtype = LindbladType::DEPHASE;
  else if (lindblad.compare("both")      == 0 ) lindbladtype = LindbladType::BOTH;
  else {
    printf("\n\n ERROR: Unnown lindblad type: %s.\n", lindblad.c_str());
    printf(" Choose either 'none', 'decay', 'dephase', or 'both'\n");
    exit(1);
  }
  if (lindbladtype != LindbladType::NONE) {
    assert(decay_time.size() >= nlevels.size());
    assert(dephase_time.size() >= nlevels.size());
  }

  // Create the oscillators 
  for (int i = 0; i < nlevels.size(); i++){
    std::vector<double> carrier_freq;
    std::string key = "carrier_frequency" + std::to_string(i);
    config.GetVecDoubleParam(key, carrier_freq, 0.0);
    oscil_vec[i] = new Oscillator(i, nlevels, nspline, trans_freq[i], selfkerr[i], rot_freq[i], decay_time[i], dephase_time[i], carrier_freq, total_time, lindbladtype);
  }

  // Get pi-pulses, if any
  std::vector<std::string> pipulse_str;
  config.GetVecStrParam("apply_pipulse", pipulse_str, "none");
  if (pipulse_str[0].compare("none") != 0) { // There is at least one pipulse to be applied!
    // sanity check
    if (pipulse_str.size() % 4 != 0) {
      printf("Wrong pi-pulse configuration. Number of elements must be multiple of 4!\n");
      printf("apply_pipulse config option: <oscilID>, <tstart>, <tstop>, <amp>, <anotherOscilID>, <anotherTstart>, <anotherTstop>, <anotherAmp> ...\n");
      exit(1);
    }
    int k=0;
    while (k < pipulse_str.size()){
      // Set pipulse for this oscillator
      int pipulse_id = atoi(pipulse_str[k+0].c_str());
      oscil_vec[pipulse_id]->pipulse.tstart.push_back(atof(pipulse_str[k+1].c_str()));
      oscil_vec[pipulse_id]->pipulse.tstop.push_back(atof(pipulse_str[k+2].c_str()));
      oscil_vec[pipulse_id]->pipulse.amp.push_back(atof(pipulse_str[k+3].c_str()));
      if (mpirank_world==0) printf("Applying PiPulse to oscillator %d in [%f,%f]: |p+iq|=%f\n", pipulse_id, oscil_vec[pipulse_id]->pipulse.tstart.back(), oscil_vec[pipulse_id]->pipulse.tstop.back(), oscil_vec[pipulse_id]->pipulse.amp.back());
      // Set zero control for all other oscillators during this pipulse
      for (int i=0; i<nlevels.size(); i++){
        if (i != pipulse_id) {
          oscil_vec[i]->pipulse.tstart.push_back(atof(pipulse_str[k+1].c_str()));
          oscil_vec[i]->pipulse.tstop.push_back(atof(pipulse_str[k+2].c_str()));
          oscil_vec[i]->pipulse.amp.push_back(0.0);
        }
      }
      k+=4;
    }
  }

  /* --- Initialize the Master Equation  --- */
  // Get self and cross kers and coupling terms 
  std::vector<double> crosskerr, Jkl;
  config.GetVecDoubleParam("crosskerr", crosskerr, 0.0);   // cross ker \xi_{kl}, zz-coupling
  config.GetVecDoubleParam("Jkl", Jkl, 0.0); // Jaynes-Cummings coupling
  // If not enough elements are given, fill up with zeros!
  int noscillators = nlevels.size();
  for (int i = crosskerr.size(); i < (noscillators-1) * noscillators / 2; i++)  crosskerr.push_back(0.0);
  for (int i = Jkl.size(); i < (noscillators-1) * noscillators / 2; i++) Jkl.push_back(0.0);
  // Sanity check for matrix free solver
  bool usematfree = config.GetBoolParam("usematfree", false);
  if ( (usematfree && nlevels.size() < 2) ||   
       (usematfree && nlevels.size() > 5)   ){
        printf("Warning: Matrix free solver is only implemented for systems with 2, 3, 4, or 5 oscillators. Switching to sparse-matrix solver now.\n");
        usematfree = false;
  }
  if (usematfree && mpisize_petsc > 1) {
    printf("ERROR: No Petsc-parallel version for the matrix free solver available!");
    exit(1);
  }
  // Compute coupling rotation frequencies eta_ij = w^r_i - w^r_j
  std::vector<double> eta(nlevels.size()*(nlevels.size()-1)/2.);
  int idx = 0;
  for (int iosc=0; iosc<nlevels.size(); iosc++){
    for (int josc=iosc+1; josc<nlevels.size(); josc++){
      eta[idx] = rot_freq[iosc] - rot_freq[josc];
      idx++;
    }
  }
  MasterEq* mastereq = new MasterEq(nlevels, nessential, oscil_vec, crosskerr, Jkl, eta, lindbladtype, usematfree);


  /* Output */
#ifdef WITH_BRAID
  Output* output = new Output(config, comm_petsc, comm_init, comm_braid, noscillators);
#else 
  Output* output = new Output(config, comm_petsc, comm_init, noscillators);
#endif

  // Some screen output 
  if (mpirank_world == 0) {
    std::cout << "Time: [0:" << total_time << "], ";
    std::cout << "N="<< ntime << ", dt=" << dt << std::endl;
    std::cout<< "System: ";
    for (int i=0; i<nlevels.size(); i++) {
      std::cout<< nlevels[i];
      if (i < nlevels.size()-1) std::cout<< "x";
    }
    std::cout<<"  (essential levels: ";
    for (int i=0; i<nlevels.size(); i++) {
      std::cout<< nessential[i];
      if (i < nlevels.size()-1) std::cout<< "x";
    }
    std::cout << ") " << std::endl;
  }

  /* --- Initialize the time-stepper --- */
  LinearSolverType linsolvetype;
  std::string linsolvestr = config.GetStrParam("linearsolver_type", "gmres");
  int linsolve_maxiter = config.GetIntParam("linearsolver_maxiter", 10);
  if      (linsolvestr.compare("gmres")   == 0) linsolvetype = LinearSolverType::GMRES;
  else if (linsolvestr.compare("neumann") == 0) linsolvetype = LinearSolverType::NEUMANN;
  else {
    printf("\n\n ERROR: Unknown linear solver type: %s.\n\n", linsolvestr.c_str());
    exit(1);
  }
  /* My time stepper */
  bool storeFWD = false;
  if (runtype == RunType::GRADIENT || runtype == RunType::OPTIMIZATION) storeFWD = true;
#if TEST_FD_GRAD
  storeFWD = true;
#endif
#if TEST_FD_HESS
  storeFWD = true;
#endif
  TimeStepper *mytimestepper = new ImplMidpoint(mastereq, ntime, total_time, linsolvetype, linsolve_maxiter, output, storeFWD);
  // TimeStepper *mytimestepper = new ExplEuler(mastereq, ntime, total_time, output, storeFWD);

  // /* Petsc's Time-stepper */
  // Vec x;
  // MatCreateVecs(mastereq->getRHS(), &x, NULL);
  // TS ts;
  // TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  // TSInit(ts, mastereq, ntime, dt, total_time, x, false);
   

#ifdef WITH_BRAID
  /* --- Create braid instances --- */
  myBraidApp* primalbraidapp = NULL;
  myAdjointBraidApp *adjointbraidapp = NULL;
  // Create primal app always, adjoint only if runtype is adjoint or optimization 
  primalbraidapp = new myBraidApp(comm_braid, total_time, ntime, mytimestepper, mastereq, &config, output);
  if (runtype == RunType::GRADIENT || runtype == RunType::OPTIMIZATION) adjointbraidapp = new myAdjointBraidApp(comm_braid, total_time, ntime, mytimestepper, mastereq, &config, primalbraidapp->getCore(), output);
  // Initialize the braid time-grids. Warning: initGrids for primal app depends on initialization of adjoint! Do not move this line up!
  primalbraidapp->InitGrids();
  if (runtype == RunType::GRADIENT || runtype == RunType::OPTIMIZATION) adjointbraidapp->InitGrids();
  
#endif

  /* --- Initialize optimization --- */
  /* Get gate rotation frequencies. Default: use rotational frequencies for the gate. */
  std::vector<double> gate_rot_freq(noscillators); 
  for (int iosc=0; iosc<noscillators; iosc++) gate_rot_freq[iosc] = rot_freq[iosc];
  /* If gate_rot_freq option is given in config file, overwrite them with input */
  std::vector<double> read_gate_rot;
  config.GetVecDoubleParam("gate_rot_freq", read_gate_rot, 1e20); 
  if (read_gate_rot[0] < 1e20) { // the config option exists
    for (int i=0; i<noscillators; i++) {
      if (i < read_gate_rot.size()) gate_rot_freq[i] = read_gate_rot[i];
      else gate_rot_freq[i] = read_gate_rot[read_gate_rot.size()-1]; // using the last element for all remaining ones
    }
  }

#ifdef WITH_BRAID
  OptimProblem* optimctx = new OptimProblem(config, mytimestepper, primalbraidapp, adjointbraidapp, comm_init, ninit, gate_rot_freq, output);
#else 
  OptimProblem* optimctx = new OptimProblem(config, mytimestepper, comm_init, ninit, gate_rot_freq, output);
#endif

  /* Set upt solution and gradient vector */
  Vec xinit;
  VecCreateSeq(PETSC_COMM_SELF, optimctx->getNdesign(), &xinit);
  VecSetFromOptions(xinit);
  Vec grad;
  VecCreateSeq(PETSC_COMM_SELF, optimctx->getNdesign(), &grad);
  VecSetUp(grad);
  VecZeroEntries(grad);
  Vec opt;

  /* Some output */
  if (mpirank_world == 0)
  {
    /* Print parameters to file */
    sprintf(filename, "%s/config_log.dat", output->datadir.c_str());
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

  double objective;
  double gnorm = 0.0;
  /* --- Solve primal --- */
  if (runtype == RunType::SIMULATION) {
    optimctx->getStartingPoint(xinit);
    if (mpirank_world == 0) printf("\nStarting primal solver... \n");
    objective = optimctx->evalF(xinit);
    if (mpirank_world == 0) printf("\nTotal objective = %1.14e, \n", objective);
    optimctx->getSolution(&opt);
  } 
  
  /* --- Solve adjoint --- */
  if (runtype == RunType::GRADIENT) {
    optimctx->getStartingPoint(xinit);
    if (mpirank_world == 0) printf("\nStarting adjoint solver...\n");
    optimctx->evalGradF(xinit, grad);
    VecNorm(grad, NORM_2, &gnorm);
    // VecView(grad, PETSC_VIEWER_STDOUT_WORLD);
    if (mpirank_world == 0) {
      printf("\nGradient norm: %1.14e\n", gnorm);
    }
    optimctx->output->writeGradient(grad);
  }

  /* --- Solve the optimization  --- */
  if (runtype == RunType::OPTIMIZATION) {
    /* Set initial starting point */
    optimctx->getStartingPoint(xinit);
    if (mpirank_world == 0) printf("\nStarting Optimization solver ... \n");
    optimctx->solve(xinit);
    optimctx->getSolution(&opt);
  }

  /* Output */
  if (runtype != RunType::OPTIMIZATION) optimctx->output->writeOptimFile(optimctx->getObjective(), gnorm, 0.0, optimctx->getFidelity(), optimctx->getCostT(), optimctx->getRegul(), optimctx->getPenalty());


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
  // printf("Rank %d: %.2fMB\n", mpirank_world, myMB );

  /* Print timing to file */
  if (mpirank_world == 0) {
    sprintf(filename, "%s/timing.dat", output->datadir.c_str());
    FILE* timefile = fopen(filename, "w");
    fprintf(timefile, "%d  %1.8e\n", mpisize_world, UsedTime);
    fclose(timefile);
  }


#if TEST_FD_GRAD
  if (mpirank_world == 0)  {
    printf("\n\n#########################\n");
    printf(" FD Testing for Gradient ... \n");
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
  for (PetscInt i=0; i<optimctx->getNdesign(); i++){
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


#if TEST_FD_HESS
  if (mpirank_world == 0)  {
    printf("\n\n#########################\n");
    printf(" FD Testing for Hessian... \n");
    printf("#########################\n\n");
  }
  optimctx->getStartingPoint(xinit);

  /* Figure out which parameters are hitting bounds */
  double bound_tol = 1e-3;
  std::vector<int> Ihess; // Index set for all elements that do NOT hit a bound
  for (PetscInt i=0; i<optimctx->getNdesign(); i++){
    // get x_i and bounds for x_i
    double xi, blower, bupper;
    VecGetValues(xinit, 1, &i, &xi);
    VecGetValues(optimctx->xlower, 1, &i, &blower);
    VecGetValues(optimctx->xupper, 1, &i, &bupper);
    // compare 
    if (fabs(xi - blower) < bound_tol || 
        fabs(xi - bupper) < bound_tol  ) {
          printf("Parameter %d hits bound: x=%f\n", i, xi);
    } else {
      Ihess.push_back(i);
    }
  }

  double grad_org;
  double grad_pert1, grad_pert2;
  Mat Hess;
  int nhess = Ihess.size();
  MatCreateSeqDense(PETSC_COMM_SELF, nhess, nhess, NULL, &Hess);
  MatSetUp(Hess);

  Vec grad1, grad2;
  VecDuplicate(grad, &grad1);
  VecDuplicate(grad, &grad2);


  /* Iterate over all params that do not hit a bound */
  for (PetscInt k=0; k< Ihess.size(); k++){
    PetscInt j = Ihess[k];
    printf("Computing column %d\n", j);

    /* Evaluate \nabla_x J(x + eps * e_j) */
    VecSetValue(xinit, j, EPS, ADD_VALUES); 
    optimctx->evalGradF(xinit, grad);        
    VecCopy(grad, grad1);

    /* Evaluate \nabla_x J(x - eps * e_j) */
    VecSetValue(xinit, j, -2.*EPS, ADD_VALUES); 
    optimctx->evalGradF(xinit, grad);
    VecCopy(grad, grad2);

    for (PetscInt l=0; l<Ihess.size(); l++){
      PetscInt i = Ihess[l];

      /* Get the derivative wrt parameter i */
      VecGetValues(grad1, 1, &i, &grad_pert1);   // \nabla_x_i J(x+eps*e_j)
      VecGetValues(grad2, 1, &i, &grad_pert2);    // \nabla_x_i J(x-eps*e_j)

      /* Finite difference for element Hess(l,k) */
      double fd = (grad_pert1 - grad_pert2) / (2.*EPS);
      MatSetValue(Hess, l, k, fd, INSERT_VALUES);
    }

    /* Restore parameters xinit */
    VecSetValue(xinit, j, EPS, ADD_VALUES);
  }
  /* Assemble the Hessian */
  MatAssemblyBegin(Hess, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Hess, MAT_FINAL_ASSEMBLY);
  
  /* Clean up */
  VecDestroy(&grad1);
  VecDestroy(&grad2);


  /* Epsilon test: compute ||1/2(H-H^T)||_F  */
  MatScale(Hess, 0.5);
  Mat HessT, Htest;
  MatDuplicate(Hess, MAT_COPY_VALUES, &Htest);
  MatTranspose(Hess, MAT_INITIAL_MATRIX, &HessT);
  MatAXPY(Htest, -1.0, HessT, SAME_NONZERO_PATTERN);
  double fnorm;
  MatNorm(Htest, NORM_FROBENIUS, &fnorm);
  printf("EPS-test: ||1/2(H-H^T)||= %1.14e\n", fnorm);

  /* symmetrize H_symm = 1/2(H+H^T) */
  MatAXPY(Hess, 1.0, HessT, SAME_NONZERO_PATTERN);

  /* --- Print Hessian to file */
  
  sprintf(filename, "%s/hessian.dat", output->datadir.c_str());
  printf("File written: %s.\n", filename);
  PetscViewer viewer;
  PetscViewerCreate(MPI_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERASCII);
  PetscViewerFileSetMode(viewer, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewer, filename);
  // PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_DENSE);
  MatView(Hess, viewer);
  PetscViewerPopFormat(viewer);
  PetscViewerDestroy(&viewer);

  // write again in binary
  sprintf(filename, "%s/hessian_bin.dat", output->datadir.c_str());
  printf("File written: %s.\n", filename);
  PetscViewerBinaryOpen(MPI_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer);
  MatView(Hess, viewer);
  PetscViewerDestroy(&viewer);

  MatDestroy(&Hess);

#endif

#if HESSIAN_DECOMPOSITION 
  /* --- Compute eigenvalues of Hessian --- */
  printf("\n\n#########################\n");
  printf(" Eigenvalue analysis... \n");
  printf("#########################\n\n");

  /* Load Hessian from file */
  Mat Hess;
  MatCreate(PETSC_COMM_SELF, &Hess);
  sprintf(filename, "%s/hessian_bin.dat", output->datadir.c_str());
  printf("Reading file: %s\n", filename);
  PetscViewer viewer;
  PetscViewerCreate(MPI_COMM_WORLD, &viewer);
  PetscViewerSetType(viewer, PETSCVIEWERBINARY);
  PetscViewerFileSetMode(viewer, FILE_MODE_READ);
  PetscViewerFileSetName(viewer, filename);
  PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_DENSE);
  MatLoad(Hess, viewer);
  PetscViewerPopFormat(viewer);
  PetscViewerDestroy(&viewer);
  int nrows, ncols;
  MatGetSize(Hess, &nrows, &ncols);


  /* Set the percentage of eigenpairs that should be computed */
  double frac = 1.0;  // 1.0 = 100%
  int neigvals = nrows * frac;     // hopefully rounds to closest int 
  printf("\nComputing %d eigenpairs now...\n", neigvals);
  
  /* Compute eigenpair */
  std::vector<double> eigvals;
  std::vector<Vec> eigvecs;
  getEigvals(Hess, neigvals, eigvals, eigvecs);

  /* Print eigenvalues to file. */
  FILE *file;
  sprintf(filename, "%s/eigvals.dat", output->datadir.c_str());
  file =fopen(filename,"w");
  for (int i=0; i<eigvals.size(); i++){
      fprintf(file, "% 1.8e\n", eigvals[i]);  
  }
  fclose(file);
  printf("File written: %s.\n", filename);

  /* Print eigenvectors to file. Columns wise */
  sprintf(filename, "%s/eigvecs.dat", output->datadir.c_str());
  file =fopen(filename,"w");
  for (PetscInt j=0; j<nrows; j++){  // rows
    for (PetscInt i=0; i<eigvals.size(); i++){
      double val;
      VecGetValues(eigvecs[i], 1, &j, &val); // j-th row of eigenvalue i
      fprintf(file, "% 1.8e  ", val);  
    }
    fprintf(file, "\n");
  }
  fclose(file);
  printf("File written: %s.\n", filename);


#endif

#ifdef SANITY_CHECK
  printf("\n\n Sanity checks have been performed. Check output for warnings and errors!\n\n");
#endif

  /* Clean up */
  for (int i=0; i<nlevels.size(); i++){
    delete oscil_vec[i];
  }
  delete [] oscil_vec;
  delete mastereq;
  delete mytimestepper;
#ifdef WITH_BRAID
  delete primalbraidapp;
  if (runtype == RunType::SIMULATION || runtype == RunType::GRADIENT) delete adjointbraidapp;
#endif
  delete optimctx;
  delete output;

  // TSDestroy(&ts);  /* TODO */

  /* Finallize Petsc */
#ifdef WITH_SLEPC
  ierr = SlepcFinalize();
#else
  ierr = PetscFinalize();
#endif


  MPI_Finalize();
  return ierr;
}


