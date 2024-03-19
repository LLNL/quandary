#include "timestepper.hpp"
#include "defs.hpp"
#include <string>
#include "oscillator.hpp" 
#include "mastereq.hpp"
#include "config.hpp"
#include <stdlib.h>
#include <sys/resource.h>
#include <cassert>
#include "optimproblem.hpp"
#include "output.hpp"
#include "petsc.h"
#include <random>
#ifdef WITH_SLEPC
#include <slepceps.h>
#endif

int test_unitarize(int argc,char **argv)
{
  char filename[255];
  PetscErrorCode ierr;

  /* Initialize MPI */
  int mpisize_world, mpirank_world;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize_world); // mpisize_world is the total number of tasks, read from cmd line arg

  /* Parse argument line for "--quiet" to enable reduced output mode */
  bool quietmode = false;
  if (argc > 2){
    for (int i=2; i<argc; i++) {
      std::string quietstring = argv[i];
      if (quietstring.length() > 2 && quietstring.substr(2,5).compare("quiet") == 0) {
        quietmode = true;
        // printf("quietmode =  %d\n", quietmode);
      }
    }
  }

  if (mpirank_world == 0 /* && !quietmode */) printf("Running Quandary on %d cores.\n", mpisize_world);

  /* Read config file */
  if (argc < 2) {
    if (mpirank_world == 0) {
      printf("\nUSAGE: ./main </path/to/configfile> \n");
    }
    MPI_Finalize();
    return 0;
  }
  std::stringstream log;
  MapParam config(MPI_COMM_WORLD, log, quietmode);
  config.ReadFile(argv[1]);

  /* Initialize random number generator: Check if rand_seed is provided from config file, otherwise set random. */
  int rand_seed = config.GetIntParam("rand_seed", -1, false, false);
  std::random_device rd;
  if (rand_seed < 0){
    rand_seed = rd();  // random non-reproducable seed
  }
  MPI_Bcast(&rand_seed, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); // Broadcast from rank 0 to all.
  std::default_random_engine rand_engine{};
  rand_engine.seed(rand_seed);
  export_param(mpirank_world, *config.log, "rand_seed", rand_seed);

  /* --- Get some options from the config file --- */
  std::vector<int> nlevels;
  config.GetVecIntParam("nlevels", nlevels, 0);
  
  int ntime = config.GetIntParam("ntime", 1000);    // number of global time steps
  // AP: This would be a good place to modify ntime to be divisible by nwindows

  double total_time = config.GetDoubleParam("duration", -1.0);
  double dt;
  if (total_time < 0.0){ // NOTE if a 'duration' line is not present in the cfg file, rollback to the original approach
    dt    = config.GetDoubleParam("dt", 0.01);
    total_time = ntime * dt;
  }
  else{
    dt = total_time/ntime;
  }

  RunType runtype;
  std::string runtypestr = config.GetStrParam("runtype", "simulation");
  if      (runtypestr.compare("simulation")      == 0) runtype = RunType::SIMULATION;
  else if (runtypestr.compare("gradient")     == 0)    runtype = RunType::GRADIENT;
  else if (runtypestr.compare("optimization")== 0)     runtype = RunType::OPTIMIZATION;
  else if (runtypestr.compare("evalcontrols")== 0)     runtype = RunType::EVALCONTROLS;
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
  else if (initcondstr[0].compare("performance") == 0 ) ninit = 1;
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
      for (int j=0; j<nlevels.size(); j++)  initcondstr.push_back(std::to_string(j));
    }
    for (int i = 1; i<initcondstr.size(); i++){
      int oscilID = atoi(initcondstr[i].c_str());
      if (oscilID < nessential.size()) ninit *= nessential[oscilID];
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
  if (mpirank_world == 0)
    printf("Number of initial conditions: %d\n", ninit);

  /* --- Split communicators for distributed initial conditions, distributed linear algebra, time-parallel optimization --- */
  int mpirank_init, mpisize_init;
  int mpirank_time, mpisize_time;
  int mpirank_petsc, mpisize_petsc;
  MPI_Comm comm_time, comm_init, comm_petsc;

  /* Get the size of communicators  */

  // Parse command line args for "-np_init <x>" and "-np_time <x>" 
  int np_init=0;
  int np_time=0;
  int np_petsc=0;
  for (int i=2; i<argc; i++) {
    std::string mystring = argv[i];
    if (mystring.length() > 2 && mystring.substr(1,7).compare("np_init") == 0) {
      np_init = std::stoi(argv[i+1]);
      np_init = std::min(np_init, mpisize_world);
    }
    if (mystring.length() > 2 && mystring.substr(1,7).compare("np_time") == 0) {
      np_time = std::stoi(argv[i+1]);
      np_time = std::min(np_time, mpisize_world);
    }
  }

  // Number of petsc cores. Hardcode 1 for now due to time-parallel development. TODO.
  np_petsc= 1;
  // If either np_init or np_time is given, use it, and compute the other one
  if (np_init > 0 && np_time <= 0) {
    np_init = std::min(ninit, np_init); // Limit by number of initial conditions
    np_time = mpisize_world / (np_init * np_petsc);
  } else if (np_time > 0 && np_init <= 0) {
    np_init = mpisize_world / (np_time * np_petsc);
    np_init = std::min(ninit, np_init); // Limit by number of initial conditions
  } else if (np_init <= 0 && np_time <= 0){
    // If none were given, default np_init = ninit
    np_init = std::min(ninit, mpisize_world); 
    np_time = mpisize_world / (np_init * np_petsc);
  }
  // If both were given, they have to match to mpisize_world -> sanity check

  /* Sanity check for communicator sizes */ 
  if (mpisize_world != np_init * np_time) {
    if (mpirank_world == 0) printf("ERROR: Can't split %d cores onto processor grid of %d (initial conditions) x %d (time-parallel) \n", mpisize_world, np_init, np_time);
    MPI_Finalize();
    exit(1);
  }
  if (ninit % np_init != 0) {
    if (mpirank_world == 0) printf("ERROR: Number of initial conditions (%d) can not be evenly distributed amongst %d cores.\n", ninit, np_init);
    MPI_Finalize();
    exit(1);
  }


  /* Split communicators */


  // Time-parallel Optimization
  int color_time = mpirank_world % (np_petsc * np_init);
  // int color_time = mpirank_world % np_petsc + mpirank_init * np_petsc;
  MPI_Comm_split(MPI_COMM_WORLD, color_time , mpirank_world, &comm_time);
  MPI_Comm_rank(comm_time, &mpirank_time);
  MPI_Comm_size(comm_time, &mpisize_time);



  // Distributed initial conditions 
  // int color_init = mpirank_world % (np_petsc * np_time);
  int color_init= mpirank_world % np_petsc + mpirank_time* np_petsc;
  MPI_Comm_split(MPI_COMM_WORLD, color_init, mpirank_world, &comm_init);
  MPI_Comm_rank(comm_init, &mpirank_init);
  MPI_Comm_size(comm_init, &mpisize_init);


  // Distributed Linear algebra: Petsc
  int color_petsc = mpirank_world / np_petsc;
  MPI_Comm_split(MPI_COMM_WORLD, color_petsc, mpirank_world, &comm_petsc);
  MPI_Comm_rank(comm_petsc, &mpirank_petsc);
  MPI_Comm_size(comm_petsc, &mpisize_petsc);

  /* Set Petsc using petsc's communicator */
  // PETSC_COMM_WORLD = comm_petsc;
  PETSC_COMM_WORLD = MPI_COMM_WORLD;  // PETSC WILL BE THE GLOBAL COMMUNICATOR

  if (mpirank_world == 0 && !quietmode)  std::cout<< "Parallel distribution: " << mpisize_init << " np_init  X  " << mpisize_time << " np_time" << std::endl;

#ifdef WITH_SLEPC
  ierr = SlepcInitialize(&argc, &argv, (char*)0, NULL);if (ierr) return ierr;
#else
  ierr = PetscInitialize(&argc,&argv,(char*)0,NULL);if (ierr) return ierr;
#endif
  PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, 	PETSC_VIEWER_ASCII_MATLAB );

//  double total_time = ntime * dt;

  /* --- Initialize the Oscillators --- */
  Oscillator** oscil_vec = new Oscillator*[nlevels.size()];
  // Get fundamental and rotation frequencies from config file 
  std::vector<double> trans_freq, rot_freq;
  config.GetVecDoubleParam("transfreq", trans_freq, 1e20);
  copyLast(trans_freq, nlevels.size());

  config.GetVecDoubleParam("rotfreq", rot_freq, 1e20);
  copyLast(rot_freq, nlevels.size());
  // Get self kerr coefficient
  std::vector<double> selfkerr;
  config.GetVecDoubleParam("selfkerr", selfkerr, 0.0);   // self ker \xi_k 
  copyLast(selfkerr, nlevels.size());
  // Get lindblad type and collapse times
  std::string lindblad = config.GetStrParam("collapse_type", "none");
  std::vector<double> decay_time, dephase_time;
  config.GetVecDoubleParam("decay_time", decay_time, 0.0, true, false);
  config.GetVecDoubleParam("dephase_time", dephase_time, 0.0, true, false);
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
  copyLast(decay_time, nlevels.size());
  copyLast(dephase_time, nlevels.size());

  // Get control segment types, carrierwaves and control initialization
  std::string default_seg_str = "spline, 10, 0.0, "+std::to_string(total_time); // Default for first oscillator control segment
  std::string default_init_str = "constant, 0.0";                               // Default for first oscillator initialization
  for (int i = 0; i < nlevels.size(); i++){
    // Get carrier wave frequencies 
    std::vector<double> carrier_freq;
    std::string key = "carrier_frequency" + std::to_string(i);
    config.GetVecDoubleParam(key, carrier_freq, 0.0);

    // Get control type. Default for second or larger oscillator is the previous one
    std::vector<std::string> controltype_str;
    config.GetVecStrParam("control_segments" + std::to_string(i), controltype_str,default_seg_str);

    // Get control initialization
    std::vector<std::string> controlinit_str;
    config.GetVecStrParam("control_initialization" + std::to_string(i), controlinit_str, default_init_str);

    // Create oscillator 
    oscil_vec[i] = new Oscillator(config, i, nlevels, controltype_str, controlinit_str, trans_freq[i], selfkerr[i], rot_freq[i], decay_time[i], dephase_time[i], carrier_freq, total_time, lindbladtype, rand_engine);
    
    // Update the default for control type
    default_seg_str = "";
    default_init_str = "";
    for (int l = 0; l<controltype_str.size(); l++) default_seg_str += controltype_str[l]+=", ";
    for (int l = 0; l<controlinit_str.size(); l++) default_init_str += controlinit_str[l]+=", ";
  }

  // Get pi-pulses, if any
  std::vector<std::string> pipulse_str;
  config.GetVecStrParam("apply_pipulse", pipulse_str, "none", true, false);
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
  config.GetVecDoubleParam("Jkl", Jkl, 0.0); // Jaynes-Cummings (dipole-dipole) coupling coeff
  copyLast(crosskerr, (nlevels.size()-1) * nlevels.size()/ 2);
  copyLast(Jkl, (nlevels.size()-1) * nlevels.size()/ 2);
  // If not enough elements are given, fill up with zeros!
  // for (int i = crosskerr.size(); i < (noscillators-1) * noscillators / 2; i++)  crosskerr.push_back(0.0);
  // for (int i = Jkl.size(); i < (noscillators-1) * noscillators / 2; i++) Jkl.push_back(0.0);
  // Sanity check for matrix free solver
  bool usematfree = config.GetBoolParam("usematfree", false);
  if (usematfree && nlevels.size() > 5){
        printf("Warning: Matrix free solver is only implemented for systems with 2, 3, 4, or 5 oscillators. Switching to sparse-matrix solver now.\n");
        usematfree = false;
  }
  if (usematfree && mpisize_petsc > 1) {
    if (mpirank_world == 0) printf("ERROR: No Petsc-parallel version for the matrix free solver available!");
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
  // Check if Hamiltonian should be read from file
  std::string hamiltonian_file = config.GetStrParam("hamiltonian_file", "none", true, false);
  if (hamiltonian_file.compare("none") != 0 && usematfree) {
    if (mpirank_world==0 && !quietmode) printf("# Warning: Matrix-free solver can not be used when Hamiltonian is read fromfile. Switching to sparse-matrix version.\n");
    usematfree = false;
  }
  // Initialize Master equation
  MasterEq* mastereq = new MasterEq(nlevels, nessential, oscil_vec, crosskerr, Jkl, eta, lindbladtype, usematfree, hamiltonian_file, quietmode);


  /* Output */
  Output* output = new Output(config, comm_petsc, comm_init, nlevels.size(), quietmode);

  // Some screen output 
  if (mpirank_world == 0 && !quietmode) {
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

    std::cout<<"State dimension (complex): " << mastereq->getDim() << std::endl;
    std::cout << "Time: [0:" << total_time << "], ";
    std::cout << "N="<< ntime << ", dt=" << dt << std::endl;
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
  int ninit_local = ninit / mpisize_init; 
  if (mastereq->lindbladtype != LindbladType::NONE &&   
     (runtype == RunType::GRADIENT || runtype == RunType::OPTIMIZATION) ) storeFWD = true;  // if NOT Schroedinger solver and running gradient optim: store forward states. Otherwise, they will be recomputed during gradient. 

  std::string timesteppertypestr = config.GetStrParam("timestepper", "IMR");
  int nwindows = mpisize_time; // default: use as many windows as time-processors. 
  int nwindows_read = config.GetIntParam("nwindows", -1, false); //If config option given, overwrite nwindows.
  if (nwindows_read > 0){
    nwindows = nwindows_read; 
  }
  int ntime_per_window = ntime / nwindows;
  if (ntime % nwindows != 0) {
    printf("Error: Need ntime MOD nwindows == 0.\n");
    exit(1);
  }
  if (mpirank_world == 0) {
    printf("Time-windows: %d, each %d steps\n", nwindows, ntime_per_window);
  }
  TimeStepper* mytimestepper;
  if (timesteppertypestr.compare("IMR")==0) mytimestepper = new ImplMidpoint(config, mastereq, ntime, ntime_per_window, total_time, dt, linsolvetype, linsolve_maxiter, output, storeFWD, comm_time);
  else if (timesteppertypestr.compare("IMR4")==0) mytimestepper = new CompositionalImplMidpoint(config, 4, mastereq, ntime, ntime_per_window, total_time, dt, linsolvetype, linsolve_maxiter, output, storeFWD, comm_time);
  else if (timesteppertypestr.compare("IMR8")==0) mytimestepper = new CompositionalImplMidpoint(config, 8, mastereq, ntime, ntime_per_window, total_time, dt, linsolvetype, linsolve_maxiter, output, storeFWD, comm_time);
  else if (timesteppertypestr.compare("EE")==0) mytimestepper = new ExplEuler(config, mastereq, ntime, ntime_per_window, total_time, dt, output, storeFWD, comm_time);
  else {
    printf("\n\n ERROR: Unknow timestepping type: %s.\n\n", timesteppertypestr.c_str());
    exit(1);
  }

  /* --- Initialize optimization --- */
  /* Get gate rotation frequencies. Default: use rotational frequencies for the gate. */
  OptimProblem* optimctx = new OptimProblem(config, mytimestepper, comm_init, comm_time, ninit, nwindows, output, quietmode);

  /* Set up vectors for the optimization variables and the gradient of the objective */
  Vec xinit = optimctx->xinit;
  Vec lambda = optimctx->lambda;
  Vec grad, v0;
  Vec opt; // for returning results for runtypes Simulation & Optimization
  VecDuplicate(xinit, &grad);
  VecZeroEntries(grad);

  bool load_optimvar = config.GetBoolParam("load_optimvar", false, false);
  double old_mu;  // discontinuity penalty strength at the previous optimization iteration.

  /* Set initial starting point */
  optimctx->getStartingPoint(xinit); 

  /* Some output */
  if (mpirank_world == 0)
  {
    /* Print parameters to file */
    snprintf(filename, 254, "%s/config_log.dat", output->datadir.c_str());
    std::ofstream logfile(filename);
    if (logfile.is_open()){
      logfile << log.str();
      logfile.close();
      if (!quietmode) printf("File written: %s\n", filename);
    }
    else std::cerr << "Unable to open " << filename;
  }

  /* Start timer */
  double StartTime = MPI_Wtime();
  double EndTime = StartTime;
  double objective;
  double gnorm = 0.0;
  std::vector<std::vector<double>> vnorm;
  
  /* --- Solve adjoint --- */
  bool verify_gradient = config.GetBoolParam("verify_grad", false);

  if (mpirank_world == 0)
    printf("\nRandomizing the optimization variables for verification...\n");

  // srand(time(0));
  srand(1234);
  PetscScalar *ptr;
  VecGetArray(xinit, &ptr);
  int local_size;
  VecGetLocalSize(xinit, &local_size);
  for (int k = 0; k < local_size; k++) {
    double randval = (double) rand() / ((double)RAND_MAX);  // random in [0,1]
    randval = -0.01 + 0.02 * randval;
    ptr[k] *= 1.0 + randval;
  }
  VecRestoreArray(xinit, &ptr);

  VecGetArray(grad, &ptr);
  for (int k = 0; k < local_size; k++) {
    if ((mpirank_world == 0) && (k < optimctx->getNdesign()))
      continue;

    double randval = (double) rand() / ((double)RAND_MAX);  // random in [0,1]
    ptr[k] = -1.0 + 2.0 * randval;
  }
  VecRestoreArray(grad, &ptr);
  VecDuplicate(grad, &v0);
  VecCopy(grad, v0);

  if (mpirank_world == 0 && !quietmode) printf("\nCheck unitarization...\n");

  Vec xinit0;
  VecDuplicate(xinit, &xinit0);
  VecCopy(xinit, xinit0);

  StartTime = MPI_Wtime(); 
  optimctx->unitarize(xinit, vnorm);
  EndTime = MPI_Wtime();

  optimctx->check_unitarity(xinit);

  double J0 = 0.0;
  VecDot(xinit, v0, &J0);
  if (mpirank_world == 0 && !quietmode)
    printf("J0: %.15E\n", J0);

  optimctx->unitarize_grad(xinit0, xinit, vnorm, grad);
  double gg;
  VecNorm(grad, NORM_2, &gnorm);
  gg = gnorm * gnorm;
  printf("gnorm: %.15E\n", gnorm);

  if (verify_gradient) {

    const int Nk = 40;
    Vec xperturb;
    VecDuplicate(xinit, &xperturb);
    double obj1[Nk], amp[Nk], dJdx[Nk], error[Nk];
    for (int k = 0; k < Nk; k++) {
      amp[k] = pow(10.0, -0.25 * k);
      VecCopy(xinit0, xperturb);
      VecAXPY(xperturb, amp[k] / gnorm, grad);

      optimctx->unitarize(xperturb, vnorm);

      double J1 = 0.0;
      VecDot(xperturb, v0, &J1);

      obj1[k] = J1;
      dJdx[k] = (J1 - J0) / amp[k];
      error[k] = abs(dJdx[k] - gnorm) / gnorm;
    }

    if (mpirank_world == 0) {
      // PetscScalar *ptr;
      // VecGetArray(xinit0, &ptr);
      // printf("x0: \n");
      // for (int k = 0; k < xdim; k++)
      //   printf("%.3E\t", ptr[k]);
      // printf("\n");
      // VecRestoreArray(x, &ptr);

      // VecGetArray(g, &ptr);
      // printf("g: \n");
      // for (int k = 0; k < xdim; k++)
      //   printf("%.3E\t", ptr[k]);
      // printf("\n");
      // VecRestoreArray(g, &ptr);
      
      printf("amp\tJ1\tdJdx\terror\n");
      for (int k = 0; k < Nk; k++)
        printf("%.4E\t%.4E\t%.4E\t%.4E\n", amp[k], obj1[k], dJdx[k], error[k]);
    }

    VecDestroy(&xperturb);
  }

  /* --- Finalize --- */

  /* Get timings */
  // #ifdef WITH_MPI
  double UsedTime = EndTime - StartTime;
  // #else
  // double UsedTime = 0.0; // TODO
  // #endif
  /* Get memory usage */
  struct rusage r_usage;
  getrusage(RUSAGE_SELF, &r_usage);
  double myMB = (double)r_usage.ru_maxrss / 1024.0;
  double globalMB = myMB;
  MPI_Allreduce(&myMB, &globalMB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  /* Print statistics */
  if (mpirank_world == 0 && !quietmode) {
    printf("\n");
    printf(" Used Time:        %.2f seconds\n", UsedTime);
    printf(" Processors used:  %d\n", mpisize_world);
    printf(" Global Memory:    %.2f MB    [~ %.2f MB per proc]\n", globalMB, globalMB / mpisize_world);
    printf(" [NOTE: The memory unit is platform dependent. If you run on MacOS, the unit will likely be KB instead of MB.]\n");
    printf("\n");
  }
  // printf("Rank %d: %.2fMB\n", mpirank_world, myMB );

  /* Print timing to file */
  if (mpirank_world == 0) {
    snprintf(filename, 254, "%s/timing.dat", output->datadir.c_str());
    FILE* timefile = fopen(filename, "w");
    fprintf(timefile, "%d  %1.8e\n", mpisize_world, UsedTime);
    fclose(timefile);
  }

  /* Clean up */
  for (int i=0; i<nlevels.size(); i++){
    delete oscil_vec[i];
  }
  delete [] oscil_vec;
  delete mastereq;
  delete mytimestepper;
  delete optimctx;
  delete output;
  VecDestroy(&xinit0);
  VecDestroy(&v0);
  
  /* Finallize Petsc */
#ifdef WITH_SLEPC
  ierr = SlepcFinalize();
#else
  PetscOptionsSetValue(NULL, "-options_left", "no"); // Remove warning about unused options.
  ierr = PetscFinalize();
#endif

  MPI_Finalize();
  return ierr;
}
