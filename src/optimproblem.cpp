#include "optimproblem.hpp"

#ifdef WITH_BRAID
OptimProblem::OptimProblem(MapParam config, TimeStepper* timestepper_, myBraidApp* primalbraidapp_, myAdjointBraidApp* adjointbraidapp_, MPI_Comm comm_init_, int ninit_, std::vector<double> gate_rot_freq, Output* output_) : OptimProblem(config, timestepper_, comm_init_, ninit_, gate_rot_freq, output_) {
  primalbraidapp  = primalbraidapp_;
  adjointbraidapp = adjointbraidapp_;
  MPI_Comm_rank(primalbraidapp->comm_braid, &mpirank_braid);
  MPI_Comm_size(primalbraidapp->comm_braid, &mpisize_braid);
}
#endif

OptimProblem::OptimProblem(MapParam config, TimeStepper* timestepper_, MPI_Comm comm_init_, int ninit_, std::vector<double> gate_rot_freq, Output* output_){

  timestepper = timestepper_;
  ninit = ninit_;
  comm_init = comm_init_;
  output = output_;
  /* Reset */
  objective = 0.0;

  /* Store ranks and sizes of communicators */
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize_world);
  MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank_space);
  MPI_Comm_size(PETSC_COMM_WORLD, &mpisize_space);
  MPI_Comm_rank(comm_init, &mpirank_init);
  MPI_Comm_size(comm_init, &mpisize_init);
  mpirank_braid = 0;
  mpisize_braid = 1;

  /* Store number of initial conditions per init-processor group */
  ninit_local = ninit / mpisize_init; 

  /* Store number of design parameters */
  int n = 0;
  for (int ioscil = 0; ioscil < timestepper->mastereq->getNOscillators(); ioscil++) {
      n += timestepper->mastereq->getOscillator(ioscil)->getNParams(); 
  }
  ndesign = n;
  if (mpirank_world == 0) std::cout<< "ndesign = " << ndesign << std::endl;

  /* Store other optimization parameters */
  gamma_tik = config.GetDoubleParam("optim_regul", 1e-4);
  gatol = config.GetDoubleParam("optim_atol", 1e-8);
  grtol = config.GetDoubleParam("optim_rtol", 1e-4);
  maxiter = config.GetIntParam("optim_maxiter", 200);
  initguess_type = config.GetStrParam("optim_init", "zero");
  config.GetVecDoubleParam("optim_init_ampl", initguess_amplitudes, 0.0);
  // sanity check
  if (initguess_type.compare("constant") == 0 || 
      initguess_type.compare("random")    == 0 ||
      initguess_type.compare("random_seed") == 0)  {
      if (initguess_amplitudes.size() < timestepper->mastereq->getNOscillators()) {
         printf("ERROR reading config file: List of initial optimization parameter amplitudes must equal the number of oscillators!\n");
         exit(1);
      }
  }

  /* Store the optimization target */
  std::vector<std::string> target_str;
  Gate* targetgate=NULL;
  int purestateID = -1;
  std::string target_filename = "";
  TargetType target_type;
  // Read from config file 
  config.GetVecStrParam("optim_target", target_str, "pure");
  if ( target_str[0].compare("gate") ==0 ) {
    target_type = TargetType::GATE;
    /* Initialize the targetgate */
    if ( target_str.size() < 2 ) {
      printf("ERROR: You want to optimize for a gate, but didn't specify which one. Check your config for 'optim_target'!\n");
      exit(1);
    }
    if      (target_str[1].compare("none") == 0)  targetgate = new Gate(); // dummy gate. do nothing
    else if (target_str[1].compare("xgate") == 0) targetgate = new XGate(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, timestepper->total_time, gate_rot_freq); 
    else if (target_str[1].compare("ygate") == 0) targetgate = new YGate(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, timestepper->total_time, gate_rot_freq); 
    else if (target_str[1].compare("zgate") == 0) targetgate = new ZGate(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, timestepper->total_time, gate_rot_freq);
    else if (target_str[1].compare("hadamard") == 0) targetgate = new HadamardGate(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, timestepper->total_time, gate_rot_freq);
    else if (target_str[1].compare("cnot") == 0) targetgate = new CNOT(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, timestepper->total_time, gate_rot_freq); 
    else if (target_str[1].compare("swap") == 0) targetgate = new SWAP(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, timestepper->total_time, gate_rot_freq); 
    else if (target_str[1].compare("swap0q") == 0) targetgate = new SWAP_0Q(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, timestepper->total_time, gate_rot_freq); 
    else if (target_str[1].compare("cqnot") == 0) targetgate = new CQNOT(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, timestepper->total_time, gate_rot_freq); 
    else {
      printf("\n\n ERROR: Unnown gate type: %s.\n", target_str[1].c_str());
      printf(" Available gates are 'none', 'xgate', 'ygate', 'zgate', 'hadamard', 'cnot', 'swap', 'swap0q', 'cqnot'.\n");
      exit(1);
    } 
  }  
  else if (target_str[0].compare("pure")==0) {
    target_type = TargetType::PURE;
    purestateID = 0;
    if (target_str.size() < 2) {
      printf("# Warning: You want to prepare a pure state, but didn't specify which one. Taking default: ground-state |0...0> \n");
    } else {
      /* Compute the index m for preparing e_m e_m^\dagger. Note that the input is given for pure states PER OSCILLATOR such as |m_1 m_2 ... m_Q> and hence m = m_1 * dimPost(oscil 1) + m_2 * dimPost(oscil 2) + ... + m_Q */
      if (target_str.size() - 1 < timestepper->mastereq->getNOscillators()) {
        printf("ERROR: List of ID's for pure-state preparation must contain %d elements! Check config option 'optim_target'.\n", timestepper->mastereq->getNOscillators());
        exit(1);
      }
      for (int i=0; i < timestepper->mastereq->getNOscillators(); i++) {
        int Qi_state = atoi(target_str[i+1].c_str());
        purestateID += Qi_state * timestepper->mastereq->getOscillator(i)->dim_postOsc;
      }
    }
    // printf("Preparing the state e_%d\n", purestateID);
  } 
  else if (target_str[0].compare("file")==0) { 
    // Get the name of the file and pass it to the OptimTarget class later.
    target_type = TargetType::FROMFILE;
    assert(target_str.size() >= 2);
    target_filename = target_str[1];
  }
  else {
      printf("\n\n ERROR: Unknown optimization target: %s\n", target_str[0].c_str());
      exit(1);
  }

  /* Get the objective function */
  ObjectiveType objective_type;
  std::string objective_str = config.GetStrParam("optim_objective", "Jfrobenius");
  if (objective_str.compare("Jfrobenius")==0)           objective_type = ObjectiveType::JFROBENIUS;
  else if (objective_str.compare("Jhilbertschmidt")==0) objective_type = ObjectiveType::JHS;
  else if (objective_str.compare("Jmeasure")==0)        objective_type = ObjectiveType::JMEASURE;
  else  {
    printf("\n\n ERROR: Unknown objective function: %s\n", objective_str.c_str());
    exit(1);
  }

  /* Finally initialize the optimization target struct */
  optim_target = new OptimTarget(timestepper->mastereq->getDim(), purestateID, target_type, objective_type, targetgate, target_filename);

  /* Get weights for the objective function (weighting the different initial conditions */
  config.GetVecDoubleParam("optim_weights", obj_weights, 1.0);
  int nfill = 0;
  if (obj_weights.size() < ninit) nfill = ninit - obj_weights.size();
  double val = obj_weights[obj_weights.size()-1];
  if (obj_weights.size() < ninit){
    for (int i = 0; i < nfill; i++) 
      obj_weights.push_back(val);
  }
  assert(obj_weights.size() >= ninit);
  double sendbuf[obj_weights.size()];
  double recvbuf[obj_weights.size()];
  for (int i = 0; i < obj_weights.size(); i++) sendbuf[i] = obj_weights[i];
  // Distribute over mpi_init processes 
  int nscatter = ninit_local;
  MPI_Scatter(sendbuf, nscatter, MPI_DOUBLE, recvbuf, nscatter,  MPI_DOUBLE, 0, comm_init);
  for (int i = 0; i < nscatter; i++) obj_weights[i] = recvbuf[i];
  for (int i=nscatter; i < obj_weights.size(); i++) obj_weights[i] = 0.0;


  /* Pass information on objective function to the time stepper needed for penalty objective function */
  gamma_penalty = config.GetDoubleParam("optim_penalty", 1e-4);
  penalty_param = config.GetDoubleParam("optim_penalty_param", 0.5);
  timestepper->penalty_param = penalty_param;
  timestepper->gamma_penalty = gamma_penalty;
  timestepper->optim_target = optim_target;

  /* Get initial condition type and involved oscillators */
  std::vector<std::string> initcondstr;
  config.GetVecStrParam("initialcondition", initcondstr, "none", false);
  for (int i=1; i<initcondstr.size(); i++) initcond_IDs.push_back(atoi(initcondstr[i].c_str()));
  if (initcondstr[0].compare("file") == 0 )          initcond_type = InitialConditionType::FROMFILE;
  else if (initcondstr[0].compare("pure") == 0 )     initcond_type = InitialConditionType::PURE;
  else if (initcondstr[0].compare("ensemble") == 0 ) initcond_type = InitialConditionType::ENSEMBLE;
  else if (initcondstr[0].compare("3states") == 0 )  initcond_type = InitialConditionType::THREESTATES;
  else if (initcondstr[0].compare("Nplus1") == 0 )   initcond_type = InitialConditionType::NPLUSONE;
  else if (initcondstr[0].compare("diagonal") == 0 ) initcond_type = InitialConditionType::DIAGONAL;
  else if (initcondstr[0].compare("basis")    == 0 ) initcond_type = InitialConditionType::BASIS;
  else {
    printf("\n\n ERROR: Wrong setting for initial condition.\n");
    exit(1);
  }

  /* Allocate the initial condition vector */
  VecCreate(PETSC_COMM_WORLD, &rho_t0); 
  VecSetSizes(rho_t0,PETSC_DECIDE,2*timestepper->mastereq->getDim());
  VecSetFromOptions(rho_t0);
  PetscInt ilow, iupp;
  VecGetOwnershipRange(rho_t0, &ilow, &iupp);

  /* If PURE or FROMFILE or ENSEMBLE initialization, store them here. Otherwise they are set inside evalF */
  if (initcond_type == InitialConditionType::PURE) { 
    /* Initialize with tensor product of unit vectors. */

    // Compute index of diagonal elements that is one.
    if (initcond_IDs.size() != timestepper->mastereq->getNOscillators()) {
      printf("ERROR during pure-state initialization: List of IDs must contain %d elements!\n", timestepper->mastereq->getNOscillators());
      exit(1);
    }
    int diag_id = 0;
    for (int k=0; k < initcond_IDs.size(); k++) {
      if (initcond_IDs[k] > timestepper->mastereq->getOscillator(k)->getNLevels()-1){
        printf("ERROR in config setting. The requested pure state initialization |%d> exceeds the number of allowed levels for that oscillator (%d).\n", initcond_IDs[k], timestepper->mastereq->getOscillator(k)->getNLevels());
        exit(1);
      }
      assert (initcond_IDs[k] < timestepper->mastereq->getOscillator(k)->getNLevels());
      int dim_postkron = 1;
      for (int m=k+1; m < initcond_IDs.size(); m++) {
        dim_postkron *= timestepper->mastereq->getOscillator(m)->getNLevels();
      }
      diag_id += initcond_IDs[k] * dim_postkron;
    }
    int ndim = (int)sqrt(timestepper->mastereq->getDim());
    int vec_id = getIndexReal(getVecID( diag_id, diag_id, ndim )); // Real part of x
    if (ilow <= vec_id && vec_id < iupp) VecSetValue(rho_t0, vec_id, 1.0, INSERT_VALUES);
  }
  else if (initcond_type == InitialConditionType::FROMFILE) { 
    /* Read initial condition from file */
    
    // int dim = timestepper->mastereq->getDim();
    int dim_ess = timestepper->mastereq->getDimEss();
    int dim_rho = timestepper->mastereq->getDimRho();
    double * vec = new double[2*dim_ess*dim_ess];
    if (mpirank_world == 0) {
      assert (initcondstr.size()==2);
      std::string filename = initcondstr[1];
      read_vector(filename.c_str(), vec, 2*dim_ess*dim_ess);
    }
    MPI_Bcast(vec, 2*dim_ess*dim_ess, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < dim_ess*dim_ess; i++) {
      int k = i % dim_ess;
      int j = (int) i / dim_ess;
      if (dim_ess*dim_ess < timestepper->mastereq->getDim()) {
        k = mapEssToFull(k, timestepper->mastereq->nlevels, timestepper->mastereq->nessential);
        j = mapEssToFull(j, timestepper->mastereq->nlevels, timestepper->mastereq->nessential);
      }
      int elemid_re = getIndexReal(getVecID(k,j,dim_rho));
      int elemid_im = getIndexImag(getVecID(k,j,dim_rho));
      if (ilow <= elemid_re && elemid_re < iupp) VecSetValue(rho_t0, elemid_re, vec[i], INSERT_VALUES);        // RealPart
      if (ilow <= elemid_im && elemid_im < iupp) VecSetValue(rho_t0, elemid_im, vec[i + dim_ess*dim_ess], INSERT_VALUES); // Imaginary Part
      // printf("  -> k=%d j=%d, elemid=%d vals=%1.4e, %1.4e\n", k, j, elemid, vec[i], vec[i+dim_ess*dim_ess]);
    }
    delete [] vec;
  } else if (initcond_type == InitialConditionType::ENSEMBLE) {
    // Sanity check for the list in initcond_IDs!
    assert(initcond_IDs.size() >= 1); // at least one element 
    assert(initcond_IDs[initcond_IDs.size()-1] < timestepper->mastereq->getNOscillators()); // last element can't exceed total number of oscillators
    for (int i=0; i < initcond_IDs.size()-1; i++){ // list should be consecutive!
      if (initcond_IDs[i]+1 != initcond_IDs[i+1]) {
        printf("ERROR: List of oscillators for ensemble initialization should be consecutive!\n");
        exit(1);
      }
    }

    // get dimension of subsystems *before* the first oscillator, and *behind* the last oscillator
    int dimpre  = timestepper->mastereq->getOscillator(initcond_IDs[0])->dim_preOsc;
    int dimpost = timestepper->mastereq->getOscillator(initcond_IDs[initcond_IDs.size()-1])->dim_postOsc;

    // get dimension of subsystems defined by initcond_IDs
    int dimsub = 1;
    for (int i=0; i<initcond_IDs.size(); i++){
      dimsub *= timestepper->mastereq->getOscillator(initcond_IDs[i])->getNLevels();  
    }
    // printf("dimpre %d sub %d post %d \n", dimpre, dimsub, dimpost);
    int dimrho = timestepper->mastereq->getDimRho();
    assert(dimsub == (int) ( dimrho / dimpre / dimpost) );

    for (int i=0; i < dimsub; i++){
      for (int j=i; j < dimsub; j++){
        int ifull = i * dimpost;
        int jfull = j * dimpost;
        // printf(" i=%d j=%d ifull %d, jfull %d\n", i, j, ifull, jfull);
        if (i == j) { // diagonal element: 1/N_sub
          int elemid_re = getIndexReal(getVecID(ifull, jfull, dimrho));
          if (ilow <= elemid_re && elemid_re < iupp) VecSetValue(rho_t0, elemid_re, 1./dimsub, INSERT_VALUES);
        } else {
          // upper diagonal (0.5 + 0.5*i) / (N_sub^2)
          int elemid_re = getIndexReal(getVecID(ifull, jfull, dimrho));
          int elemid_im = getIndexImag(getVecID(ifull, jfull, dimrho));
          if (ilow <= elemid_re && elemid_re < iupp) VecSetValue(rho_t0, elemid_re, 0.5/(dimsub*dimsub), INSERT_VALUES);
          if (ilow <= elemid_im && elemid_im < iupp) VecSetValue(rho_t0, elemid_im, 0.5/(dimsub*dimsub), INSERT_VALUES);
          // lower diagonal (0.5 - 0.5*i) / (N_sub^2)
          elemid_re = getIndexReal(getVecID(jfull, ifull, dimrho));
          elemid_im = getIndexImag(getVecID(jfull, ifull, dimrho));
          if (ilow <= elemid_re && elemid_re < iupp) VecSetValue(rho_t0, elemid_re,  0.5/(dimsub*dimsub), INSERT_VALUES);
          if (ilow <= elemid_im && elemid_im < iupp) VecSetValue(rho_t0, elemid_im, -0.5/(dimsub*dimsub), INSERT_VALUES);
        } 
      }
    }
  }
  VecAssemblyBegin(rho_t0); VecAssemblyEnd(rho_t0);

  /* Initialize adjoint */
  VecDuplicate(rho_t0, &rho_t0_bar);
  VecZeroEntries(rho_t0_bar);
  VecAssemblyBegin(rho_t0_bar); VecAssemblyEnd(rho_t0_bar);

  /* Store optimization bounds */
  VecCreateSeq(PETSC_COMM_SELF, ndesign, &xlower);
  VecSetFromOptions(xlower);
  VecDuplicate(xlower, &xupper);
  std::vector<double> bounds;
  config.GetVecDoubleParam("optim_bounds", bounds, 1e20);
  for (int i = bounds.size(); i < timestepper->mastereq->getNOscillators(); i++) bounds.push_back(1e+12); // fill up with zeros
  int col = 0;
  for (int iosc = 0; iosc < timestepper->mastereq->getNOscillators(); iosc++){
    // Scale bounds by 1/sqrt(2) * (number of carrier waves) */
    std::vector<double> carrier_freq;
    std::string key = "carrier_frequency" + std::to_string(iosc);
    config.GetVecDoubleParam(key, carrier_freq, 0.0, false);
    bounds[iosc] = bounds[iosc] / ( sqrt(2) * carrier_freq.size()) ;
    // set bounds for all parameters in this oscillator
    for (int i=0; i<timestepper->mastereq->getOscillator(iosc)->getNParams(); i++){
      double bound = bounds[iosc];

      /* for the first and last two splines, overwrite the bound with zero to ensure that control at t=0 and t=T is zero. */
      int ibegin = 2*2*carrier_freq.size();
      int iend = (timestepper->mastereq->getOscillator(iosc)->getNSplines()-2)*2*carrier_freq.size();
      if (i < ibegin || i >= iend) bound = 0.0;

      // set the bound
      VecSetValue(xupper, col, bound, INSERT_VALUES);
      VecSetValue(xlower, col, -1. * bound, INSERT_VALUES);
      col++;
    }
  }
  VecAssemblyBegin(xlower); VecAssemblyEnd(xlower);
  VecAssemblyBegin(xupper); VecAssemblyEnd(xupper);
 
  /* Create Petsc's optimization solver */
  TaoCreate(PETSC_COMM_WORLD, &tao);
  /* Set optimization type and parameters */
  TaoSetType(tao,TAOBQNLS);         // Optim type: taoblmvm vs BQNLS ??
  TaoSetMaximumIterations(tao, maxiter);
  TaoSetTolerances(tao, gatol, PETSC_DEFAULT, grtol);
  TaoSetMonitor(tao, TaoMonitor, (void*)this, NULL);
  TaoSetVariableBounds(tao, xlower, xupper);
  TaoSetFromOptions(tao);
  /* Set user-defined objective and gradient evaluation routines */
  TaoSetObjectiveRoutine(tao, TaoEvalObjective, (void *)this);
  TaoSetGradientRoutine(tao, TaoEvalGradient,(void *)this);
  TaoSetObjectiveAndGradientRoutine(tao, TaoEvalObjectiveAndGradient, (void*) this);

  /* Allocate auxiliary vector */
  mygrad = new double[ndesign];
}


OptimProblem::~OptimProblem() {
  delete [] mygrad;
  delete optim_target;
  VecDestroy(&rho_t0);
  VecDestroy(&rho_t0_bar);

  VecDestroy(&xlower);
  VecDestroy(&xupper);

  TaoDestroy(&tao);
}



double OptimProblem::evalF(const Vec x) {

  // OptimProblem* ctx = (OptimProblem*) ptr;
  MasterEq* mastereq = timestepper->mastereq;

  if (mpirank_world == 0) printf("EVAL F... \n");
  Vec finalstate = NULL;

  /* Pass design vector x to oscillators */
  mastereq->setControlAmplitudes(x); 

  /*  Iterate over initial condition */
  obj_cost  = 0.0;
  obj_regul = 0.0;
  obj_penal = 0.0;
  fidelity = 0.0;
  double obj_cost_max = 0.0;
  for (int iinit = 0; iinit < ninit_local; iinit++) {
      
    /* Prepare the initial condition in [rank * ninit_local, ... , (rank+1) * ninit_local - 1] */
    int iinit_global = mpirank_init * ninit_local + iinit;
    int initid = timestepper->mastereq->getRhoT0(iinit_global, ninit, initcond_type, initcond_IDs, rho_t0);
    if (mpirank_braid == 0) printf("%d: Initial condition id=%d ...\n", mpirank_init, initid);

    /* If gate optimiztion, compute the target state rho^target = Vrho(0)V^dagger */
    optim_target->prepare(rho_t0);

    /* Run forward with initial condition initid */
#ifdef WITH_BRAID
      primalbraidapp->PreProcess(initid, rho_t0, 0.0);
      primalbraidapp->Drive();
      finalstate = primalbraidapp->PostProcess(); // this return NULL for all but the last time processor
#else
      finalstate = timestepper->solveODE(initid, rho_t0);
#endif

    /* Add to integral penalty term */
    obj_penal += gamma_penalty * timestepper->penalty_integral;

    /* Evaluate J(finalstate) and add to final-time cost */
    double obj_iinit = optim_target->evalJ(finalstate);
    obj_cost +=  obj_weights[iinit] * obj_iinit;
    obj_cost_max = std::max(obj_cost_max, obj_iinit);

    /* Add to final-time fidelity */
    double fidelity_iinit = optim_target->evalFidelity(finalstate);
    fidelity += fidelity_iinit;

    // printf("%d, %d: iinit objective: %f * %1.14e, Fid=%1.14e\n", mpirank_world, mpirank_init, obj_weights[iinit], obj_iinit, fidelity_iinit);
  }

#ifdef WITH_BRAID
  /* Communicate over braid processors: Sum up penalty, broadcast final time cost */
  double mine = obj_penal;
  MPI_Allreduce(&mine, &obj_penal, 1, MPI_DOUBLE, MPI_SUM, primalbraidapp->comm_braid);
  MPI_Bcast(&obj_cost, 1, MPI_DOUBLE, mpisize_braid-1, primalbraidapp->comm_braid);
#endif

  /* Average over initial conditions processors */
  double mypen = 1./ninit * obj_penal;
  double mycost = 1./ninit * obj_cost;
  double myfidelity = 1./ninit * fidelity;
  MPI_Allreduce(&mypen, &obj_penal, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mycost, &obj_cost, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&myfidelity, &fidelity, 1, MPI_DOUBLE, MPI_SUM, comm_init);

  /* Evaluate regularization objective += gamma/2 * ||x||^2*/
  double xnorm;
  VecNorm(x, NORM_2, &xnorm);
  obj_regul = gamma_tik / 2. * pow(xnorm,2.0);

  /* Sum, store and return objective value */
  objective = obj_cost + obj_regul + obj_penal;

  /* Output */
  if (mpirank_world == 0) {
    std::cout<< "Objective = " << std::scientific<<std::setprecision(14) << obj_cost << " + " << obj_regul << " + " << obj_penal << std::endl;
    std::cout<< "Fidelity = " << fidelity  << std::endl;
    // std::cout<< "Max. costT = " << obj_cost_max << std::endl;
  }

  return objective;
}



void OptimProblem::evalGradF(const Vec x, Vec G){

  MasterEq* mastereq = timestepper->mastereq;

  if (mpirank_world == 0) std::cout<< "EVAL GRAD F... " << std::endl;
  Vec finalstate = NULL;

  /* Pass design vector x to oscillators */
  mastereq->setControlAmplitudes(x); 

  /* Reset Gradient */
  VecZeroEntries(G);

  /* Derivative of regulatization term gamma / 2 ||x||^2 (ADD ON ONE PROC ONLY!) */
  if (mpirank_init == 0 && mpirank_braid == 0) {
    VecAXPY(G, gamma_tik, x);
  }

  /*  Iterate over initial condition */
  obj_cost = 0.0;
  obj_regul = 0.0;
  obj_penal = 0.0;
  fidelity = 0.0;
  for (int iinit = 0; iinit < ninit_local; iinit++) {

    /* Prepare the initial condition */
    int iinit_global = mpirank_init * ninit_local + iinit;
    int initid = timestepper->mastereq->getRhoT0(iinit_global, ninit, initcond_type, initcond_IDs, rho_t0);

    /* If gate optimiztion, compute the target state rho^target = Vrho(0)V^dagger */
    optim_target->prepare(rho_t0);

    /* --- Solve primal --- */
    // if (mpirank_braid == 0) printf("%d: %d FWD. ", mpirank_init, initid);

    /* Run forward with initial condition rho_t0 */
#ifdef WITH_BRAID 
      primalbraidapp->PreProcess(initid, rho_t0, 0.0);
      primalbraidapp->Drive();
      finalstate = primalbraidapp->PostProcess(); // this return NULL for all but the last time processor
#else 
      finalstate = timestepper->solveODE(initid, rho_t0);
#endif

    /* Add to integral penalty term */
    obj_penal += gamma_penalty * timestepper->penalty_integral;

    /* Evaluate J(finalstate) and add to final-time cost */
    double obj_iinit = optim_target->evalJ(finalstate);
    obj_cost += obj_weights[iinit] * obj_iinit;
    // if (mpirank_braid == 0) printf("%d: iinit objective: %1.14e\n", mpirank_init, obj_iinit);

    /* Add to final-time fidelity */
    fidelity += optim_target->evalFidelity(finalstate);

    /* --- Solve adjoint --- */
    // if (mpirank_braid == 0) printf("%d: %d BWD.", mpirank_init, initid);

    /* Reset adjoint */
    VecZeroEntries(rho_t0_bar);

    /* Derivative of final time objective J */
    optim_target->evalJ_diff(finalstate, rho_t0_bar, 1.0 / ninit * obj_weights[iinit]);

    /* Derivative of time-stepping */
#ifdef WITH_BRAID
      adjointbraidapp->PreProcess(initid, rho_t0_bar, 1.0 / ninit * gamma_penalty);
      adjointbraidapp->Drive();
      adjointbraidapp->PostProcess();
#else
      timestepper->solveAdjointODE(initid, rho_t0_bar, 1.0 / ninit * gamma_penalty);
#endif

    /* Add to optimizers's gradient */
    VecAXPY(G, 1.0, timestepper->redgrad);
  }

#ifdef WITH_BRAID
  /* Communicate over braid processors: Sum up penalty, broadcast final time cost */
  double mine = obj_penal;
  MPI_Allreduce(&mine, &obj_penal, 1, MPI_DOUBLE, MPI_SUM, primalbraidapp->comm_braid);
  MPI_Bcast(&obj_cost, 1, MPI_DOUBLE, mpisize_braid-1, primalbraidapp->comm_braid);
  #endif

  /* Average over initial conditions processors */
  double mypen = 1./ninit * obj_penal;
  double mycost = 1./ninit * obj_cost;
  double myfidelity = 1./ninit * fidelity;
  MPI_Allreduce(&mypen, &obj_penal, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mycost, &obj_cost, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&myfidelity, &fidelity, 1, MPI_DOUBLE, MPI_SUM, comm_init);

  /* Evaluate regularization gamma/2 * ||x||^2*/
  double xnorm;
  VecNorm(x, NORM_2, &xnorm);
  obj_regul = gamma_tik / 2. * pow(xnorm,2.0);

  /* Sum, store and return objective function value*/
  objective = obj_cost + obj_regul + obj_penal;

  /* Sum up the gradient from all initial condition processors */
  PetscScalar* grad; 
  VecGetArray(G, &grad);
  for (int i=0; i<ndesign; i++) {
    mygrad[i] = grad[i];
  }
  MPI_Allreduce(mygrad, grad, ndesign, MPI_DOUBLE, MPI_SUM, comm_init);
  VecRestoreArray(G, &grad);

#ifdef WITH_BRAID
  /* Sum up the gradient from all braid processors */
  VecGetArray(G, &grad);
  for (int i=0; i<ndesign; i++) {
    mygrad[i] = grad[i];
  }
  MPI_Allreduce(mygrad, grad, ndesign, MPI_DOUBLE, MPI_SUM, primalbraidapp->comm_braid);
  VecRestoreArray(G, &grad);
#endif

  /* Compute and store gradient norm */
  VecNorm(G, NORM_2, &(gnorm));

  /* Output */
  if (mpirank_world == 0) {
    std::cout<< "Objective = " << std::scientific<<std::setprecision(14) << obj_cost << " + " << obj_regul << " + " << obj_penal << std::endl;
    std::cout<< "Fidelity = " << fidelity << std::endl;
  }
}


void OptimProblem::solve(Vec xinit) {
  TaoSetInitialVector(tao, xinit);
  TaoSolve(tao);
}

void OptimProblem::getStartingPoint(Vec xinit){
  MasterEq* mastereq = timestepper->mastereq;

  if (initguess_type.compare("constant") == 0 ){ // set constant initial design
    // set values
    int j = 0;
    for (int ioscil = 0; ioscil < mastereq->getNOscillators(); ioscil++) {
      int nparam = mastereq->getOscillator(ioscil)->getNParams();
      for (int i = 0; i < nparam; i++) {
        VecSetValue(xinit, j, initguess_amplitudes[ioscil], INSERT_VALUES);
        j++;
      }
    }
  } else if ( initguess_type.compare("random")      == 0 ||       // init random, fixed seed
              initguess_type.compare("random_seed") == 0)  { // init random with new seed
    /* Create vector with random elements between [-1:1] */
    if ( initguess_type.compare("random") == 0) srand(1);  // fixed seed
    else srand(time(0)); // random seed
    double* randvec = new double[ndesign];
    for (int i=0; i<ndesign; i++) {
      randvec[i] = (double) rand() / ((double)RAND_MAX);
      randvec[i] = 2.*randvec[i] - 1.;
    }
    /* Broadcast random vector from rank 0 to all, so that all have the same starting point (necessary?) */
    MPI_Bcast(randvec, ndesign, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Scale vector by the initial amplitudes */
    int shift = 0;
    for (int ioscil = 0; ioscil < mastereq->getNOscillators(); ioscil++) {
      int nparam_iosc = mastereq->getOscillator(ioscil)->getNParams();
      for (int i=0; i<nparam_iosc; i++) {
        randvec[shift + i] *= initguess_amplitudes[ioscil];
      }
      shift+= nparam_iosc;
    }

    /* Set the initial guess */
    for (int i=0; i<ndesign; i++) {
      VecSetValue(xinit, i, randvec[i], INSERT_VALUES);
    }
    delete [] randvec;

  }  else { // Read from file 
    double* vecread = new double[ndesign];

    if (mpirank_world == 0) read_vector(initguess_type.c_str(), vecread, ndesign);
    MPI_Bcast(vecread, ndesign, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Set the initial guess */
    for (int i=0; i<ndesign; i++) {
      VecSetValue(xinit, i, vecread[i], INSERT_VALUES);
    }
    delete [] vecread;
  }

  /* for the first and last two splines, overwrite the parameters with zero to ensure that control at t=0 and t=T is zero. */
  PetscInt col = 0.0;
  for (int iosc = 0; iosc < timestepper->mastereq->getNOscillators(); iosc++){
    int ncarrier = timestepper->mastereq->getOscillator(iosc)->getNCarrierwaves();
    PetscInt ibegin = 2*2*ncarrier;
    PetscInt iend = (timestepper->mastereq->getOscillator(iosc)->getNSplines()-2)*2*ncarrier;

    for (int i = 0; i < timestepper->mastereq->getOscillator(iosc)->getNParams(); i++) {
      if (i < ibegin || i >= iend) VecSetValue(xinit, col, 0.0, INSERT_VALUES);
      col++;
    }
  }


  /* Assemble initial guess */
  VecAssemblyBegin(xinit);
  VecAssemblyEnd(xinit);

  /* Pass to oscillator */
  timestepper->mastereq->setControlAmplitudes(xinit);
  
  /* Write initial control functions to file */
  output->writeControls(xinit, timestepper->mastereq, timestepper->ntime, timestepper->dt);

}


void OptimProblem::getSolution(Vec* param_ptr){
  
  /* Get ref to optimized parameters */
  Vec params;
  TaoGetSolutionVector(tao, &params);
  *param_ptr = params;
}

PetscErrorCode TaoMonitor(Tao tao,void*ptr){
  OptimProblem* ctx = (OptimProblem*) ptr;

  /* Get information from Tao optimization */
  PetscInt iter;
  PetscScalar deltax;
  Vec params;
  TaoConvergedReason reason;
  PetscScalar f, gnorm;
  TaoGetSolutionStatus(tao, &iter, &f, &gnorm, NULL, &deltax, &reason);
  TaoGetSolutionVector(tao, &params);

  /* Pass current iteration number to output manager */
  ctx->output->optim_iter = iter;

  /* Grab some output stuff */
  double obj_cost = ctx->getCostT();
  double obj_regul = ctx->getRegul();
  double obj_penal = ctx->getPenalty();
  double F_avg = ctx->getFidelity();

  /* Print to optimization file */
  ctx->output->writeOptimFile(f, gnorm, deltax, F_avg, obj_cost, obj_regul, obj_penal);

  /* Print parameters and controls to file */
  ctx->output->writeControls(params, ctx->timestepper->mastereq, ctx->timestepper->ntime, ctx->timestepper->dt);

  return 0;
}


PetscErrorCode TaoEvalObjectiveAndGradient(Tao tao, Vec x, PetscReal *f, Vec G, void*ptr){

  TaoEvalGradient(tao, x, G, ptr);
  OptimProblem* ctx = (OptimProblem*) ptr;
  *f = ctx->getObjective();

  return 0;
}

PetscErrorCode TaoEvalObjective(Tao tao, Vec x, PetscReal *f, void*ptr){

  OptimProblem* ctx = (OptimProblem*) ptr;
  *f = ctx->evalF(x);
  
  return 0;
}


PetscErrorCode TaoEvalGradient(Tao tao, Vec x, Vec G, void*ptr){

  OptimProblem* ctx = (OptimProblem*) ptr;
  ctx->evalGradF(x, G);
  
  return 0;
}
