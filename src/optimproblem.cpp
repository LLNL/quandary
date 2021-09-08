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
  TargetType target_type;
  // Read from config file 
  config.GetVecStrParam("optim_target", target_str, "pure");
  if ( target_str[0].compare("gate") ==0 ) {
    target_type = GATE;
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
    target_type = PUREM;
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
  else {
      printf("\n\n ERROR: Unknown optimization target: %s\n", target_str[0].c_str());
      exit(1);
  }

  /* Get the objective function */
  ObjectiveType objective_type;
  std::string objective_str = config.GetStrParam("optim_objective", "Jfrobenius");
  if (objective_str.compare("Jfrobenius")==0)           objective_type = JFROBENIUS;
  else if (objective_str.compare("Jhilbertschmidt")==0) objective_type = JHS;
  else if (objective_str.compare("Jmeasure")==0)        objective_type = JMEASURE;
  else  {
    printf("\n\n ERROR: Unknown objective function: %s\n", objective_str.c_str());
    exit(1);
  }

  /* Finally initialize the optimization target struct */
  optim_target = new OptimTarget(purestateID, target_type, objective_type, targetgate);

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
  if (initcondstr[0].compare("file") == 0 )          initcond_type = FROMFILE;
  else if (initcondstr[0].compare("pure") == 0 )     initcond_type = PURE;
  else if (initcondstr[0].compare("3states") == 0 )  initcond_type = THREESTATES;
  else if (initcondstr[0].compare("Nplus1") == 0 )   initcond_type = NPLUSONE;
  else if (initcondstr[0].compare("diagonal") == 0 ) initcond_type = DIAGONAL;
  else if (initcondstr[0].compare("basis")    == 0 ) initcond_type = BASIS;
  else {
    printf("\n\n ERROR: Wrong setting for initial condition.\n");
    exit(1);
  }

  /* Allocate the initial condition vector */
  VecCreate(PETSC_COMM_WORLD, &rho_t0); 
  VecSetSizes(rho_t0,PETSC_DECIDE,2*timestepper->mastereq->getDim());
  VecSetFromOptions(rho_t0);
  int ilow, iupp;
  VecGetOwnershipRange(rho_t0, &ilow, &iupp);

  /* If PURE or FROMFILE initialization, store them here. Otherwise they are set inside evalF */
  if (initcond_type == PURE) { 
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
  else if (initcond_type == FROMFILE) { 
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
  assert (bounds.size() >= timestepper->mastereq->getNOscillators());
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

    /* Compute and add final-time cost */
    double obj_iinit = objectiveT(optim_target, timestepper->mastereq, finalstate, rho_t0);
    obj_cost +=  obj_weights[iinit] * obj_iinit;
    obj_cost_max = std::max(obj_cost_max, obj_iinit);
    // printf("%d, %d: iinit objective: %f * %1.14e\n", mpirank_world, mpirank_init, obj_weights[iinit], obj_iinit);

    /* Add to final-time fidelity */
    fidelity += getFidelity(finalstate);
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

    /* Compute and add final-time cost */
    double obj_iinit = objectiveT(optim_target, timestepper->mastereq, finalstate, rho_t0);
    obj_cost += obj_weights[iinit] * obj_iinit;
    // if (mpirank_braid == 0) printf("%d: iinit objective: %1.14e\n", mpirank_init, obj_iinit);

    /* Add to final-time fidelity */
    fidelity += getFidelity(finalstate);

    /* --- Solve adjoint --- */
    // if (mpirank_braid == 0) printf("%d: %d BWD.", mpirank_init, initid);

    /* Reset adjoint */
    VecZeroEntries(rho_t0_bar);

    /* Derivative of average over initial conditions */
    double Jbar = 1.0 / ninit * obj_weights[iinit];

    /* Derivative of final time objective */
    objectiveT_diff(optim_target, timestepper->mastereq, finalstate, rho_t0_bar, rho_t0, Jbar);

    /* Derivative of time-stepping */
#ifdef WITH_BRAID
      adjointbraidapp->PreProcess(initid, rho_t0_bar, Jbar*gamma_penalty);
      adjointbraidapp->Drive();
      adjointbraidapp->PostProcess();
#else
      timestepper->solveAdjointODE(initid, rho_t0_bar, Jbar*gamma_penalty);
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
  int iter;
  double deltax;
  Vec params;
  TaoConvergedReason reason;
  double f, gnorm;
  TaoGetSolutionStatus(tao, &iter, &f, &gnorm, NULL, &deltax, &reason);
  TaoGetSolutionVector(tao, &params);

  /* Pass current iteration number to output manager */
  ctx->output->optim_iter = iter;

  /* Grab some output stuff */
  double obj_cost = ctx->obj_cost;
  double obj_regul = ctx->obj_regul;
  double obj_penal = ctx->obj_penal;
  double F_avg = ctx->fidelity;

  /* Print to optimization file */
  ctx->output->writeOptimFile(f, gnorm, deltax, F_avg, obj_cost, obj_regul, obj_penal);

  /* Print parameters and controls to file */
  ctx->output->writeControls(params, ctx->timestepper->mastereq, ctx->timestepper->ntime, ctx->timestepper->dt);

  return 0;
}


PetscErrorCode TaoEvalObjectiveAndGradient(Tao tao, Vec x, PetscReal *f, Vec G, void*ptr){

  TaoEvalGradient(tao, x, G, ptr);
  OptimProblem* ctx = (OptimProblem*) ptr;
  *f = ctx->objective;

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



double objectiveT(OptimTarget* optim_target, MasterEq* mastereq, const Vec state, const Vec rho_t0) {
  double obj_local = 0.0;
  int diagID;
  double sum, mine, rhoii, lambdai, norm;
  int ilo, ihi;

  if (state != NULL) {

    switch (optim_target->target_type) {
      case GATE: // Gate optimization: Target \rho_target = V\rho(0)V^\dagger
        
        switch(optim_target->objective_type) {
          case JFROBENIUS:
            /* J_T = 1/2 * || rho_target - rho(T)||^2_F  */
            optim_target->targetgate->compare_frobenius(state, rho_t0, obj_local);
            break;
          case JHS:
            /* J_T = 1 - 1/purity * Tr(rho_target^\dagger * rho(T)) */
            optim_target->targetgate->compare_trace(state, rho_t0, obj_local, true);
            break;
          case JMEASURE: // JMEASURE is only for pure-state preparation!
            printf("ERROR: Check settings for optim_target and optim_objective.\n");
            exit(1);
            break;
        }
        break; // case gate

      case PUREM:

        int dim;
        VecGetSize(state, &dim);
        dim = (int) sqrt(dim/2.0);  // dim = N with \rho \in C^{N\times N}
        VecGetOwnershipRange(state, &ilo, &ihi);

        switch(optim_target->objective_type) {

          case JMEASURE:
            /* J_T = Tr(O_m rho(T)) = \sum_i |i-m| rho_ii(T) */
            // iterate over diagonal elements 
            sum = 0.0;
            for (int i=0; i<dim; i++){
              diagID = getIndexReal(getVecID(i,i,dim));
              rhoii = 0.0;
              if (ilo <= diagID && diagID < ihi) VecGetValues(state, 1, &diagID, &rhoii);
              lambdai = fabs(i - optim_target->purestateID);
              sum += lambdai * rhoii;
            }
            mine = sum;
            MPI_Allreduce(&mine, &sum, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
            obj_local = sum;
            break;
            
          case JFROBENIUS:
            /* J_T = 1/2 * || rho(T) - e_m e_m^\dagger||_F^2 */
            // substract 1.0 from m-th diagonal element then take the vector norm 
            diagID = getIndexReal(getVecID(optim_target->purestateID,optim_target->purestateID,dim));
            if (ilo <= diagID && diagID < ihi) VecSetValue(state, diagID, -1.0, ADD_VALUES);
            norm = 0.0;
            VecNorm(state, NORM_2, &norm);
            obj_local = pow(norm, 2.0) / 2.0;
            if (ilo <= diagID && diagID < ihi) VecSetValue(state, diagID, +1.0, ADD_VALUES); // restore original state!
            break;
            
          case JHS:
            /* J_T = 1 - Tr(e_m e_m^\dagger \rho(T)) = 1 - rho_mm(T) */
            diagID = getIndexReal(getVecID(optim_target->purestateID,optim_target->purestateID,dim));
            rhoii = 0.0;
            if (ilo <= diagID && diagID < ihi) VecGetValues(state, 1, &diagID, &rhoii);
            mine = 1. - rhoii;
            MPI_Allreduce(&mine, &obj_local, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
            break;
        } 
      break; // break pure1
    }
  }
  return obj_local;
}



void objectiveT_diff(OptimTarget *optim_target, MasterEq* mastereq, const Vec state, Vec statebar, const Vec rho_t0, const double obj_bar){
  int ilo, ihi;
  double lambdai, val;
  int diagID;

  if (state != NULL) {

    switch (optim_target->target_type) {
      case GATE:
        switch (optim_target->objective_type) {
          case JFROBENIUS:
            optim_target->targetgate->compare_frobenius_diff(state, rho_t0, statebar, obj_bar);
            break;
          case JHS:
            optim_target->targetgate->compare_trace_diff(state, rho_t0, statebar, obj_bar, true);
            break;
          case JMEASURE: // Will never happen
            printf("ERROR: Check settings for optim_target and optim_objective.\n");
            exit(1);
            break;
        }
        break; // case gate

      case PUREM:
        int dim;
        VecGetSize(state, &dim);
        dim = (int) sqrt(dim/2.0);  // dim = N with \rho \in C^{N\times N}
        VecGetOwnershipRange(state, &ilo, &ihi);

        switch (optim_target->objective_type) {

          case JMEASURE:
            // iterate over diagonal elements 
            for (int i=0; i<dim; i++){
              lambdai = fabs(i - optim_target->purestateID);
              diagID = getIndexReal(getVecID(i,i,dim));
              val = lambdai * obj_bar;
              if (ilo <= diagID && diagID < ihi) VecSetValue(statebar, diagID, val, ADD_VALUES);
            }
            break;

          case JFROBENIUS:
            // Derivative of J = 1/2||x||^2 is xbar += x * Jbar, where x = rho(t) - E_mm
            VecAXPY(statebar, obj_bar, state);
            // now substract 1.0*obj_bar from m-th diagonal element
            diagID = getIndexReal(getVecID(optim_target->purestateID,optim_target->purestateID,dim));
            if (ilo <= diagID && diagID < ihi) VecSetValue(statebar, diagID, -1.0*obj_bar, ADD_VALUES);
            break;

          case JHS:
            diagID = getIndexReal(getVecID(optim_target->purestateID,optim_target->purestateID,dim));
            val = -1. * obj_bar;
            if (ilo <= diagID && diagID < ihi) VecSetValue(statebar, diagID, val, ADD_VALUES);
            break;
        }
        break; // case pure1
    }
  }
}


double OptimProblem::getFidelity(const Vec finalstate){
  double fidel = 0.0;
  int dimrho = timestepper->mastereq->getDimRho(); // N
  int vecID, ihi, ilo;
  double rho_mm, mine;

  switch(optim_target->target_type){
    case PUREM: // fidelity = rho(T)_mm
      vecID = getIndexReal(getVecID(optim_target->purestateID, optim_target->purestateID, dimrho));
      VecGetOwnershipRange(finalstate, &ilo, &ihi);
      rho_mm = 0.0;
      if (ilo <= vecID && vecID < ihi) VecGetValues(finalstate, 1, &vecID, &rho_mm); // local!
      fidel = rho_mm;
      // Communicate over all petsc processors.
      mine = fidel;
      MPI_Allreduce(&mine, &fidel, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    break;

    case GATE: // fidelity = Tr(Vrho(0)V^\dagger \rho(T))
      optim_target->targetgate->compare_trace(finalstate, rho_t0, fidel, false);
      fidel = 1. - fidel; // because compare_trace computes infidelity 1-x and we want x.
    break;
  }
 
  return fidel;
}