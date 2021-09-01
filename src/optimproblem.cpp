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

  /* Store objective function type */
  std::vector<std::string> objective_str;
  config.GetVecStrParam("optim_objective", objective_str);
  targetgate = NULL;
  if ( objective_str[0].compare("gate") ==0 ) {
    /* Read and initialize the targetgate */
    assert ( objective_str.size() >=2 );
    if      (objective_str[1].compare("none") == 0)  targetgate = new Gate(); // dummy gate. do nothing
    else if (objective_str[1].compare("xgate") == 0) targetgate = new XGate(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, timestepper->total_time, gate_rot_freq); 
    else if (objective_str[1].compare("ygate") == 0) targetgate = new YGate(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, timestepper->total_time, gate_rot_freq); 
    else if (objective_str[1].compare("zgate") == 0) targetgate = new ZGate(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, timestepper->total_time, gate_rot_freq);
    else if (objective_str[1].compare("hadamard") == 0) targetgate = new HadamardGate(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, timestepper->total_time, gate_rot_freq);
    else if (objective_str[1].compare("cnot") == 0) targetgate = new CNOT(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, timestepper->total_time, gate_rot_freq); 
    else if (objective_str[1].compare("swap") == 0) targetgate = new SWAP(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, timestepper->total_time, gate_rot_freq); 
    else if (objective_str[1].compare("swap0q") == 0) {
      assert(objective_str.size() >= 3);
      targetgate = new SWAP_0Q(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, stoi(objective_str[2]), timestepper->total_time, gate_rot_freq); 
    }
    else if (objective_str[1].compare("cqnot") == 0) targetgate = new CQNOT(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, timestepper->total_time, gate_rot_freq); 
    else {
      printf("\n\n ERROR: Unnown gate type: %s.\n", objective_str[1].c_str());
      printf(" Available gates are 'none', 'xgate', 'ygate', 'zgate', 'hadamard', 'cnot', 'swap', 'swap0q', 'cqnot'.\n");
      exit(1);
    }
    /* Get gate measure */
    std::string gate_measure = config.GetStrParam("gate_measure", "frobenius");
    if (gate_measure.compare("frobenius")==0) objective_type = GATE_FROBENIUS;
    else if (gate_measure.compare("trace")==0) objective_type = GATE_TRACE;
    else  {
      printf("\n\n ERROR: Unknown gate measure: %s\n", gate_measure.c_str());
      exit(1);
    }
  }  
  else if (objective_str[0].compare("expectedEnergy")==0) objective_type = EXPECTEDENERGY;
  else if (objective_str[0].compare("expectedEnergya")==0) objective_type = EXPECTEDENERGYa;
  else if (objective_str[0].compare("expectedEnergyb")==0) objective_type = EXPECTEDENERGYb;
  else if (objective_str[0].compare("expectedEnergyc")==0) objective_type = EXPECTEDENERGYc;
  else if (objective_str[0].compare("groundstate")   ==0) objective_type = GROUNDSTATE;
  else {
      printf("\n\n ERROR: Unknown objective function: %s\n", objective_str[0].c_str());
      exit(1);
  }

  /* Get the IDs of oscillators that are considered in the objective function and corresponding weights */
  std::vector<std::string> oscilIDstr;
  config.GetVecStrParam("optim_oscillators", oscilIDstr);
  if (oscilIDstr[0].compare("all") == 0) {
    for (int iosc = 0; iosc < timestepper->mastereq->getNOscillators(); iosc++) 
      obj_oscilIDs.push_back(iosc);
  } else {
    config.GetVecIntParam("optim_oscillators", obj_oscilIDs, 0);
  }
  config.GetVecDoubleParam("optim_weights", obj_weights, 1.0);
  int nfill = 0;
  if (obj_weights.size() < ninit) nfill = ninit - obj_weights.size();
  double val = obj_weights[obj_weights.size()-1];
  if (obj_weights.size() < ninit){
    for (int i = 0; i < nfill; i++) 
      obj_weights.push_back(val);
  }
  assert(obj_weights.size() >= ninit);
  /* Sanity check for oscillator IDs */
  bool err = false;
  assert(obj_oscilIDs.size() > 0);
  for (int i=0; i<obj_oscilIDs.size(); i++){
    if ( obj_oscilIDs[i] >= timestepper->mastereq->getNOscillators() ) err = true;
    if ( i>0 &&  ( obj_oscilIDs[i] != obj_oscilIDs[i-1] + 1 ) ) err = true;
  }
  if (err) {
    printf("ERROR: List of oscillator IDs for objective function invalid\n"); 
    exit(1);
  }

  /* Pass information on objective function to the time stepper needed for penalty objective function */
  penalty_coeff = config.GetDoubleParam("optim_penalty", 1e-4);
  penalty_weightparam = config.GetDoubleParam("optim_penalty_param", 0.5);
  timestepper->objective_type = objective_type;
  timestepper->obj_oscilIDs = obj_oscilIDs;
  timestepper->penalty_weightparam = penalty_weightparam;
  timestepper->penalty_coeff = penalty_coeff;

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
    int diag_id = 0.0;
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
  double obj_cost_max = 0.0;
  obj_regul = 0.0;
  obj_penal = 0.0;
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
    obj_penal += penalty_coeff * timestepper->penalty_integral;

    /* Compute and add final-time cost */
    double obj_iinit = objectiveT(timestepper->mastereq, objective_type, obj_oscilIDs, finalstate, rho_t0, targetgate);
    obj_cost +=  obj_weights[iinit] * obj_iinit;
    obj_cost_max = std::max(obj_cost_max, obj_iinit);
    // printf("%d, %d: iinit objective: %f * %1.14e\n", mpirank_world, mpirank_init, obj_weights[iinit], obj_iinit);
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
  MPI_Allreduce(&mypen, &obj_penal, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mycost, &obj_cost, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  // if (mpirank_init == 0) printf("%d: global sum objective: %1.14e\n\n", mpirank_init, obj);

  /* Evaluate regularization objective += gamma/2 * ||x||^2*/
  double xnorm;
  VecNorm(x, NORM_2, &xnorm);
  obj_regul = gamma_tik / 2. * pow(xnorm,2.0);

  /* Sum, store and return objective value */
  objective = obj_cost + obj_regul + obj_penal;

  /* Output */
  if (mpirank_world == 0) {
    std::cout<< "Objective = " << std::scientific<<std::setprecision(14) << obj_cost << " + " << obj_regul << " + " << obj_penal << std::endl;
    std::cout<< "Max. costT = " << obj_cost_max << std::endl;
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
    obj_penal += penalty_coeff * timestepper->penalty_integral;

    /* Compute and add final-time cost */
    double obj_iinit = objectiveT(timestepper->mastereq, objective_type, obj_oscilIDs, finalstate, rho_t0, targetgate);
    obj_cost += obj_weights[iinit] * obj_iinit;
    // if (mpirank_braid == 0) printf("%d: iinit objective: %1.14e\n", mpirank_init, obj_iinit);

    /* --- Solve adjoint --- */
    // if (mpirank_braid == 0) printf("%d: %d BWD.", mpirank_init, initid);

    /* Reset adjoint */
    VecZeroEntries(rho_t0_bar);

    /* Derivative of average over initial conditions */
    double Jbar = 1.0 / ninit * obj_weights[iinit];

    /* Derivative of final time objective */
    objectiveT_diff(timestepper->mastereq, objective_type, obj_oscilIDs, finalstate, rho_t0_bar, rho_t0, Jbar, targetgate);

    /* Derivative of time-stepping */
#ifdef WITH_BRAID
      adjointbraidapp->PreProcess(initid, rho_t0_bar, Jbar*penalty_coeff);
      adjointbraidapp->Drive();
      adjointbraidapp->PostProcess();
#else
      timestepper->solveAdjointODE(initid, rho_t0_bar, Jbar*penalty_coeff);
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
  MPI_Allreduce(&mypen, &obj_penal, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mycost, &obj_cost, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  // if (mpirank_init == 0) printf("%d: global sum objective: %1.14e\n\n", mpirank_init, obj);

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


  /* Average fidelity. */
  /* For gates, this is 1.-\sum_ij Tr(V \rho_ij(0) V^\dagger \rho_ij(T)) */
  /* For groundstate optimization, this is the objective function value */
  double F_avg = -1.0;
  if (ctx->initcond_type == BASIS && ctx->objective_type == GATE_TRACE) F_avg = 1. - ctx->obj_cost;
  if (ctx->objective_type == EXPECTEDENERGY ||
      ctx->objective_type == EXPECTEDENERGYa ||
      ctx->objective_type == EXPECTEDENERGYb ||
      ctx->objective_type == EXPECTEDENERGYc ) F_avg = ctx->obj_cost;

  /* If average fidelity is not the objective function, it can be computed at each optimization iteration here. 
   * However, this involves the entire basis be propagated forward (or at last the N+1 states from Koch's paper). We omit it here to save compute time during optimization. 
   * Average fidelity can be computed AFTER optimization has finished by propagating the the basis and evaluating the Gate_trace. The commented code below needs to be checked before use!
   */
  // if (ctx->initcond_type != BASIS)  ctx->obj_cost = -1.0;  // -1 is used to indicate that average fidelity has not been computed
  // if (ctx->objective_type == GATE_FROBENIUS ||
      // (ctx->objective_type == GATE_TRACE && ctx->initcond_type != BASIS) ){
  //   InitialConditionType inittype_org = ctx->initcond_type; 
  //   ObjectiveType objtype_org = ctx->objective_type; 
  //   int ninit_local_org = ctx->ninit_local;
  //   int ninit_org = ctx->ninit;
  //   ctx->initcond_type = BASIS;
  //   ctx->ninit_local = 16;
  //   ctx->ninit= 16;
  //   ctx->objective_type = GATE_TRACE;
  //   double obj = ctx->evalF(params);    // this sets ctx->obj_cost
  //   // double F_avg = 1.0 - ctx->obj_cost;
  //   ctx->initcond_type = inittype_org;
  //   ctx->objective_type = objtype_org;
  //   ctx->ninit_local = ninit_local_org;
  //   ctx->ninit = ninit_org;
  // }

  // /* Estimate Favg by propagating N+1 initial states */
  // if (iter % ctx->output->optim_monitor_freq == 0) {

  //   // save original optimization setting
  //   InitialConditionType inittype_org = ctx->initcond_type; 
  //   ObjectiveType objtype_org = ctx->objective_type; 
  //   int ninit_local_org = ctx->ninit_local;
  //   int ninit_org = ctx->ninit;
  //   std::vector<double> obj_weights_org;
  //   for (int i=0; i<ctx->obj_weights.size(); i++){
  //     obj_weights_org.push_back(ctx->obj_weights[i]);
  //   }

  //   // setting for propagating N+1 initial states
  //   ctx->objective_type = GATE_TRACE;
  //   ctx->initcond_type = NPLUSONE;
  //   ctx->ninit_local = ctx->targetgate->getDimRho() + 1; // N+1
  //   ctx->ninit= ctx->ninit_local;
  //   // Make sure obj_weights is long enough and all weights are 1.0
  //   for (int i=0; i < ctx->ninit; i++){
  //     if (i < ctx->obj_weights.size()) ctx->obj_weights[i] = 1.0;
  //     else                             ctx->obj_weights.push_back(1.0); 
  //   }
  //   printf("\n Monitor: Eval Favg(Nplus1)...\n");
  //   ctx->evalF(params);    // this sets ctx->obj_cost
  //   F_avg = ctx->obj_cost;

  //   // Reset to original setting
  //   ctx->initcond_type = inittype_org;
  //   ctx->objective_type = objtype_org;
  //   ctx->ninit_local = ninit_local_org;
  //   ctx->ninit = ninit_org;
  //   for (int i=0; i < obj_weights_org.size(); i++){
  //     ctx->obj_weights[i] = obj_weights_org[i];
  //   }
  // }

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



double objectiveT(MasterEq* mastereq, ObjectiveType objective_type, const std::vector<int>& obj_oscilIDs, const Vec state, const Vec rho_t0, Gate* targetgate) {
  double obj_local = 0.0;
  double sum;

  if (state != NULL) {

    switch (objective_type) {
      case GATE_FROBENIUS:
        /* compare state to linear transformation of initial conditions using Frobenius norm */
        /* J_T = 1/2 || rho(T) - Vrho(0)V^d||^2 */
        targetgate->compare_frobenius(state, rho_t0, obj_local);
        break;

      case GATE_TRACE:
        /* compare state to linear transformation of initial conditions using Trace overlap */
        /* J_T = 1 - Tr[Vrho(0)V^d rho(T)] */
        targetgate->compare_trace(state, rho_t0, obj_local);
        break;

      case EXPECTEDENERGY:
        /* Weighted sum of expected energy levels */
        obj_local = 0.0;
        for (int i=0; i<obj_oscilIDs.size(); i++) {
          obj_local += mastereq->getOscillator(obj_oscilIDs[i])->expectedEnergy(state);
        }
        break;

      case EXPECTEDENERGYa:
        /* Squared average of expected energy level f = ( sum_{k=0}^Q < N_k(rho(T)) > )^2 */
        sum = 0.0;
        for (int i=0; i<obj_oscilIDs.size(); i++) {
          sum += mastereq->getOscillator(obj_oscilIDs[i])->expectedEnergy(state);
        }
        obj_local = pow(sum / obj_oscilIDs.size(), 2.0);
        break;

      case EXPECTEDENERGYb:
        /* average of Squared expected energy level f = 1/Q sum_{k=0}^Q < N_k(rho(T))>^2 */
        double g;
        sum = 0.0;
        for (int i=0; i<obj_oscilIDs.size(); i++) {
          g = mastereq->getOscillator(obj_oscilIDs[i])->expectedEnergy(state);
          sum += pow(g,2.0);
        }
        obj_local = sum / obj_oscilIDs.size();
        break;

      case EXPECTEDENERGYc:
        /* average of expected energy level f = 1/Q sum_{k=0}^Q < N_k(rho(T))> */
        sum = 0.0;
        for (int i=0; i<obj_oscilIDs.size(); i++) {
          sum += mastereq->getOscillator(obj_oscilIDs[i])->expectedEnergy(state);
        }
        obj_local = sum / obj_oscilIDs.size();
        break;

      case GROUNDSTATE:
        /* compare full state to groundstate */

        /* If sub-system is requested, compute reduced density operator first */
        if (obj_oscilIDs.size() < mastereq->getNOscillators()) { 
          printf("ERROR: Computing reduced density matrix is currently not available and needs testing!\n");
          exit(1);
        }

        /* Compute frobenius norm: frob = 1/2|| q(T) - e_0 ||^2 */
        int ilo, ihi;
        VecGetOwnershipRange(state, &ilo, &ihi);
        if (ilo <= 0 && 0 < ihi) VecSetValue(state, 0, -1.0, ADD_VALUES); // substract 1.0 from (0,0) element
        VecNorm(state, NORM_2, &obj_local);
        obj_local = 0.5*pow(obj_local, 2.0);
        if (ilo <= 0 && 0 < ihi) VecSetValue(state, 0, 1.0, ADD_VALUES); // restore state 
        VecAssemblyBegin(state);
        VecAssemblyEnd(state);
        break;
    }
  }
  return obj_local;
}



void objectiveT_diff(MasterEq* mastereq, ObjectiveType objective_type, const std::vector<int>& obj_oscilIDs, Vec state, Vec statebar, const Vec rho_t0, const double obj_bar, Gate* targetgate){

  if (state != NULL) {
    switch (objective_type) {

      case GATE_FROBENIUS:
        targetgate->compare_frobenius_diff(state, rho_t0, statebar, obj_bar);
        break;

      case GATE_TRACE:
        targetgate->compare_trace_diff(state, rho_t0, statebar, obj_bar);
        break;

      case EXPECTEDENERGY:
        for (int i=0; i<obj_oscilIDs.size(); i++) {
          mastereq->getOscillator(obj_oscilIDs[i])->expectedEnergy_diff(state, statebar, obj_bar);
        }
        break;

      case EXPECTEDENERGYa:
        double Jbar, sum;
        // Recompute sum over energy levels 
        sum = 0.0;
        for (int i=0; i<obj_oscilIDs.size(); i++) {
          sum += mastereq->getOscillator(obj_oscilIDs[i])->expectedEnergy(state);
        } 
        sum = sum / obj_oscilIDs.size();
        // Derivative of expected energy levels 
        Jbar = 2. * sum * obj_bar / obj_oscilIDs.size();
        for (int i=0; i<obj_oscilIDs.size(); i++) {
          mastereq->getOscillator(obj_oscilIDs[i])->expectedEnergy_diff(state, statebar, Jbar);
        }
        break;

      case EXPECTEDENERGYb:
        for (int i=0; i<obj_oscilIDs.size(); i++) {
          Jbar = 2. / obj_oscilIDs.size() * mastereq->getOscillator(obj_oscilIDs[i])->expectedEnergy(state) * obj_bar;
          mastereq->getOscillator(obj_oscilIDs[i])->expectedEnergy_diff(state, statebar, Jbar);
        }
        break;

      case EXPECTEDENERGYc:
        Jbar = obj_bar / obj_oscilIDs.size();
        for (int i=0; i<obj_oscilIDs.size(); i++) {
          mastereq->getOscillator(obj_oscilIDs[i])->expectedEnergy_diff(state, statebar, Jbar);
        }
        break;

    case GROUNDSTATE:
      int ilo, ihi;
      VecGetOwnershipRange(statebar, &ilo, &ihi);

      /* Derivative of frobenius norm: (q(T) - e_0) * frob_bar */
      VecAXPY(statebar, obj_bar, state);
      if (ilo <= 0 && 0 < ihi) VecSetValue(statebar, 0, -1.0*obj_bar, ADD_VALUES);
      VecAssemblyBegin(statebar); VecAssemblyEnd(statebar);
      break;
    }
  }
}
