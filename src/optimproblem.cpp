#include "optimproblem.hpp"


OptimProblem::OptimProblem(MapParam config, myBraidApp* primalbraidapp_, myAdjointBraidApp* adjointbraidapp_, MPI_Comm comm_hiop_, MPI_Comm comm_init_, int ninit_) {

  primalbraidapp  = primalbraidapp_;
  adjointbraidapp = adjointbraidapp_;
  ninit = ninit_;
  comm_hiop = comm_hiop_;
  comm_init = comm_init_;

  /* Store ranks and sizes of communicators */
  MPI_Comm_rank(primalbraidapp->comm_braid, &mpirank_braid);
  MPI_Comm_size(primalbraidapp->comm_braid, &mpisize_braid);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize_world);
  MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank_space);
  MPI_Comm_size(PETSC_COMM_WORLD, &mpisize_space);
  MPI_Comm_rank(comm_hiop, &mpirank_optim);
  MPI_Comm_size(comm_hiop, &mpisize_optim);
  MPI_Comm_rank(comm_init, &mpirank_init);
  MPI_Comm_size(comm_init, &mpisize_init);

  /* Store number of initial conditions per init-processor group */
  ninit_local = ninit / mpisize_init; 

  /* Store number of design parameters */
  int n = 0;
  for (int ioscil = 0; ioscil < primalbraidapp->mastereq->getNOscillators(); ioscil++) {
      n += primalbraidapp->mastereq->getOscillator(ioscil)->getNParams(); 
  }
  ndesign = n;
  if (mpirank_world == 0) std::cout<< "ndesign = " << ndesign << std::endl;

  /* Store other optimization parameters */
  gamma_tik = config.GetDoubleParam("optim_regul", 1e-4);
  gatol = config.GetDoubleParam("optim_atol", 1e-8);
  grtol = config.GetDoubleParam("optim_rtol", 1e-4);
  maxiter = config.GetIntParam("optim_maxiter", 200);

  /* Reset */
  objective = 0.0;

  /* Store objective function type */
  std::vector<std::string> objective_str;
  config.GetVecStrParam("optim_objective", objective_str);
  targetgate = NULL;
  if ( objective_str[0].compare("gate") ==0 ) {
    objective_type = GATE;
    /* Read and initialize the targetgate */
    assert ( objective_str.size() >=2 );
    if      (objective_str[1].compare("none") == 0)  targetgate = new Gate(); // dummy gate. do nothing
    else if (objective_str[1].compare("xgate") == 0) targetgate = new XGate(); 
    else if (objective_str[1].compare("ygate") == 0) targetgate = new YGate(); 
    else if (objective_str[1].compare("zgate") == 0) targetgate = new ZGate();
    else if (objective_str[1].compare("hadamard") == 0) targetgate = new HadamardGate();
    else if (objective_str[1].compare("cnot") == 0) targetgate = new CNOT(); 
    else {
      printf("\n\n ERROR: Unnown gate type: %s.\n", objective_str[1].c_str());
      printf(" Available gates are 'none', 'xgate', 'ygate', 'zgate', 'hadamard', 'cnot'\n");
      exit(1);
    }
  }  
  else if (objective_str[0].compare("expectedEnergy")==0) objective_type = EXPECTEDENERGY;
  else if (objective_str[0].compare("groundstate")   ==0) objective_type = GROUNDSTATE;
  else {
      printf("\n\n ERROR: Unknown objective function: %s\n", objective_str[0].c_str());
      exit(1);
  }

  /* Get the IDs of oscillators that are considered in the objective function */
  std::vector<std::string> oscilIDstr;
  config.GetVecStrParam("optim_oscillators", oscilIDstr);
  if (oscilIDstr[0].compare("all") == 0) {
    for (int iosc = 0; iosc < primalbraidapp->mastereq->getNOscillators(); iosc++) 
      obj_oscilIDs.push_back(iosc);
  } else {
    config.GetVecIntParam("optim_oscillators", obj_oscilIDs, 0);
  }
  /* Sanity check for oscillator IDs */
  bool err = false;
  assert(obj_oscilIDs.size() > 0);
  for (int i=0; i<obj_oscilIDs.size(); i++){
    if ( obj_oscilIDs[i] >= primalbraidapp->mastereq->getNOscillators() ) err = true;
    if ( i>0 &&  ( obj_oscilIDs[i] != obj_oscilIDs[i-1] + 1 ) ) err = true;
  }
  if (err) {
    printf("ERROR: List of oscillator IDs for objective function invalid\n"); 
    exit(1);
  }

  /* Get initial condition type and involved oscillators */
  std::vector<std::string> initcondstr;
  config.GetVecStrParam("initialcondition", initcondstr);
  for (int i=1; i<initcondstr.size(); i++) initcond_IDs.push_back(atoi(initcondstr[i].c_str()));
  ninit = 1;
  if (initcondstr[0].compare("file") == 0 )      initcond_type = FROMFILE;
  else if (initcondstr[0].compare("pure") == 0 ) initcond_type = PURE;
  else if (initcondstr[0].compare("diagonal") == 0 ) {
    initcond_type = DIAGONAL;
    /* Compute ninit = dim(subsystem defined by initcond_IDs) */
    ninit = 1;
    for (int i = 1; i<initcondstr.size(); i++){
      int oscilID = atoi(initcondstr[i].c_str());
      ninit *= primalbraidapp->mastereq->getOscillator(oscilID)->getNLevels();
    }
  }
  else if (initcondstr[0].compare("basis")    == 0 ) {
    initcond_type = BASIS;
    /* Compute ninit = dim(subsystem defined by obj_oscilIDs)^2 */
    ninit = 1;
    for (int i = 1; i<initcondstr.size(); i++){
      int oscilID = atoi(initcondstr[i].c_str());
      ninit *= primalbraidapp->mastereq->getOscillator(oscilID)->getNLevels();
    }
    ninit = (int) pow(ninit, 2);
  }
  else {
    printf("\n\n ERROR: Wrong setting for initial condition.\n");
    exit(1);
  }

  /* Allocate the initial condition vector */
  VecCreate(PETSC_COMM_WORLD, &rho_t0); 
  VecSetSizes(rho_t0,PETSC_DECIDE,2*primalbraidapp->mastereq->getDim());
  VecSetFromOptions(rho_t0);

  /* If PURE or FROMFILE initialization, store them here. Otherwise they are set inside evalF */
  if (initcond_type == PURE) { 
    /* Initialize with tensor product of unit vectors. */

    // Compute index of diagonal elements that is one.
    assert (initcond_IDs.size() == primalbraidapp->mastereq->getNOscillators());
    int diag_id = 0.0;
    for (int k=0; k < initcond_IDs.size(); k++) {
      assert (initcond_IDs[k] < primalbraidapp->mastereq->getOscillator(k)->getNLevels());
      int dim_postkron = 1;
      for (int m=k+1; m < initcond_IDs.size(); m++) {
        dim_postkron *= primalbraidapp->mastereq->getOscillator(m)->getNLevels();
      }
      diag_id += initcond_IDs[k] * dim_postkron;
    }
    int vec_id = diag_id * (int)sqrt(primalbraidapp->mastereq->getDim()) + diag_id;
    vec_id = getIndexReal(vec_id); // Real part of x
    VecSetValue(rho_t0, vec_id, 1.0, INSERT_VALUES);
  }
  else if (initcond_type == FROMFILE) { 
    /* Read initial condition from file */
    
    int dim = primalbraidapp->mastereq->getDim();
    double * vec = new double[2*dim];
    std::vector<std::string> initcondstr;
    config.GetVecStrParam("initialcondition", initcondstr);
    if (mpirank_world == 0) {
      assert (initcondstr.size()==2);
      std::string filename = initcondstr[1];
      read_vector(filename.c_str(), vec, 2*dim);
    }
    MPI_Bcast(vec, 2*dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < dim; i++) {
      VecSetValue(rho_t0, getIndexReal(i), vec[i], INSERT_VALUES);        // RealPart
      VecSetValue(rho_t0, getIndexImag(i), vec[i + dim ], INSERT_VALUES); // Imaginary Part
    }
    delete [] vec;
  }
  VecAssemblyBegin(rho_t0); VecAssemblyEnd(rho_t0);

  /* Initialize adjoint */
  VecDuplicate(rho_t0, &rho_t0_bar);
  VecZeroEntries(rho_t0_bar);
  VecAssemblyBegin(rho_t0_bar); VecAssemblyEnd(rho_t0_bar);

  /* Output */
  printlevel = config.GetIntParam("optim_printlevel", 1);
  if (mpirank_world == 0 && printlevel > 0) {
    char filename[255];
    sprintf(filename, "%s/optimTao.dat", primalbraidapp->datadir.c_str());
    optimfile = fopen(filename, "w");
    fprintf(optimfile, "#iter    obj_value           ||grad||               LS step          Regularization\n");
  } 

  /* Store optimization bounds */
  VecCreateSeq(PETSC_COMM_SELF, ndesign, &xlower);
  VecSetFromOptions(xlower);
  VecDuplicate(xlower, &xupper);
  std::vector<double> bounds;
  config.GetVecDoubleParam("optim_bounds", bounds, 1e20);
  assert (bounds.size() >= primalbraidapp->mastereq->getNOscillators());
  int col = 0;
  for (int iosc = 0; iosc < primalbraidapp->mastereq->getNOscillators(); iosc++){
    // Scale bounds by 1/sqrt(2) * (number of carrier waves) */
    std::vector<double> carrier_freq;
    std::string key = "carrier_frequency" + std::to_string(iosc);
    config.GetVecDoubleParam(key, carrier_freq, 0.0);
    bounds[iosc] = bounds[iosc] / ( sqrt(2) * carrier_freq.size()) ;
    for (int i=0; i<primalbraidapp->mastereq->getOscillator(iosc)->getNParams(); i++){
      VecSetValue(xupper, col, bounds[iosc], INSERT_VALUES);
      VecSetValue(xlower, col, -1. * bounds[iosc], INSERT_VALUES);
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

  /* Set initial starting point */
  initguess_type = config.GetStrParam("optim_init", "zero");
  if (initguess_type.compare("constant") == 0 ){ 
    config.GetVecDoubleParam("optim_init_const", initguess_amplitudes, 0.0);
    assert(initguess_amplitudes.size() == primalbraidapp->mastereq->getNOscillators());
  }
  VecDuplicate(xlower, &xinit);
  getStartingPoint(xinit);
  TaoSetInitialVector(tao, xinit);

}


OptimProblem::~OptimProblem() {
  VecDestroy(&rho_t0);
  VecDestroy(&rho_t0_bar);
  if (mpirank_world == 0 && printlevel > 0) fclose(optimfile);

  VecDestroy(&xinit);
  VecDestroy(&xlower);
  VecDestroy(&xupper);

  TaoDestroy(&tao);
}



double OptimProblem::evalF(const Vec x) {

  // OptimProblem* ctx = (OptimProblem*) ptr;
  MasterEq* mastereq = primalbraidapp->mastereq;

  if (mpirank_world == 0) printf(" EVAL F... \n");
  Vec finalstate = NULL;

  /* Pass design vector x to oscillators */
  mastereq->setControlAmplitudes(x); 

  /*  Iterate over initial condition */
  obj_cost = 0.0;
  for (int iinit = 0; iinit < ninit_local; iinit++) {
      
    /* Prepare the initial condition in [rank * ninit_local, ... , (rank+1) * ninit_local - 1] */
    int iinit_global = mpirank_init * ninit_local + iinit;
    int initid = primalbraidapp->mastereq->getRhoT0(iinit_global, ninit, initcond_type, initcond_IDs, rho_t0);
    if (mpirank_braid == 0) printf("%d: %d FWD. \n", mpirank_init, initid);

    /* Run forward with initial condition initid*/
    primalbraidapp->PreProcess(initid);
    primalbraidapp->setInitCond(rho_t0);
    primalbraidapp->Drive();
    finalstate = primalbraidapp->PostProcess(); // this return NULL for all but the last time processor
    

    /* Add to objective function */
    double obj_iinit = objectiveT(finalstate);
    obj_cost += obj_iinit;
    // printf("%d, %d: iinit objective: %1.14e\n", mpirank_world, mpirank_init, obj_iinit);
  }

  /* Broadcast objective from last to all time processors */
  MPI_Bcast(&obj_cost, 1, MPI_DOUBLE, mpisize_braid-1, primalbraidapp->comm_braid);

  /* Sum up objective from all initial conditions */
  obj_cost = 1./ninit * obj_cost;
  double myobj = obj_cost;
  MPI_Allreduce(&myobj, &obj_cost, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  // if (mpirank_init == 0) printf("%d: global sum objective: %1.14e\n\n", mpirank_init, obj);

  /* Add regularization objective += gamma/2 * ||x||^2*/
  double xnorm;
  VecNorm(x, NORM_2, &xnorm);
  obj_regul = gamma_tik / 2. * pow(xnorm,2.0);

  /* Store and return objective value */
  objective = obj_cost + obj_regul;

  /* Output */
  // if (mpirank_world == 0) {
    // std::cout<< mpirank_world << ": Obj = " << std::scientific<<std::setprecision(14) << obj_cost << " + " << obj_regul << std::endl;
  // }

  return objective;
}



void OptimProblem::evalGradF(const Vec x, Vec G){

  MasterEq* mastereq = primalbraidapp->mastereq;

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
  for (int iinit = 0; iinit < ninit_local; iinit++) {
      

    /* Prepare the initial condition */
    int iinit_global = mpirank_init * ninit_local + iinit;
    int initid = primalbraidapp->mastereq->getRhoT0(iinit_global, ninit, initcond_type, initcond_IDs, rho_t0);

    /* --- Solve primal --- */
    // if (mpirank_braid == 0) printf("%d: %d FWD. ", mpirank_init, initid);

    /* Run forward with initial condition initid*/
    primalbraidapp->PreProcess(initid);
    primalbraidapp->setInitCond(rho_t0);
    primalbraidapp->Drive();
    finalstate = primalbraidapp->PostProcess(); // this return NULL for all but the last time processor

    /* Add final-time objective */
    double obj_iinit = objectiveT(finalstate);
    obj_cost += obj_iinit;
      // if (mpirank_braid == 0) printf("%d: iinit objective: %1.14e\n", mpirank_init, obj_iinit);

    /* --- Solve adjoint --- */
    // if (mpirank_braid == 0) printf("%d: %d BWD.", mpirank_init, initid);

    /* Derivative of final time objective */
    double Jbar = 1.0 / ninit;
    objectiveT_diff(finalstate, obj_iinit, Jbar);

    adjointbraidapp->PreProcess(initid);
    adjointbraidapp->setInitCond(rho_t0_bar);
    adjointbraidapp->Drive();
    adjointbraidapp->PostProcess();

    /* Add to optimizers's gradient */
    VecAXPY(G, 1.0, adjointbraidapp->redgrad);

  }

  /* Broadcast objective from last to all time processors */
  MPI_Bcast(&obj_cost, 1, MPI_DOUBLE, mpisize_braid-1, primalbraidapp->comm_braid);

  /* Sum up objective from all initial conditions */
  double myobj = obj_cost;
  MPI_Allreduce(&myobj, &obj_cost, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  // if (mpirank_init == 0) printf("%d: global sum objective: %1.14e\n\n", mpirank_init, obj);

  /* Evaluate regularization gamma/2 * ||x||^2*/
  double xnorm;
  VecNorm(x, NORM_2, &xnorm);
  obj_regul = gamma_tik / 2. * pow(xnorm,2.0);

  /* Store objective function value*/
  objective = obj_cost + obj_regul;


  /* Sum up the gradient from all braid processors */
  PetscScalar* grad; 
  VecGetArray(G, &grad);
  double* mygrad = new double[ndesign];
  for (int i=0; i<ndesign; i++) {
    mygrad[i] = grad[i];
  }
  MPI_Allreduce(mygrad, grad, ndesign, MPI_DOUBLE, MPI_SUM, primalbraidapp->comm_braid);
  /* Sum up the gradient from all initial condition processors */
  for (int i=0; i<ndesign; i++) {
    mygrad[i] = grad[i];
  }
  MPI_Allreduce(mygrad, grad, ndesign, MPI_DOUBLE, MPI_SUM, comm_init);
  VecRestoreArray(G, &grad);
  delete [] mygrad;


  /* Compute and store gradient norm */
  VecNorm(G, NORM_2, &(gnorm));

  /* Output */
  if (mpirank_world == 0) {
    std::cout<< "Obj = " << obj_cost << " + " << obj_regul << " ||grad|| = " << gnorm << std::endl;
  }
}


void OptimProblem::solve() {
  TaoSolve(tao);
}

void OptimProblem::getStartingPoint(Vec xinit){
  MasterEq* mastereq = primalbraidapp->mastereq;

  if (initguess_type.compare("constant") == 0 ){ // set constant initial design
    int j = 0;
    for (int ioscil = 0; ioscil < mastereq->getNOscillators(); ioscil++) {
      int nparam = mastereq->getOscillator(ioscil)->getNParams();
      for (int i = 0; i < nparam; i++) {
        VecSetValue(xinit, j, initguess_amplitudes[ioscil], INSERT_VALUES);
        j++;
      }
    }
  } else if ( initguess_type.compare("zero") == 0)  { // init design with zero
    VecZeroEntries(xinit);

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

    /* Scale vector to be at 10% of the parameter bounds */
    int shift = 0;
    for (int ioscil = 0; ioscil < mastereq->getNOscillators(); ioscil++) {
      /* Get upper bound value */
      double bound;
      VecGetValues(xupper, 1, &shift, &bound);
      /* Scale all parameters */
      int nparam_iosc = mastereq->getOscillator(ioscil)->getNParams();
      for (int i=0; i<nparam_iosc; i++) {
        randvec[shift + i] *= 0.1 * bound;
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
  primalbraidapp->mastereq->setControlAmplitudes(xinit);
  
  /* Flush initial control functions */
  if (mpirank_world == 0 ) {
    int ntime = primalbraidapp->ntime;
    double dt = primalbraidapp->total_time / ntime;
    char filename[255];
    for (int ioscil = 0; ioscil < mastereq->getNOscillators(); ioscil++) {
        sprintf(filename, "%s/control_init_%02d.dat", primalbraidapp->datadir.c_str(), ioscil+1);
        mastereq->getOscillator(ioscil)->flushControl(ntime, dt, filename);
    }
  }

}



double OptimProblem::objectiveT(Vec finalstate){
  double obj_local = 0.0;

  if (finalstate != NULL) {

    switch (objective_type) {
      case GATE:
        /* compare state to linear transformation of initial conditions */
        targetgate->compare(finalstate, rho_t0, obj_local);
        break;

      case EXPECTEDENERGY:
        /* Squared average of expected energy level f = ( sum_{k=0}^Q < N_k(rho(T)) > )^2 */
        double sum;
        sum = 0.0;
        for (int i=0; i<obj_oscilIDs.size(); i++) {
          /* compute the expected value of energy levels for each oscillator */
          sum += primalbraidapp->mastereq->getOscillator(obj_oscilIDs[i])->expectedEnergy(finalstate);
        }
        obj_local = pow(sum / obj_oscilIDs.size(), 2.0);
        break;

      case GROUNDSTATE:
        /* compare full or pariatl state to groundstate */
        MasterEq *meq= primalbraidapp->mastereq;
        Vec state;

        /* If sub-system is requested, compute reduced density operator first */
        if (obj_oscilIDs.size() < meq->getNOscillators()) { 
          meq->createReducedDensity(finalstate, &state, obj_oscilIDs);
        } else { // full density matrix system 
           state = finalstate; 
        }


        /* Compute frobenius norm: frob = || q(T) - e_1 ||^2 */
        int ilo, ihi;
        VecGetOwnershipRange(state, &ilo, &ihi);
        if (ilo <= 0 && 0 < ihi) VecSetValue(state, 0, -1.0, ADD_VALUES); // substract 1.0 from (0,0) element
        VecNorm(state, NORM_2, &obj_local);
        obj_local = pow(obj_local, 2.0);
        if (ilo <= 0 && 0 < ihi) VecSetValue(state, 0, 1.0, ADD_VALUES); // restore state 
        VecAssemblyBegin(state);
        VecAssemblyEnd(state);

        /* Destroy reduced density matrix, if it has been created */
        if (obj_oscilIDs.size() < primalbraidapp->mastereq->getNOscillators()) { 
          VecDestroy(&state);
        }

        break;
    }

  }

  return obj_local;
}


void OptimProblem::objectiveT_diff(Vec finalstate, const double obj, const double obj_bar){

  /* Reset adjoints */
  VecZeroEntries(rho_t0_bar);

  if (finalstate != NULL) {
    switch (objective_type) {
      case GATE:
        targetgate->compare_diff(finalstate, rho_t0, rho_t0_bar, obj_bar);
        break;

      case EXPECTEDENERGY:
        double tmp;
        tmp = 2. * sqrt(obj) * obj_bar / obj_oscilIDs.size();
        // tmp = obj_bar;
        for (int i=0; i<obj_oscilIDs.size(); i++) {
          primalbraidapp->mastereq->getOscillator(obj_oscilIDs[i])->expectedEnergy_diff(finalstate, rho_t0_bar, tmp);
        }
        break;

    case GROUNDSTATE:

      MasterEq *meq= primalbraidapp->mastereq;
      Vec state;

      /* If sub-system is requested, compute reduced density operator first */
      if (obj_oscilIDs.size() < primalbraidapp->mastereq->getNOscillators()) { 
        /* Create reduced density matrix */
        meq->createReducedDensity(finalstate, &state, obj_oscilIDs);
      } else { // full density matrix system
         state = finalstate;
      }

      const PetscScalar *stateptr;
      PetscScalar *statebarptr;
      Vec statebar;
      VecDuplicate(state, &statebar);

      int ilo, ihi;
      VecGetOwnershipRange(statebar, &ilo, &ihi);

      /* Derivative of frobenius norm: 2 * (q(T) - e_1) * frob_bar */
      VecAXPY(statebar, 2.0*obj_bar, state);
      if (ilo <= 0 && 0 < ihi) VecSetValue(statebar, 0, -2.0*obj_bar, ADD_VALUES);
      VecAssemblyBegin(statebar); VecAssemblyEnd(statebar);
      
      /* Pass derivative from statebar to rho_t0_bar */
      if (obj_oscilIDs.size() < meq->getNOscillators()) {
        /* Derivative of partial trace  */
        meq->createReducedDensity_diff(rho_t0_bar, statebar, obj_oscilIDs);
        VecDestroy(&state);
      } else {
        VecCopy(statebar, rho_t0_bar);
      }

      VecDestroy(&statebar);
    }
  }
}


void OptimProblem::getSolution(Vec* param_ptr){
  
  /* Get ref to optimized parameters */
  Vec params;
  TaoGetSolutionVector(tao, &params);
  *param_ptr = params;

  /* Print if needed */
  if (mpirank_world == 0 && printlevel > 0) {
    char filename[255];
    FILE *paramfile;

    int iter;
    double obj, gnorm, cnorm, dx;
    TaoConvergedReason reason;
    TaoGetSolutionStatus(tao, &iter, &obj, &gnorm, &cnorm, &dx, &reason);
    std::cout<< "\n Optimization finished!\n";
    std::cout<< " TaoSolve termination reason: " << reason << std::endl;

    /* Print optimized parameters */
    const PetscScalar* params_ptr;
    VecGetArrayRead(params, &params_ptr);
    sprintf(filename, "%s/param_optimized.dat", primalbraidapp->datadir.c_str());
    paramfile = fopen(filename, "w");
    for (int i=0; i<ndesign; i++){
      fprintf(paramfile, "%1.14e\n", params_ptr[i]);
    }
    fclose(paramfile);
    VecRestoreArrayRead(params, &params_ptr);

    /* Print control functions */
    primalbraidapp->mastereq->setControlAmplitudes(params);
    int ntime = primalbraidapp->ntime;
    double dt = primalbraidapp->total_time / ntime;
    MasterEq* mastereq = primalbraidapp->mastereq;
    for (int ioscil = 0; ioscil < mastereq->getNOscillators(); ioscil++) {
        sprintf(filename, "%s/control_optimized_%02d.dat", primalbraidapp->datadir.c_str(), ioscil+1);
        mastereq->getOscillator(ioscil)->flushControl(ntime, dt, filename);
    }
  }
  // return param_ptr;
}


PetscErrorCode TaoMonitor(Tao tao,void*ptr){
  OptimProblem* ctx = (OptimProblem*) ptr;

  /* Output */
  if (ctx->mpirank_world == 0 && ctx->printlevel > 0) {

    int iter;
    double deltax;
    TaoConvergedReason reason;
    TaoGetSolutionStatus(tao, &iter, NULL, NULL, NULL, &deltax, &reason);

    /* Print to optimization file */
    fprintf(ctx->optimfile, "%05d  %1.14e  %1.14e  %.8f  %1.14e\n", iter, ctx->objective, ctx->gnorm, deltax, ctx->obj_regul);
    fflush(ctx->optimfile);

    /* Print parameters and controls to file */
    if ( ctx->printlevel > 1 || iter % 10 == 0 ) {
      char filename[255];

      /* Print current parameters to file */
      Vec params;
      const PetscScalar* params_ptr;
      TaoGetSolutionVector(tao, &params);
      VecGetArrayRead(params, &params_ptr);
      FILE *paramfile;
      sprintf(filename, "%s/param_iter%04d.dat", ctx->primalbraidapp->datadir.c_str(), iter);
      paramfile = fopen(filename, "w");
      for (int i=0; i<ctx->ndesign; i++){
        fprintf(paramfile, "%1.14e\n", params_ptr[i]);
      }
      fclose(paramfile);
      VecRestoreArrayRead(params, &params_ptr);

      /* Print control functions */
      ctx->primalbraidapp->mastereq->setControlAmplitudes(params);
      int ntime = ctx->primalbraidapp->ntime;
      double dt = ctx->primalbraidapp->total_time / ntime;
      MasterEq* mastereq = ctx->primalbraidapp->mastereq;
      for (int ioscil = 0; ioscil < mastereq->getNOscillators(); ioscil++) {
          sprintf(filename, "%s/control_iter%04d_%02d.dat", ctx->primalbraidapp->datadir.c_str(), iter, ioscil+1);
          mastereq->getOscillator(ioscil)->flushControl(ntime, dt, filename);
      }
    }
  }

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
