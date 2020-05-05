#include "optimizer_petsc.hpp"


OptimCtx::OptimCtx(MapParam config, myBraidApp* primalbraidapp_, myAdjointBraidApp* adjointbraidapp_, MPI_Comm comm_hiop_, MPI_Comm comm_init_, std::vector<int> obj_oscilIDs_, InitialConditionType initcondtype_, int ninit_) {

  primalbraidapp  = primalbraidapp_;
  adjointbraidapp = adjointbraidapp_;
  initcond_type = initcondtype_;
  ninit = ninit_;
  comm_hiop = comm_hiop_;
  comm_init = comm_init_;
  obj_oscilIDs = obj_oscilIDs_;

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

  /* Store other optimization parameters */
  gamma_tik = config.GetDoubleParam("optim_regul", 1e-4);

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

  /* Set up and store initial condition vectors */
  VecCreate(PETSC_COMM_WORLD, &initcond_re); 
  VecSetSizes(initcond_re,PETSC_DECIDE,primalbraidapp->mastereq->getDim());
  VecSetFromOptions(initcond_re);
  VecDuplicate(initcond_re, &initcond_im);
  VecDuplicate(initcond_re, &initcond_re_bar);
  VecDuplicate(initcond_re, &initcond_im_bar);
  
  if (initcond_type == PURE) { /* Initialize with tensor product of unit vectors. */
    std::vector<std::string> initcondstr;
    config.GetVecStrParam("optim_initialcondition", initcondstr);
    std::vector<int> unitids;
    for (int i=1; i<initcondstr.size(); i++) unitids.push_back(atoi(initcondstr[i].c_str()));
    assert (unitids.size() == primalbraidapp->mastereq->getNOscillators());
    // Compute index of diagonal elements that is one.
    int diag_id = 0.0;
    for (int k=0; k < unitids.size(); k++) {
      assert (unitids[k] < primalbraidapp->mastereq->getOscillator(k)->getNLevels());
      int dim_postkron = 1;
      for (int m=k+1; m < unitids.size(); m++) {
        dim_postkron *= primalbraidapp->mastereq->getOscillator(m)->getNLevels();
      }
      diag_id += unitids[k] * dim_postkron;
    }
    int vec_id = diag_id * (int)sqrt(primalbraidapp->mastereq->getDim()) + diag_id;
    VecSetValue(initcond_re, vec_id, 1.0, INSERT_VALUES);
  }
  else if (initcond_type == FROMFILE) { /* Read initial condition from file */
    int dim = primalbraidapp->mastereq->getDim();
    double * vec = new double[2*dim];

    std::vector<std::string> initcondstr;
    config.GetVecStrParam("optim_initialcondition", initcondstr);
    if (mpirank_world == 0) {
      assert (initcondstr.size()==2);
      std::string filename = initcondstr[1];
      read_vector(filename.c_str(), vec, 2*dim);
    }
    MPI_Bcast(vec, 2*dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < dim; i++) {
      if (vec[i]     != 0.0) VecSetValue(initcond_re, i, vec[i],     INSERT_VALUES);
      if (vec[i+dim] != 0.0) VecSetValue(initcond_im, i, vec[i+dim], INSERT_VALUES);
    }
    delete [] vec;
  }
  VecAssemblyBegin(initcond_re); VecAssemblyEnd(initcond_re);
  VecAssemblyBegin(initcond_im); VecAssemblyEnd(initcond_im);
  VecAssemblyBegin(initcond_re_bar); VecAssemblyEnd(initcond_re_bar);
  VecAssemblyBegin(initcond_im_bar); VecAssemblyEnd(initcond_im_bar);

  /* Reset */
  objective = 0.0;

}


OptimCtx::~OptimCtx() {
  VecDestroy(&initcond_re);
  VecDestroy(&initcond_im);
  VecDestroy(&initcond_re_bar);
  VecDestroy(&initcond_im_bar);
}

void OptimCtx_Setup(OptimCtx* ctx, MapParam config, myBraidApp* primalbraidapp_, myAdjointBraidApp* adjointbraidapp_, MPI_Comm comm_hiop_, MPI_Comm comm_init_, std::vector<int> obj_oscilIDs_, InitialConditionType inittype_, int ninit_){
  ctx->primalbraidapp  = primalbraidapp_;
  ctx->adjointbraidapp = adjointbraidapp_;
  ctx->initcond_type = inittype_;
  ctx->ninit = ninit_;
  ctx->comm_hiop = comm_hiop_;
  ctx->comm_init = comm_init_;
  ctx->obj_oscilIDs = obj_oscilIDs_;

  /* Store ranks and sizes of communicators */
  MPI_Comm_rank(ctx->primalbraidapp->comm_braid, &(ctx->mpirank_braid));
  MPI_Comm_size(ctx->primalbraidapp->comm_braid, &(ctx->mpisize_braid));
  MPI_Comm_rank(MPI_COMM_WORLD, &(ctx->mpirank_world));
  MPI_Comm_size(MPI_COMM_WORLD, &(ctx->mpisize_world));
  MPI_Comm_rank(PETSC_COMM_WORLD, &(ctx->mpirank_space));
  MPI_Comm_size(PETSC_COMM_WORLD, &(ctx->mpisize_space));
  MPI_Comm_rank(ctx->comm_hiop, &(ctx->mpirank_optim));
  MPI_Comm_size(ctx->comm_hiop, &(ctx->mpisize_optim));
  MPI_Comm_rank(ctx->comm_init, &(ctx->mpirank_init));
  MPI_Comm_size(ctx->comm_init, &(ctx->mpisize_init));

  /* Store number of initial conditions per init-processor group */
  ctx->ninit_local = ctx->ninit / ctx->mpisize_init; 

  /* Store number of design parameters */
  int n = 0;
  for (int ioscil = 0; ioscil < ctx->primalbraidapp->mastereq->getNOscillators(); ioscil++) {
      n += ctx->primalbraidapp->mastereq->getOscillator(ioscil)->getNParams(); 
  }
  ctx->ndesign = n;

  /* Store other optimization parameters */
  ctx->gamma_tik = config.GetDoubleParam("optim_regul", 1e-4);

  /* Store objective function type */
  std::vector<std::string> objective_str;
  config.GetVecStrParam("optim_objective", objective_str);
  ctx->targetgate = NULL;
  if ( objective_str[0].compare("gate") ==0 ) {
    ctx->objective_type = GATE;
    /* Read and initialize the targetgate */
    assert ( objective_str.size() >=2 );
    if      (objective_str[1].compare("none") == 0)  ctx->targetgate = new Gate(); // dummy gate. do nothing
    else if (objective_str[1].compare("xgate") == 0) ctx->targetgate = new XGate(); 
    else if (objective_str[1].compare("ygate") == 0) ctx->targetgate = new YGate(); 
    else if (objective_str[1].compare("zgate") == 0) ctx->targetgate = new ZGate();
    else if (objective_str[1].compare("hadamard") == 0) ctx->targetgate = new HadamardGate();
    else if (objective_str[1].compare("cnot") == 0) ctx->targetgate = new CNOT(); 
    else {
      printf("\n\n ERROR: Unnown gate type: %s.\n", objective_str[1].c_str());
      printf(" Available gates are 'none', 'xgate', 'ygate', 'zgate', 'hadamard', 'cnot'\n");
      exit(1);
    }
  }  
  else if (objective_str[0].compare("expectedEnergy")==0) ctx->objective_type = EXPECTEDENERGY;
  else if (objective_str[0].compare("groundstate")   ==0) ctx->objective_type = GROUNDSTATE;
  else {
      printf("\n\n ERROR: Unknown objective function: %s\n", objective_str[0].c_str());
      exit(1);
  }

  /* Set up and store initial condition vectors */
  VecCreate(PETSC_COMM_WORLD, &ctx->initcond_re); 
  VecSetSizes(ctx->initcond_re,PETSC_DECIDE,ctx->primalbraidapp->mastereq->getDim());
  VecSetFromOptions(ctx->initcond_re);
  VecDuplicate(ctx->initcond_re, &ctx->initcond_im);
  VecDuplicate(ctx->initcond_re, &ctx->initcond_re_bar);
  VecDuplicate(ctx->initcond_re, &ctx->initcond_im_bar);
  
  if (ctx->initcond_type == PURE) { /* Initialize with tensor product of unit vectors. */
    std::vector<std::string> initcondstr;
    config.GetVecStrParam("optim_initialcondition", initcondstr);
    std::vector<int> unitids;
    for (int i=1; i<initcondstr.size(); i++) unitids.push_back(atoi(initcondstr[i].c_str()));
    assert (unitids.size() == ctx->primalbraidapp->mastereq->getNOscillators());
    // Compute index of diagonal elements that is one.
    int diag_id = 0.0;
    for (int k=0; k < unitids.size(); k++) {
      assert (unitids[k] < ctx->primalbraidapp->mastereq->getOscillator(k)->getNLevels());
      int dim_postkron = 1;
      for (int m=k+1; m < unitids.size(); m++) {
        dim_postkron *= ctx->primalbraidapp->mastereq->getOscillator(m)->getNLevels();
      }
      diag_id += unitids[k] * dim_postkron;
    }
    int vec_id = diag_id * (int)sqrt(ctx->primalbraidapp->mastereq->getDim()) + diag_id;
    VecSetValue(ctx->initcond_re, vec_id, 1.0, INSERT_VALUES);
  }
  else if (ctx->initcond_type == FROMFILE) { /* Read initial condition from file */
    int dim = ctx->primalbraidapp->mastereq->getDim();
    double * vec = new double[2*dim];

    std::vector<std::string> initcondstr;
    config.GetVecStrParam("optim_initialcondition", initcondstr);
    if (ctx->mpirank_world == 0) {
      assert (initcondstr.size()==2);
      std::string filename = initcondstr[1];
      read_vector(filename.c_str(), vec, 2*dim);
    }
    MPI_Bcast(vec, 2*dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for (int i = 0; i < dim; i++) {
      if (vec[i]     != 0.0) VecSetValue(ctx->initcond_re, i, vec[i],     INSERT_VALUES);
      if (vec[i+dim] != 0.0) VecSetValue(ctx->initcond_im, i, vec[i+dim], INSERT_VALUES);
    }
    delete [] vec;
  }
  VecAssemblyBegin(ctx->initcond_re); VecAssemblyEnd(ctx->initcond_re);
  VecAssemblyBegin(ctx->initcond_im); VecAssemblyEnd(ctx->initcond_im);
  VecAssemblyBegin(ctx->initcond_re_bar); VecAssemblyEnd(ctx->initcond_re_bar);
  VecAssemblyBegin(ctx->initcond_im_bar); VecAssemblyEnd(ctx->initcond_im_bar);

}

void OptimTao_Setup(Tao* tao, OptimCtx* ctx, MapParam config, Vec xinit, Vec xlower, Vec xupper){

  TaoCreate(PETSC_COMM_WORLD, tao);
  TaoSetType(*tao,TAOBLMVM);         // Optim type: taoblmvm vs BQNLS ??
  TaoSetObjectiveRoutine(*tao, optim_evalObjective, (void *)ctx);
  TaoSetGradientRoutine(*tao, optim_evalGradient,(void *)ctx);

  /* Set the optimization bounds */
  std::vector<double> bounds;
  config.GetVecDoubleParam("optim_bounds", bounds, 1e20);
  assert (bounds.size() >= ctx->primalbraidapp->mastereq->getNOscillators());
  for (int iosc = 0; iosc < ctx->primalbraidapp->mastereq->getNOscillators(); iosc++){
    // Scale bounds by number of carrier waves */
    std::vector<double> carrier_freq;
    std::string key = "carrier_frequency" + std::to_string(iosc);
    config.GetVecDoubleParam(key, carrier_freq, 0.0);
    for (int i=0; i<ctx->primalbraidapp->mastereq->getOscillator(iosc)->getNParams(); i++){
      bounds[iosc] = bounds[iosc] / carrier_freq.size();
      VecSetValue(xupper, i, bounds[iosc], INSERT_VALUES);
      VecSetValue(xlower, i, -1. * bounds[iosc], INSERT_VALUES);
    }
  }
  VecAssemblyBegin(xlower); VecAssemblyEnd(xlower);
  VecAssemblyBegin(xupper); VecAssemblyEnd(xupper);
  TaoSetVariableBounds(*tao, xlower, xupper);

  /* Set initial starting point */
  std::string start_type;
  std::vector<double> start_amplitudes;
  start_type = config.GetStrParam("optim_init", "zero");
  if (start_type.compare("constant") == 0 ){ 
    config.GetVecDoubleParam("optim_init_const", start_amplitudes, 0.0);
    assert(start_amplitudes.size() == ctx->primalbraidapp->mastereq->getNOscillators());
  }
  getStartingPoint(xinit, ctx, start_type, start_amplitudes, bounds);
  TaoSetInitialVector(*tao, xinit);

  /* Set runtime options */
  TaoSetFromOptions(*tao);
}


void getStartingPoint(Vec xinit, OptimCtx* ctx, std::string start_type, std::vector<double> start_amplitudes, std::vector<double> bounds){
  MasterEq* mastereq = ctx->primalbraidapp->mastereq;

  if (start_type.compare("constant") == 0 ){ // set constant initial design
    int j = 0;
    for (int ioscil = 0; ioscil < mastereq->getNOscillators(); ioscil++) {
      int nparam = mastereq->getOscillator(ioscil)->getNParams();
      for (int i = 0; i < nparam; i++) {
        VecSetValue(xinit, j, start_amplitudes[ioscil], INSERT_VALUES);
        j++;
      }
    }
  } else if ( start_type.compare("zero") == 0)  { // init design with zero
    VecZeroEntries(xinit);

  } else if ( start_type.compare("random")      == 0 ||       // init random, fixed seed
              start_type.compare("random_seed") == 0)  { // init random with new seed

    /* Create random vector on one processor only, then broadcast to all, so that all have the same initial guess */
    if (ctx->mpirank_world == 0) {

      /* Seed */
      if ( start_type.compare("random") == 0) srand(1);  // fixed seed
      else srand(time(0)); // random seed

      /* Create vector with random elements between [-1:1] */
      double* randvec = new double[ctx->ndesign];
      for (int i=0; i<ctx->ndesign; i++) {
        randvec[i] = (double) rand() / ((double)RAND_MAX);
        randvec[i] = 2.*randvec[i] - 1.;
      }
      /* Broadcast random vector to all */
      MPI_Bcast(randvec, ctx->ndesign, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      /* Trimm back to the box constraints */
      int j = 0;
      for (int ioscil = 0; ioscil < mastereq->getNOscillators(); ioscil++) {
        if (bounds[ioscil] >= 1.0) continue;
        int nparam = mastereq->getOscillator(ioscil)->getNParams();
        for (int i = 0; i < nparam; i++) {
          randvec[j] = randvec[j] * bounds[ioscil];
          j++;
        }
      }

      /* Set the initial guess */
      for (int i=0; i<ctx->ndesign; i++) {
        VecSetValue(xinit, i, randvec[i], INSERT_VALUES);
      }
      delete [] randvec;
    }

  }  else { // Read from file 
    double* vecread = new double[ctx->ndesign];

    if (ctx->mpirank_world == 0) read_vector(start_type.c_str(), vecread, ctx->ndesign);
    MPI_Bcast(vecread, ctx->ndesign, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Set the initial guess */
    for (int i=0; i<ctx->ndesign; i++) {
      VecSetValue(xinit, i, vecread[i], INSERT_VALUES);
    }
    delete [] vecread;
  }

  /* Assemble initial guess */
  VecAssemblyBegin(xinit);
  VecAssemblyEnd(xinit);

  /* Pass to oscillator */
  ctx->primalbraidapp->mastereq->setControlAmplitudes(xinit);
  
  /* Flush initial control functions */
  if (ctx->mpirank_world == 0 ) {
    int ntime = ctx->primalbraidapp->ntime;
    double dt = ctx->primalbraidapp->total_time / ntime;
    char filename[255];
    for (int ioscil = 0; ioscil < mastereq->getNOscillators(); ioscil++) {
        sprintf(filename, "%s/control_init_%02d.dat", ctx->primalbraidapp->datadir.c_str(), ioscil+1);
        mastereq->getOscillator(ioscil)->flushControl(ntime, dt, filename);
    }
  }

}



int optim_assembleInitCond(int iinit_local, OptimCtx* ctx){
  int dim_post;

  int initID = -1;    // Output: ID for this initial condition */
  int dim_rho = (int) sqrt(ctx->primalbraidapp->mastereq->getDim()); // N

  /* Translate local iinit to this processor's domain [rank * ninit_local, (rank+1) * ninit_local - 1] */
  int iinit = ctx->mpirank_init * ctx->ninit_local + iinit_local;

  /* Switch over type of initial condition */
  switch (ctx->initcond_type) {

    case FROMFILE:
      /* Do nothing. Init cond is already stored in initcond_re, initcond_im */
      break;

    case PURE:
      /* Do nothing. Init cond is already stored in initcond_re, initcond_im */
      break;

    case DIAGONAL:
      int row, diagelem;

      /* Reset the initial conditions */
      VecZeroEntries(ctx->initcond_re); 
      VecZeroEntries(ctx->initcond_im); 

      /* Get dimension of partial system behind last oscillator ID */
      dim_post = 1;
      for (int k = ctx->obj_oscilIDs[ctx->obj_oscilIDs.size()-1] + 1; k < ctx->primalbraidapp->mastereq->getNOscillators(); k++) {
        dim_post *= ctx->primalbraidapp->mastereq->getOscillator(k)->getNLevels();
      }

      /* Compute index of the nonzero element in rho_m(0) = E_pre \otimes |m><m| \otimes E_post */
      diagelem = iinit * dim_post;
      // /* Position in vectorized q(0) */
      row = diagelem * dim_rho + diagelem;

      /* Assemble */
      VecSetValue(ctx->initcond_re, row, 1.0, INSERT_VALUES);
      VecAssemblyBegin(ctx->initcond_re);
      VecAssemblyEnd(ctx->initcond_re);

      /* Set initial conditon ID */
      initID = iinit * ( (int) sqrt(ctx->ninit) ) + iinit;

      break;

    case BASIS:

      /* Reset the initial conditions */
      VecZeroEntries(ctx->initcond_re); 
      VecZeroEntries(ctx->initcond_im); 

      /* Get dimension of partial system behind last oscillator ID */
      dim_post = 1;
      for (int k = ctx->obj_oscilIDs[ctx->obj_oscilIDs.size()-1] + 1; k < ctx->primalbraidapp->mastereq->getNOscillators(); k++) {
        dim_post *= ctx->primalbraidapp->mastereq->getOscillator(k)->getNLevels();
      }

      // /* Get index (k,j) of basis element B_{k,j} for this initial condition index iinit */
      int k, j;
      k = iinit % ( (int) sqrt(ctx->ninit) );
      j = (int) iinit / ( (int) sqrt(ctx->ninit) );   

      if (k == j) {
        /* B_{kk} = E_{kk} -> set only one element at (k,k) */
        int elemID = j * dim_post * dim_rho + k * dim_post;
        double val = 1.0;
        VecSetValues(ctx->initcond_re, 1, &elemID, &val, INSERT_VALUES);
        VecAssemblyBegin(ctx->initcond_re);
        VecAssemblyEnd(ctx->initcond_re);
      } else {
      //   /* B_{kj} contains four non-zeros, two per row */
        int* rows = new int[4];
        double* vals = new double[4];

        rows[0] = k * dim_post * dim_rho + k * dim_post; // (k,k)
        rows[1] = j * dim_post * dim_rho + j * dim_post; // (j,j)
        rows[2] = j * dim_post * dim_rho + k * dim_post; // (k,j)
        rows[3] = k * dim_post * dim_rho + j * dim_post; // (j,k)

        if (k < j) { // B_{kj} = 1/2(E_kk + E_jj) + 1/2(E_kj + E_jk)
          vals[0] = 0.5;
          vals[1] = 0.5;
          vals[2] = 0.5;
          vals[3] = 0.5;
          VecSetValues(ctx->initcond_re, 4, rows, vals, INSERT_VALUES);
          VecAssemblyBegin(ctx->initcond_re);
          VecAssemblyEnd(ctx->initcond_re);
        } else {  // B_{kj} = 1/2(E_kk + E_jj) + i/2(E_jk - E_kj)
          vals[0] = 0.5;
          vals[1] = 0.5;
          VecSetValues(ctx->initcond_re, 2, rows, vals, INSERT_VALUES); // diagonal, real
          VecAssemblyBegin(ctx->initcond_re);
          VecAssemblyEnd(ctx->initcond_re);
          vals[0] = -0.5;
          vals[1] = 0.5;
          VecSetValues(ctx->initcond_im, 2, rows+2, vals, INSERT_VALUES); // off-diagonals, imaginary
          VecAssemblyBegin(ctx->initcond_im);
          VecAssemblyEnd(ctx->initcond_im);
        }

        delete [] rows; 
        delete [] vals;
      }

      /* Set initial condition ID */
      initID = j * ( (int) sqrt(ctx->ninit)) + k;

      break;

    default:
      printf("ERROR! Wrong initial condition type: %d\n This should never happen!\n", ctx->initcond_type);
      exit(1);
  }

  // printf("InitCond %d \n", iinit);
  // VecView(initcond_re, PETSC_VIEWER_STDOUT_WORLD);
  // VecView(initcond_im, PETSC_VIEWER_STDOUT_WORLD);
  
  return initID;
}


PetscErrorCode optim_evalObjective(Tao tao, Vec x, PetscReal *f, void*ptr){
  OptimCtx* ctx = (OptimCtx*) ptr;
  MasterEq* mastereq = ctx->primalbraidapp->mastereq;

  if (ctx->mpirank_world == 0) printf(" EVAL F... \n");
  Vec finalstate = NULL;

  /* Pass design vector x to oscillators */
  mastereq->setControlAmplitudes(x); 

  /*  Iterate over initial condition */
  double objective = 0.0;
  for (int iinit = 0; iinit < ctx->ninit_local; iinit++) {
      
    /* Prepare the initial condition */
    int initid = optim_assembleInitCond(iinit, ctx);
    // if (ctx->mpirank_braid == 0) printf("%d: %d FWD. \n", ctx->mpirank_init, initid);

    /* Run forward with initial condition initid*/
    ctx->primalbraidapp->PreProcess(initid);
    ctx->primalbraidapp->setInitialCondition(ctx->initcond_re, ctx->initcond_im);
    ctx->primalbraidapp->Drive();
    finalstate = ctx->primalbraidapp->PostProcess(); // this return NULL for all but the last time processor

    /* Add to objective function */
    objective += optim_objectiveT(finalstate, ctx);
      // if (mpirank_braid == 0) printf("%d: local objective: %1.14e\n", mpirank_init, obj_local);
  }

  /* Broadcast objective from last to all time processors */
  MPI_Bcast(&objective, 1, MPI_DOUBLE, ctx->mpisize_braid-1, ctx->primalbraidapp->comm_braid);

  /* Sum up objective from all initial conditions */
  double myobj = objective;
  MPI_Allreduce(&myobj, &objective, 1, MPI_DOUBLE, MPI_SUM, ctx->comm_init);
  // if (ctx->mpirank_init == 0) printf("%d: global sum objective: %1.14e\n\n", ctx->mpirank_init, objective);

  /* Add regularization objective += gamma/2 * ||x||^2*/
  double xnorm;
  VecNorm(x, NORM_2, &xnorm);
  objective += ctx->gamma_tik / 2. * pow(xnorm,2.0);

  /* Store and return objective value */
  ctx->objective = objective;
  *f = objective;

  return 0;
}


PetscErrorCode optim_evalGradient(Tao tao, Vec x, Vec G, void*ptr){
  OptimCtx* ctx = (OptimCtx*) ptr;
  MasterEq* mastereq = ctx->primalbraidapp->mastereq;

  if (ctx->mpirank_world == 0) printf(" EVAL GRAD F...\n");
  Vec finalstate = NULL;

  /* Pass design vector x to oscillators */
  mastereq->setControlAmplitudes(x); 

  /*  Iterate over initial condition */
  double objective = 0.0;
  for (int iinit = 0; iinit < ctx->ninit_local; iinit++) {
      
    /* Prepare the initial condition */
    int initid = optim_assembleInitCond(iinit, ctx);

    /* --- Solve primal --- */
    // if (ctx->mpirank_braid == 0) printf("%d: %d FWD. ", ctx->mpirank_init, initid);

    /* Run forward with initial condition initid*/
    ctx->primalbraidapp->PreProcess(initid);
    ctx->primalbraidapp->setInitialCondition(ctx->initcond_re, ctx->initcond_im);
    ctx->primalbraidapp->Drive();
    finalstate = ctx->primalbraidapp->PostProcess(); // this return NULL for all but the last time processor

    /* Add final-time objective */
    double obj_local = optim_objectiveT(finalstate, ctx);
    objective += obj_local;
      // if (ctx->mpirank_braid == 0) printf("%d: local objective: %1.14e\n", ctx->mpirank_init, obj_local);

    /* --- Solve adjoint --- */
    // if (ctx->mpirank_braid == 0) printf("%d: %d BWD.", ctx->mpirank_init, initid);

    /* Derivative of final time objective */
    optim_objectiveT_diff(finalstate, obj_local, 1.0, ctx);

    ctx->adjointbraidapp->PreProcess(initid);
    ctx->adjointbraidapp->setInitialCondition(ctx->initcond_re_bar, ctx->initcond_im_bar);
    ctx->adjointbraidapp->Drive();
    ctx->adjointbraidapp->PostProcess();

    /* Add to Ipopt's gradient */
    const double* grad_ptr = ctx->adjointbraidapp->getReducedGradientPtr();
    VecAXPY(G, 1.0, ctx->adjointbraidapp->redgrad);

  }

  /* Broadcast objective from last to all time processors */
  MPI_Bcast(&objective, 1, MPI_DOUBLE, ctx->mpisize_braid-1, ctx->primalbraidapp->comm_braid);

  /* Sum up objective from all initial conditions */
  double myobj = objective;
  MPI_Allreduce(&myobj, &objective, 1, MPI_DOUBLE, MPI_SUM, ctx->comm_init);
  // if (ctx->mpirank_init == 0) printf("%d: global sum objective: %1.14e\n\n", ctx->mpirank_init, objective);

  /* Add regularization objective += gamma/2 * ||x||^2*/
  double xnorm;
  VecNorm(x, NORM_2, &xnorm);
  objective += ctx->gamma_tik / 2. * pow(xnorm,2.0);

  /* Sum up the gradient from all braid processors */
  PetscScalar* grad; 
  VecGetArray(G, &grad);
  double* mygrad = new double[ctx->ndesign];
  for (int i=0; i<ctx->ndesign; i++) {
    mygrad[i] = grad[i];
  }
  MPI_Allreduce(mygrad, grad, ctx->ndesign, MPI_DOUBLE, MPI_SUM, ctx->primalbraidapp->comm_braid);
  VecRestoreArray(G, &grad);
  delete [] mygrad;


  /* Compute and store gradient norm */
  double gradnorm = 0.0;
  VecNorm(G, NORM_2, &gradnorm);
  if (ctx->mpirank_world == 0) printf("%d: ||grad|| = %1.14e\n", ctx->mpirank_init, gradnorm);


  return 0;
}


double optim_objectiveT(Vec finalstate, OptimCtx* ctx){
  double obj_local = 0.0;

  if (finalstate != NULL) {

    switch (ctx->objective_type) {
      case GATE:
        /* compare state to linear transformation of initial conditions */
        ctx->targetgate->compare(finalstate, ctx->initcond_re, ctx->initcond_im, obj_local);
        break;

      case EXPECTEDENERGY:
        /* compute the expected value of energy levels for each oscillator */
        obj_local = 0.0;
        for (int i=0; i<ctx->obj_oscilIDs.size(); i++) {
          obj_local += ctx->primalbraidapp->mastereq->getOscillator(ctx->obj_oscilIDs[i])->expectedEnergy(finalstate);
        }
        obj_local = pow(obj_local, 2.0);
        break;

      case GROUNDSTATE:
        /* compare full or pariatl state to groundstate */
        MasterEq *meq= ctx->primalbraidapp->mastereq;
        Vec state;

        /* If sub-system is requested, compute reduced density operator first */
        if (ctx->obj_oscilIDs.size() < meq->getNOscillators()) { 
          
          /* Get dimensions of preceding and following subsystem */
          int dim_pre  = 1; 
          int dim_post = 1;
          for (int iosc = 0; iosc < meq->getNOscillators(); iosc++) {
            if ( iosc < ctx->obj_oscilIDs[0])                      
              dim_pre  *= meq->getOscillator(iosc)->getNLevels();
            if ( iosc > ctx->obj_oscilIDs[ctx->obj_oscilIDs.size()-1])   
              dim_post *= meq->getOscillator(iosc)->getNLevels();
          }

          /* Create reduced density matrix */
          int dim_reduced = 1;
          for (int i = 0; i < ctx->obj_oscilIDs.size();i++) {
            dim_reduced *= meq->getOscillator(ctx->obj_oscilIDs[i])->getNLevels();
          }
          VecCreate(PETSC_COMM_WORLD, &state);
          VecSetSizes(state, PETSC_DECIDE, 2*dim_reduced*dim_reduced);
          VecSetFromOptions(state);

          /* Fill reduced density matrix */
          meq->reducedDensity(finalstate, &state, dim_pre, dim_post, dim_reduced);

        } else { // full density matrix system 

           state = finalstate; 

        }

        // PetscViewerSetFormat(PETSC_VIEWER_STDOUT_WORLD, 	PETSC_VIEWER_ASCII_MATLAB );
        // VecView(state, PETSC_VIEWER_STDOUT_WORLD);

        /* Compute frobenius norm: frob = || q(T) - e_1 ||^2 */
        int dimstate;
        const PetscScalar *stateptr;
        VecGetSize(state, &dimstate);
        VecGetArrayRead(state, &stateptr);
        obj_local = 0.0;
        obj_local += pow(stateptr[0] - 1.0, 2); 
        for (int i = 1; i < dimstate; i++){
           obj_local += pow(stateptr[i], 2);
        }
        VecRestoreArrayRead(state, &stateptr);

        /* Destroy reduced density matrix, if it has been created */
        if (ctx->obj_oscilIDs.size() < ctx->primalbraidapp->mastereq->getNOscillators()) { 
          VecDestroy(&state);
        }

        break;
    }

  }

  return obj_local;
}


void optim_objectiveT_diff(Vec finalstate, double obj, double obj_bar, OptimCtx *ctx){
  /* Reset adjoints */
  VecZeroEntries(ctx->initcond_re_bar);
  VecZeroEntries(ctx->initcond_im_bar);

  if (finalstate != NULL) {
    switch (ctx->objective_type) {
      case GATE:
        ctx->targetgate->compare_diff(finalstate, ctx->initcond_re, ctx->initcond_im, ctx->initcond_re_bar, ctx->initcond_im_bar, obj_bar);
        break;

      case EXPECTEDENERGY:
        double tmp;
        tmp = 2. * sqrt(obj) * obj_bar;
        // tmp = obj_bar;
        for (int i=0; i<ctx->obj_oscilIDs.size(); i++) {
          ctx->primalbraidapp->mastereq->getOscillator(ctx->obj_oscilIDs[i])->expectedEnergy_diff(finalstate, ctx->initcond_re_bar, ctx->initcond_im_bar, tmp);
        }
        break;

    case GROUNDSTATE:

        MasterEq *meq= ctx->primalbraidapp->mastereq;
        Vec state;
        int dim_pre = 1;
        int dim_post = 1;
        int dim_reduced = 1;

        /* If sub-system is requested, compute reduced density operator first */
        if (ctx->obj_oscilIDs.size() < ctx->primalbraidapp->mastereq->getNOscillators()) { 
          
          /* Get dimensions of preceding and following subsystem */
          for (int iosc = 0; iosc < meq->getNOscillators(); iosc++) {
            if ( iosc < ctx->obj_oscilIDs[0])                      
              dim_pre  *= meq->getOscillator(iosc)->getNLevels();
            if ( iosc > ctx->obj_oscilIDs[ctx->obj_oscilIDs.size()-1])   
              dim_post *= meq->getOscillator(iosc)->getNLevels();
          }

          /* Create reduced density matrix */
          for (int i = 0; i < ctx->obj_oscilIDs.size();i++) {
            dim_reduced *= meq->getOscillator(ctx->obj_oscilIDs[i])->getNLevels();
          }
          VecCreate(PETSC_COMM_WORLD, &state);
          VecSetSizes(state, PETSC_DECIDE, 2*dim_reduced*dim_reduced);
          VecSetFromOptions(state);

          /* Fill reduced density matrix */
          meq->reducedDensity(finalstate, &state, dim_pre, dim_post, dim_reduced);

        } else { // full density matrix system 

           state = finalstate; 

        }

       /* Get real and imag part of final and initial primal and adjoint states, x = [u,v] */
      const PetscScalar *stateptr;
      PetscScalar *statebarptr;
      Vec statebar;
      VecDuplicate(state, &statebar);
      VecGetArrayRead(state, &stateptr);
      VecGetArray(statebar, &statebarptr);

      /* Derivative of frobenius norm: 2 * (q(T) - e_1) * frob_bar */
      int dimstate;
      VecGetSize(state, &dimstate);
      statebarptr[0] += 2. * ( stateptr[0] - 1.0 ) * obj_bar;
      for (int i=1; i<dimstate; i++) {
        statebarptr[i] += 2. * stateptr[i] * obj_bar;
      }
      VecRestoreArrayRead(state, &stateptr);

      /* Derivative of partial trace */
      if (ctx->obj_oscilIDs.size() < meq->getNOscillators()) {
        meq->reducedDensity_diff(statebar, ctx->initcond_re_bar, ctx->initcond_im_bar, dim_pre, dim_post, dim_reduced);
        VecDestroy(&state);
      } else {
        /* Split statebar into initcond_rebar, initcond_im_bar */
        PetscScalar *u0_barptr, *v0_barptr;
        VecGetArray(ctx->initcond_re_bar, &u0_barptr);
        VecGetArray(ctx->initcond_im_bar, &v0_barptr);
        const PetscScalar *statebarptr;
        VecGetArrayRead(statebar, &statebarptr);
        for (int i=0; i<dimstate/2; i++){
          u0_barptr[i] += statebarptr[i];
          v0_barptr[i] += statebarptr[i + dimstate/2];
        }
        VecRestoreArrayRead(statebar, &statebarptr);
        VecRestoreArray(ctx->initcond_re_bar, &u0_barptr);
        VecRestoreArray(ctx->initcond_im_bar, &v0_barptr);
      }

      VecDestroy(&statebar);
    }
  }
}