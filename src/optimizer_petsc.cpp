#include "optimizer_petsc.hpp"


void optim_CtxSetup(OptimCtx* ctx, MapParam config, myBraidApp* primalbraidapp_, myAdjointBraidApp* adjointbraidapp_, MPI_Comm comm_hiop_, MPI_Comm comm_init_, std::vector<int> obj_oscilIDs_, InitialConditionType inittype_, int ninit_){
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
}

void optim_TaoSetup(Tao* tao, OptimCtx* ctx, MapParam config, Vec xinit, Vec xlower, Vec xupper){

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

PetscErrorCode optim_evalObjective(Tao tao, Vec x, PetscReal *f, void*ptr){
    OptimCtx* optimctx = (OptimCtx*) ptr;

    /* TODO: Eval objective */

    return 0;
}


PetscErrorCode optim_evalGradient(Tao tao, Vec x, Vec G, void*ptr){
    OptimCtx* optimctx = (OptimCtx*) ptr;

    /* TODO: Eval objective */

    return 0;
}

