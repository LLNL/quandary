#include "optimproblem.hpp"

OptimProblem::OptimProblem(MapParam config, TimeStepper* timestepper_, MPI_Comm comm_init_, MPI_Comm comm_optim_, int ninit_, Output* output_, bool quietmode_){

  timestepper = timestepper_;
  ninit = ninit_;
  output = output_;
  quietmode = quietmode_;
  /* Reset */
  objective = 0.0;

  /* Store communicators */
  comm_init = comm_init_;
  comm_optim = comm_optim_;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize_world);
  MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank_space);
  MPI_Comm_size(PETSC_COMM_WORLD, &mpisize_space);
  MPI_Comm_rank(comm_init, &mpirank_init);
  MPI_Comm_size(comm_init, &mpisize_init);
  MPI_Comm_rank(comm_optim, &mpirank_optim);
  MPI_Comm_size(comm_optim, &mpisize_optim);

  /* Store number of initial conditions per init-processor group */
  ninit_local = ninit / mpisize_init; 

  /*  If Schroedingers solver, allocate storage for the final states at time T for each initial condition. Schroedinger's solver does not store the time-trajectories during forward ODE solve, but instead recomputes the primal states during the adjoint solve. Therefore we need to store the terminal condition for the backwards primal solve. Be aware that the final states stored here will be overwritten during backwards computation!! */
  if (timestepper->mastereq->lindbladtype == LindbladType::NONE) {
    for (int i = 0; i < ninit_local; i++) {
      Vec state;
      VecCreate(PETSC_COMM_WORLD, &state);
      VecSetSizes(state, PETSC_DECIDE, 2*timestepper->mastereq->getDim());
      VecSetFromOptions(state);
      store_finalstates.push_back(state);
    }
  }

  /* Store number of design parameters */
  int n = 0;
  for (int ioscil = 0; ioscil < timestepper->mastereq->getNOscillators(); ioscil++) {
      n += timestepper->mastereq->getOscillator(ioscil)->getNParams(); 
  }
  ndesign = n;
  if (mpirank_world == 0 && !quietmode) std::cout<< "Number of control parameters: " << ndesign << std::endl;

  /* Allocate the initial condition vector and adjoint terminal state */
  VecCreate(PETSC_COMM_WORLD, &rho_t0); 
  VecSetSizes(rho_t0,PETSC_DECIDE,2*timestepper->mastereq->getDim());
  VecSetFromOptions(rho_t0);
  VecZeroEntries(rho_t0);
  VecAssemblyBegin(rho_t0); VecAssemblyEnd(rho_t0);
  VecDuplicate(rho_t0, &rho_t0_bar);
  VecZeroEntries(rho_t0_bar);
  VecAssemblyBegin(rho_t0_bar); VecAssemblyEnd(rho_t0_bar);

  /* Initialize the optimization target, including setting of initial state rho_t0 if read from file or pure state or ensemble */
  std::vector<std::string> target_str;
  config.GetVecStrParam("optim_target", target_str, "pure");
  std::string objective_str = config.GetStrParam("optim_objective", "Jfrobenius");
  std::vector<double> read_gate_rot;
  config.GetVecDoubleParam("gate_rot_freq", read_gate_rot, 1e20); 
  std::vector<std::string> initcond_str;
  config.GetVecStrParam("initialcondition", initcond_str, "none", false);
  optim_target = new OptimTarget(target_str, objective_str, initcond_str, timestepper->mastereq, timestepper->total_time, read_gate_rot, rho_t0, quietmode);

  /* Get weights for the objective function (weighting the different initial conditions. Default: 1/ninit */
  config.GetVecDoubleParam("optim_weights", obj_weights, 1.0);
  copyLast(obj_weights, ninit);
  // Scale the weights such that they sum up to one: beta_i <- beta_i / (\sum_i beta_i)
  double scaleweights = 0.0;
  for (int i=0; i<ninit; i++) scaleweights += obj_weights[i];
  for (int i=0; i<ninit; i++) obj_weights[i] = obj_weights[i] / scaleweights;
  // Distribute over mpi_init processes 
  double sendbuf[obj_weights.size()];
  double recvbuf[obj_weights.size()];
  for (int i = 0; i < obj_weights.size(); i++) sendbuf[i] = obj_weights[i];
  for (int i = 0; i < obj_weights.size(); i++) recvbuf[i] = obj_weights[i];
  int nscatter = ninit_local;
  MPI_Scatter(sendbuf, nscatter, MPI_DOUBLE, recvbuf, nscatter,  MPI_DOUBLE, 0, comm_init);
  for (int i = 0; i < nscatter; i++) obj_weights[i] = recvbuf[i];
  for (int i=nscatter; i < obj_weights.size(); i++) obj_weights[i] = 0.0;


  /* Store other optimization parameters */
  gamma_tik = config.GetDoubleParam("optim_regul", 1e-4);
  gatol = config.GetDoubleParam("optim_atol", 1e-8);
  fatol = config.GetDoubleParam("optim_ftol", 1e-8);
  inftol = config.GetDoubleParam("optim_inftol", 1e-5);
  grtol = config.GetDoubleParam("optim_rtol", 1e-4);
  maxiter = config.GetIntParam("optim_maxiter", 200);
  gamma_penalty = config.GetDoubleParam("optim_penalty", 0.0);
  penalty_param = config.GetDoubleParam("optim_penalty_param", 0.5);
  gamma_penalty_energy = config.GetDoubleParam("optim_penalty_energy", 0.0);
  gamma_tik_interpolate = config.GetBoolParam("optim_regul_interpolate", false, false);
  gamma_penalty_dpdm = config.GetDoubleParam("optim_penalty_dpdm", 0.0);

  if (gamma_penalty_dpdm > 1e-13 && timestepper->mastereq->lindbladtype != LindbladType::NONE){
    if (mpirank_world == 0) {
      printf("Warning: Disabling DpDm penalty term because it is not implemented for the Lindblad solver.\n");
    }
    gamma_penalty_dpdm = 0.0;
  }

  /* Pass information on objective function to the time stepper needed for penalty objective function */
  timestepper->penalty_param = penalty_param;
  timestepper->gamma_penalty = gamma_penalty;
  timestepper->gamma_penalty_dpdm = gamma_penalty_dpdm;
  timestepper->gamma_penalty_energy = gamma_penalty_energy;
  timestepper->optim_target = optim_target;

  /* Store optimization bounds */
  VecCreateSeq(PETSC_COMM_SELF, ndesign, &xlower);
  VecSetFromOptions(xlower);
  VecDuplicate(xlower, &xupper);
  int col = 0;
  for (int iosc = 0; iosc < timestepper->mastereq->getNOscillators(); iosc++){
    std::vector<std::string> bound_str;
    config.GetVecStrParam("control_bounds" + std::to_string(iosc), bound_str, "10000.0");
    for (int iseg = 0; iseg < timestepper->mastereq->getOscillator(iosc)->getNSegments(); iseg++){
      double boundval = 0.0;
      if (bound_str.size() <= iseg) boundval =  atof(bound_str[bound_str.size()-1].c_str());
      else boundval = atof(bound_str[iseg].c_str());
      // Scale bounds by the number of carrier waves, and convert to radians */
      boundval = boundval / (sqrt(2) * timestepper->mastereq->getOscillator(iosc)->getNCarrierfrequencies());
      boundval = boundval * 2.0*M_PI;
      for (int i=0; i<timestepper->mastereq->getOscillator(iosc)->getNSegParams(iseg); i++){
        VecSetValue(xupper, col + i, boundval, INSERT_VALUES);
        VecSetValue(xlower, col + i, -1. * boundval, INSERT_VALUES);
      }
      // Disable bound for phase if this is spline_amplitude control
      if (timestepper->mastereq->getOscillator(iosc)->getControlType() == ControlType::BSPLINEAMP) {
        for (int f = 0; f < timestepper->mastereq->getOscillator(iosc)->getNCarrierfrequencies(); f++){
          int nsplines = timestepper->mastereq->getOscillator(iosc)->getNSplines();
          boundval = 1e+10;
          VecSetValue(xupper, col + f*(nsplines+1) + nsplines, boundval, INSERT_VALUES);
          VecSetValue(xlower, col + f*(nsplines+1) + nsplines, -1.*boundval, INSERT_VALUES);
        }
      }
      col = col + timestepper->mastereq->getOscillator(iosc)->getNSegParams(iseg);
    }
  }
  VecAssemblyBegin(xlower); VecAssemblyEnd(xlower);
  VecAssemblyBegin(xupper); VecAssemblyEnd(xupper);

  /* Store the initial guess if read from file */
  std::vector<std::string> controlinit_str;
  config.GetVecStrParam("control_initialization0", controlinit_str, "constant, 0.0");
  if ( controlinit_str.size() > 0 && controlinit_str[0].compare("file") == 0 ) {
    assert(controlinit_str.size() >=2);
    for (int i=0; i<ndesign; i++) initguess_fromfile.push_back(0.0);
    if (mpirank_world == 0) read_vector(controlinit_str[1].c_str(), initguess_fromfile.data(), ndesign, quietmode);
    MPI_Bcast(initguess_fromfile.data(), ndesign, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
 
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
  TaoSetObjective(tao, TaoEvalObjective, (void *)this);
  TaoSetGradient(tao, NULL, TaoEvalGradient,(void *)this);
  TaoSetObjectiveAndGradient(tao, NULL, TaoEvalObjectiveAndGradient, (void*) this);

  /* Allocate auxiliary vector */
  mygrad = new double[ndesign];

  /* Allocat xinit, xtmp */
  VecCreateSeq(PETSC_COMM_SELF, ndesign, &xinit);
  VecSetFromOptions(xinit);
  VecZeroEntries(xinit);
  VecCreateSeq(PETSC_COMM_SELF, ndesign, &xtmp);
  VecSetFromOptions(xtmp);
  VecZeroEntries(xtmp);
}


OptimProblem::~OptimProblem() {
  delete [] mygrad;
  delete optim_target;
  VecDestroy(&rho_t0);
  VecDestroy(&rho_t0_bar);
  ic_seed.clear();

  VecDestroy(&xlower);
  VecDestroy(&xupper);
  VecDestroy(&xinit);
  VecDestroy(&xtmp);

  for (int i = 0; i < store_finalstates.size(); i++) {
    VecDestroy(&(store_finalstates[i]));
  }

  TaoDestroy(&tao);
}



double OptimProblem::evalTikhonov_(const double* x, int ndesign){
  double xnorm = 0.0;
  if (!gamma_tik_interpolate){  // ||x||^2
    for (int i=0; i<ndesign; i++){
      xnorm += pow(x[i],2.0);
    }
  } else {
    printf("Tikhonov interpolate disabled currently.\n");
    exit(1);
  //   VecCopy(x, xtmp);
  //   VecAXPY(xtmp, -1.0, xinit);    // xtmp =  x - x_0
  //   VecNorm(xtmp, NORM_2, &xnorm);
  }
  return 0.5 * xnorm;
}

void OptimProblem::evalTikhonov_diff_(const double* x, double* xbar, int ndesign, double factor){
  if (!gamma_tik_interpolate){  // ||x||^2
    for (int i=0; i<ndesign; i++){
      xbar[i] += x[i]*factor;
    }
  } else {
    printf("Tikhonov interpolate disabled currently.\n");
    exit(1);
  }
}

// Petsc wrapper for evalF
double OptimProblem::evalF(const Vec x) {
    const PetscScalar* x_ptr;
    VecGetArrayRead(x, &x_ptr);
    double F = evalF_(x_ptr, 0, ninit_local);
    VecRestoreArrayRead(x, &x_ptr);
    return F;
 }

// Evaluate F(x)
double OptimProblem::evalF_(const double* x, const size_t i, const size_t ninit_local) {

  MasterEq* mastereq = timestepper->mastereq;

  if (mpirank_world == 0 && !quietmode) printf("EVAL F... \n");

  /* Pass design vector x to oscillators */
  mastereq->setControlAmplitudes_(x); 

  /*  Iterate over initial condition */
  obj_cost  = 0.0;
  obj_regul = 0.0;
  obj_penal = 0.0;
  obj_penal_dpdm = 0.0;
  obj_penal_energy = 0.0;
  fidelity = 0.0;
  double obj_cost_re = 0.0;
  double obj_cost_im = 0.0;
  double fidelity_re = 0.0;
  double fidelity_im = 0.0;
  for (int iinit = i; iinit < i+ninit_local; iinit++) {
      
    /* Prepare the initial condition in [rank * ninit_local, ... , (rank+1) * ninit_local - 1] */
    int iinit_global = mpirank_init * ninit_local + iinit;
    // FOR STOCHASTIC OPTIM: Pass the random number generator seed for this initial condition
    if (ic_seed.size() > 0 ) {
      iinit_global = ic_seed[iinit];
    }
    int initid = optim_target->prepareInitialState(iinit_global, ninit, timestepper->mastereq->nlevels, timestepper->mastereq->nessential, rho_t0);
    if (mpirank_optim == 0 && !quietmode) printf("%d: iinit=%d, i=%zu, ninit_local=%zu. Initial condition id=%d ...\n", mpirank_init, iinit, i, ninit_local, initid);

    /* If gate optimiztion, compute the target state rho^target = Vrho(0)V^dagger */
    optim_target->prepareTargetState(rho_t0);

    /* Run forward with initial condition initid */
    Vec finalstate = timestepper->solveODE(initid, rho_t0);

    /* Add to integral penalty term */
    obj_penal += obj_weights[iinit-i] * gamma_penalty * timestepper->penalty_integral;

    /* Add to second derivative penalty term */
    obj_penal_dpdm += obj_weights[iinit-i] * gamma_penalty_dpdm * timestepper->penalty_dpdm;
    
    /* Add to energy integral penalty term */
    obj_penal_energy += obj_weights[iinit-i] * gamma_penalty_energy* timestepper->energy_penalty_integral;

    /* Evaluate J(finalstate) and add to final-time cost */
    double obj_iinit_re = 0.0;
    double obj_iinit_im = 0.0;
    optim_target->evalJ(finalstate,  &obj_iinit_re, &obj_iinit_im);
    obj_cost_re += obj_weights[iinit-i] * obj_iinit_re;
    obj_cost_im += obj_weights[iinit-i] * obj_iinit_im;

    // If stochastic optimizer, take the absolute value here and add to objective function
    if (optim_target->getInitCondType() == InitialConditionType::RANDOM) {
      obj_cost += obj_weights[iinit-i]*(1.0 - (pow(obj_iinit_re,2.0) + pow(obj_iinit_im, 2.0)));
    }

    /* Add to final-time fidelity */
    double fidelity_iinit_re = 0.0;
    double fidelity_iinit_im = 0.0;
    optim_target->HilbertSchmidtOverlap(finalstate, false, &fidelity_iinit_re, &fidelity_iinit_im);
    double scaling = 1./ninit;
    if (optim_target->getInitCondType() == InitialConditionType::RANDOM) {
      scaling = 1.0;
    }
    fidelity_re += scaling * fidelity_iinit_re;
    fidelity_im += scaling * fidelity_iinit_im;

    // printf("%d, %d: iinit obj_iinit: %f * (%1.14e + i %1.14e), obj_cost=%1.14e, Overlap=%1.14e + i %1.14e\n", mpirank_world, mpirank_init, obj_weights[iinit-i], obj_iinit_re, obj_iinit_im, obj_cost, fidelity_iinit_re, fidelity_iinit_im);
  }

  /* Sum up from initial conditions processors */
  double mypen = obj_penal;
  double mypen_dpdm = obj_penal_dpdm;
  double mypenen = obj_penal_energy;
  double mycost_re = obj_cost_re;
  double mycost_im = obj_cost_im;
  double mycost = obj_cost;
  double myfidelity_re = fidelity_re;
  double myfidelity_im = fidelity_im;
  MPI_Allreduce(&mypen, &obj_penal, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mypen_dpdm, &obj_penal_dpdm, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mypenen, &obj_penal_energy, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mycost_re, &obj_cost_re, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mycost_im, &obj_cost_im, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mycost, &obj_cost, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&myfidelity_re, &fidelity_re, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&myfidelity_im, &fidelity_im, 1, MPI_DOUBLE, MPI_SUM, comm_init);

  /* Set the fidelity: If Schroedinger, need to compute the absolute value: Fid= |\sum_i \phi^\dagger \phi_target|^2 */
  if (timestepper->mastereq->lindbladtype == LindbladType::NONE) {
    fidelity = pow(fidelity_re, 2.0) + pow(fidelity_im, 2.0);
  } else {
    fidelity = fidelity_re; 
  }
 
  /* Finalize the objective function */
  if (optim_target->getInitCondType() != InitialConditionType::RANDOM) {
    obj_cost = optim_target->finalizeJ(obj_cost_re, obj_cost_im);
  }

  /* Evaluate regularization objective += gamma/2 * ||x-x0||^2*/
  obj_regul = gamma_tik * evalTikhonov_(x, ndesign);

  /* Sum, store and return objective value */
  objective = obj_cost + obj_regul + obj_penal + obj_penal_dpdm + obj_penal_energy;

  /* Output */
  if (mpirank_world == 0) {
    std::cout<< "Objective = " << std::scientific<<std::setprecision(14) << obj_cost << " + " << obj_regul << " + " << obj_penal << " + " << obj_penal_dpdm << " + " << obj_penal_energy << std::endl;
    std::cout<< "Fidelity = " << fidelity  << std::endl;
  }

  return objective;
}


// Petsc wrapper for evalGradF_
double OptimProblem::evalGradF(const Vec x, Vec G){
  const PetscScalar* x_ptr;
  PetscScalar* G_ptr;
  VecGetArrayRead(x, &x_ptr);
  VecGetArray(G, &G_ptr);
  double F = evalGradF_(x_ptr, 0, G_ptr, ninit_local);
  VecRestoreArrayRead(x, &x_ptr);
  VecRestoreArray(G, &G_ptr);
  return F;
}

/* Evaluate the gradient F'(x), and return F(x) */
double OptimProblem::evalGradF_(const double* x, const size_t i, double* G, const size_t ninit_local){

  MasterEq* mastereq = timestepper->mastereq;

  if (mpirank_world == 0 && !quietmode) std::cout<< "EVAL GRAD F... " << std::endl;

  /* Pass design vector x to oscillators */
  mastereq->setControlAmplitudes_(x); 

  /* Reset Gradient */
  for (int i=0; i<ndesign; i++){
    G[i] = 0.0;
  }

  /* Derivative of regulatization term gamma / 2 ||x||^2 (ADD ON ONE PROC ONLY!) */
  if (mpirank_init == 0 ) {
    evalTikhonov_diff_(x, G, ndesign, gamma_tik); // G += gamma_tik * x
  }

  /*  Iterate over initial condition */
  obj_cost = 0.0;
  obj_regul = 0.0;
  obj_penal = 0.0;
  obj_penal_dpdm = 0.0;
  obj_penal_energy = 0.0;
  fidelity = 0.0;
  double obj_cost_re = 0.0;
  double obj_cost_im = 0.0;
  double fidelity_re = 0.0;
  double fidelity_im = 0.0;
  for (int iinit = i; iinit < i+ninit_local; iinit++) {


    /* Prepare the initial condition */
    int iinit_global = mpirank_init * ninit_local + iinit;
    // FOR STOCHASTIC OPTIM: Pass the random number generator seed for this initial condition
    if (ic_seed.size() > 0 ) {
      iinit_global = ic_seed[iinit];
    }
    int initid = optim_target->prepareInitialState(iinit_global, ninit, timestepper->mastereq->nlevels, timestepper->mastereq->nessential, rho_t0);
    if (mpirank_optim == 0 && !quietmode) printf("%d: iinit=%d, i=%zu, ninit_local=%zu. Initial condition id=%d ...\n", mpirank_init, iinit, i, ninit_local, initid);

    /* If gate optimiztion, compute the target state rho^target = Vrho(0)V^dagger */
    optim_target->prepareTargetState(rho_t0);

    /* --- Solve primal --- */
    // if (mpirank_optim == 0) printf("%d: %d FWD. ", mpirank_init, initid);

    /* Run forward with initial condition rho_t0 */
    Vec finalstate = timestepper->solveODE(initid, rho_t0);

    /* Store the final state for the Schroedinger solver */
    if (timestepper->mastereq->lindbladtype == LindbladType::NONE) VecCopy(finalstate, store_finalstates[iinit-i]);

    /* Add to integral penalty term */
    obj_penal += obj_weights[iinit-i] * gamma_penalty * timestepper->penalty_integral;

    /* Add to second derivative dpdm integral penalty term */
    obj_penal_dpdm += obj_weights[iinit-i] * gamma_penalty_dpdm * timestepper->penalty_dpdm;
    /* Add to energy integral penalty term */
    obj_penal_energy += obj_weights[iinit-i] * gamma_penalty_energy * timestepper->energy_penalty_integral;

    /* Evaluate J(finalstate) and add to final-time cost */
    double obj_iinit_re = 0.0;
    double obj_iinit_im = 0.0;
    optim_target->evalJ(finalstate,  &obj_iinit_re, &obj_iinit_im);
    obj_cost_re += obj_weights[iinit-i] * obj_iinit_re;
    obj_cost_im += obj_weights[iinit-i] * obj_iinit_im;

    // If stochastic optimizer, take the absolute value here and add to objective function
    if (optim_target->getInitCondType() == InitialConditionType::RANDOM) {
      obj_cost += obj_weights[iinit-i] * (1.0 - (pow(obj_iinit_re,2.0) + pow(obj_iinit_im, 2.0)));
    }
    /* Add to final-time fidelity */
    double fidelity_iinit_re = 0.0;
    double fidelity_iinit_im = 0.0;
    optim_target->HilbertSchmidtOverlap(finalstate, false, &fidelity_iinit_re, &fidelity_iinit_im);
    double scaling = 1./ninit;
    if (optim_target->getInitCondType() == InitialConditionType::RANDOM) {
      scaling = 1.0;
    }
    fidelity_re += scaling * fidelity_iinit_re;
    fidelity_im += scaling * fidelity_iinit_im;


    // printf("%d, %d: iinit obj_iinit: %f * (%1.14e + i %1.14e), obj_cost=%1.14e, Overlap=%1.14e + i %1.14e\n", mpirank_world, mpirank_init, obj_weights[iinit-i], obj_iinit_re, obj_iinit_im, obj_cost, fidelity_iinit_re, fidelity_iinit_im);

    /* If Lindblas solver, compute adjoint for this initial condition. Otherwise (Schroedinger solver), compute adjoint only after all initial conditions have been propagated through (separate loop below) */
    if (timestepper->mastereq->lindbladtype != LindbladType::NONE) {
      // if (mpirank_optim == 0) printf("%d: %d BWD.", mpirank_init, initid);

      /* Reset adjoint */
      VecZeroEntries(rho_t0_bar);

      /* Terminal condition for adjoint variable: Derivative of final time objective J */
      double obj_cost_re_bar, obj_cost_im_bar;
      optim_target->finalizeJ_diff(obj_cost_re, obj_cost_im, &obj_cost_re_bar, &obj_cost_im_bar);
      optim_target->evalJ_diff(finalstate, rho_t0_bar, obj_weights[iinit-i]*obj_cost_re_bar, obj_weights[iinit-i]*obj_cost_im_bar);

      /* Derivative of time-stepping */
      timestepper->solveAdjointODE(initid, rho_t0_bar, finalstate, obj_weights[iinit-i] * gamma_penalty, obj_weights[iinit-i]*gamma_penalty_dpdm, obj_weights[iinit-i]*gamma_penalty_energy);

      /* Add to optimizers's gradient */
      const PetscScalar* g_ptr;
      VecGetArrayRead(timestepper->redgrad, &g_ptr);
      for (int i=0; i<ndesign;i++){
        G[i] += g_ptr[i]; 
      }
      VecRestoreArrayRead(timestepper->redgrad, &g_ptr);
    }
  }

  /* Sum up from initial conditions processors */
  double mypen = obj_penal;
  double mypen_dpdm = obj_penal_dpdm;
  double mypenen = obj_penal_energy;
  double mycost_re = obj_cost_re;
  double mycost_im = obj_cost_im;
  double mycost = obj_cost;
  double myfidelity_re = fidelity_re;
  double myfidelity_im = fidelity_im;
  MPI_Allreduce(&mypen, &obj_penal, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mypen_dpdm, &obj_penal_dpdm, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mypenen, &obj_penal_energy, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mycost_re, &obj_cost_re, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mycost_im, &obj_cost_im, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mycost, &obj_cost, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&myfidelity_re, &fidelity_re, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&myfidelity_im, &fidelity_im, 1, MPI_DOUBLE, MPI_SUM, comm_init);

  /* Set the fidelity: If Schroedinger, need to compute the absolute value: Fid= |\sum_i \phi^\dagger \phi_target|^2 */
  if (timestepper->mastereq->lindbladtype == LindbladType::NONE) {
    fidelity = pow(fidelity_re, 2.0) + pow(fidelity_im, 2.0);
  } else {
    fidelity = fidelity_re; 
  }
 
  /* Finalize the objective function Jtrace to get the infidelity. */
  if (optim_target->getInitCondType() != InitialConditionType::RANDOM) {
    obj_cost = optim_target->finalizeJ(obj_cost_re, obj_cost_im);
  }

  /* Evaluate regularization objective += gamma/2 * ||x||^2*/
  obj_regul = gamma_tik * evalTikhonov_(x, ndesign);

  /* Sum, store and return objective value */
  objective = obj_cost + obj_regul + obj_penal + obj_penal_dpdm + obj_penal_energy;

  /* For Schroedinger solver: Solve adjoint equations for all initial conditions here. */
  if (timestepper->mastereq->lindbladtype == LindbladType::NONE) {

    double obj_cost_re_bar=0.0, obj_cost_im_bar=0.0;
    double obj_cost_bar = 0.0;
    if (optim_target->getInitCondType() == InitialConditionType::RANDOM) {
      obj_cost_bar = -1.0;
    } else {
      optim_target->finalizeJ_diff(obj_cost_re, obj_cost_im, &obj_cost_re_bar, &obj_cost_im_bar);
    }

    // Iterate over all initial conditions 
    for (int iinit = i; iinit < i+ninit_local; iinit++) {
      int iinit_global = mpirank_init * ninit_local + iinit;

      // FOR STOCHASTIC OPTIM: Pass the random number generator seed for this initial condition
      if (ic_seed.size() > 0 ) {
        iinit_global = ic_seed[iinit];
      }

      /* Recompute the initial state and target */
      int initid = optim_target->prepareInitialState(iinit_global, ninit, timestepper->mastereq->nlevels, timestepper->mastereq->nessential, rho_t0);
      optim_target->prepareTargetState(rho_t0);
     
      /* Reset adjoint */
      VecZeroEntries(rho_t0_bar);

      /* Terminal condition for adjoint variable: Derivative of final time objective J */
      if (optim_target->getInitCondType() == InitialConditionType::RANDOM) {
        double obj_iinit_re = 0.0;
        double obj_iinit_im = 0.0;
        optim_target->evalJ(store_finalstates[iinit],  &obj_iinit_re, &obj_iinit_im);
        obj_cost_re_bar = 2.0*obj_iinit_re*obj_cost_bar;
        obj_cost_im_bar = 2.0*obj_iinit_im*obj_cost_bar;
      }
      optim_target->evalJ_diff(store_finalstates[iinit], rho_t0_bar, obj_weights[iinit-i]*obj_cost_re_bar, obj_weights[iinit-i]*obj_cost_im_bar);

      /* Derivative of time-stepping */
      timestepper->solveAdjointODE(initid, rho_t0_bar, store_finalstates[iinit], obj_weights[iinit-i] * gamma_penalty, obj_weights[iinit-i]*gamma_penalty_dpdm, obj_weights[iinit-i]*gamma_penalty_energy);

      /* Add to optimizers's gradient */
      const PetscScalar* g_ptr;
      VecGetArrayRead(timestepper->redgrad, &g_ptr);
      for (int i=0; i<ndesign;i++){
        G[i] += g_ptr[i]; 
      }
      VecRestoreArrayRead(timestepper->redgrad, &g_ptr);
    } // end of initial condition loop 
  } // end of adjoint for Schroedinger

  /* Sum up the gradient from all initial condition processors */
  for (int i=0; i<ndesign; i++) {
    mygrad[i] = G[i];
  }
  MPI_Allreduce(mygrad, G, ndesign, MPI_DOUBLE, MPI_SUM, comm_init);
  
  mastereq->setControlAmplitudes_diff_(G);

  /* Compute and store gradient norm */
  gnorm = 0.0;
  for (int i=0; i<ndesign; i++){
    gnorm += G[i]*G[i];
  }
  gnorm = sqrt(gnorm);

  /* Output */
  if (mpirank_world == 0 && !quietmode) {
    std::cout<< "Objective = " << std::scientific<<std::setprecision(14) << obj_cost << " + " << obj_regul << " + " << obj_penal << " + " << obj_penal_dpdm << " + " << obj_penal_energy << std::endl;
    std::cout<< "Fidelity = " << fidelity << std::endl;
  }

  // printf("Writing controls %1.14e\n", x[0]);
  output->writeControls_(x, ndesign, timestepper->mastereq, timestepper->ntime, timestepper->dt);

  return objective;
}


void OptimProblem::solve(Vec xinit) {
  TaoSetSolution(tao, xinit);
  TaoSolve(tao);
}

void OptimProblem::getStartingPoint(Vec xinit){
  MasterEq* mastereq = timestepper->mastereq;

  if (initguess_fromfile.size() > 0) {
    /* Set the initial guess from file */
    for (int i=0; i<initguess_fromfile.size(); i++) {
      VecSetValue(xinit, i, initguess_fromfile[i], INSERT_VALUES);
    }

  } else { // copy from initialization in oscillators contructor
    PetscScalar* xptr;
    VecGetArray(xinit, &xptr);
    int shift = 0;
    for (int ioscil = 0; ioscil<mastereq->getNOscillators(); ioscil++){
      mastereq->getOscillator(ioscil)->getParams(xptr + shift);
      shift += mastereq->getOscillator(ioscil)->getNParams();
    }
    VecRestoreArray(xinit, &xptr);
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
  TaoGetSolution(tao, &params);
  *param_ptr = params;
}

void OptimProblem::setObjWeights(double value){
  for (int i=0; i<obj_weights.size(); i++){
    obj_weights[i] = value;
  }
}

bool OptimProblem::Monitor(const double objective, const double* x, const size_t iter, const double stepsize){
  /* Pass current iteration number to output manager */
  output->optim_iter = iter;

  /* Grab some output stuff */
  double obj_cost = getCostT();
  double obj_regul = getRegul();
  double obj_penal = getPenalty();
  double obj_penal_dpdm = getPenaltyDpDm();
  double obj_penal_energy = getPenaltyEnergy();
  double F_avg = getFidelity();
  double gn = getGnorm();

  /* Print to optimization file */
  output->writeOptimFile(objective, gn, stepsize, F_avg, obj_cost, obj_regul, obj_penal, obj_penal_dpdm, obj_penal_energy);

  /* Print parameters and controls to file */
  /* DOES NOT WORK WITH ENSMALLEN, because MONITOR is called AFTER the control update. Instead, writing controls at the end of evalGradF_ */
  // if ( optim_iter % optim_monitor_freq == 0 ) {
  // printf("Writing controls %1.14e\n", x[0]);
  // output->writeControls_(x, ndesign, timestepper->mastereq, timestepper->ntime, timestepper->dt);
  // }

  /* Additional Stopping criteria */
  bool lastIter = false;
  std::string finalReason_str = "";
  if (1.0 - F_avg <= getInfTol()) {
    finalReason_str = "Optimization converged with small infidelity.";
    lastIter = true;
  } else if (obj_cost <= getFaTol()) {
    finalReason_str = "Optimization converged with small final time cost.";
    lastIter = true;
  } 

  /* Screen output header */
  if (getMPIrank_world() == 0 && iter == 0) {
    std::cout<<  "    Objective             Tikhonov                Penalty-Leakage        Penalty-StateVar       Penalty-TotalEnergy " << std::endl;
  }

  /* Screen output per iteration */
  if (getMPIrank_world() == 0 && (iter == getMaxIter() || lastIter || iter % output->optim_monitor_freq == 0)) {
    std::cout<< iter <<  "  " << std::scientific<<std::setprecision(14) << obj_cost << " + " << obj_regul << " + " << obj_penal << " + " << obj_penal_dpdm << " + " << obj_penal_energy;
    std::cout<< "  Fidelity = " << F_avg;
    std::cout<< "  ||Grad|| = " << gn;
    std::cout<< std::endl;
  }

  if (getMPIrank_world() == 0 && lastIter){
    std::cout<< finalReason_str << std::endl;
  }
 
  return lastIter;
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
  TaoGetSolution(tao, &params);

  // printf("TAO Gnorm = %1.14e\n", gnorm);
  // printf("TAO f = %1.14e\n", f);
  ctx->setGnorm(gnorm); // TODO: For some reason, Tao's gnorm is smaller than the reported Gnorm from EvalGradF_...

  /* Call optimproblem->Monitor() */
  const PetscScalar* params_ptr;
  VecGetArrayRead(params, &params_ptr);
  bool isLastIter = ctx->Monitor(f, params_ptr, iter, deltax);
  VecRestoreArrayRead(params, &params_ptr);


  if (isLastIter)
    TaoSetConvergedReason(tao, TAO_CONVERGED_USER);

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


// #ifdef WITH_ENSMALLEN
/* ENSMALLEN interface */

// Constructor
EnsmallenFunction::EnsmallenFunction(OptimProblem* optimctx_, int ndata_, std::default_random_engine rand_engine_){
  optimctx = optimctx_;
  ndata = ndata_;
  rand_engine = rand_engine_;
  
  /* Create the 'data set': <ndata> random initial conditions, encoded as random seeds for generating random initial conditions */
  std::uniform_int_distribution<> unit_int_dist(1000, 1000000000);
  for (int ic = 0; ic<ndata; ic++){
    int val = unit_int_dist(rand_engine);
    optimctx->ic_seed.push_back(val);
    // printf("Rand seed = %d\n", optimctx->ic_seed[ic]);
  }

  /* Overwrite weights in objective function to be 1/ndata */
  // optimctx->setObjWeights(1.0 / ndata);
  optimctx->setObjWeights(1.0);
}

EnsmallenFunction::~EnsmallenFunction(){}


double EnsmallenFunction::Evaluate(const arma::mat& x, const size_t i, const size_t batchSize){

  double F = optimctx->evalF_(x.memptr(), i, batchSize);

  return F;
}

void EnsmallenFunction::Shuffle(){
  std::shuffle(std::begin(optimctx->ic_seed), std::end(optimctx->ic_seed), rand_engine);
}

// Return the number of functions
size_t EnsmallenFunction::NumFunctions(){ 
  return ndata;
}

double EnsmallenFunction::EvaluateWithGradient(const arma::mat& x, const size_t i, arma::mat& g, const size_t batchSize){

  double F = optimctx->evalGradF_(x.memptr(), i, g.memptr(), batchSize);
  return F;
}

// #endif

