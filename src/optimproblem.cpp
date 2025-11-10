#include "optimproblem.hpp"

OptimProblem::OptimProblem(Config config, TimeStepper* timestepper_, MPI_Comm comm_init_, MPI_Comm comm_optim_, int ninit_, Output* output_, bool quietmode_){

  timestepper = timestepper_;
  ninit = ninit_;
  output = output_;
  quietmode = quietmode_;
  /* Reset */
  objective = 0.0;
  lastIter = false;

  /* Store communicators */
  comm_init = comm_init_;
  comm_optim = comm_optim_;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize_world);
  MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank_petsc);
  MPI_Comm_size(PETSC_COMM_WORLD, &mpisize_petsc);
  MPI_Comm_rank(comm_init, &mpirank_init);
  MPI_Comm_size(comm_init, &mpisize_init);
  MPI_Comm_rank(comm_optim, &mpirank_optim);
  MPI_Comm_size(comm_optim, &mpisize_optim);

  /* Store number of initial conditions per init-processor group */
  ninit_local = ninit / mpisize_init; 

  /*  If Schroedingers solver, allocate storage for the final states at time T for each initial condition. Schroedinger's solver does not store the time-trajectories during forward ODE solve, but instead recomputes the primal states during the adjoint solve. Therefore we need to store the terminal condition for the backwards primal solve. Be aware that the final states stored here will be overwritten during backwards computation!! */
  if (timestepper->mastereq->lindbladtype == LindbladType::NONE) {
    for (int i = 0; i < ninit_local; i++) {

      PetscInt globalsize = 2 * timestepper->mastereq->getDim();  // Global state vector: 2 for real and imaginary part
      PetscInt localsize = globalsize / mpisize_petsc;  // Local vector per processor
      Vec state;
      VecCreate(PETSC_COMM_WORLD, &state);
      VecSetSizes(state, localsize, globalsize);
      VecSetFromOptions(state);
      store_finalstates.push_back(state);
    }
  }

  /* Store number of design parameters */
  int n = 0;
  for (size_t ioscil = 0; ioscil < timestepper->mastereq->getNOscillators(); ioscil++) {
      n += timestepper->mastereq->getOscillator(ioscil)->getNParams(); 
  }
  ndesign = n;
  if (mpirank_world == 0 && !quietmode) std::cout<< "Number of control parameters: " << ndesign << std::endl;

  /* Allocate the initial condition vector and adjoint terminal state */
  VecCreate(PETSC_COMM_WORLD, &rho_t0); 
  PetscInt globalsize = 2 * timestepper->mastereq->getDim();  // Global state vector: 2 for real and imaginary part
  PetscInt localsize = globalsize / mpisize_petsc;  // Local vector per processor
  VecSetSizes(rho_t0, localsize, globalsize);
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
  config.GetVecDoubleParam("gate_rot_freq", read_gate_rot, 1e20, true, false); 
  std::vector<std::string> initcond_str;
  config.GetVecStrParam("initialcondition", initcond_str, "none", false);
  optim_target = new OptimTarget(target_str, objective_str, initcond_str, timestepper->mastereq, timestepper->total_time, read_gate_rot, rho_t0, quietmode);

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
  // Scale the weights such that they sum up to one: beta_i <- beta_i / (\sum_i beta_i)
  double scaleweights = 0.0;
  for (size_t i=0; i<ninit; i++) scaleweights += obj_weights[i];
  for (size_t i=0; i<ninit; i++) obj_weights[i] = obj_weights[i] / scaleweights;
  // Distribute over mpi_init processes 
  double sendbuf[obj_weights.size()];
  double recvbuf[obj_weights.size()];
  for (size_t i = 0; i < obj_weights.size(); i++) sendbuf[i] = obj_weights[i];
  for (size_t i = 0; i < obj_weights.size(); i++) recvbuf[i] = obj_weights[i];
  int nscatter = ninit_local;
  MPI_Scatter(sendbuf, nscatter, MPI_DOUBLE, recvbuf, nscatter,  MPI_DOUBLE, 0, comm_init);
  for (int i = 0; i < nscatter; i++) obj_weights[i] = recvbuf[i];
  for (size_t i=nscatter; i < obj_weights.size(); i++) obj_weights[i] = 0.0;


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
  gamma_penalty_variation = config.GetDoubleParam("optim_penalty_variation", 0.01); 
  

  if (gamma_penalty_dpdm > 1e-13 && timestepper->mastereq->lindbladtype != LindbladType::NONE){
    if (mpirank_world == 0 && !quietmode) {
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
  for (size_t iosc = 0; iosc < timestepper->mastereq->getNOscillators(); iosc++){
    std::vector<std::string> bound_str;
    config.GetVecStrParam("control_bounds" + std::to_string(iosc), bound_str, "10000.0");
    for (size_t iseg = 0; iseg < timestepper->mastereq->getOscillator(iosc)->getNSegments(); iseg++){
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
        for (size_t f = 0; f < timestepper->mastereq->getOscillator(iosc)->getNCarrierfrequencies(); f++){
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
  TaoSetObjective(tao, TaoEvalObjective, (void *)this);
  TaoSetGradient(tao, NULL, TaoEvalGradient,(void *)this);
  TaoSetObjectiveAndGradient(tao, NULL, TaoEvalObjectiveAndGradient, (void*) this);
  bool use_hessian = config.GetBoolParam("optim_use_hessian", false, false);

  if (use_hessian) {
    // TaoSetType(tao, TAONLS);     // Newton line search, unconstrained. TODO: Check Bounds!
    TaoSetType(tao, TAOBNLS);     // Bounded Newton with line search
    TaoSetHessian(tao, Hessian, Hessian, TaoEvalHessian, (void*) this);
    // Create Hessian matrix
    MatCreateDense(PETSC_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, ndesign, ndesign, NULL, &Hessian);
    MatSetFromOptions(Hessian);

    // TaoSetType(tao,TAOBQNLS);   // Bounded LBFGS with line search  
    // // Disable LBFGS history to use just the (projected!) gradient
    // Mat H_lmvm;
    // TaoGetLMVMMatrix(tao, &H_lmvm);
    // MatLMVMSetHistorySize(H_lmvm, 0); 
  } else {
    TaoSetType(tao,TAOBQNLS);   // Bounded LBFGS with line search  
  }

  TaoSetMaximumIterations(tao, maxiter);
  TaoSetTolerances(tao, gatol, PETSC_DEFAULT, grtol);
  TaoMonitorSet(tao, TaoMonitor, (void*)this, NULL);
  TaoSetVariableBounds(tao, xlower, xupper);
  TaoSetFromOptions(tao);

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
  MatDestroy(&Hessian);

  // VecDestroy(&xlower);
  // VecDestroy(&xupper);
  VecDestroy(&xinit);
  VecDestroy(&xtmp);

  for (size_t i = 0; i < store_finalstates.size(); i++) {
    VecDestroy(&(store_finalstates[i]));
  }

  TaoDestroy(&tao);
}



double OptimProblem::evalF(const Vec x) {

  MasterEq* mastereq = timestepper->mastereq;

  if (mpirank_world == 0 && !quietmode) printf("EVAL F... \n");

  /* Pass design vector x to oscillators */
  mastereq->setControlAmplitudes(x); 

  /*  Iterate over initial condition */
  obj_cost  = 0.0;
  obj_regul = 0.0;
  obj_penal = 0.0;
  obj_penal_dpdm = 0.0;
  obj_penal_energy = 0.0;
  obj_penal_variation = 0.0;
  fidelity = 0.0;
  double obj_cost_re = 0.0;
  double obj_cost_im = 0.0;
  double fidelity_re = 0.0;
  double fidelity_im = 0.0;
  for (int iinit = 0; iinit < ninit_local; iinit++) {
      
    /* Prepare the initial condition in [rank * ninit_local, ... , (rank+1) * ninit_local - 1] */
    int iinit_global = mpirank_init * ninit_local + iinit;
    optim_target->prepareInitialState(iinit_global, ninit, timestepper->mastereq->nlevels, timestepper->mastereq->nessential, rho_t0);
    // if (mpirank_optim == 0 && !quietmode) printf("%d: Initial condition id=%d ...\n", mpirank_init, initid);

    /* If gate optimiztion, compute the target state rho^target = Vrho(0)V^dagger */
    optim_target->prepareTargetState(rho_t0);

    /* Run forward with initial condition initid */
    Vec finalstate = timestepper->solveODE(iinit, rho_t0);

    /* Add to integral penalty term */
    obj_penal += obj_weights[iinit] * gamma_penalty * timestepper->penalty_integral;

    /* Add to second derivative penalty term */
    obj_penal_dpdm += obj_weights[iinit] * gamma_penalty_dpdm * timestepper->penalty_dpdm;
    
    /* Add to energy integral penalty term */
    obj_penal_energy += obj_weights[iinit] * gamma_penalty_energy* timestepper->energy_penalty_integral;

    /* Evaluate J(finalstate) and add to final-time cost */
    double obj_iinit_re = 0.0;
    double obj_iinit_im = 0.0;
    optim_target->evalJ(finalstate,  &obj_iinit_re, &obj_iinit_im);
    obj_cost_re += obj_weights[iinit] * obj_iinit_re;
    obj_cost_im += obj_weights[iinit] * obj_iinit_im;

    /* Add to final-time fidelity */
    double fidelity_iinit_re = 0.0;
    double fidelity_iinit_im = 0.0;
    optim_target->HilbertSchmidtOverlap(finalstate, false, &fidelity_iinit_re, &fidelity_iinit_im);
    fidelity_re += 1./ ninit * fidelity_iinit_re;
    fidelity_im += 1./ ninit * fidelity_iinit_im;

    // printf("%d, %d: iinit obj_iinit: %f * (%1.14e + i %1.14e, Overlap=%1.14e + i %1.14e\n", mpirank_world, mpirank_init, obj_weights[iinit], obj_iinit_re, obj_iinit_im, fidelity_iinit_re, fidelity_iinit_im);
  }

  /* Sum up from initial conditions processors */
  double mypen = obj_penal;
  double mypen_dpdm = obj_penal_dpdm;
  double mypenen = obj_penal_energy;
  double mycost_re = obj_cost_re;
  double mycost_im = obj_cost_im;
  double myfidelity_re = fidelity_re;
  double myfidelity_im = fidelity_im;
  MPI_Allreduce(&mypen, &obj_penal, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mypen_dpdm, &obj_penal_dpdm, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mypenen, &obj_penal_energy, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mycost_re, &obj_cost_re, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mycost_im, &obj_cost_im, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&myfidelity_re, &fidelity_re, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&myfidelity_im, &fidelity_im, 1, MPI_DOUBLE, MPI_SUM, comm_init);

  /* Set the fidelity: If Schroedinger, need to compute the absolute value: Fid= |\sum_i \phi^\dagger \phi_target|^2 */
  if (timestepper->mastereq->lindbladtype == LindbladType::NONE) {
    fidelity = pow(fidelity_re, 2.0) + pow(fidelity_im, 2.0);
  } else {
    fidelity = fidelity_re; 
  }
 
  /* Finalize the objective function */
  obj_cost = optim_target->finalizeJ(obj_cost_re, obj_cost_im);

  /* Evaluate Tikhonov regularization term: gamma/2 * ||x-x0||^2*/
  double xnorm;
  if (!gamma_tik_interpolate){  // ||x||^2
    VecNorm(x, NORM_2, &xnorm);
  } else {
    VecCopy(x, xtmp);
    VecAXPY(xtmp, -1.0, xinit);    // xtmp =  x - x_0
    VecNorm(xtmp, NORM_2, &xnorm);
  }
  obj_regul = gamma_tik / 2. * pow(xnorm,2.0);

  /* Evaluate penality term for control variation */
  double var_reg = 0.0;
  for (size_t iosc = 0; iosc < timestepper->mastereq->getNOscillators(); iosc++){
    var_reg += timestepper->mastereq->getOscillator(iosc)->evalControlVariation(); // uses Oscillator::params instead of 'x'
  }
  obj_penal_variation = 0.5*gamma_penalty_variation*var_reg; 

  /* Sum, store and return objective value */
  objective = obj_cost + obj_regul + obj_penal + obj_penal_dpdm + obj_penal_energy + obj_penal_variation;

  /* Output */
  if (mpirank_world == 0 && !quietmode) {
    std::cout<< "Objective = " << std::scientific<<std::setprecision(14) << obj_cost << " + " << obj_regul << " + " << obj_penal << " + " << obj_penal_dpdm << " + " << obj_penal_energy << " + " << obj_penal_variation << std::endl;
    std::cout<< "Fidelity = " << fidelity  << std::endl;
  }

  return objective;
}



void OptimProblem::evalGradF(const Vec x, Vec G){

  MasterEq* mastereq = timestepper->mastereq;

  if (mpirank_world == 0 && !quietmode) std::cout<< "EVAL GRAD F... " << std::endl;

  /* Pass design vector x to oscillators */
  mastereq->setControlAmplitudes(x); 

  /* Reset Gradient */
  VecZeroEntries(G);

  /* Derivative of regulatization terms (ADD ON ONE PROC ONLY!) */
  // if (mpirank_init == 0 && mpirank_optim == 0) { // TODO: Which one?? 
  if (mpirank_init == 0 ) {

    // Derivative of Tikhonov 0.5*gamma * ||x||^2 
    VecAXPY(G, gamma_tik, x);   // + gamma_tik * x
    if (gamma_tik_interpolate){
      VecAXPY(G, -1.0*gamma_tik, xinit); // -gamma_tik * xinit
    }

    // Derivative of penalization of control variation 
    double var_reg_bar = 0.5*gamma_penalty_variation;
    int skip_to_oscillator = 0;
    for (size_t iosc = 0; iosc < timestepper->mastereq->getNOscillators(); iosc++){
      Oscillator* osc = timestepper->mastereq->getOscillator(iosc);
      osc->evalControlVariationDiff(G, var_reg_bar, skip_to_oscillator);
      skip_to_oscillator += osc->getNParams();
    }
  }

  /*  Iterate over initial condition */
  obj_cost = 0.0;
  obj_regul = 0.0;
  obj_penal = 0.0;
  obj_penal_dpdm = 0.0;
  obj_penal_energy = 0.0;
  obj_penal_variation = 0.0;
  fidelity = 0.0;
  double obj_cost_re = 0.0;
  double obj_cost_im = 0.0;
  double fidelity_re = 0.0;
  double fidelity_im = 0.0;
  for (int iinit = 0; iinit < ninit_local; iinit++) {

    /* Prepare the initial condition */
    int iinit_global = mpirank_init * ninit_local + iinit;
    optim_target->prepareInitialState(iinit_global, ninit, timestepper->mastereq->nlevels, timestepper->mastereq->nessential, rho_t0);

    /* If gate optimiztion, compute the target state rho^target = Vrho(0)V^dagger */
    optim_target->prepareTargetState(rho_t0);

    /* --- Solve primal --- */
    // if (mpirank_optim == 0) printf("%d: %d FWD. ", mpirank_init, iinit_global);

    /* Run forward with initial condition rho_t0 */
    Vec finalstate = timestepper->solveODE(iinit, rho_t0);

    /* Store the final state for the Schroedinger solver */
    if (timestepper->mastereq->lindbladtype == LindbladType::NONE) VecCopy(finalstate, store_finalstates[iinit]);

    /* Add to integral penalty term */
    obj_penal += obj_weights[iinit] * gamma_penalty * timestepper->penalty_integral;

    /* Add to second derivative dpdm integral penalty term */
    obj_penal_dpdm += obj_weights[iinit] * gamma_penalty_dpdm * timestepper->penalty_dpdm;
    /* Add to energy integral penalty term */
    obj_penal_energy += obj_weights[iinit] * gamma_penalty_energy * timestepper->energy_penalty_integral;

    /* Evaluate J(finalstate) and add to final-time cost */
    double obj_iinit_re = 0.0;
    double obj_iinit_im = 0.0;
    optim_target->evalJ(finalstate,  &obj_iinit_re, &obj_iinit_im);
    obj_cost_re += obj_weights[iinit] * obj_iinit_re;
    obj_cost_im += obj_weights[iinit] * obj_iinit_im;
    // printf("evalGradF: iinit %d: objcost = %f * (%1.8e + i %1.8e)\n", iinit, obj_weights[iinit], obj_iinit_re, obj_iinit_im);

    /* Add to final-time fidelity */
    double fidelity_iinit_re = 0.0;
    double fidelity_iinit_im = 0.0;
    optim_target->HilbertSchmidtOverlap(finalstate, false, &fidelity_iinit_re, &fidelity_iinit_im);
    fidelity_re += 1./ ninit * fidelity_iinit_re;
    fidelity_im += 1./ ninit * fidelity_iinit_im;

    /* If Lindblas solver, compute adjoint for this initial condition. Otherwise (Schroedinger solver), compute adjoint only after all initial conditions have been propagated through (separate loop below) */
    if (timestepper->mastereq->lindbladtype != LindbladType::NONE) {
      // if (mpirank_optim == 0) printf("%d: %d BWD.", mpirank_init, initid);

      /* Reset adjoint */
      VecZeroEntries(rho_t0_bar);

      /* Terminal condition for adjoint variable: Derivative of final time objective J */
      double obj_cost_re_bar, obj_cost_im_bar;
      optim_target->finalizeJ_diff(obj_cost_re, obj_cost_im, &obj_cost_re_bar, &obj_cost_im_bar);
      optim_target->evalJ_diff(finalstate, rho_t0_bar, obj_weights[iinit]*obj_cost_re_bar, obj_weights[iinit]*obj_cost_im_bar);

      /* Derivative of time-stepping */
      timestepper->solveAdjointODE(iinit, rho_t0_bar, finalstate, obj_weights[iinit] * gamma_penalty, obj_weights[iinit]*gamma_penalty_dpdm, obj_weights[iinit]*gamma_penalty_energy);

      /* Add to optimizers's gradient */
      VecAXPY(G, 1.0, timestepper->redgrad);
    }
  }

  /* Sum up from initial conditions processors */
  double mypen = obj_penal;
  double mypen_dpdm = obj_penal_dpdm;
  double mypenen = obj_penal_energy;
  double mycost_re = obj_cost_re;
  double mycost_im = obj_cost_im;
  double myfidelity_re = fidelity_re;
  double myfidelity_im = fidelity_im;
  MPI_Allreduce(&mypen, &obj_penal, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mypen_dpdm, &obj_penal_dpdm, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mypenen, &obj_penal_energy, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mycost_re, &obj_cost_re, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mycost_im, &obj_cost_im, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&myfidelity_re, &fidelity_re, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&myfidelity_im, &fidelity_im, 1, MPI_DOUBLE, MPI_SUM, comm_init);

  /* Set the fidelity: If Schroedinger, need to compute the absolute value: Fid= |\sum_i \phi^\dagger \phi_target|^2 */
  if (timestepper->mastereq->lindbladtype == LindbladType::NONE) {
    fidelity = pow(fidelity_re, 2.0) + pow(fidelity_im, 2.0);
  } else {
    fidelity = fidelity_re; 
  }
 
  /* Finalize the objective function Jtrace to get the infidelity. 
     If Schroedingers solver, need to take the absolute value */
  obj_cost = optim_target->finalizeJ(obj_cost_re, obj_cost_im);

  /* Evaluate Tikhonov regularization term += gamma/2 * ||x||^2*/
  double xnorm;
  if (!gamma_tik_interpolate){  // ||x||^2
    VecNorm(x, NORM_2, &xnorm);
  } else {
    VecCopy(x, xtmp);
    VecAXPY(xtmp, -1.0, xinit);    // xtmp =  x_k - x_0
    VecNorm(xtmp, NORM_2, &xnorm);
  }
  obj_regul = gamma_tik / 2. * pow(xnorm,2.0);

  /* Evaluate penalty term for control parameter variation */
  double var_reg = 0.0;
  for (size_t iosc = 0; iosc < timestepper->mastereq->getNOscillators(); iosc++){
    var_reg += timestepper->mastereq->getOscillator(iosc)->evalControlVariation(); // uses Oscillator::params instead of 'x'
  }
  obj_penal_variation = 0.5*gamma_penalty_variation*var_reg; 

  /* Sum, store and return objective value */
  objective = obj_cost + obj_regul + obj_penal + obj_penal_dpdm + obj_penal_energy + obj_penal_variation;

  /* For Schroedinger solver: Solve adjoint equations for all initial conditions here. */
  if (timestepper->mastereq->lindbladtype == LindbladType::NONE) {

    // Iterate over all initial conditions 
    for (int iinit = 0; iinit < ninit_local; iinit++) {
      int iinit_global = mpirank_init * ninit_local + iinit;

      /* Recompute the initial state and target */
      optim_target->prepareInitialState(iinit_global, ninit, timestepper->mastereq->nlevels, timestepper->mastereq->nessential, rho_t0);
      optim_target->prepareTargetState(rho_t0);
     
      /* Reset adjoint */
      VecZeroEntries(rho_t0_bar);

      /* Terminal condition for adjoint variable: Derivative of final time objective J */
      double obj_cost_re_bar, obj_cost_im_bar;
      optim_target->finalizeJ_diff(obj_cost_re, obj_cost_im, &obj_cost_re_bar, &obj_cost_im_bar);
      optim_target->evalJ_diff(store_finalstates[iinit], rho_t0_bar, obj_weights[iinit]*obj_cost_re_bar, obj_weights[iinit]*obj_cost_im_bar);

      /* Derivative of time-stepping */
      timestepper->solveAdjointODE(iinit, rho_t0_bar, store_finalstates[iinit], obj_weights[iinit] * gamma_penalty, obj_weights[iinit]*gamma_penalty_dpdm, obj_weights[iinit]*gamma_penalty_energy);

      /* Add to optimizers's gradient */
      VecAXPY(G, 1.0, timestepper->redgrad);
    } // end of initial condition loop 
  } // end of adjoint for Schroedinger

  /* Sum up the gradient from all initial condition processors */
  PetscScalar* grad; 
  VecGetArray(G, &grad);
  for (int i=0; i<ndesign; i++) {
    mygrad[i] = grad[i];
  }
  MPI_Allreduce(mygrad, grad, ndesign, MPI_DOUBLE, MPI_SUM, comm_init);
  VecRestoreArray(G, &grad);

  /* Compute and store gradient norm */
  VecNorm(G, NORM_2, &(gnorm));

  /* Output */
  // if (mpirank_world == 0 && !quietmode) {
  //   std::cout<< "Objective = " << std::scientific<<std::setprecision(14) << obj_cost << " + " << obj_regul << " + " << obj_penal << " + " << obj_penal_dpdm << " + " << obj_penal_energy << " + " << obj_penal_variation << std::endl;
  //   std::cout<< "Fidelity = " << fidelity << std::endl;
  // }
}


void OptimProblem::evalHessVec(const Vec x, const Vec v, Vec Hv){
  if (mpirank_world == 0 && !quietmode) std::cout<< "EVAL HESS VEC... " << std::endl;

  timestepper->mastereq->setControlAmplitudes(x); 

  // Reset
  VecZeroEntries(Hv);

  /* Solve forward and adjoint ODE, storing the (adjoint) states at each timestep and for each initial condition */
  VecZeroEntries(xtmp);
  timestepper->storeFWD = true;
  evalGradF(x, xtmp);

  /* Solve linearized ODE forward in time */
  for (int iinit = 0; iinit < ninit_local; iinit++) {
    timestepper->solveLinearizedODE(iinit, v);
  }

  /* Solve linearized adjoint ODE while accumulating Hessian-vector product */

  // First, compute the objective function using the linearized state which is needed for the terminal conditions, 
  double obj_cost_re = 0.0;
  double obj_cost_im = 0.0;
  for (int iinit=0; iinit<ninit_local; iinit++){
    // First compute the target state (which needs the initial state)
    int iinit_global = mpirank_init * ninit_local + iinit;
    optim_target->prepareInitialState(iinit_global, ninit, timestepper->mastereq->nlevels, timestepper->mastereq->nessential, rho_t0);
    optim_target->prepareTargetState(rho_t0);
    Vec wT = timestepper->getLinearizedState(iinit, timestepper->getNTimeSteps());
    double obj_lin_iinit_re = 0.0;
    double obj_lin_iinit_im = 0.0;
    optim_target->evalJ(wT,  &obj_lin_iinit_re, &obj_lin_iinit_im);
    obj_cost_re += obj_weights[iinit] * obj_lin_iinit_re;
    obj_cost_im += obj_weights[iinit] * obj_lin_iinit_im;
  }
  double mycost_re = obj_cost_re;
  double mycost_im = obj_cost_im;
  MPI_Allreduce(&mycost_re, &obj_cost_re, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mycost_im, &obj_cost_im, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  // Now solve adjoint backwards for each terminal condition 
  for (int iinit = 0; iinit < ninit_local; iinit++) {
    int iinit_global = mpirank_init * ninit_local + iinit;
    // get the terminal condition 
    optim_target->prepareInitialState(iinit_global, ninit, timestepper->mastereq->nlevels, timestepper->mastereq->nessential, rho_t0);
    optim_target->prepareTargetState(rho_t0);
    VecZeroEntries(rho_t0_bar);
    double obj_cost_re_bar, obj_cost_im_bar;
    optim_target->finalizeJ_diff(obj_cost_re, obj_cost_im, &obj_cost_re_bar, &obj_cost_im_bar);
    Vec wT = timestepper->getLinearizedState(iinit, timestepper->getNTimeSteps());
    optim_target->evalJ_diff(wT, rho_t0_bar, obj_weights[iinit]*obj_cost_re_bar, obj_weights[iinit]*obj_cost_im_bar);

    // solve backwards while accumulating hessian-vector product 
    timestepper->solveLinearizedAdjointODE(iinit, rho_t0_bar, v, Hv);
  }

  /* Hessian of Tikhonov regularization */
  if (mpirank_init == 0) { // ADD ON ONE PROC ONLY!
    // Hessian of Tikhonov 0.5*gamma * ||x||^2 : H = gamma_tik I
    VecAXPY(Hv, gamma_tik, v);   // hv += gamma_tik v
  }

  /* Sum up from all initial condition processors */
  PetscScalar* hess; 
  VecGetArray(Hv, &hess);
  for (int i=0; i<ndesign; i++) {
    mygrad[i] = hess[i];
  }
  MPI_Allreduce(mygrad, hess, ndesign, MPI_DOUBLE, MPI_SUM, comm_init);
  VecRestoreArray(Hv, &hess);

  // double hnorm;
  // VecNorm(Hv, NORM_2, &(hnorm));
  // printf("Hessian vector product norm = %1.14e\n", hnorm);
}

void OptimProblem::solve(Vec xinit) {
  TaoSetSolution(tao, xinit);
  TaoSolve(tao);

  // TaoView(tao, NULL);
}

void OptimProblem::getStartingPoint(Vec xinit){
  MasterEq* mastereq = timestepper->mastereq;

  if (initguess_fromfile.size() > 0) {
    /* Set the initial guess from file */
    for (size_t i=0; i<initguess_fromfile.size(); i++) {
      VecSetValue(xinit, i, initguess_fromfile[i], INSERT_VALUES);
    }

  } else { // copy from initialization in oscillators contructor
    PetscScalar* xptr;
    VecGetArray(xinit, &xptr);
    int shift = 0;
    for (size_t ioscil = 0; ioscil<mastereq->getNOscillators(); ioscil++){
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


bool OptimProblem::monitor(int iter, double deltax, Vec params){

  /* Grab some output stuff */
  double obj_cost = getCostT();
  double obj_regul = getRegul();
  double obj_penal = getPenalty();
  double obj_penal_dpdm = getPenaltyDpDm();
  double obj_penal_energy = getPenaltyEnergy();
  double obj_penal_variation= getPenaltyVariation();
  double F_avg = getFidelity();

  /* Additional Stopping criteria */
  bool TAOCONVERGED = false;
  std::string finalReason_str = "";
  if (1.0 - F_avg <= getInfTol()) {
    finalReason_str = "Optimization converged with small infidelity.";
    TAOCONVERGED = true;
    lastIter = true;
  } else if (obj_cost <= getFaTol()) {
    finalReason_str = "Optimization converged with small final time cost.";
    TAOCONVERGED = true;
    lastIter = true;
  } else if (iter == getMaxIter()) {
    finalReason_str = "Optimization stopped at maximum number of iterations.";
    lastIter = true;
  } else if (gnorm < getGaTol()) {
    finalReason_str = "OPtimization converged with small gradient norm.";
    lastIter=true;
  }

  /* First iteration: Header for screen output of optimization history */
  if (iter == 0 && getMPIrank_world() == 0) {
    std::cout<<  "    Objective             Tikhonov                Penalty-Leakage        Penalty-StateVar       Penalty-TotalEnergy    Penalty-CtrlVar" << std::endl;
  }

  /* Every <optim_monitor_freq> iterations: Output of optimization history */
  if (iter % output->optim_monitor_freq == 0 ||lastIter) {
    // Add to optimization history file 
    // ctx->output->writeOptimFile(iter, f, gnorm, deltax, F_avg, obj_cost, obj_regul, obj_penal, obj_penal_dpdm, obj_penal_energy, obj_penal_variation);
    output->writeOptimFile(iter, objective, gnorm, deltax, F_avg, obj_cost, obj_regul, obj_penal, obj_penal_dpdm, obj_penal_energy, obj_penal_variation);
    // Screen output 
    if (getMPIrank_world() == 0) {
      std::cout<< iter <<  "  " << std::scientific<<std::setprecision(14) << obj_cost << " + " << obj_regul << " + " << obj_penal << " + " << obj_penal_dpdm << " + " << obj_penal_energy << " + " << obj_penal_variation;
      std::cout<< "  Fidelity = " << F_avg;
      std::cout<< "  ||Grad|| = " << gnorm;
      std::cout<< std::endl;
    }
  }

  /* Last iteration: Print solution, controls and trajectory data to files */
  if (lastIter) {
    output->writeControls(params, timestepper->mastereq, timestepper->ntime, timestepper->dt);

    // do one last forward evaluation while writing trajectory files
    timestepper->writeTrajectoryDataFiles = true;
    evalF(params); 

    // Print stopping reason to screen
    if (getMPIrank_world() == 0){
      std::cout<< finalReason_str << std::endl;
    }
  }

  return TAOCONVERGED;
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

  bool TAOCONVERGED = ctx->monitor(iter, deltax, params);

  if (TAOCONVERGED) TaoSetConvergedReason(tao, TAO_CONVERGED_USER);

  return 0;
}


PetscErrorCode TaoEvalObjectiveAndGradient(Tao tao, Vec x, PetscReal *f, Vec G, void*ptr){

  TaoEvalGradient(tao, x, G, ptr);
  OptimProblem* ctx = (OptimProblem*) ptr;
  *f = ctx->getObjective();

  // /* Project gradient onto dominant subspace */
  // PetscInt ncut = 25;    // Number of dominant eigenvalues/vectors to compute TODO: READ FROM CONFIG
  // PetscInt nextra = 10;  // Oversampling
  // Vec grad_proj;
  // VecDuplicate(G, &grad_proj);
  // ctx->ProjectGradient(x, G, grad_proj, ncut, nextra);
  // VecCopy(grad_proj, G);
  // VecDestroy(&grad_proj);
 

  return 0;
}

PetscErrorCode TaoEvalObjective(Tao /*tao*/, Vec x, PetscReal *f, void*ptr){

  OptimProblem* ctx = (OptimProblem*) ptr;
  *f = ctx->evalF(x);
  
  return 0;
}

PetscErrorCode TaoEvalHessian(Tao /* tao */, Vec x, Mat H, Mat /* Hpre */, void*ptr){

  PetscInt ncut = 25;    // Number of dominant eigenvalues/vectors to compute
  PetscInt nextra = 10;  // Oversampling
  // PetscInt ncut = 2; 
  // PetscInt nextra = 1; 

  OptimProblem* ctx = (OptimProblem*) ptr;
  ctx->evalHessian(x, ncut, nextra, H);

  return 0;
}

PetscErrorCode TaoEvalGradient(Tao /*tao*/, Vec x, Vec G, void*ptr){

  OptimProblem* ctx = (OptimProblem*) ptr;
  ctx->evalGradF(x, G);

  // /* Project gradient onto dominant subspace */
  // PetscInt ncut = 25;    // Number of dominant eigenvalues/vectors to compute TODO: READ FROM CONFIG
  // PetscInt nextra = 10;  // Oversampling
  // Vec grad_proj;
  // VecDuplicate(G, &grad_proj);
  // ctx->ProjectGradient(x, G, grad_proj, ncut, nextra);
  // VecCopy(grad_proj, G);
  // VecDestroy(&grad_proj);
  
  return 0;
}



/**** ROL interface ****/

myVec::myVec(Vec vec) {

  VecDuplicate(vec, &petscVec_);
  VecCopy(vec, petscVec_);
} 
myVec::~myVec() { VecDestroy(&petscVec_);  }


int myVec::dimension() const {
  PetscInt size;
  VecGetSize(petscVec_, &size);
  return size;
}

double myVec::dot(const ROL::Vector<double> &x) const {
    const myVec &ex = dynamic_cast<const myVec&>(x);
    double result;
    VecDot(petscVec_, ex.petscVec_, &result); 
    return result;
}

void myVec::plus(const ROL::Vector<double> &x) {
    const myVec &ex = dynamic_cast<const myVec&>(x);
    VecAXPY(petscVec_, 1.0, ex.petscVec_);  
}

void myVec::applyUnary( const ROL::Elementwise::UnaryFunction<double> &f ) {
    PetscInt dim = dimension();
    double* vecptr;
    VecGetArray(petscVec_, &vecptr);
    for(int i=0; i<dim; ++i) {
      vecptr[i] = f.apply(vecptr[i]);
    }
    VecRestoreArray(petscVec_, &vecptr);
}

void myVec::applyBinary(const ROL::Elementwise::BinaryFunction<double> &f, const ROL::Vector<double> &x ){
  const myVec &ex = dynamic_cast<const myVec&>(x);

  int dim = dimension();
  ROL_TEST_FOR_EXCEPTION( dim != x.dimension(), std::invalid_argument, "Error: Vectors must have the same dimension." );

  double *selfptr, *xptr;
  Vec exVec_ = ex.getVector();
  VecGetArray(petscVec_, &selfptr);
  VecGetArray(exVec_, &xptr);
  for (int i=0; i<dim; i++) {
    selfptr[i] = f.apply(selfptr[i],xptr[i]);
  }
  VecRestoreArray(petscVec_, &selfptr);
  VecRestoreArray(exVec_, &xptr);
}

double myVec::reduce(const ROL::Elementwise::ReductionOp<double> &r)const{
  double result = r.initialValue();
  int dim  = dimension();
  const double* selfptr;
  VecGetArrayRead(petscVec_, &selfptr);
  for(int i=0; i<dim; ++i) {
    r.reduce(selfptr[i],result);
  }
  VecRestoreArrayRead(petscVec_, &selfptr);

  return result;
}

double myVec::norm() const {
    double result;
    VecNorm(petscVec_, NORM_2, &result); 
    return result;
  }

void myVec::scale(double alpha) { 
    VecScale(petscVec_, alpha);  
}

ROL::Ptr<ROL::Vector<double>> myVec::clone (void) const {
  Vec clonedVec;
  VecDuplicate(petscVec_, &clonedVec);
  return ROL::makePtr<myVec>(clonedVec);
}

void myVec::set(const ROL::Vector<double> &x) {
  const myVec &ex = dynamic_cast<const myVec&>(x);
  VecCopy(ex.petscVec_, petscVec_); 
}

Vec myVec::getVector() const {
  return petscVec_;
}

void myVec::axpy(double alpha, const ROL::Vector<double> &x) {
  const myVec &ex = dynamic_cast<const myVec&>(x);
  VecAXPY(petscVec_, alpha, ex.petscVec_); 
}

void myVec::zero() {
  VecZeroEntries(petscVec_);
}

void myVec::view() { 
  VecView(petscVec_, NULL);
}

myObjective::myObjective(OptimProblem* optimctx) : optimctx_(optimctx) {
  myAcceptIter=0;
}
myObjective::~myObjective() {}

double myObjective::value(const ROL::Vector<double> &x, double & /*tol*/) {

  // Cast the input and evalF on the petsc vectors
  const myVec& ex = dynamic_cast<const myVec&>(x); 
  double f = optimctx_->evalF(ex.getVector());
  return f;
}

void myObjective::gradient(ROL::Vector<double> &g, const ROL::Vector<double> &x, double & /*tol*/) {

  // Hack: If last iter, set gradient to zero so that ROL stops.
  if (optimctx_->lastIter) {
    g.zero();
    return;
  }

  // Cast the input and evalGradF on the petsc vectors 
  const myVec& ex = dynamic_cast<const myVec&>(x); 
  myVec& eg = dynamic_cast<myVec&>(g); 
  optimctx_->evalGradF(ex.getVector(), eg.getVector());
}

void myObjective::hessVec( ROL::Vector<double> &hv, const ROL::Vector<double> &v, const ROL::Vector<double> &x, double& /*tol*/ ){

  // Cast the input and evalHessVec on the petsc vectors 
  const myVec& ev = dynamic_cast<const myVec&>(v); 
  const myVec& ex = dynamic_cast<const myVec&>(x); 
  myVec& ehv = dynamic_cast<myVec&>(hv); 
  optimctx_->evalHessVec(ex.getVector(), ev.getVector(), ehv.getVector());
}

void myObjective::update(const ROL::Vector<double> &x, ROL::UpdateType type, int /*iter*/){
  std::string out; 
  if (type == ROL::UpdateType::Initial)  {
    // This is the first call to update
    out = "Initial";
  }
  else if (type == ROL::UpdateType::Accept) {
    // u_ was set to u=S(x) during a trial update and has been accepted as the new iterate
    out = "Accept";
  }
  else if (type == ROL::UpdateType::Revert) {
    // u_ was set to u=S(x) during a trial update and has been rejected as the new iterate
    out = "Revert";
  }
  else if (type == ROL::UpdateType::Trial) {
    // This is a new value of x
    out = "Trial";
  }
  else { 
    out = "Else";
  }
  // printf("Update was called at iter %d. Status: %s\n", iter, out.c_str());

  if (type == ROL::UpdateType::Accept) {

    myAcceptIter = myAcceptIter+1;
    const myVec& ex = dynamic_cast<const myVec&>(x); 
    optimctx_->monitor(myAcceptIter, 0.0, ex.getVector());
  }
}

void OptimProblem::HessianRandRangeFinder(const Vec x, PetscInt ncut, PetscInt nextra, Mat* U_out, Vec* lambda_out){
  // Sample a random matrix Omega
  // Apply Hessian-vector product on each column of Q -> Y = H * Omega
  // Find basis with economy SVD and take top-k left singular vectors
  //   -> U,S,V = SVD(Y), Q = U[:,1:k]
  // Solve B * (Q' * Omega) = (Q' * Y) for B: least squares problem 
  //   -> B = lstsq( (Q.T*Omega).T, (W.T*Y).T ).T
  // Eigenvalue decomposition of B -> B = V Lambda V'
  // Project: U = Q * V
  //  -> Hessian \approx U * Lambda * U'
  //  -> grad_proj = U * Lambda^{-1} * U' * grad
  // Returns Mat U and Vec Lambda

  PetscInt n;
  VecGetSize(x, &n);

  Mat Omega, Y, Q;
  SVD svd;
  PetscInt nconv;
  Mat QtOmega, QtY, B, U;
  EPS eps;
  PetscScalar *lambda;
  PetscRandom rctx;
  
  /* Sample random matrix Omega (n x ncut+nextra) */
  PetscInt nsample = ncut + nextra;
  MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, nsample, NULL, &Omega);
  PetscRandomCreate(PETSC_COMM_WORLD, &rctx);
  PetscRandomSetFromOptions(rctx);
  PetscRandomSetSeed(rctx, 42);
  PetscRandomSeed(rctx);
  MatSetRandom(Omega, rctx);
  MatSetUp(Omega);
  MatAssemblyBegin(Omega, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Omega, MAT_FINAL_ASSEMBLY);

  /* Apply Hessian-vector product Y = H * Omega */
  MatDuplicate(Omega, MAT_DO_NOT_COPY_VALUES, &Y);
  Vec omega_col, y_col;
  for (int i = 0; i < nsample; i++) {
    if (mpirank_world==0) printf("Applying Hessian-vector product %d / %d\n", i, nsample);
    MatDenseGetColumnVecRead(Omega, i, &omega_col);
    MatDenseGetColumnVecWrite(Y, i, &y_col);
    evalHessVec(x, omega_col, y_col);
    MatDenseRestoreColumnVecRead(Omega, i, &omega_col);
    MatDenseRestoreColumnVecWrite(Y, i, &y_col);
  }
  
  /* Find orthonormal basis for Y */
  // Economy SVD of Y, keep ncut top left singular vectors
  SVDCreate(PETSC_COMM_WORLD, &svd); 
  SVDSetOperators(svd, Y, NULL); 
  SVDSetType(svd, SVDTRLANCZOS);  // or SVDCROSS, SVDLAPACK
  SVDSetDimensions(svd, ncut, PETSC_DEFAULT, PETSC_DEFAULT); 
  SVDSetFromOptions(svd); 
  SVDSolve(svd); 
  SVDGetConverged(svd, &nconv); 
  if (nconv < ncut) {
    printf("ERROR: SVD converged to %D singular values, needed %D", nconv, ncut);
    exit(1);
  }
  // Extract Q = U[:,1:ncut] (left singular vectors)
  MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, ncut, NULL, &Q); 
  Vec u_col;
  for (int i = 0; i < ncut; i++) {
    MatDenseGetColumnVecWrite(Q, i, &u_col);
    SVDGetSingularTriplet(svd, i, NULL, u_col, NULL);
    MatDenseRestoreColumnVecWrite(Q, i, &u_col);
  }
  
  /*  Compute Q^T * Omega and Q^T * Y */
  MatTransposeMatMult(Q, Omega, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &QtOmega); 
  MatTransposeMatMult(Q, Y, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &QtY);
  
  /* Solve least squares: B * (Q^T * Omega) = (Q^T * Y) */
  // Transpose: (Q^T*Omega)^T * B^T = (Q^T*Y)^T
  // Solve for B^T:  (Q^T*Omega)(Q^T*Omega)^T * B^T = (Q^T*Omega)(Q^T*Y)^T

  // Compute Gram = QtOmega * QtOmega^T
  Mat Gram;
  MatMatTransposeMult(QtOmega, QtOmega, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &Gram);

  // Compute RHS = QtOmega * QtY^T
  Mat RHS;
  MatMatTransposeMult(QtOmega, QtY, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &RHS);

  // Set up the solution matrix
  Mat X;
  MatDuplicate(Gram, MAT_DO_NOT_COPY_VALUES, &X);

  // Solve Gram * X = RHS for X (=B^T) using Petsc KSP solver
  KSP ksp;
  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetOperators(ksp, Gram, Gram);
  // PC pc;
  // KSPGetPC(ksp, &pc);
  // PCSetType(pc, PCLU);
  KSPSetFromOptions(ksp);
  KSPMatSolve(ksp, RHS, X);

  // Transpose X to get B
  MatTranspose(X, MAT_INITIAL_MATRIX, &B);

  /* Eigenvalue decomposition of B */
  EPSCreate(PETSC_COMM_WORLD, &eps);
  EPSSetOperators(eps, B, NULL);
  EPSSetProblemType(eps, EPS_NHEP); // Symmetric system matrix?
  EPSSetDimensions(eps, ncut, PETSC_DEFAULT, PETSC_DEFAULT);
  EPSSetFromOptions(eps);
  EPSSolve(eps);
  EPSGetConverged(eps, &nconv); 
  if (nconv < ncut) {
    printf("ERROR: EPS converged to %D eigenvalues, needed %D", nconv, ncut);
  }
  
  // Extract eigenvectors V and eigenvalues Lambda
  MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, ncut, ncut, NULL, &U); // This will be U from B = U*Lambda*V'
  PetscMalloc1(ncut, &lambda);
  
  PetscScalar eigval;
  for (int i = 0; i < ncut; i++) {
    MatDenseGetColumnVecWrite(U, i, &u_col);
    EPSGetEigenpair(eps, i, &eigval, NULL, u_col, NULL);
    lambda[i] = eigval;  // Store inverse eigenvalue
    MatDenseRestoreColumnVecWrite(U, i, &u_col);
  }
  // print the eigenvalues
  if (mpirank_world==0) {
    printf("Dominant eigenvalues of the Hessian approximation:\n");
    for (int i = 0; i < ncut; i++) {
      printf("%1.14e\n", lambda[i]);
    }
  }
  
  /* Project and store U_out = Q * V */
  MatMatMult(Q, U, MAT_INITIAL_MATRIX, PETSC_DEFAULT, U_out); 

  /* Store lambda*/
  MatCreateVecs(*U_out, lambda_out, NULL);
  PetscScalar *lambda_ptr;
  VecGetArrayWrite(*lambda_out, &lambda_ptr);
  for (int i = 0; i < ncut; i++) {
    lambda_ptr[i] = lambda[i];
  }
  VecRestoreArrayWrite(*lambda_out, &lambda_ptr);

  // Cleanup
  PetscFree(lambda);
  MatDestroy(&Q);
  MatDestroy(&Y);
  MatDestroy(&QtOmega);
  MatDestroy(&Omega);
  MatDestroy(&QtY);
  MatDestroy(&Gram);
  MatDestroy(&RHS);
  MatDestroy(&X);
  MatDestroy(&B);
  MatDestroy(&U);
  SVDDestroy(&svd);
  EPSDestroy(&eps);
  KSPDestroy(&ksp);
  PetscRandomDestroy(&rctx);
}

void OptimProblem::evalHessian(const Vec x, PetscInt ncut, PetscInt nextra, Mat H){

  // Get U, Lambda s.t. H \approx U * Lambda * U^T
  Mat U;
  Vec lambda;
  HessianRandRangeFinder(x, ncut, nextra, &U, &lambda);

  /* Assemble H = U lambda U^T */
  Mat U_scaled;
  MatDuplicate(U, MAT_COPY_VALUES, &U_scaled);  // U_scaled = U
  MatDiagonalScale(U_scaled, NULL, lambda);  // Right scaling U_scaled = U * diag(lambda)
  MatMatTransposeMult(U_scaled, U, MAT_REUSE_MATRIX, PETSC_DEFAULT, &H); // H= U_scaled * U^T

  MatDestroy(&U);
  VecDestroy(&lambda);
}
  
void OptimProblem::ProjectGradient(const Vec x, const Vec grad, Vec grad_proj, PetscInt ncut, PetscInt nextra){

  // Get U, Lambda s.t. H \approx U * Lambda * U^T
  Mat U;
  Vec lambda;
  HessianRandRangeFinder(x, ncut, nextra, &U, &lambda);

  /* Gradient projection: grad_proj = U*Lambda^{-1}*U^T * grad */
  
  Vec tmp;
  VecDuplicate(lambda, &tmp);
  MatMultTranspose(U, grad, tmp); // grad_proj = U^T * grad
  // Scale rows by Lambda^{-1}
  PetscScalar *tmp_ptr;
  const PetscScalar *lambda_ptr;
  VecGetArrayWrite(grad_proj, &tmp_ptr);
  VecGetArrayRead(lambda, &lambda_ptr);
  for (int i = 0; i < ncut; i++) {
    tmp_ptr[i] = 1.0 / lambda_ptr[i] * tmp_ptr[i];
  }
  VecRestoreArrayWrite(grad_proj, &tmp_ptr);
  VecRestoreArrayRead(lambda, &lambda_ptr);
  
  // Project graident: grad_proj = U * tmp
  MatMult(U, tmp, grad_proj);
  
  MatDestroy(&U);
  VecDestroy(&lambda);
  VecDestroy(&tmp);
}