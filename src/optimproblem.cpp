#include "optimproblem.hpp"

OptimProblem::OptimProblem(MapParam config, TimeStepper* timestepper_, MPI_Comm comm_init_, MPI_Comm comm_optim_, int ninit_, Output* output_, bool x_is_control_, bool quietmode_){

  timestepper = timestepper_;
  ninit = ninit_;
  output = output_;
  quietmode = quietmode_;
  x_is_control = x_is_control_;
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

  /* Store number of optimization parameters */
  if (x_is_control) { // Optimizing on the control parameters
    int n = 0;
    for (int ioscil = 0; ioscil < timestepper->mastereq->getNOscillators(); ioscil++) {
        n += timestepper->mastereq->getOscillator(ioscil)->getNParams(); 
    }
    ndesign = n;
    if (mpirank_world == 0 && !quietmode) std::cout<< "Number of control parameters: " << ndesign << std::endl;
  } else { // Optimizing on the learnable parameters
    ndesign = timestepper->mastereq->learning->getNParams();
  }

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
  gamma_tik_onenorm = config.GetBoolParam("optim_regul_onenorm", false, false);
  gamma_penalty_dpdm = config.GetDoubleParam("optim_penalty_dpdm", 0.0);
  gamma_penalty_variation = config.GetDoubleParam("optim_penalty_variation", 0.0); 
  

  if (gamma_penalty_dpdm > 1e-13 && timestepper->mastereq->lindbladtype != LindbladType::NONE){
    if (mpirank_world == 0) {
      // printf("Warning: Disabling DpDm penalty term because it is not implemented for the Lindblad solver.\n");
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
  if (x_is_control) { // Optimize on control parameters, set bounds from config
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
  } else { // Optimize on learnable parameters. Hamiltonian: no bounds, Lindblad: >=0, Transfer: >= 0
    int nparamsH = timestepper->mastereq->learning->getNParamsHamiltonian();
    int nparamsL = timestepper->mastereq->learning->getNParamsLindblad();
    int nparamsT = timestepper->mastereq->learning->getNParamsTransfer();
    assert(ndesign = nparamsH + nparamsL + nparamsT);
    for (int i=0; i<nparamsH; i++) {
      double boundval = 1e6;
      VecSetValue(xupper, i,     boundval, INSERT_VALUES);
      VecSetValue(xlower, i, -1.*boundval, INSERT_VALUES);
    }
    for (int i=nparamsH; i<nparamsH+nparamsL; i++) {
      double boundval = 1e6;
      VecSetValue(xupper, i, boundval, INSERT_VALUES);
      VecSetValue(xlower, i, 0.0, INSERT_VALUES);
    }
    for (int i=nparamsH+nparamsL; i<nparamsH+nparamsL+nparamsT; i++) {
      double boundval = 1e6;
      VecSetValue(xupper, i, boundval, INSERT_VALUES);
      VecSetValue(xlower, i, 0.0, INSERT_VALUES);
    }
  }
  VecAssemblyBegin(xlower); VecAssemblyEnd(xlower);
  VecAssemblyBegin(xupper); VecAssemblyEnd(xupper);
  // VecView(xlower, NULL);

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

  /* Allocat xinit, xtmp */
  VecCreateSeq(PETSC_COMM_SELF, ndesign, &xinit);
  VecSetFromOptions(xinit);
  VecZeroEntries(xinit);
  VecCreateSeq(PETSC_COMM_SELF, ndesign, &xtmp);
  VecSetFromOptions(xtmp);
  VecZeroEntries(xtmp);

  /* Allocat reduce gradient for time-stepper */
  VecCreateSeq(PETSC_COMM_SELF, ndesign, &(timestepper->redgrad));
  VecSetFromOptions(timestepper->redgrad);
  VecAssemblyBegin(timestepper->redgrad);
  VecAssemblyEnd(timestepper->redgrad);


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
  VecDestroy(&xinit);
  VecDestroy(&xtmp);
  VecDestroy(&(timestepper->redgrad));

  for (int i = 0; i < store_finalstates.size(); i++) {
    VecDestroy(&(store_finalstates[i]));
  }

  TaoDestroy(&tao);
}



double OptimProblem::evalF(const Vec x) {

  MasterEq* mastereq = timestepper->mastereq;
  if (mpirank_world == 0 && !quietmode) printf("EVAL F... \n");

  /* Reset optimizer's objective function measures*/
  obj_cost  = 0.0;
  obj_loss = 0.0;
  obj_regul = 0.0;
  obj_penal = 0.0;
  obj_penal_dpdm = 0.0;
  obj_penal_energy = 0.0;
  obj_penal_variation = 0.0;
  fidelity = 0.0;

  /* Iterate over control pulses */
  int npulseiters = 1;
  if (!x_is_control) npulseiters = mastereq->learning->data->getNPulses_local();
  for (int ipulse_local = 0; ipulse_local < npulseiters; ipulse_local++){

    /* Get global id if the pulse */
    int ipulse = mpirank_optim * npulseiters + ipulse_local;
    // if (!quietmode) printf("%dx%d: evalF: Pulse number ipulse=%d ...\n", mpirank_optim, mpirank_init, ipulse);

    /* Set current optimization vector x */
    if (x_is_control) { // Optimize on control parameters
      mastereq->setControlAmplitudes(x); 
    } else { // Optimize on learnable parameters
      mastereq->learning->setLearnParams(x); 

      /* Make sure the control pulse matches the data, if given */
      mastereq->setControlFromData(ipulse);

      // TEST: write expected energy of the Training data.
      for (int iosc=0; iosc<mastereq->nlevels.size(); iosc++){
        std::string filename_expEnergy = output->datadir + "/TrainingData_pulse"+std::to_string(ipulse)+"_expectedEnergy"+std::to_string(iosc)+".dat"; 
        mastereq->learning->data->writeExpectedEnergy(filename_expEnergy.c_str(), ipulse, iosc);
      }
      std::string filename_rho_Re = output->datadir + "/TrainingData_pulse"+std::to_string(ipulse)+"_rho_Re.dat"; 
      std::string filename_rho_Im = output->datadir + "/TrainingData_pulse"+std::to_string(ipulse)+"_rho_Im.dat"; 
      mastereq->learning->data->writeFullstate(filename_rho_Re.c_str(), filename_rho_Im.c_str(), ipulse);
    }

    /*  Iterate over initial condition */
    double obj_cost_re = 0.0;
    double obj_cost_im = 0.0;
    double fidelity_re = 0.0;
    double fidelity_im = 0.0;
    for (int iinit = 0; iinit < ninit_local; iinit++) {
      /* Prepare the initial condition in [rank * ninit_local, ... , (rank+1) * ninit_local - 1] */
      int iinit_global = mpirank_init * ninit_local + iinit;
      int initid = optim_target->prepareInitialState(iinit_global, ninit, timestepper->mastereq->nlevels, timestepper->mastereq->nessential, rho_t0);
      // if (!quietmode) printf("%dx%d:    -> Initial condition id=%d ...\n",mpirank_optim, mpirank_init, initid);

      /* Run forward with initial condition initid */
      optim_target->prepareTargetState(rho_t0);
      Vec finalstate = timestepper->solveODE(initid, rho_t0, ipulse);

      /* If learning: add to loss function */
      double loss_local = mastereq->learning->getLoss();
      if (!x_is_control)
        printf("%dx%d: Local loss for pulse %d = %1.14e\n", mpirank_optim, mpirank_init, ipulse, loss_local);
      obj_loss += obj_weights[iinit] * loss_local;

      /* Add to integral penalty terms */
      obj_penal += obj_weights[iinit] * gamma_penalty * timestepper->penalty_integral;
      obj_penal_dpdm += obj_weights[iinit] * gamma_penalty_dpdm * timestepper->penalty_dpdm;
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

    } // end of loop over initial conditions 

    /* Sum the fidelity cost function from initial conditions processors, for the current pulse iteration */
    double mycost_re = obj_cost_re;
    double mycost_im = obj_cost_im;
    double myfidelity_re = fidelity_re;
    double myfidelity_im = fidelity_im;
    MPI_Allreduce(&mycost_re, &obj_cost_re, 1, MPI_DOUBLE, MPI_SUM, comm_init);
    MPI_Allreduce(&mycost_im, &obj_cost_im, 1, MPI_DOUBLE, MPI_SUM, comm_init);
    MPI_Allreduce(&myfidelity_re, &fidelity_re, 1, MPI_DOUBLE, MPI_SUM, comm_init);
    MPI_Allreduce(&myfidelity_im, &fidelity_im, 1, MPI_DOUBLE, MPI_SUM, comm_init);
    // Add the fidelity of this pulse: If Schroedinger, need to compute the absolute value: Fid= |\sum_i \phi^\dagger \phi_target|^2
    if (timestepper->mastereq->lindbladtype == LindbladType::NONE) {
      fidelity += pow(fidelity_re, 2.0) + pow(fidelity_im, 2.0);
    } else {
      fidelity += fidelity_re; 
    }
    obj_cost += optim_target->finalizeJ(obj_cost_re, obj_cost_im);

  } // end of loop over pulses

  /* Sum up penalty and loss from all processors */
  double mypen = obj_penal;
  double myloss= obj_loss;
  double mypen_dpdm = obj_penal_dpdm;
  double mypenen = obj_penal_energy;
  MPI_Allreduce(&myloss, &obj_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mypen, &obj_penal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mypen_dpdm, &obj_penal_dpdm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mypenen, &obj_penal_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // Scale loss with respect to the number of pulses 
  obj_loss /= mastereq->learning->data->getNPulses();

  /* Evaluate regularization objective += gamma/2 * ||x-x0||^2*/
  double xnorm=0.0;
  if (!gamma_tik_interpolate){  // ||x||^2
    if (!gamma_tik_onenorm) {
      VecNorm(x, NORM_2, &xnorm);
    } else {
      int size = 0;
      const double *xptr;
      VecGetSize(x, &size);
      VecGetArrayRead(x, &xptr);
      for (int i=0; i<size; i++){
        xnorm += fabs(xptr[i]);
      }
      VecRestoreArrayRead(x, &xptr);
    }
  } else {
    VecCopy(x, xtmp);
    VecAXPY(xtmp, -1.0, xinit);    // xtmp =  x - x_0
    VecNorm(xtmp, NORM_2, &xnorm);
  }
  if (!gamma_tik_onenorm) obj_regul = gamma_tik / 2. * pow(xnorm,2.0);
  else obj_regul = gamma_tik * xnorm;

  /* Evaluate penality term for control variation */
  double var_reg = 0.0;
  if (x_is_control) {
    for (int iosc = 0; iosc < timestepper->mastereq->getNOscillators(); iosc++){
      var_reg += timestepper->mastereq->getOscillator(iosc)->evalControlVariation(); // uses Oscillator::params instead of 'x'
    }
  }
  obj_penal_variation = 0.5*gamma_penalty_variation*var_reg; 

  /* Sum, store and return objective value */
  if (x_is_control) {
    objective = obj_cost + obj_regul + obj_penal + obj_penal_dpdm + obj_penal_energy + obj_penal_variation;
  } else {
    objective = obj_loss + obj_regul;
  }

  /* Output */
  if (mpirank_world == 0 && !quietmode) {
    if (x_is_control){
      std::cout<< "Objective = " << std::scientific<<std::setprecision(14) << obj_cost << " + " << obj_regul << " + " << obj_penal << " + " << obj_penal_dpdm << " + " << obj_penal_energy << " + " << obj_penal_variation << std::endl;
      std::cout<< "Fidelity = " << fidelity  << std::endl;
    } else {
      std::cout<< "Learning loss = " << std::scientific<<std::setprecision(14) << obj_loss << " + " << obj_regul << std::endl;
    }
  }

  return objective;
}



void OptimProblem::evalGradF(const Vec x, Vec G){

  MasterEq* mastereq = timestepper->mastereq;
  if (mpirank_world == 0 && !quietmode) std::cout<< "EVAL GRAD F... " << std::endl;

  /* Reset optimizer's objective function measures*/
  obj_cost = 0.0;
  obj_loss = 0.0;
  obj_regul = 0.0;
  obj_penal = 0.0;
  obj_penal_dpdm = 0.0;
  obj_penal_energy = 0.0;
  obj_penal_variation = 0.0;
  fidelity = 0.0;

   /* Reset Gradient */
  VecZeroEntries(G);

  /* Derivative of regulatization term gamma / 2 ||x||^2 (ADD ON ONE PROC ONLY!) */
  // if (mpirank_init == 0 && mpirank_optim == 0) { // TODO: Which one?? 
  if (mpirank_world== 0 ) {
    if (!gamma_tik_onenorm) {
      VecAXPY(G, gamma_tik, x);   // + gamma_tik * x
    } else {
      double *Gptr; 
      const double *xptr;
      int size;
      VecGetSize(G, &size);
      VecGetArray(G, &Gptr);
      VecGetArrayRead(x, &xptr);
      for (int i=0; i<size; i++){
        if (xptr[i] > 0) {
          Gptr[i] += gamma_tik;
        }
        else if (xptr[i] < 0){
          Gptr[i] -= gamma_tik;
        }
      }
      VecRestoreArray(G, &Gptr);
      VecRestoreArrayRead(x, &xptr);
    }
    if (gamma_tik_interpolate){
      VecAXPY(G, -1.0*gamma_tik, xinit); // -gamma_tik * xinit
    }
  }

  // Derivative of penalization of control variation 
  if (mpirank_init == 0 && x_is_control) {
    double var_reg_bar = 0.5*gamma_penalty_variation;
    int skip_to_oscillator = 0;
    for (int iosc = 0; iosc < timestepper->mastereq->getNOscillators(); iosc++){
      Oscillator* osc = timestepper->mastereq->getOscillator(iosc);
      osc->evalControlVariationDiff(G, var_reg_bar, skip_to_oscillator);
      skip_to_oscillator += osc->getNParams();
    }
  }

  /* Iterate over control pulses */
  int npulseiters = 1;
  if (!x_is_control) npulseiters = mastereq->learning->data->getNPulses_local();
  for (int ipulse_local = 0; ipulse_local < npulseiters; ipulse_local++){

    /* Get global id if the pulse */
    int ipulse = mpirank_optim * npulseiters + ipulse_local;
    // if (!quietmode) printf("%dx%d: evalGradF: Pulse number ipulse=%d ...\n", mpirank_optim, mpirank_init, ipulse);

    /* Set current optimization vector x */
    if (x_is_control) { // Optimize on control parameters
      mastereq->setControlAmplitudes(x); 
    } else { // Optimize on learnable parameters
      mastereq->learning->setLearnParams(x); 

      /* Make sure the control pulse matches the data */
      mastereq->setControlFromData(ipulse);
    }

    /*  Iterate over initial condition */
    double obj_cost_re = 0.0;
    double obj_cost_im = 0.0;
    double fidelity_re = 0.0;
    double fidelity_im = 0.0;
    for (int iinit = 0; iinit < ninit_local; iinit++) {

      /* Prepare the initial condition */
      int iinit_global = mpirank_init * ninit_local + iinit;
      int initid = optim_target->prepareInitialState(iinit_global, ninit, timestepper->mastereq->nlevels, timestepper->mastereq->nessential, rho_t0);

      /* --- Solve primal --- */

      /* Run forward with initial condition rho_t0 */
      optim_target->prepareTargetState(rho_t0);
      Vec finalstate = timestepper->solveODE(initid, rho_t0, ipulse);

      /* Store the final state for the Schroedinger solver */
      if (timestepper->mastereq->lindbladtype == LindbladType::NONE) VecCopy(finalstate, store_finalstates[iinit]);

      /* If learning: add to loss function */
      obj_loss += obj_weights[iinit] * mastereq->learning->getLoss();

      /* Add to integral penalty terms */
      obj_penal += obj_weights[iinit] * gamma_penalty * timestepper->penalty_integral;
      obj_penal_dpdm += obj_weights[iinit] * gamma_penalty_dpdm * timestepper->penalty_dpdm;
      obj_penal_energy += obj_weights[iinit] * gamma_penalty_energy * timestepper->energy_penalty_integral;

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

      /* If Lindblas solver, compute adjoint for this initial condition. Otherwise (Schroedinger solver), compute adjoint only after all initial conditions have been propagated through (separate loop below) */
      if (timestepper->mastereq->lindbladtype != LindbladType::NONE) {

        /* Reset adjoint */
        VecZeroEntries(rho_t0_bar);

        double Jbar_loss = 0.0;
        if (x_is_control) {
          /* Terminal condition for adjoint variable: Derivative of final time objective J */
          double obj_cost_re_bar, obj_cost_im_bar;
          optim_target->finalizeJ_diff(obj_cost_re, obj_cost_im, &obj_cost_re_bar, &obj_cost_im_bar);
          optim_target->evalJ_diff(finalstate, rho_t0_bar, obj_weights[iinit]*obj_cost_re_bar, obj_weights[iinit]*obj_cost_im_bar);
        } else {
          Jbar_loss = obj_weights[iinit]/mastereq->learning->data->getNPulses();
        }

        /* Derivative of time-stepping */
        timestepper->solveAdjointODE(initid, rho_t0_bar, finalstate, obj_weights[iinit] * gamma_penalty, obj_weights[iinit]*gamma_penalty_dpdm, obj_weights[iinit]*gamma_penalty_energy, Jbar_loss, ipulse);

        /* Add to optimizers's gradient */
        VecAXPY(G, 1.0, timestepper->redgrad);
      }
    } // end of loop over initial conditions

    /* Sum up from initial conditions processors for the current pulse iteration */
    double mycost_re = obj_cost_re;
    double mycost_im = obj_cost_im;
    double myfidelity_re = fidelity_re;
    double myfidelity_im = fidelity_im;
    MPI_Allreduce(&mycost_re, &obj_cost_re, 1, MPI_DOUBLE, MPI_SUM, comm_init);
    MPI_Allreduce(&mycost_im, &obj_cost_im, 1, MPI_DOUBLE, MPI_SUM, comm_init);
    MPI_Allreduce(&myfidelity_re, &fidelity_re, 1, MPI_DOUBLE, MPI_SUM, comm_init);
    MPI_Allreduce(&myfidelity_im, &fidelity_im, 1, MPI_DOUBLE, MPI_SUM, comm_init);

    /* Add the fidelity: If Schroedinger, need to compute the absolute value: Fid= |\sum_i \phi^\dagger \phi_target|^2 */
    if (timestepper->mastereq->lindbladtype == LindbladType::NONE) {
      fidelity += pow(fidelity_re, 2.0) + pow(fidelity_im, 2.0);
    } else {
      fidelity += fidelity_re; 
    }
    obj_cost += optim_target->finalizeJ(obj_cost_re, obj_cost_im);

  } // end of loop over npulses 

  /* Sum up penalty and loss from all processors */
  double mypen = obj_penal;
  double myloss= obj_loss;
  double mypen_dpdm = obj_penal_dpdm;
  double mypenen = obj_penal_energy;
  MPI_Allreduce(&myloss, &obj_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mypen, &obj_penal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mypen_dpdm, &obj_penal_dpdm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mypenen, &obj_penal_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // Scale loss with respect to the number of pulses 
  obj_loss /= mastereq->learning->data->getNPulses();

  /* Evaluate regularization objective += gamma/2 * ||x||^2*/
  double xnorm=0.0;
  if (!gamma_tik_interpolate){  // ||x||^2
    if (!gamma_tik_onenorm) {
      VecNorm(x, NORM_2, &xnorm);
    } else {
      int size = 0;
      const double *xptr;
      VecGetSize(x, &size);
      VecGetArrayRead(x, &xptr);
      for (int i=0; i<size; i++){
        xnorm += fabs(xptr[i]);
      }
      VecRestoreArrayRead(x, &xptr);
    }
  } else {
    VecCopy(x, xtmp);
    VecAXPY(xtmp, -1.0, xinit);    // xtmp =  x_k - x_0
    VecNorm(xtmp, NORM_2, &xnorm);
  }
  if (!gamma_tik_onenorm) obj_regul = gamma_tik / 2. * pow(xnorm,2.0);
  else obj_regul = gamma_tik * xnorm;

  /* Evaluate penalty term for control parameter variation */
  double var_reg = 0.0;
  if (x_is_control) {
    for (int iosc = 0; iosc < timestepper->mastereq->getNOscillators(); iosc++){
      var_reg += timestepper->mastereq->getOscillator(iosc)->evalControlVariation(); // uses Oscillator::params instead of 'x'
    }
  }
  obj_penal_variation = 0.5*gamma_penalty_variation*var_reg; 

  /* Sum, store and return objective value */
  if (x_is_control) {
    objective = obj_cost + obj_regul + obj_penal + obj_penal_dpdm + obj_penal_energy + obj_penal_variation;
  } else {
    objective = obj_loss + obj_regul;
  }

  /* For Schroedinger solver: Solve adjoint equations for all initial conditions here. */
  if (timestepper->mastereq->lindbladtype == LindbladType::NONE) {
    printf("ERROR, THIS NEEDS CHANGE! !\n");
    exit(1);

    // Iterate over all initial conditions 
    for (int iinit = 0; iinit < ninit_local; iinit++) {
      int iinit_global = mpirank_init * ninit_local + iinit;

      /* Recompute the initial state and target */
      int initid = optim_target->prepareInitialState(iinit_global, ninit, timestepper->mastereq->nlevels, timestepper->mastereq->nessential, rho_t0);
      optim_target->prepareTargetState(rho_t0);
     
      /* Reset adjoint */
      VecZeroEntries(rho_t0_bar);

      double Jbar_loss = 0.0;
      if (x_is_control) {
        /* Terminal condition for adjoint variable: Derivative of final time objective J */
        double obj_cost_re_bar, obj_cost_im_bar;
        // optim_target->finalizeJ_diff(obj_cost_re, obj_cost_im, &obj_cost_re_bar, &obj_cost_im_bar);
        // optim_target->evalJ_diff(store_finalstates[iinit], rho_t0_bar, obj_weights[iinit]*obj_cost_re_bar, obj_weights[iinit]*obj_cost_im_bar);
      } else {
        Jbar_loss = obj_weights[iinit];
      }

      /* Derivative of time-stepping */
      printf("TODO. PUlsenum.\n");
      exit(1);
      timestepper->solveAdjointODE(initid, rho_t0_bar, store_finalstates[iinit], obj_weights[iinit] * gamma_penalty, obj_weights[iinit]*gamma_penalty_dpdm, obj_weights[iinit]*gamma_penalty_energy, Jbar_loss, 0);

      /* Add to optimizers's gradient */
      VecAXPY(G, 1.0, timestepper->redgrad);
    } // end of initial condition loop 
  } // end of adjoint for Schroedinger

  /* Sum up the gradient from all processors */
  PetscScalar* grad; 
  VecGetArray(G, &grad);
  for (int i=0; i<ndesign; i++) {
    mygrad[i] = grad[i];
  }
  MPI_Allreduce(mygrad, grad, ndesign, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  VecRestoreArray(G, &grad);

  /* Compute and store gradient norm */
  VecNorm(G, NORM_2, &(gnorm));

  /* Output */
  if (mpirank_world == 0 && !quietmode) {
    if (x_is_control){
      std::cout<< "Objective = " << std::scientific<<std::setprecision(14) << obj_cost << " + " << obj_regul << " + " << obj_penal << " + " << obj_penal_dpdm << " + " << obj_penal_energy << " + " << obj_penal_variation << std::endl;
      std::cout<< "Fidelity = " << fidelity << std::endl;
    } else {
      std::cout<< "Learning loss = " << std::scientific<<std::setprecision(14) << obj_loss << " + " << obj_regul << std::endl;
    }
  }
}

void OptimProblem::solve(Vec xinit) {
  TaoSetSolution(tao, xinit);
  TaoSolve(tao);
}

void OptimProblem::getStartingPoint(Vec xinit){
  MasterEq* mastereq = timestepper->mastereq;

  if (x_is_control) {// optim wrt controls 
    // copy from initialization of oscillator constructor
    PetscScalar* xptr;
    VecGetArray(xinit, &xptr);
    int shift = 0;
    for (int ioscil = 0; ioscil<mastereq->getNOscillators(); ioscil++){
      mastereq->getOscillator(ioscil)->getParams(xptr + shift);
      shift += mastereq->getOscillator(ioscil)->getNParams();
    }
    VecRestoreArray(xinit, &xptr);

  } else { // optim wrt learning parameters 
    // Copy from initialization in learning constructor
    PetscScalar* xptr;
    VecGetArray(xinit, &xptr);
    mastereq->learning->getLearnParams(xptr);
    VecRestoreArray(xinit, &xptr);
  }

  /* Write initial optimization paramters to file */
  output->writeParams(xinit);
}


void OptimProblem::getSolution(Vec* param_ptr){
  
  /* Get ref to optimized parameters */
  Vec params;
  TaoGetSolution(tao, &params);
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
  TaoGetSolution(tao, &params);

  /* Grab some output stuff */
  double obj_cost = ctx->getCostT();
  double obj_loss = ctx->getLoss();
  double obj_regul = ctx->getRegul();
  double obj_penal = ctx->getPenalty();
  double obj_penal_dpdm = ctx->getPenaltyDpDm();
  double obj_penal_energy = ctx->getPenaltyEnergy();
  double obj_penal_variation= ctx->getPenaltyVariation();
  double F_avg = ctx->getFidelity();

  /* Additional Stopping criteria */
  bool lastIter = false;
  std::string finalReason_str = "";
  if (ctx->x_is_control) {
    if (1.0 - F_avg <= ctx->getInfTol()) {
      finalReason_str = "Optimization converged with small infidelity.";
      TaoSetConvergedReason(tao, TAO_CONVERGED_USER);
      lastIter = true;
    } else if (obj_cost <= ctx->getFaTol()) {
      finalReason_str = "Optimization converged with small final time cost.";
      TaoSetConvergedReason(tao, TAO_CONVERGED_USER);
      lastIter = true;
    }
  } else {
    if (obj_loss <= ctx->getFaTol()) {
      finalReason_str = "Optimization converged with small Loss.";
      TaoSetConvergedReason(tao, TAO_CONVERGED_USER);
      lastIter = true;
    }
  }
  if (iter == ctx->getMaxIter()) {
    finalReason_str = "Optimization stopped at maximum number of iterations.";
    lastIter = true;
  } else if (gnorm < ctx->getGaTol()) {
    finalReason_str = "OPtimization converged with small gradient norm.";
    lastIter=true;
  }

  /* First iteration: Header for screen output of optimization history */
  if (iter == 0 && ctx->getMPIrank_world() == 0) {
    if (ctx->x_is_control) {
      std::cout<<  "    Objective             Tikhonov                Penalty-Leakage        Penalty-StateVar       Penalty-TotalEnergy    Penalty-CtrlVar" << std::endl;
    } else {
      std::cout<<  "    Objective             Tikhonov                GradNorm " << std::endl;
    }
  }

  /* Every <optim_monitor_freq> iterations: Output of optimization history */
  if (iter % ctx->output->optim_monitor_freq == 0 ||lastIter) {
    // Add to optimization history file 
    double costT_output = obj_cost;
    if (!ctx->x_is_control) costT_output = obj_loss;
    ctx->output->writeOptimFile(iter, f, gnorm, deltax, F_avg, costT_output, obj_regul, obj_penal, obj_penal_dpdm, obj_penal_energy, obj_penal_variation);
    // Screen output 
    if (ctx->getMPIrank_world() == 0) {
      std::cout<< iter <<  "  " << std::scientific<<std::setprecision(14);
      if (ctx->x_is_control){
        std::cout << obj_cost << " + " << obj_regul << " + " << obj_penal << " + " << obj_penal_dpdm << " + " << obj_penal_energy << " + " << obj_penal_variation;
        std::cout<< "  Fidelity = " << F_avg;
        std::cout<< "  ||Grad|| = " << gnorm;
      } else {
        std::cout << obj_loss << " + " << obj_regul << ".   " << gnorm;
      }
      std::cout<< std::endl;
    }
  }

  /* Print optimization parameters to file */
  ctx->output->writeParams(params);

  /* Last iteration: Print solution, controls and trajectory data to files */
  if (lastIter) {
    // ctx->output->writeControls(params, ctx->timestepper->mastereq, ctx->timestepper->ntime, ctx->timestepper->dt);

    // do one last forward evaluation while writing trajectory files
    ctx->timestepper->writeDataFiles = true;
    ctx->evalF(params); 

    // Print stopping reason to screen
    if (ctx->getMPIrank_world() == 0){
      std::cout<< finalReason_str << std::endl;
    }
  }


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
