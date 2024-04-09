#include "optimproblem.hpp"

OptimProblem::OptimProblem(MapParam config, TimeStepper* timestepper_, MPI_Comm comm_init_, MPI_Comm comm_time_, int ninit_, int nwindows_, Output* output_, bool quietmode_){

  timestepper = timestepper_;
  ninit = ninit_;
  nwindows = nwindows_;
  output = output_;
  quietmode = quietmode_;
  /* Reset */
  objective = 0.0;

  /* Store communicators */
  comm_init = comm_init_;
  comm_time = comm_time_;
  /* Store ranks and sizes of communicators */
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize_world);
  MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank_space);
  MPI_Comm_size(PETSC_COMM_WORLD, &mpisize_space);
  MPI_Comm_rank(comm_init, &mpirank_init);
  MPI_Comm_size(comm_init, &mpisize_init);
  MPI_Comm_rank(comm_time, &mpirank_time);
  MPI_Comm_size(comm_time, &mpisize_time);

  /* Store other optimization parameters */
  gamma_tik = config.GetDoubleParam("optim_regul", 1e-4);
  gamma_tik_interpolate = config.GetBoolParam("optim_regul_interpolate", false, false);
  gamma_penalty_energy = config.GetDoubleParam("optim_penalty_energy", 0.0);
  gamma_penalty = config.GetDoubleParam("optim_penalty", 0.0);
  gamma_penalty_dpdm = config.GetDoubleParam("optim_penalty_dpdm", 0.0);
  penalty_param = config.GetDoubleParam("optim_penalty_param", 0.5);
  gatol = config.GetDoubleParam("optim_atol", 1e-8);
  fatol = config.GetDoubleParam("optim_ftol", 1e-8);
  inftol = config.GetDoubleParam("optim_inftol", 1e-5);
  grtol = config.GetDoubleParam("optim_rtol", 1e-4);
  interm_tol = config.GetDoubleParam("optim_interm_tol", 1e-4, false);
  maxiter = config.GetIntParam("optim_maxiter", 200);
  mu = config.GetDoubleParam("optim_mu", 0.0, false);
  unitarize_interm_ic = config.GetBoolParam("optim_unitarize", false);
  al_max_outer = config.GetIntParam("optim_maxouter", 1);
  scalefactor_states = config.GetDoubleParam("scalefactor_states", 1.0);
  if (gamma_penalty_dpdm > 1e-13 && timestepper->mastereq->lindbladtype != LindbladType::NONE){
    if (mpirank_world == 0) {
      printf("Warning: Disabling DpDm penalty term because it is not implemented for the Lindblad solver.\n");
    }
    gamma_penalty_dpdm = 0.0;
  }

 

  /* Store number of initial conditions per init-processor group */
  ninit_local = ninit / mpisize_init; 

  /* Store number of local windows per time-processor group */
  nwindows_local = nwindows / mpisize_time;
  // TODO. For now sanity check whether nwindows is equally divisible by number or processors for time.
  if (nwindows % mpisize_time != 0){
    printf("ERROR: For now, need nwindows MOD mpisize_time == 0.\n");
    exit(1);
  }

  /*  Allocate storage for the final states at the end of each time window */
  store_finalstates.resize(nwindows_local);
  for (int m = 0; m < nwindows_local; m++) {
    store_finalstates[m].clear();
    for (int i = 0; i < ninit_local; i++) {
      Vec state;
      VecCreateSeq(PETSC_COMM_SELF, 2*timestepper->mastereq->getDim(), &state);
      VecSetFromOptions(state);
      store_finalstates[m].push_back(state);
    }
  }

  /* Store number of optimization parameters */
  // First add all that correspond to the splines
  ndesign = 0;
  for (int ioscil = 0; ioscil < timestepper->mastereq->getNOscillators(); ioscil++) {
      ndesign += timestepper->mastereq->getOscillator(ioscil)->getNParams(); 
  }
  // Then all state variables N*(M-1) for each of the initial conditions (real and imag)
  nstate = (int) 2*timestepper->mastereq->getDimRho() * (nwindows-1) * ninit;
  if (mpirank_world == 0 && !quietmode) std::cout<< "noptimvars = " << ndesign << "(controls) + " << nstate << "(states) = " << ndesign + nstate  << std::endl;

  /* Allocate optimization variable xinit and lambda */
  // Determine local rank sizes. Adding design to the very first processor. TODO: Divide by number of local time windows, add ghost layers for alpha
  PetscInt local_size = ninit_local * 2*timestepper->mastereq->getDimRho() * nwindows_local ;
  if (mpirank_time == mpisize_time-1) local_size -= ninit_local * 2*timestepper->mastereq->getDimRho(); // remove last windows states
  VecCreateMPI(PETSC_COMM_WORLD, local_size, PETSC_DETERMINE, &lambda);
  VecSetFromOptions(lambda);
  VecSet(lambda, 0.0);
  VecAssemblyBegin(lambda);
  VecAssemblyEnd(lambda);

  // xinit also has the control parameters
  local_size = ninit_local * 2*timestepper->mastereq->getDimRho() * nwindows_local ;
  if (mpirank_time == 0) {
    local_size -= ninit_local * 2*timestepper->mastereq->getDimRho(); // remove first windows states
  }
  if (mpirank_world == 0) {
    local_size += ndesign;  // Add design to very first processor for the state
  }
  VecCreateMPI(PETSC_COMM_WORLD, local_size, PETSC_DETERMINE, &xinit);
  VecSetFromOptions(xinit);
  VecAssemblyBegin(xinit);
  VecAssemblyEnd(xinit);

  // VecSet(lambda, 1.0);
  // VecSet(xinit, 1.0);

  // test sizes
  int global_size;
  VecGetSize(xinit, &global_size);
  assert(global_size == getNoptimvars());

  // Create scatter context for x_alpha
  int *ids_all = new int[ndesign];
  for (int i=0; i< ndesign; i++){
    ids_all[i] = i;
  }
  IS IS_alldesign;
  ISCreateGeneral(PETSC_COMM_SELF,ndesign,ids_all,PETSC_COPY_VALUES, &IS_alldesign);
  delete [] ids_all;
  VecCreateSeq(PETSC_COMM_SELF, ndesign, &x_alpha);
  VecScatterCreate(xinit, IS_alldesign, x_alpha, IS_alldesign, &scatter_alpha);

  /* allocate reduced gradient of timestepper */
  VecDuplicate(x_alpha, &(timestepper->redgrad));

  // Gather the processor distribution of xinit and lambda
  int istart_x, istart_lambda, istop_x, istop_lambda;
  VecGetOwnershipRange(xinit, &istart_x, &istop_x); 
  VecGetOwnershipRange(lambda, &istart_lambda, &istop_lambda); 
  int istart_x_all[mpisize_time];
  MPI_Allgather(&istart_x, 1, MPI_INT, istart_x_all, 1, MPI_INT, comm_time);

  /* Create index set to access intermediate states from global vector */
  int xdim = timestepper->mastereq->getDimRho()*2; // state dimension. x2 for real and imag. 
  int *ids_m_ic = new int[xdim];
  IS_interm_states.resize(nwindows);
  IS_interm_lambda.resize(nwindows);
  for (int m=0; m<nwindows; m++) {
    IS_interm_states[m].clear();
    IS_interm_lambda[m].clear();
    for (int ic=0; ic<ninit; ic++){
      IS state_m_ic;
      IS lambda_m_ic;
      int nelems = 0;
      // figure out which processor stores this part of the vector. Leave other index sets empty.
      if (mpirank_init == ic / ninit_local && mpirank_time == m / nwindows_local 
         && m > 0){
        nelems = xdim;
      }
      int skip = istart_x + (ic % ninit_local)*xdim;
      if (mpirank_time > 0) skip += ( m    % nwindows_local)*ninit_local *xdim;
      else                  skip += ((m-1) % nwindows_local)*ninit_local *xdim;
      if (mpirank_init==0 && mpirank_time==0) skip += ndesign;
      for (int i=0; i<xdim; i++){
        ids_m_ic[i] = skip + i;
        // ids_m_ic[i] = skip_alpha + globalID_x + i;
      }
      ISCreateGeneral(PETSC_COMM_SELF, nelems, ids_m_ic, PETSC_COPY_VALUES, &state_m_ic);
      IS_interm_states[m].push_back(state_m_ic);
      // if (nelems>0) printf("%d: P_%d^%d: Created state IS[%d][%d] nelems %d, first id=%d \n", mpirank_world, mpirank_time, mpirank_init, m, ic, nelems, ids_m_ic[0]);

      // set the global IDs for lambda. Note how thiis does not skip ndesign,
      nelems=0;
      if (mpirank_init == ic / ninit_local && mpirank_time == m / nwindows_local && m<nwindows-1){
        nelems=xdim;
      }
      skip = istart_lambda + (ic % ninit_local)*xdim;
      skip += ( m % nwindows_local)*ninit_local *xdim;
      for (int i=0; i<xdim; i++){
        ids_m_ic[i] = skip + i;
      }
      ISCreateGeneral(PETSC_COMM_SELF, nelems, ids_m_ic, PETSC_COPY_VALUES, &lambda_m_ic);
      // if (nelems>0) printf("%d: P_%d^%d: Created lambda IS[%d][%d] nelems %d, first id=%d \n", mpirank_world, mpirank_time, mpirank_init, m, ic, nelems, ids_m_ic[0]);
      IS_interm_lambda[m].push_back(lambda_m_ic);
    }
  } 

  // Create scatter contexts for xnext
  VecCreateSeq(PETSC_COMM_SELF, xdim, &x_next);
  int *ids_state = new int[xdim];
  for (int i=0; i< xdim; i++){
    ids_state[i] = i;
  }
  scatter_xnext.resize(nwindows_local);  // THESE NEED TO BE LOCAL!!
  for (int m=0; m<nwindows_local; m++) {
    scatter_xnext[m].clear();
    for (int ic=0; ic<ninit_local; ic++){
      int nelems= 0;
      int skip = 0;
      int ic_global = mpirank_init * ninit_local + ic;
      int m_global  = mpirank_time * nwindows_local + m;

      if ( m_global < nwindows-1 ){ // dont scatter on the last interval
        nelems = xdim;

        // figure out which global starting index the next time window state has
        int mpirank_time_next = (m_global+1)/nwindows_local;
        int mpirank_init_next = (ic_global)/ninit_local;
        skip = istart_x_all[mpirank_time_next];
        skip += ic * xdim; 
        if (mpirank_time_next > 0) skip += ((m_global+1)%nwindows_local) * ninit_local *xdim;
        else                       skip += ( ( (m_global+1)%nwindows_local) -1) * ninit_local *xdim;
        if (mpirank_init_next==0 && mpirank_time_next==0) skip += ndesign;
      }
      for (int i=0; i<xdim; i++){
        ids_m_ic[i] = skip + i;
      }
      // if (nelems > 0)
      //   printf("%d: P_%d^%d: Scatter at local [%d][%d] nelems %d, taking from id=%d \n", mpirank_world, mpirank_time, mpirank_init, m, ic, nelems, ids_m_ic[0]);

      // finally create the scatter contexts
      IS IS_to, IS_from;
      VecScatter ctx_state;
      ISCreateGeneral(PETSC_COMM_SELF,nelems,ids_state,PETSC_COPY_VALUES, &IS_to);
      ISCreateGeneral(PETSC_COMM_SELF,nelems,ids_m_ic,PETSC_COPY_VALUES, &IS_from);
      VecScatterCreate(xinit, IS_from, x_next, IS_to, &ctx_state);
      scatter_xnext[m].push_back(ctx_state);

      // if (nelems > 0)
        // VecScatterView(scatter_xnext[m][ic], PETSC_VIEWER_STDOUT_SELF);
    }
  }
  delete [] ids_state;
  delete [] ids_m_ic;
   
  /* Store the optimization target */
  /* Allocate the initial condition vector and adjoint terminal state */
  VecCreateSeq(PETSC_COMM_SELF, 2*timestepper->mastereq->getDim(), &rho_t0); 
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

  /* Pass information on objective function to the time stepper needed for penalty objective function */
  timestepper->penalty_param = penalty_param;
  timestepper->gamma_penalty = gamma_penalty;
  timestepper->gamma_penalty_dpdm = gamma_penalty_dpdm;
  timestepper->gamma_penalty_energy = gamma_penalty_energy;
  timestepper->optim_target = optim_target;

  /* allocate storage for unscaled version of the optimization variable */
  VecDuplicate(xinit, &x);
  VecZeroEntries(x);

  /* allocate temporary storage needed for unitarization and its gradient */
  if (unitarize_interm_ic) {
    VecDuplicate(x, &x_b4unit); // Parallel vector, entire optimization variable
    VecDuplicate(x_next, &x0j);  // Sequential vector, only one initial state at one time-window
  }

  /* Store optimization bounds */
  VecDuplicate(xinit, &xlower);
  VecDuplicate(xinit, &xupper);
  /* bounds for control */
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
  /* intermediate conditions must be unbounded. Setting a large range. */
  double very_large = 1.0e15;
  int ilow, iupp;
  VecGetOwnershipRange(xlower, &ilow, &iupp);
  if (mpirank_world==0) ilow += getNdesign(); // skip design on first processor
  for (int k = ilow; k < iupp; k++) {
    VecSetValue(xlower, k, -very_large, INSERT_VALUES);
    VecSetValue(xupper, k, very_large, INSERT_VALUES);
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
  // mygrad = new double[getNoptimvars()];

  /* Allocate temporary storage of a state vector */
  VecCreateSeq(PETSC_COMM_SELF,2*timestepper->mastereq->getDim(), &disc);
  VecSetFromOptions(disc);
  VecSet(disc, 0.0);
  VecAssemblyBegin(disc);
  VecAssemblyEnd(disc);

  if (gamma_tik_interpolate) {
    // DISABLE FOR NOW
    printf("Warning: Disabling gamma_tik_interpolate for multiple shooting.\n");
    gamma_tik_interpolate = false;
  }
}


OptimProblem::~OptimProblem() {
  // delete [] mygrad;
  delete optim_target;
  VecDestroy(&rho_t0);
  VecDestroy(&rho_t0_bar);

  VecDestroy(&xinit);
  VecDestroy(&x_next);
  VecDestroy(&x);
  if (unitarize_interm_ic) {
    VecDestroy(&x_b4unit);
    VecDestroy(&x0j);
  }
  VecDestroy(&lambda);
  VecDestroy(&xlower);
  VecDestroy(&xupper);
  VecDestroy(&disc);

  for (int m = 0; m < store_finalstates.size(); m++) {
    for (int i = 0; i < store_finalstates[m].size(); i++) {
      VecDestroy(&(store_finalstates[m][i]));
    }
  }

  for (int m=0; m<IS_interm_states.size(); m++){
    for (int ic=0; ic<IS_interm_states[m].size(); ic++){
      ISDestroy(&(IS_interm_states[m][ic]));
      ISDestroy(&(IS_interm_lambda[m][ic]));
    }
  }

  VecScatterDestroy(&scatter_alpha);
  for (int m=0; m<scatter_xnext.size();m++){
    for (int ic=0; ic<scatter_xnext[m].size();ic++){
      VecScatterDestroy(&(scatter_xnext[m][ic]));
    }
  }
  TaoDestroy(&tao);
}

void OptimProblem::setGradTol(double newAtol, double newRtol){
  gatol = newAtol;
  grtol = newRtol;
  TaoSetTolerances(tao, gatol, PETSC_DEFAULT, grtol);
}

// EvalF optim var. x = (alpha, interm.states), 
double OptimProblem::evalF(const Vec xin, const Vec lambda_) {

  MasterEq* mastereq = timestepper->mastereq;

  if (mpirank_world == 0 && !quietmode) printf("EVAL F... \n");
  Vec finalstate = NULL;

  /* unscale the state part of 'x' */
  VecCopy(xin,x);
  VecScale(x, 1.0/scalefactor_states);
  // undo scaling of alpha
  double* xptr;
  VecGetArray(x, &xptr);
  if (mpirank_world==0) {
    for (int i=0; i<ndesign; i++)
      xptr[i] *=scalefactor_states;
  }
  VecRestoreArray(x, &xptr);

  /* Pass control vector to oscillators */
  // x_alpha is set only on first processor. Need to communicate x_alpha to all processors here. 
  VecScatterBegin(scatter_alpha, x, x_alpha, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(scatter_alpha, x, x_alpha, INSERT_VALUES, SCATTER_FORWARD);
  mastereq->setControlAmplitudes(x_alpha); 
 
  // Unitarize the initial conditions. Note: This modifies x!
  std::vector<std::vector<double>> vnorms;
  if (unitarize_interm_ic)
    unitarize(x, vnorms);

  /*  Iterate over initial condition */
  obj_cost  = 0.0;
  obj_regul = 0.0;
  obj_penal = 0.0;
  obj_penal_dpdm = 0.0;
  obj_penal_energy = 0.0;
  fidelity = 0.0;
  interm_discontinuity = 0.0; 
  obj_constraint = 0.0;
  double obj_cost_re = 0.0;
  double obj_cost_im = 0.0;
  double fidelity_re = 0.0;
  double fidelity_im = 0.0;
  double frob2 = 0.0; // For generalized infidelity, stores Tr(U'U)

  for (int iinit = 0; iinit < ninit_local; iinit++) {
    /* Iterate over local time windows */
    for (int iwindow=0; iwindow<nwindows_local; iwindow++){

      int iinit_global = mpirank_init * ninit_local + iinit;
      int iwindow_global = mpirank_time*nwindows_local + iwindow ; 
      // printf("%d:%d|%d: --> LOOP m = %d(%d) ic = %d(%d)\n", mpirank_world, mpirank_time, mpirank_init, iwindow_global, iwindow, iinit_global, iinit);

      /* Start sending next window's initial state, will be received by this processor */
      VecScatterBegin(scatter_xnext[iwindow][iinit], x, x_next, INSERT_VALUES, SCATTER_FORWARD);

      /* Get local state and lambda for this window. Probably safer to make a copy VecISCopy. */
      Vec x0, lag;  
      VecGetSubVector(x, IS_interm_states[iwindow_global][iinit_global], &x0);
      VecGetSubVector(lambda_, IS_interm_lambda[iwindow_global][iinit_global], &lag);

      /* Prepare the initial condition. Last window will also need it for J. */
      if ( iwindow_global == 0 || iwindow_global == nwindows -1 ) {
        optim_target->prepareInitialState(iinit_global, ninit, timestepper->mastereq->nlevels, timestepper->mastereq->nessential, rho_t0);
      }

      /* Solve the ODE in the local time window and store Su_i */
      if (iwindow_global == 0) {
        finalstate = timestepper->solveODE(1, rho_t0, iwindow_global * timestepper->ntime);
      } else {
        finalstate = timestepper->solveODE(1, x0, iwindow_global * timestepper->ntime);
      }
      VecCopy(finalstate, store_finalstates[iwindow][iinit]);


      /* Add to integral penalty term */
      obj_penal += obj_weights[iinit] * gamma_penalty * timestepper->penalty_integral;
      obj_penal_dpdm += obj_weights[iinit] * gamma_penalty_dpdm * timestepper->penalty_dpdm;
      obj_penal_energy += obj_weights[iinit] * gamma_penalty_energy* timestepper->energy_penalty_integral;

      /* Receive next window's initial state and lagrangian variable */
      VecScatterEnd(scatter_xnext[iwindow][iinit], x, x_next, INSERT_VALUES, SCATTER_FORWARD); 

      /* Evaluate J(finalstate) and add to final-time cost */
      if (iwindow_global == nwindows-1){
        double obj_iinit_re = 0.0;
        double obj_iinit_im = 0.0;
        double frob2_iinit; 
        optim_target->prepareTargetState(rho_t0);
        optim_target->evalJ(finalstate,  &obj_iinit_re, &obj_iinit_im, &frob2_iinit);
        obj_cost_re += obj_weights[iinit] * obj_iinit_re; 
        obj_cost_im += obj_weights[iinit] * obj_iinit_im;
        frob2 += frob2_iinit / ninit;
        /* Contributions to final-time (regular) fidelity */
        double fidelity_iinit_re = 0.0;
        double fidelity_iinit_im = 0.0;
        // NOTE: scalebypurity = false. TODO: Check.
        optim_target->HilbertSchmidtOverlap(finalstate, false, &fidelity_iinit_re, &fidelity_iinit_im);
        fidelity_re += fidelity_iinit_re / ninit; // Scale by 1/N
        fidelity_im += fidelity_iinit_im / ninit;
      }
      /* Else, add to constraint. */
      else {
        // eval || (Su - u) ||^2 and lambda^T(Su-u)
        double cdot, qnorm2;
        VecAXPY(finalstate, -1.0, x_next);  // finalstate = S(u_{i-1}) - u_i
        VecDot(finalstate, finalstate, &qnorm2); // q = || (Su - u) ||^2
        interm_discontinuity += qnorm2;
        VecDot(finalstate, lag, &cdot);   // c = lambda^T (Su - u)
        obj_constraint += 0.5 * mu * qnorm2 - cdot;
      }


      /* Restore local subvectors */
      VecRestoreSubVector(x, IS_interm_states[iwindow_global][iinit_global], &x0);
      VecRestoreSubVector(lambda_, IS_interm_lambda[iwindow_global][iinit_global], &lag);
    } // end for iwindow
    // printf("%d, (%d, %d): iinit obj_iinit: %f * (%1.14e + i %1.14e, Overlap=%1.14e + i %1.14e, Constraint=%1.14e\n", mpirank_world, mpirank_init, mpirank_time, obj_weights[iinit], obj_iinit_re, obj_iinit_im, fidelity_iinit_re, fidelity_iinit_im, constraint);
  } // end for iinit

  /* Sum up from initial conditions processors */
  double mypen = obj_penal;
  double mypen_dpdm = obj_penal_dpdm;
  double mypenen = obj_penal_energy;
  double mycost_re = obj_cost_re;
  double mycost_im = obj_cost_im;
  double my_frob2 = frob2; 
  double myfidelity_re = fidelity_re;
  double myfidelity_im = fidelity_im;
  double myconstraint = obj_constraint;
  double my_interm_disc = interm_discontinuity;
  // Should be comm_init and also comm_time! Currently, no Petsc Parallelization possible, hence (comm-init AND comm_time) = COMM_WORLD
  MPI_Allreduce(&mypen, &obj_penal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mypen_dpdm, &obj_penal_dpdm, 1, MPI_DOUBLE, MPI_SUM, comm_init); // SG: Penalty DPDM currently disabled.
  MPI_Allreduce(&mypenen, &obj_penal_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mycost_re, &obj_cost_re, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mycost_im, &obj_cost_im, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&myfidelity_re, &fidelity_re, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&myfidelity_im, &fidelity_im, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&myconstraint, &obj_constraint, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&my_interm_disc, &interm_discontinuity, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&my_frob2, &frob2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  /* Set the fidelity: If Schroedinger, need to compute the absolute value: Fid= |\sum_i \phi^\dagger \phi_target|^2 */
  if (timestepper->mastereq->lindbladtype == LindbladType::NONE) {
    fidelity = fidelity_re*fidelity_re + fidelity_im*fidelity_im;
  } else {
    fidelity = fidelity_re; 
  }
 
  /* Finalize the objective function */
  obj_cost = optim_target->finalizeJ(obj_cost_re, obj_cost_im, frob2);

  /* Evaluate regularization objective += gamma/2 * ||x-x0||^2*/
  double x_alpha_norm;
  if (!gamma_tik_interpolate){  // ||x_alpha||^2
    VecNorm(x_alpha, NORM_2, &x_alpha_norm);
  } 
  obj_regul = gamma_tik / 2. * pow(x_alpha_norm,2.0); // scale by 1/ndesign

  /* Sum, store and return objective value */
  objective = obj_cost + obj_regul + obj_penal + obj_penal_dpdm + obj_penal_energy + obj_constraint;

  /* Output */
  if (mpirank_world == 0) {
    std::cout<< "Objective = " << std::scientific<<std::setprecision(14) << obj_cost << " + " << obj_regul << " + " << obj_penal << " + " << obj_penal_dpdm << " + " << obj_penal_energy << " + " << obj_constraint <<std::endl;
    std::cout<< "Discontinuity = " << interm_discontinuity << std::endl;
    if (nwindows == 1) // Fidelity only makes sense with one window
      std::cout<< "Fidelity = " << fidelity  << std::endl;
  }

  return objective;
} // end evalF()



void OptimProblem::evalGradF(const Vec xin, const Vec lambda_, Vec G){

  MasterEq* mastereq = timestepper->mastereq;

  if (mpirank_world == 0 && !quietmode) std::cout<< "EVAL GRAD F... " << std::endl;
  Vec finalstate = NULL;
  Vec adjoint_ic = NULL;


  /* unscale the state part of 'x' */
  VecCopy(xin,x);
  VecScale(x, 1.0/scalefactor_states);
  // undo scaling of alpha
  double* xptr;
  VecGetArray(x, &xptr);
  if (mpirank_world==0) {
    for (int i=0; i<ndesign; i++)
      xptr[i] *=scalefactor_states;
  }
  VecRestoreArray(x, &xptr);

  /* Pass design vector x to oscillators */
  // x_alpha is set only on first processor. Need to communicate x_alpha to all processors here. 
  VecScatterBegin(scatter_alpha, x, x_alpha, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(scatter_alpha, x, x_alpha, INSERT_VALUES, SCATTER_FORWARD);
  mastereq->setControlAmplitudes(x_alpha); 

 
  // Unitarize the initial conditions. Note: This modifies x!
  std::vector<std::vector<double>> vnorms;
  if (unitarize_interm_ic)
  {
    // original x is required for gradient computation
    VecCopy(x, x_b4unit);
    unitarize(x, vnorms);
  }

  /* Reset Gradient */
  VecZeroEntries(G);

  /* Derivative of regulatization term gamma / 2 ||x||^2 */
  VecScatterBegin(scatter_alpha, x_alpha, G, INSERT_VALUES, SCATTER_REVERSE);
  VecScatterEnd(scatter_alpha, x_alpha, G, INSERT_VALUES, SCATTER_REVERSE);
  VecScale(G, gamma_tik);

  /*  Iterate over initial condition */
  obj_cost = 0.0;
  obj_regul = 0.0;
  obj_penal = 0.0;
  obj_penal_dpdm = 0.0;
  obj_penal_energy = 0.0;
  fidelity = 0.0;
  interm_discontinuity = 0.0; // for TaoMonitor
  obj_constraint = 0.0; 
  double obj_cost_re = 0.0;
  double obj_cost_im = 0.0;
  double fidelity_re = 0.0;
  double fidelity_im = 0.0;
  double constraint = 0.0;
  double frob2 = 0.0; 
  for (int iinit = 0; iinit < ninit_local; iinit++) {
    /* Iterate over local time windows */
    for (int iwindow=0; iwindow<nwindows_local; iwindow++){

      int iinit_global = mpirank_init * ninit_local + iinit;
      int iwindow_global = mpirank_time * nwindows_local + iwindow ;

      /* Start sending next window's initial state, will be received by this processor */
      VecScatterBegin(scatter_xnext[iwindow][iinit], x, x_next, INSERT_VALUES, SCATTER_FORWARD);

      /* Get local state and lambda for this window */
      Vec x0, lag;
      VecGetSubVector(x, IS_interm_states[iwindow_global][iinit_global], &x0);
      VecGetSubVector(lambda_, IS_interm_lambda[iwindow_global][iinit_global], &lag);

      /* Prepare the initial condition */
      if ( iwindow_global == 0 || iwindow_global == nwindows -1) {
        optim_target->prepareInitialState(iinit_global, ninit, timestepper->mastereq->nlevels, timestepper->mastereq->nessential, rho_t0);
      }

      /* Solve the ODE in the local time window */
      if (iwindow_global == 0) {
        finalstate = timestepper->solveODE(1, rho_t0, iwindow_global * timestepper->ntime);
      } else {
        finalstate = timestepper->solveODE(1, x0, iwindow_global * timestepper->ntime);
      }
      VecCopy(finalstate, store_finalstates[iwindow][iinit]);

      /* Add to integral penalty term */
      obj_penal += obj_weights[iinit] * gamma_penalty * timestepper->penalty_integral;
      obj_penal_dpdm += obj_weights[iinit] * gamma_penalty_dpdm * timestepper->penalty_dpdm;
      obj_penal_energy += obj_weights[iinit] * gamma_penalty_energy * timestepper->energy_penalty_integral;

      /* Receive next window's initial state and lagrangian variable */
      VecScatterEnd(scatter_xnext[iwindow][iinit], x, x_next, INSERT_VALUES, SCATTER_FORWARD); 

      /* Evaluate J(finalstate) and add to final-time cost */
      if (iwindow_global == nwindows-1) {

        double obj_iinit_re = 0.0;
        double obj_iinit_im = 0.0;
        double frob2_iinit;    
        optim_target->prepareTargetState(rho_t0);
        optim_target->evalJ(finalstate,  &obj_iinit_re, &obj_iinit_im, &frob2_iinit);
        obj_cost_re += obj_weights[iinit] * obj_iinit_re;
        obj_cost_im += obj_weights[iinit] * obj_iinit_im;
        frob2 += frob2_iinit / ninit;
        /* Add to final-time fidelity */
        double fidelity_iinit_re = 0.0;
        double fidelity_iinit_im = 0.0;
        optim_target->HilbertSchmidtOverlap(finalstate, false, &fidelity_iinit_re, &fidelity_iinit_im);
        fidelity_re += fidelity_iinit_re / ninit;
        fidelity_im += fidelity_iinit_im / ninit;
      }
      /* Else, add to constraint. */
      else {
        // eval || (Su - u) ||^2 and lambda^T(Su-u)
        double cdot, qnorm2;
        VecAXPY(finalstate, -1.0, x_next);  // finalstate = Su - u
        VecDot(finalstate, finalstate, &qnorm2); // q = || (Su - u) ||^2
        interm_discontinuity += qnorm2;
        VecDot(finalstate, lag, &cdot);   // c = lambda^T (Su - u)
        obj_constraint += 0.5 * mu * qnorm2 - cdot;
      }
      
      /* Restore local state and lambda*/
      VecRestoreSubVector(x, IS_interm_states[iwindow_global][iinit_global], &x0);
      VecRestoreSubVector(lambda_, IS_interm_lambda[iwindow_global][iinit_global], &lag);

      /* If Lindblas solver, compute adjoint for this initial condition. Otherwise (Schroedinger solver), compute adjoint only after all initial conditions have been propagated through (separate loop below) */
      if (timestepper->mastereq->lindbladtype != LindbladType::NONE) {
        if (mpirank_time == 0)
          printf("Multiple shooting adjoint is not yet implemented for Lindblas solver!\n");
          exit(1);
      //   // if (mpirank_time == 0) printf("%d: %d BWD.", mpirank_init, initid);

      //   /* Reset adjoint */
      //   VecZeroEntries(rho_t0_bar);

      //   /* Terminal condition for adjoint variable: Derivative of final time objective J */
      //   double obj_cost_re_bar, obj_cost_im_bar, frob2_bar;
      //   optim_target->finalizeJ_diff(obj_cost_re, obj_cost_im, &obj_cost_re_bar, &obj_cost_im_bar, &frob2_bar);
      //   optim_target->evalJ_diff(finalstate, rho_t0_bar, obj_weights[iinit]*obj_cost_re_bar, obj_weights[iinit]*obj_cost_im_bar, 1./ninit*frob2_bar);

      //   /* Derivative of time-stepping */
      //   adjoint_ic = timestepper->solveAdjointODE(initid, rho_t0_bar, finalstate, obj_weights[iinit] * gamma_penalty, obj_weights[iinit]*gamma_penalty_dpdm, obj_weights[iinit]*gamma_penalty_energy, n0);

      //   if (!(mpirank_time == 0 && iwindow == 0)) {
      //     // int id = iinit_global*(nwindows-1) + index-1;
      //     VecISAXPY(G, IS_interm_states[iwindow_global][iinit_global], 1.0, adjoint_ic);
      //   }
      //   /* Add to optimizers's gradient */
      //   VecAXPY(G, 1.0, timestepper->redgrad);
      }
    } // for (int iwindow=0; iwindow<nwindows_local; iwindow++)
  } // for (int iinit = 0; iinit < ninit_local; iinit++) {

  /* Sum up from initial conditions processors */
  double mypen = obj_penal;
  double mypen_dpdm = obj_penal_dpdm;
  double mypenen = obj_penal_energy;
  double mycost_re = obj_cost_re;
  double mycost_im = obj_cost_im;
  double myfidelity_re = fidelity_re;
  double myfidelity_im = fidelity_im;
  double myconstraint = obj_constraint;
  double my_interm_disc = interm_discontinuity;
  double my_frob2 = frob2;
  MPI_Allreduce(&mypen, &obj_penal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mypen_dpdm, &obj_penal_dpdm, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mypenen, &obj_penal_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  MPI_Allreduce(&mycost_re, &obj_cost_re, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mycost_im, &obj_cost_im, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&myfidelity_re, &fidelity_re, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&myfidelity_im, &fidelity_im, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&myconstraint, &obj_constraint, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&my_interm_disc, &interm_discontinuity, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&my_frob2, &frob2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  /* Set the fidelity: If Schroedinger, need to compute the absolute value: Fid= |\sum_i \phi^\dagger \phi_target|^2 */
  if (timestepper->mastereq->lindbladtype == LindbladType::NONE) {
    fidelity = fidelity_re*fidelity_re + fidelity_im*fidelity_im;
  } else {
    fidelity = fidelity_re; 
  }
 
  /* Finalize the objective function Jtrace to get the infidelity. 
     If Schroedingers solver, need to take the absolute value */
  obj_cost = optim_target->finalizeJ(obj_cost_re, obj_cost_im, frob2);

  /* Evaluate regularization objective += gamma/2 * ||x||^2*/
  double x_alpha_norm;
  if (!gamma_tik_interpolate){  // ||x||^2
    VecNorm(x_alpha, NORM_2, &x_alpha_norm);
  } 
  obj_regul = gamma_tik / 2. * pow(x_alpha_norm,2.0);

  /* Sum, store and return objective value */
  objective = obj_cost + obj_regul + obj_penal + obj_penal_dpdm + obj_penal_energy + obj_constraint;

  /* For Schroedinger solver: Solve adjoint equations for all initial conditions here. */
  if (timestepper->mastereq->lindbladtype == LindbladType::NONE) {

    // Iterate over all initial conditions 
    for (int iinit = 0; iinit < ninit_local; iinit++) {
      /* Iterate over local time windows */
      for (int iwindow=0; iwindow<nwindows_local; iwindow++) {

        int iinit_global = mpirank_init * ninit_local + iinit;
        int iwindow_global = mpirank_time*nwindows_local + iwindow ; 

        /* Reset adjoint */
        VecZeroEntries(rho_t0_bar);

        /* Recompute the initial state */
        if ( iwindow_global == 0 || iwindow_global == nwindows -1 ) {
          optim_target->prepareInitialState(iinit_global, ninit, timestepper->mastereq->nlevels, timestepper->mastereq->nessential, rho_t0);
        }

        /* Get next window's initial state and lagrangian variable */
        VecScatterBegin(scatter_xnext[iwindow][iinit], x, x_next, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(scatter_xnext[iwindow][iinit], x, x_next, INSERT_VALUES, SCATTER_FORWARD); 
        Vec lag;
        VecGetSubVector(lambda_, IS_interm_lambda[iwindow_global][iinit_global], &lag);

        /* Get adjoint terminal condition */
        if (iwindow_global == nwindows-1) {
          /* Set terminal adjoint condition from derivative of final time objective J */
          double obj_cost_re_bar, obj_cost_im_bar, frob2_bar;
          optim_target->prepareTargetState(rho_t0);
          optim_target->finalizeJ_diff(obj_cost_re, obj_cost_im, &obj_cost_re_bar, &obj_cost_im_bar, &frob2_bar);
          optim_target->evalJ_diff(store_finalstates[iwindow][iinit], rho_t0_bar, obj_weights[iinit]*obj_cost_re_bar, obj_weights[iinit]*obj_cost_im_bar, frob2_bar/ninit);

          /* Reset temporary disc variable, in case it had been set in prev iteration */
          VecSet(disc, 0.0);
        }
        else {
          /* Set terminal adjoint condition from discontinuity*/
          VecCopy(store_finalstates[iwindow][iinit], disc);
          VecAXPY(disc, -1.0, x_next);     // disc = final - xnext
          VecScale(disc, -mu);
          VecAXPY(rho_t0_bar, -1.0, disc);  // rho_t0_bar = mu *( final - xnext) = d q / d Su
          VecAXPY(rho_t0_bar, -1.0, lag); // rho_t0_bar -= lambda => - d c / d Su
        }

        /* Start scattering discontinuity gradient w.r.t xnext. Everyone needs to participate in the scatter. */
        int lag_exists_here;
        VecGetSize(lag, &lag_exists_here);
        if (lag_exists_here> 0)
          VecAXPY(disc, 1.0, lag);
        VecScatterBegin(scatter_xnext[iwindow][iinit], disc, G, ADD_VALUES, SCATTER_REVERSE);

        /* Solve adjoint ODE */
        adjoint_ic = timestepper->solveAdjointODE(1, rho_t0_bar, store_finalstates[iwindow][iinit], obj_weights[iinit] * gamma_penalty, obj_weights[iinit]*gamma_penalty_dpdm, obj_weights[iinit]*gamma_penalty_energy, iwindow_global * timestepper->ntime);

        /* Finalize the scatter */
        VecScatterEnd(scatter_xnext[iwindow][iinit], disc, G, ADD_VALUES, SCATTER_REVERSE);

        // Add adjoint initial conditions to gradient. This is local, each proc sets its own windows/initialconditions. 
        if (!(iwindow_global==0)) {
          VecISAXPY(G, IS_interm_states[iwindow_global][iinit_global], 1.0, adjoint_ic);
        }

        /* Add time-stepping reduced gradient to G. All scatter to the first processor. */
        VecScatterBegin(scatter_alpha, timestepper->redgrad, G, ADD_VALUES, SCATTER_REVERSE);
        VecScatterEnd(scatter_alpha, timestepper->redgrad, G, ADD_VALUES, SCATTER_REVERSE);

        /* Restore lambda */
        VecRestoreSubVector(lambda_, IS_interm_lambda[iwindow_global][iinit_global], &lag);
      } // end of local iwindow loop
    } // end of local iinit loop 
  } // end of adjoint for Schroedinger

  // TODO: REALLY UNSURE ABOUT HOW TO DO setControlAmplitudes_diff. Disabling.
  if (timestepper->mastereq->getOscillator(0)->control_enforceBC &&
      nwindows>1) {
    printf("TODO: enforce control boundary currently not available with multiple time windows. Exiting now.\n");
    exit(1);
  }
  // Vec g_alpha;
  // VecScatterBegin(scatter_alpha, G, g_alpha, INSERT_VALUES, SCATTER_FORWARD);
  // VecScatterEnd(scatter_alpha, G, g_alpha, INSERT_VALUES, SCATTER_FORWARD);
  // mastereq->setControlAmplitudes_diff(g_alpha);
  // VecScatterBegin(scatter_alpha, g_alpha, G, ADD_VALUES, SCATTER_REVERSE);
  // VecScatterEnd(scatter_alpha, g_alpha, G ADD_VALUES, SCATTER_REVERSE);

  /* Apply chain rule for unitarization */
  if (unitarize_interm_ic && (nwindows > 1))
    unitarize_grad(x_b4unit, x, vnorms, G); 

  /* Compute and store gradient norm */
  VecNorm(G, NORM_2, &(gnorm));

  // scale the state part of the gradient by 1/scalefactor_states
  VecScale(G, 1.0/scalefactor_states);
  // undo scaling of alpha
  double* ptr;
  VecGetArray(G, &ptr);
  if (mpirank_world==0) {
    for (int i=0; i<ndesign; i++)
      ptr[i] *=scalefactor_states;
  }
  VecRestoreArray(G, &ptr);

  /* Output */
  if (mpirank_world == 0 && !quietmode) {
    std::cout<< "Objective = " << std::scientific<<std::setprecision(14) << obj_cost << " + " << obj_regul << " + " << obj_penal << " + " << obj_penal_dpdm << " + " << obj_penal_energy << " + " << obj_constraint;
    if (nwindows == 1) // Fidelity only makes sense with one window
      std::cout<< " Fidelity = " << fidelity << std::endl;
    std::cout<< " Discontinuity = " << interm_discontinuity << std::endl;
  }
}


void OptimProblem::solve(Vec xinit) {
  TaoSetSolution(tao, xinit);
  TaoSolve(tao);
}

void OptimProblem::getStartingPoint(Vec xinit){
  MasterEq* mastereq = timestepper->mastereq;

  /* This is for the controls alphas! TODO: also read the intermediate states from file. */
  if (initguess_fromfile.size() > 0) {
    /* Set the initial guess from file */
    for (int i=0; i<initguess_fromfile.size(); i++) {
      VecSetValue(xinit, i, initguess_fromfile[i], INSERT_VALUES);
    }

  } else { // copy alpha from control initialization in oscillators contructor
    PetscScalar* xptr;
    VecGetArray(xinit, &xptr);
    if (mpirank_world == 0) { // only first processor stores x_alpha
      int shift = 0;
      for (int ioscil = 0; ioscil<mastereq->getNOscillators(); ioscil++){
        mastereq->getOscillator(ioscil)->getParams(xptr + shift);
        shift += mastereq->getOscillator(ioscil)->getNParams();
      }
    }
    VecRestoreArray(xinit, &xptr);
  }

  /* Assemble initial guess */
  VecAssemblyBegin(xinit);
  VecAssemblyEnd(xinit);

  /* Pass control alphas to the oscillator */
  // First communicate from P0 to each proc's sequential vector x_alpha
  VecScatterBegin(scatter_alpha, xinit, x_alpha, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(scatter_alpha, xinit, x_alpha, INSERT_VALUES, SCATTER_FORWARD);
  // now pass x_alpha to oscillators
  timestepper->mastereq->setControlAmplitudes(x_alpha);

  /* Write initial control functions to file TODO: Multiple time windows */
  // output->writeControls(xinit, timestepper->mastereq, timestepper->ntime, timestepper->dt);

  /* Set the initial guess for the intermediate states. Here, roll-out forward propagation. TODO: Read from file */
  rollOut(xinit);
}

void OptimProblem::rollOut(Vec x){
  if (mpirank_world==0) {
  printf(" -> Rollout initialization of intermediate states (sequential in time windows parallel in initial conditions. This might take a while...\n");
  }

  /* Roll-out forward propagation. */
  for (int iinit = 0; iinit < ninit_local; iinit++) {
    int iinit_global = mpirank_init * ninit_local + iinit;

    // Get the initial condition
    optim_target->prepareInitialState(iinit_global, ninit, timestepper->mastereq->nlevels, timestepper->mastereq->nessential, rho_t0);

    // Iterate over all (global) time windows
    for (int iwindow_global=0; iwindow_global<nwindows-1; iwindow_global++){

      // Solve forward from starting point.
      Vec xfinal = timestepper->solveODE(1, rho_t0, iwindow_global * timestepper->ntime); 
      VecCopy(xfinal, rho_t0);

      /* Store the final state of this window in x. Only the processor who owns this window/initial contidion should do the copy */
      int size;
      ISGetLocalSize(IS_interm_states[iwindow_global+1][iinit_global], &size);
      if (size > 0) {
        // printf("%d|%d|%d -> Copying size %d into iwindow %d iinit %d\n", mpirank_world, mpirank_time, mpirank_init, size, iwindow+1, iinit);
        VecISCopy(x, IS_interm_states[iwindow_global+1][iinit_global], SCATTER_FORWARD, rho_t0);
      }

      PetscBarrier((PetscObject) x);
    } // end for iwindow global
  } // end for initial condition local
  // VecView(x, PETSC_VIEWER_STDOUT_WORLD);

  /* Scale the state part of x by scalefactor_states */
  VecScale(x, scalefactor_states);
  // undo scaling of alpha
  double* xptr;
  VecGetArray(x, &xptr);
  if (mpirank_world==0) {
    for (int i=0; i<ndesign; i++)
      xptr[i] /=scalefactor_states;
  }
  VecRestoreArray(x, &xptr);
}

/* lag += - prev_mu * ( S(u_{i-1}) - u_i ) */
void OptimProblem::updateLagrangian(const double prev_mu, const Vec x_a, Vec lambda_a) {
  // NOTE: lambda_incre = 0 on entry
  
  /* Forward solve to store intermediate discontinuities in store_initial_states (local sizes)*/
  evalF(x_a, lambda_a);

  /* Iterate over local initial conditions */
  for (int iinit = 0; iinit < ninit_local; iinit++) {
    /* Iterate over local time windows */
    for (int iwindow = 0; iwindow < nwindows_local; iwindow++){

      int iinit_global = mpirank_init * ninit_local + iinit;
      int iwindow_global = mpirank_time*nwindows_local + iwindow ; 

      /* Get next time-windows initial state and eval discontinuity */
      VecScatterBegin(scatter_xnext[iwindow][iinit], x_a, x_next, INSERT_VALUES, SCATTER_FORWARD);
      VecScatterEnd(scatter_xnext[iwindow][iinit], x_a, x_next, INSERT_VALUES, SCATTER_FORWARD);

      /* lag += - prev_mu * ( S(u_{i-1}) - u_i ) */
      Vec lag;
      VecGetSubVector(lambda_a, IS_interm_lambda[iwindow_global][iinit_global], &lag);
      if (iwindow_global < nwindows-1) {
        VecCopy(store_finalstates[iwindow][iinit], disc);
        VecAXPY(disc, -1.0, x_next);  // disc = S(u_{i-1}) - u_i
        VecAXPY(lag, -prev_mu, disc);
      }
      VecRestoreSubVector(lambda_a, IS_interm_lambda[iwindow_global][iinit_global], &lag);
      // printf("\n\n TODO: Check the updateLagrangian function!\n");
    }
  }
}

void OptimProblem::getSolution(Vec* param_ptr){
  
  /* Get ref to optimized parameters */
  Vec params;
  TaoGetSolution(tao, &params);
  *param_ptr = params;
}

void OptimProblem::getExitReason(TaoConvergedReason *reason_ptr){
  
  /* Get ref to optimized parameters */
  TaoConvergedReason reason;
  TaoGetConvergedReason(tao, &reason);
  *reason_ptr = reason;
}

int OptimProblem::getTotalIterations(){
  int iter;
  TaoGetTotalIterationNumber(tao, &iter);
  return iter;
}

void OptimProblem::setTaoWarmStart(PetscBool yes_no){
  TaoSetRecycleHistory(tao, yes_no);
}

// void OptimProblem::output_interm_ic(){
//   for (int iinit = 0; iinit < ninit_local; iinit++) {
//     for (int iwindow=0; iwindow<nwindows_local; iwindow++){
//       int iwin_global = mpirank_time*nwindows_local + iwindow; 
//       int iinit_global = mpirank_init * ninit_local + iinit; 

//       if (iwin_global < nwindows-1) {
//         printf("iinit_g = %d, iwind_g = %d, initial_win_state:\n", iinit_global, iwin_global);
//         VecView(store_initial_state[iinit][iwindow], PETSC_VIEWER_STDOUT_WORLD);
//       }
//     }
//   }
// }

// void OptimProblem::output_states(Vec x){
//   //    tmp
//   Vec x_ic;
//   VecCreate(PETSC_COMM_WORLD, &x_ic);
//   VecSetSizes(x_ic, PETSC_DECIDE, nstate);
//   VecSetFromOptions(x_ic);
  
//   if (mpirank_world == 0) {
//     VecISCopy(x, IS_initialcond, SCATTER_REVERSE, x_ic); // assign x_ic[i] = x[IS[i]]
//     printf("output_states(): initial states in vector x:\n");
//     VecView(x_ic, PETSC_VIEWER_STDOUT_WORLD);
//   }
// }


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

  /* Pass current iteration number to output manager */
  ctx->output->optim_iter = iter;

  /* Grab some output stuff */
  double objective = ctx->getObjective();
  double obj_cost = ctx->getCostT();
  double obj_regul = ctx->getRegul();
  double obj_penal = ctx->getPenalty();
  double obj_penal_dpdm = ctx->getPenaltyDpDm();
  double obj_penal_energy = ctx->getPenaltyEnergy();
  double obj_constraint = ctx->getConstraint();
  double interm_discontinuity = ctx->getDiscontinuity();
  double F_avg = ctx->getFidelity();

  /* Print to optimization file */
  ctx->output->writeOptimFile(f, gnorm, deltax, F_avg, obj_cost, obj_regul, obj_penal, obj_penal_dpdm, obj_penal_energy, obj_constraint, interm_discontinuity);

  /* Print parameters and controls to file */
  // if ( optim_iter % optim_monitor_freq == 0 ) {
  // ctx->output->writeControls(params, ctx->timestepper->mastereq, ctx->timestepper->ntime, ctx->timestepper->dt);
  // }

  /* Screen output */
  if (ctx->getMPIrank_world() == 0 && iter == 0) {
    std::cout<<  "    Objective             Tikhonov                Penalty-Leakage        Penalty-StateVar       Penalty-TotalEnergy    Constraint" << std::endl;
  }

  /* Additional Stopping criteria */
  bool lastIter = false;
  std::string finalReason_str = "";
  if ((1.0 - F_avg <= ctx->getInfTol()) && (interm_discontinuity < ctx->getIntermTol())) {
    finalReason_str = "Optimization converged to a continuous trajectory with small infidelity and small discontinuity.";
    TaoSetConvergedReason(tao, TAO_CONVERGED_USER);
    lastIter = true;
  } else if ((obj_cost <= ctx->getFaTol()) && (interm_discontinuity < ctx->getIntermTol())) {
    finalReason_str = "Optimization converged to a continuous trajectory with small final time cost and small discontinuity.";
    TaoSetConvergedReason(tao, TAO_CONVERGED_USER);
    lastIter = true;
  } 
  // else if (iter > 1) { // TODO: NEEDS UPDATE?? Stop if delta x is smaller than tolerance (relative)
    // // Compute ||x - xprev||/||xprev||
    // double xnorm, dxnorm;
    // VecNorm(ctx->xprev, NORM_2, &xnorm);  // xnorm = ||x_k-1||
    // VecAXPY(ctx->xprev, -1.0, params);    // xprev =  x_k - x_k-1
    // VecNorm(ctx->xprev, NORM_2, &dxnorm);  // dxnorm = || x_k - x_k-1 ||
    // if (fabs(xnorm > 1e-15)) dxnorm = dxnorm / xnorm; 
    // // Stopping 
    // if (dxnorm <= ctx->getDxTol()) {
    //    finalReason_str = "Optimization finished with small parameter update (" + std::to_string(dxnorm) + "rel. update).";
    //   TaoSetConvergedReason(tao, TAO_CONVERGED_USER);
    //   lastIter = true;
    // }
  // }

  // output progress on std out
  if (ctx->getMPIrank_world() == 0 && (iter == ctx->getMaxIter() || lastIter || iter % ctx->output->optim_monitor_freq == 0)) {
    std::cout<< iter <<  "  " << std::scientific<<std::setprecision(14) << obj_cost << " + " << obj_regul << " + " << obj_penal << " + " << obj_penal_dpdm << " + " << obj_penal_energy << " + " << obj_constraint;
    if (ctx->getNwindows() == 1) // Fidelity only makes sense with one window
      std::cout<< "  Fidelity = " << F_avg;
    std::cout<< "  ||Grad|| = " << gnorm;
    std::cout<< " norm2(disc) = " << interm_discontinuity << std::endl;
  }

  if (ctx->getMPIrank_world() == 0 && lastIter){
    std::cout<< "Rank: " << ctx->getMPIrank_world() << " " << finalReason_str << std::endl;
  }
 
  // /* Update xprev for next iteration */
  // VecCopy(params, ctx->xprev);

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
  assert(ctx->lambda);
  *f = ctx->evalF(x, ctx->lambda);
  
  return 0;
}


PetscErrorCode TaoEvalGradient(Tao tao, Vec x, Vec G, void*ptr){

  OptimProblem* ctx = (OptimProblem*) ptr;
  assert(ctx->lambda);
  ctx->evalGradF(x, ctx->lambda, G);
  
  return 0;
}


/* Unitarize the intermediate initial staets */
// could be in util.hpp/cpp, but then needs args for dim, isu,isv, is_interm_states
void OptimProblem::unitarize(Vec &x, std::vector<std::vector<double>> &vnorms) {

  const int nwindows = IS_interm_states.size();
  const int ninit = IS_interm_states[0].size();
  int dim = 2*timestepper->mastereq->getDim();

  /* vnorms do not need to be broadcast. The global size is kept just for convenient indexing. */
  vnorms.resize(ninit);
  for (int iinit = 0; iinit < ninit; iinit++)
  {
    vnorms[iinit].resize(nwindows);
    for (int iwindow = 0; iwindow < nwindows; iwindow++)
      vnorms[iinit][iwindow] = 0.0;
  }

  Vec vre, vim;
  double uv_re, uv_im, vnorm;
  IS IS_re = timestepper->mastereq->isu;
  IS IS_im = timestepper->mastereq->isv;

  int *ids_from= new int[dim];
  int *ids_to = new int[dim];
  for (int i=0; i< dim; i++){
    ids_to[i] = i;
  }

  /* classical GS */
  for (int iwindow = 1; iwindow < nwindows; iwindow++) {
    for (int iinit = 0; iinit < ninit; iinit++) {

      /* Get local state for this window: Only one proc will have it, otherones will have an empty vector. */
      int isize;
      ISGetLocalSize(IS_interm_states[iwindow][iinit], &isize);
      if (isize>0) {
        VecISCopy(x, IS_interm_states[iwindow][iinit], SCATTER_REVERSE, x_next); 
      }

      /* Iterate over all remaining initial conditions in this window */
      for (int jinit = 0; jinit < iinit; jinit++) {
        /* Scatter j'th initial condition to this processor */
        // only one processor has a nonzero IS size! That one sends his first index. Receive it if isize>0.
        int jstart, jstop;
        ISGetMinMax(IS_interm_states[iwindow][jinit], &jstart, &jstop);
        if (jstart == PETSC_MAX_INT) jstart = 0;
        MPI_Allreduce(&jstart, &jstart, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        int nelems = 0;
        if (jstart > 0 && isize > 0)  {
          nelems = dim;
          for (int i=0; i< dim; i++){
            ids_from[i] = jstart + i;
          }
        }

        IS IS_from, IS_to;
        VecScatter ctx_x0j;
        /* Now scatter the next initial condition to this processor */
        ISCreateGeneral(PETSC_COMM_SELF,nelems,ids_from,PETSC_COPY_VALUES, &IS_from);
        ISCreateGeneral(PETSC_COMM_SELF,nelems,ids_to,PETSC_COPY_VALUES, &IS_to);
        VecScatterCreate(x, IS_from, x0j, IS_to, &ctx_x0j);
        VecScatterBegin(ctx_x0j, x, x0j, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(ctx_x0j, x, x0j, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterDestroy(&ctx_x0j);
        ISDestroy(&IS_from);
        ISDestroy(&IS_to);

        /* On this processor, compute unitarization of local state */
        if (isize>0) {
          complex_inner_product(x0j, x_next, uv_re, uv_im);

          VecGetSubVector(x0j, IS_re, &vre);
          VecGetSubVector(x0j, IS_im, &vim);

          // Re[v] -= Re[u.v] * Re[u] - Im[u.v] * Im[u]
          VecISAXPY(x_next, IS_re, -uv_re, vre);
          VecISAXPY(x_next, IS_re, uv_im, vim);
          // Im[v] -= Re[u.v] * Im[u] + Im[u.v] * Re[u]
          VecISAXPY(x_next, IS_im, -uv_re, vim);
          VecISAXPY(x_next, IS_im, -uv_im, vre);

          VecRestoreSubVector(x0j, IS_re, &vre);
          VecRestoreSubVector(x0j, IS_im, &vim);
        }
      } // for (int jinit = 0; jinit < iinit; jinit++)

      if (isize>0) {
        VecNorm(x_next, NORM_2, &vnorm);
        VecScale(x_next, 1.0 / vnorm);
        vnorms[iinit][iwindow] = vnorm;

        /* Finally update the global vector entry */
        VecISCopy(x, IS_interm_states[iwindow][iinit], SCATTER_FORWARD, x_next); 
      }
    } // for (int iinit = 0; iinit < ninit; iinit++)
  } // for (int iwindow = 1; iwindow < nwindows; iwindow++)

  delete [] ids_from;
  delete [] ids_to;
}

/* The adjoint for unitarize function based on the classic Gram-Schmidt. */
void OptimProblem::unitarize_grad(const Vec &x_b4unit, Vec &x, const std::vector<std::vector<double>> &vnorms, Vec &G) {

  const int nwindows = IS_interm_states.size();
  const int ninit = IS_interm_states[0].size();
  int dim = 2*timestepper->mastereq->getDim();
  int dim_half = dim / 2;

  IS IS_re = timestepper->mastereq->isu;
  IS IS_im = timestepper->mastereq->isv;

  int *ids_from= new int[dim];
  int *ids_to = new int[dim];
  for (int i=0; i< dim; i++){
    ids_to[i] = i;
  }

  Vec us, ws;
  VecCreateSeq(PETSC_COMM_SELF, dim, &us);
  VecCreateSeq(PETSC_COMM_SELF, dim, &ws);

  Vec wc, vsc;
  VecCreateSeq(PETSC_COMM_SELF, dim, &wc);
  VecCreateSeq(PETSC_COMM_SELF, dim, &vsc);
  
  Vec wre, wim, vsre, vsim, ure, uim;
  double wu_re, wu_im, vsu_re, vsu_im;
  for (int iwindow = 1; iwindow < nwindows; iwindow++) {
    for (int iinit = ninit-1; iinit >= 0; iinit--) {
      /* Get local state for this window: Only one proc will have it, otherones will have an empty vector. */
      int isize;
      ISGetLocalSize(IS_interm_states[iwindow][iinit], &isize);
      if (isize>0) {
        VecISCopy(x, IS_interm_states[iwindow][iinit], SCATTER_REVERSE, x_next);
        VecISCopy(G, IS_interm_states[iwindow][iinit], SCATTER_REVERSE, us); 
      }

      for (int cinit = iinit+1; cinit < ninit; cinit++) {
        /* Scatter c'th initial condition/vs to this processor */
        // only one processor has a nonzero IS size! That one sends his first index. Receive it if isize>0.
        int cstart, cstop;
        ISGetMinMax(IS_interm_states[iwindow][cinit], &cstart, &cstop);
        if (cstart == PETSC_MAX_INT) cstart = 0;
        MPI_Allreduce(&cstart, &cstart, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        int nelems = 0;
        if (cstart > 0 && isize > 0)  {
          nelems = dim;
          for (int i=0; i< dim; i++){
            ids_from[i] = cstart + i;
          }
        }
        IS IS_from, IS_to;
        VecScatter ctx_xc;
        /* Now scatter the next initial condition to this processor */
        ISCreateGeneral(PETSC_COMM_SELF,nelems,ids_from,PETSC_COPY_VALUES, &IS_from);
        ISCreateGeneral(PETSC_COMM_SELF,nelems,ids_to,PETSC_COPY_VALUES, &IS_to);
        VecScatterCreate(x_b4unit, IS_from, wc, IS_to, &ctx_xc);
        VecScatterBegin(ctx_xc, x_b4unit, wc, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(ctx_xc, x_b4unit, wc, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterBegin(ctx_xc, x, vsc, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(ctx_xc, x, vsc, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterDestroy(&ctx_xc);
        ISDestroy(&IS_from);
        ISDestroy(&IS_to);

        /* On this processor, compute unitarization of local state */
        if (isize>0) {
          complex_inner_product(x_next, wc, wu_re, wu_im);
          complex_inner_product(x_next, vsc, vsu_re, vsu_im);

          VecGetSubVector(wc, IS_re, &wre);
          VecGetSubVector(wc, IS_im, &wim);
          VecGetSubVector(vsc, IS_re, &vsre);
          VecGetSubVector(vsc, IS_im, &vsim);

          VecISAXPY(us, IS_re, -wu_re, vsre);
          VecISAXPY(us, IS_re, -wu_im, vsim);
          VecISAXPY(us, IS_re, -vsu_re, wre);
          VecISAXPY(us, IS_re, -vsu_im, wim);

          VecISAXPY(us, IS_im, wu_im, vsre);
          VecISAXPY(us, IS_im, -wu_re, vsim);
          VecISAXPY(us, IS_im, vsu_im, wre);
          VecISAXPY(us, IS_im, -vsu_re, wim);

          VecRestoreSubVector(wc, IS_re, &wre);
          VecRestoreSubVector(wc, IS_im, &wim);
          VecRestoreSubVector(vsc, IS_re, &vsre);
          VecRestoreSubVector(vsc, IS_im, &vsim);
        } // if (isize>0)
      } // for (int cinit = iinit+1; cinit < ninit; cinit++)

      if (isize>0) {
        double vnorm = vnorms[iinit][iwindow];
        double usu, dummy;

        complex_inner_product(x_next, us, usu, dummy);
        VecAXPY(us, -usu, x_next);
        VecScale(us, 1.0 / vnorm);

        /* update the global vs entry */
        VecISCopy(x, IS_interm_states[iwindow][iinit], SCATTER_FORWARD, us); 

        VecCopy(us, ws);
      }
      
      for (int cinit = 0; cinit < iinit; cinit++) {
        /* Scatter c'th initial condition/vs to this processor */
        // only one processor has a nonzero IS size! That one sends his first index. Receive it if isize>0.
        int cstart, cstop;
        ISGetMinMax(IS_interm_states[iwindow][cinit], &cstart, &cstop);
        if (cstart == PETSC_MAX_INT) cstart = 0;
        MPI_Allreduce(&cstart, &cstart, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        int nelems = 0;
        if (cstart > 0 && isize > 0)  {
          nelems = dim;
          for (int i=0; i< dim; i++){
            ids_from[i] = cstart + i;
          }
        }
        IS IS_from, IS_to;
        VecScatter ctx_xc;
        /* Now scatter the next initial condition to this processor */
        ISCreateGeneral(PETSC_COMM_SELF,nelems,ids_from,PETSC_COPY_VALUES, &IS_from);
        ISCreateGeneral(PETSC_COMM_SELF,nelems,ids_to,PETSC_COPY_VALUES, &IS_to);
        VecScatterCreate(x, IS_from, wc, IS_to, &ctx_xc);
        VecScatterBegin(ctx_xc, x, wc, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(ctx_xc, x, wc, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterDestroy(&ctx_xc);
        ISDestroy(&IS_from);
        ISDestroy(&IS_to);

        if (isize>0)
        {
          complex_inner_product(wc, us, vsu_re, vsu_im);

          VecGetSubVector(wc, IS_re, &ure);
          VecGetSubVector(wc, IS_im, &uim);

          VecISAXPY(ws, IS_re, -vsu_re, ure);
          VecISAXPY(ws, IS_re, +vsu_im, uim);
          VecISAXPY(ws, IS_im, -vsu_im, ure);
          VecISAXPY(ws, IS_im, -vsu_re, uim);

          VecRestoreSubVector(wc, IS_re, &ure);
          VecRestoreSubVector(wc, IS_im, &uim);
        } // if (isize>0)
      } // for (int cinit = 0; cinit < iinit; cinit++)

      if (isize>0) {
        VecISCopy(G, IS_interm_states[iwindow][iinit], SCATTER_FORWARD, ws); 
      }
    }
  }

  VecDestroy(&ws);
  VecDestroy(&us);

  VecDestroy(&wc);
  VecDestroy(&vsc);
}

void OptimProblem::check_unitarity(const Vec &x)
{
  const int nwindows = IS_interm_states.size();
  const int ninit = IS_interm_states[0].size();
  int dim = 2*timestepper->mastereq->getDim();

  Vec vre, vim;
  double uv_re, uv_im, vnorm;
  IS IS_re = timestepper->mastereq->isu;
  IS IS_im = timestepper->mastereq->isv;

  int *ids_from= new int[dim];
  int *ids_to = new int[dim];
  for (int i=0; i< dim; i++){
    ids_to[i] = i;
  }

  /* Iterate over local time windows */
  for (int iwindow = 1; iwindow < nwindows; iwindow++) {
    /* IS_interm_states[0][...] will have zero size */
    for (int iinit = 0; iinit < ninit; iinit++) {

      /* Get local state for this window: Only one proc will have it, otherones will have an empty vector. */
      int isize;
      ISGetLocalSize(IS_interm_states[iwindow][iinit], &isize);
      if (isize>0) {
        VecISCopy(x, IS_interm_states[iwindow][iinit], SCATTER_REVERSE, x_next); 
      }

      /* Iterate over all remaining initial conditions in this window */
      for (int jinit = 0; jinit < ninit; jinit++) {
      
        /* Scatter j'th initial condition to this processor */
        // only one processor has a nonzero IS size! That one sends his first index. Receive it if isize>0.
        int jstart, jstop;
        ISGetMinMax(IS_interm_states[iwindow][jinit], &jstart, &jstop);
        if (jstart == PETSC_MAX_INT) jstart = 0;
        MPI_Allreduce(&jstart, &jstart, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        int nelems = 0;
        if (jstart > 0 && isize > 0)  {
          nelems = dim;
          for (int i=0; i< dim; i++){
            ids_from[i] = jstart + i;
          }
        }
        IS IS_from, IS_to;
        VecScatter ctx_x0j;
        /* Now scatter the next initial condition to this processor */
        ISCreateGeneral(PETSC_COMM_SELF,nelems,ids_from,PETSC_COPY_VALUES, &IS_from);
        ISCreateGeneral(PETSC_COMM_SELF,nelems,ids_to,PETSC_COPY_VALUES, &IS_to);
        VecScatterCreate(x, IS_from, x0j, IS_to, &ctx_x0j);
        VecScatterBegin(ctx_x0j, x, x0j, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterEnd(ctx_x0j, x, x0j, INSERT_VALUES, SCATTER_FORWARD);
        VecScatterDestroy(&ctx_x0j);
        ISDestroy(&IS_from);
        ISDestroy(&IS_to);

        /* On this processor, compute unitarization of local state */
        if (isize>0) {
          complex_inner_product(x0j, x_next, uv_re, uv_im);
          printf("%2.1e+%2.1ei\t", uv_re, uv_im);
        }
      } // for (int jinit = 0; jinit < ninit; jinit++)
      if (isize>0) printf("\n");
    } // for (int iinit = 0; iinit < ninit; iinit++)
    printf("\n");
  } // for (int iwindow = 1; iwindow < nwindows; iwindow++)
}
