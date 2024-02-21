#include "optimproblem.hpp"

OptimProblem::OptimProblem(MapParam config, TimeStepper* timestepper_, MPI_Comm comm_init_, MPI_Comm comm_time_, int ninit_, int nwindows_, double total_time_, std::vector<double> gate_rot_freq, Output* output_, bool quietmode_){

  timestepper = timestepper_;
  ninit = ninit_;
  nwindows = nwindows_;
  output = output_;
  quietmode = quietmode_;
  /* Reset */
  objective = 0.0;

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

  VecSet(lambda, 1.0);
  VecSet(xinit, 1.0);


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
  
  
  /* Store other optimization parameters */
  gamma_tik = config.GetDoubleParam("optim_regul", 1e-4);
  gamma_tik_interpolate = config.GetBoolParam("optim_regul_interpolate", false, false);
  gatol = config.GetDoubleParam("optim_atol", 1e-8);
  fatol = config.GetDoubleParam("optim_ftol", 1e-8);
  inftol = config.GetDoubleParam("optim_inftol", 1e-5);
  grtol = config.GetDoubleParam("optim_rtol", 1e-4);
  interm_tol = config.GetDoubleParam("optim_interm_tol", 1e-4, false);
  maxiter = config.GetIntParam("optim_maxiter", 200);
  mu = config.GetDoubleParam("optim_mu", 0.0, false);
  
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
    else if (target_str[1].compare("xgate") == 0) targetgate = new XGate(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, total_time_, gate_rot_freq, timestepper->mastereq->lindbladtype, quietmode); 
    else if (target_str[1].compare("ygate") == 0) targetgate = new YGate(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, total_time_, gate_rot_freq, timestepper->mastereq->lindbladtype, quietmode); 
    else if (target_str[1].compare("zgate") == 0) targetgate = new ZGate(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, total_time_, gate_rot_freq, timestepper->mastereq->lindbladtype, quietmode);
    else if (target_str[1].compare("hadamard") == 0) targetgate = new HadamardGate(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, total_time_, gate_rot_freq, timestepper->mastereq->lindbladtype, quietmode);
    else if (target_str[1].compare("cnot") == 0) targetgate = new CNOT(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, total_time_, gate_rot_freq, timestepper->mastereq->lindbladtype, quietmode); 
    else if (target_str[1].compare("swap") == 0) targetgate = new SWAP(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, total_time_, gate_rot_freq, timestepper->mastereq->lindbladtype, quietmode); 
    else if (target_str[1].compare("swap0q") == 0) targetgate = new SWAP_0Q(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, total_time_, gate_rot_freq, timestepper->mastereq->lindbladtype, quietmode); 
    else if (target_str[1].compare("cqnot") == 0) targetgate = new CQNOT(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, total_time_, gate_rot_freq, timestepper->mastereq->lindbladtype, quietmode); 
    else if (target_str[1].compare("qft") == 0) targetgate = new QFT(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, total_time_, gate_rot_freq, timestepper->mastereq->lindbladtype, quietmode); 
    else if (target_str[1].compare("fromfile") == 0) targetgate = new FromFile(timestepper->mastereq->nlevels, timestepper->mastereq->nessential, total_time_, gate_rot_freq, timestepper->mastereq->lindbladtype, target_str[2], quietmode); 
    else {
      printf("\n\n ERROR: Unnown gate type: %s.\n", target_str[1].c_str());
      printf(" Available gates are 'none', 'xgate', 'ygate', 'zgate', 'hadamard', 'cnot', 'swap', 'swap0q', 'cqnot', 'qft'.\n");
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
        copyLast(target_str, timestepper->mastereq->getNOscillators()+1);
      }
      for (int i=0; i < timestepper->mastereq->getNOscillators(); i++) {
        int Qi_state = atoi(target_str[i+1].c_str());
        if (Qi_state >= timestepper->mastereq->getOscillator(i)->getNLevels()) {
          printf("ERROR in config setting. The requested pure state target |%d> exceeds the number of modeled levels for that oscillator (%d).\n", Qi_state, timestepper->mastereq->getOscillator(i)->getNLevels());
          exit(1);
        }
        purestateID += Qi_state * timestepper->mastereq->getOscillator(i)->dim_postOsc;
      }
    }
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
  if (objective_str.compare("Jfrobenius")==0)     objective_type = ObjectiveType::JFROBENIUS;
  else if (objective_str.compare("Jtrace")==0)    objective_type = ObjectiveType::JTRACE;
  else if (objective_str.compare("Jmeasure")==0)  objective_type = ObjectiveType::JMEASURE;
  else  {
    printf("\n\n ERROR: Unknown objective function: %s\n", objective_str.c_str());
    exit(1);
  }

  /* Finally initialize the optimization target struct */
  optim_target = new OptimTarget(timestepper->mastereq->getDim(), purestateID, target_type, objective_type, targetgate, target_filename, timestepper->mastereq->lindbladtype, quietmode);

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
  gamma_penalty_energy = config.GetDoubleParam("optim_penalty_energy", 0.0);
  gamma_penalty = config.GetDoubleParam("optim_penalty", 0.0);
  penalty_param = config.GetDoubleParam("optim_penalty_param", 0.5);
  timestepper->penalty_param = penalty_param;
  timestepper->gamma_penalty = gamma_penalty;
  gamma_penalty_dpdm = timestepper->gamma_penalty_dpdm;
  timestepper->gamma_penalty_energy = gamma_penalty_energy;
  timestepper->optim_target = optim_target;

  if ((gamma_penalty_dpdm > 1.0e-13) && (nwindows > 1)) {
    if (mpirank_world == 0) {
      printf("\nMultiple shooting optimization currently does not support dpdm penalty!\n");
      printf("dpdm penalty adjoint calculation needs time step index adjustion!\n");
    }
    exit(-1);
  }

  /* Get initial condition type and involved oscillators */
  std::vector<std::string> initcondstr;
  config.GetVecStrParam("initialcondition", initcondstr, "none", false);
  if (initcondstr.size() < 2) for (int j=0; j<timestepper->mastereq->getNOscillators(); j++)  initcondstr.push_back(std::to_string(j));
  for (int i=1; i<initcondstr.size(); i++) initcond_IDs.push_back(atoi(initcondstr[i].c_str()));
  if (initcondstr[0].compare("file") == 0 )          initcond_type = InitialConditionType::FROMFILE;
  else if (initcondstr[0].compare("pure") == 0 )     initcond_type = InitialConditionType::PURE;
  else if (initcondstr[0].compare("ensemble") == 0 ) initcond_type = InitialConditionType::ENSEMBLE;
  else if (initcondstr[0].compare("performance") == 0 ) initcond_type = InitialConditionType::PERFORMANCE;
  else if (initcondstr[0].compare("3states") == 0 )  initcond_type = InitialConditionType::THREESTATES;
  else if (initcondstr[0].compare("Nplus1") == 0 )   initcond_type = InitialConditionType::NPLUSONE;
  else if (initcondstr[0].compare("diagonal") == 0 ) initcond_type = InitialConditionType::DIAGONAL;
  else if (initcondstr[0].compare("basis")    == 0 ) initcond_type = InitialConditionType::BASIS;
  else {
    printf("\n\n ERROR: Wrong setting for initial condition.\n");
    exit(1);
  }

  /* Sanity check for Schrodinger solver initial conditions */
  if (timestepper->mastereq->lindbladtype == LindbladType::NONE){
    if (initcond_type == InitialConditionType::ENSEMBLE ||
        initcond_type == InitialConditionType::THREESTATES ||
        initcond_type == InitialConditionType::NPLUSONE ){
          printf("\n\n ERROR for initial condition setting: \n When running Schroedingers solver (collapse_type == NONE), the initial condition needs to be either 'pure' or 'from file' or 'diagonal' or 'basis'. Note that 'diagonal' and 'basis' in the Schroedinger case are the same (all unit vectors).\n\n");
          exit(1);
    } else if (initcond_type == InitialConditionType::BASIS) {
      // DIAGONAL and BASIS initial conditions in the Schroedinger case are the same. Overwrite it to DIAGONAL
      initcond_type = InitialConditionType::DIAGONAL;  
    }
  }

  /* Allocate the initial condition vector */
  VecCreateSeq(PETSC_COMM_SELF, 2*timestepper->mastereq->getDim(), &rho_t0); 
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
    int ndim = timestepper->mastereq->getDimRho();
    int vec_id = -1;
    if (timestepper->mastereq->lindbladtype != LindbladType::NONE) vec_id = getIndexReal(getVecID( diag_id, diag_id, ndim )); // Real part of x
    else vec_id = getIndexReal(diag_id);
    if (ilow <= vec_id && vec_id < iupp) VecSetValue(rho_t0, vec_id, 1.0, INSERT_VALUES);
  }
  else if (initcond_type == InitialConditionType::FROMFILE) { 
    /* Read initial condition from file */
    
    // int dim = timestepper->mastereq->getDim();
    int dim_ess = timestepper->mastereq->getDimEss();
    int dim_rho = timestepper->mastereq->getDimRho();
    int nelems = 0;
    if (timestepper->mastereq->lindbladtype != LindbladType::NONE) nelems = 2*dim_ess*dim_ess;
    else nelems = 2 * dim_ess;
    double * vec = new double[nelems];
    if (mpirank_world == 0) {
      assert (initcondstr.size()==2);
      std::string filename = initcondstr[1];
      read_vector(filename.c_str(), vec, nelems, quietmode);
    }
    MPI_Bcast(vec, nelems, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (timestepper->mastereq->lindbladtype != LindbladType::NONE) { // Lindblad solver, fill density matrix
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
    } else { // Schroedinger solver, fill vector 
      for (int i = 0; i < dim_ess; i++) {
        int k = i;
        if (dim_ess < timestepper->mastereq->getDim()) 
          k = mapEssToFull(i, timestepper->mastereq->nlevels, timestepper->mastereq->nessential);
        int elemid_re = getIndexReal(k);
        int elemid_im = getIndexImag(k);
        if (ilow <= elemid_re && elemid_re < iupp) VecSetValue(rho_t0, elemid_re, vec[i], INSERT_VALUES);        // RealPart
        if (ilow <= elemid_im && elemid_im < iupp) VecSetValue(rho_t0, elemid_im, vec[i + dim_ess], INSERT_VALUES); // Imaginary Part
      }
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

    // get dimension of subsystems defined by initcond_IDs, as well as the one before and after. Span in essential levels only.
    int dimpre = 1;
    int dimpost = 1;
    int dimsub = 1;
    for (int i=0; i<timestepper->mastereq->getNOscillators(); i++){
      if (i < initcond_IDs[0]) dimpre *= timestepper->mastereq->nessential[i];
      else if (initcond_IDs[0] <= i && i <= initcond_IDs[initcond_IDs.size()-1]) dimsub *= timestepper->mastereq->nessential[i];
      else dimpost *= timestepper->mastereq->nessential[i];
    }
    int dimrho = timestepper->mastereq->getDimRho();
    int dimrhoess = timestepper->mastereq->getDimEss();

    // Loop over ensemble state elements in essential level dimensions of the subsystem defined by the initcond_ids:
    for (int i=0; i < dimsub; i++){
      for (int j=i; j < dimsub; j++){
        int ifull = i * dimpost; // account for the system behind
        int jfull = j * dimpost;
        if (dimrhoess < dimrho) ifull = mapEssToFull(ifull, timestepper->mastereq->nlevels, timestepper->mastereq->nessential);
        if (dimrhoess < dimrho) jfull = mapEssToFull(jfull, timestepper->mastereq->nlevels, timestepper->mastereq->nessential);
        // printf(" i=%d j=%d ifull %d, jfull %d\n", i, j, ifull, jfull);
        if (i == j) { 
          // diagonal element: 1/N_sub
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
      // If spline controls: Scale bounds by 1/sqrt(2) * (number of carrier waves) */
      if (timestepper->mastereq->getOscillator(iosc)->getControlType() == ControlType::BSPLINE)
        boundval = boundval / (sqrt(2) * timestepper->mastereq->getOscillator(iosc)->getNCarrierfrequencies());
      // If spline for amplitude only: Scale bounds by 1/sqrt(2) * (number of carrier waves) */
      else if (timestepper->mastereq->getOscillator(iosc)->getControlType() == ControlType::BSPLINEAMP)
        boundval = boundval / timestepper->mastereq->getOscillator(iosc)->getNCarrierfrequencies();
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


// EvalF optim var. x = (alpha, interm.states), 
double OptimProblem::evalF(const Vec x, const Vec lambda_) {

  MasterEq* mastereq = timestepper->mastereq;

  if (mpirank_world == 0 && !quietmode) printf("EVAL F... \n");
  Vec finalstate = NULL;

  /* Pass control vector to oscillators */
  // x_alpha is set only on first processor. Need to communicate x_alpha to all processors here. 
  VecScatterBegin(scatter_alpha, x, x_alpha, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(scatter_alpha, x, x_alpha, INSERT_VALUES, SCATTER_FORWARD);
  // Finally all processors set the controls
  mastereq->setControlAmplitudes(x_alpha); 

  /*  Iterate over initial condition */
  obj_cost  = 0.0;
  obj_regul = 0.0;
  obj_penal = 0.0;
  obj_penal_dpdm = 0.0;
  obj_penal_energy = 0.0;
  fidelity = 0.0;
  interm_discontinuity = 0.0; // For TaoMonitor only.
  double obj_cost_re = 0.0;
  double obj_cost_im = 0.0;
  double fidelity_re = 0.0;
  double fidelity_im = 0.0;
  double frob2 = 0.0; // For generalized infidelity, stores Tr(U'U)
  double constraint = 0.0;
  for (int iinit = 0; iinit < ninit_local; iinit++) {
    /* Iterate over local time windows */
    for (int iwindow=0; iwindow<nwindows_local; iwindow++){

      int iinit_global = mpirank_init * ninit_local + iinit;
      int iwindow_global = mpirank_time*nwindows_local + iwindow ; 
      // printf("%d:%d|%d: --> LOOP m = %d(%d) ic = %d(%d)\n", mpirank_world, mpirank_time, mpirank_init, iwindow_global, iwindow, iinit_global, iinit);

      /* Start sending next window's initial state, will be received by this processor */
      VecScatterBegin(scatter_xnext[iwindow][iinit], x, x_next, INSERT_VALUES, SCATTER_FORWARD);

      /* Get local state and lambda for this window */
      Vec x0, lag;
      VecGetSubVector(x, IS_interm_states[iwindow_global][iinit_global], &x0);
      VecGetSubVector(lambda_, IS_interm_lambda[iwindow_global][iinit_global], &lag);

      /* Prepare the initial condition. Last window will also need it for J. */
      if ( iwindow_global == 0 || iwindow_global == nwindows -1 ) {
        timestepper->mastereq->getRhoT0(iinit_global, ninit, initcond_type, initcond_IDs, rho_t0);
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
        optim_target->prepare(rho_t0);
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
        constraint += 0.5 * mu * qnorm2 - cdot;
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
  double myconstraint = constraint;
  double my_interm_disc = interm_discontinuity;
  // Should be comm_init and also comm_time! Currently, no Petsc Parallelization possible, hence (comm-init AND comm_time) = COMM_WORLD
  MPI_Allreduce(&mypen, &obj_penal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mypen_dpdm, &obj_penal_dpdm, 1, MPI_DOUBLE, MPI_SUM, comm_init); // SG: Penalty DPDM currently disabled.
  MPI_Allreduce(&mypenen, &obj_penal_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mycost_re, &obj_cost_re, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mycost_im, &obj_cost_im, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&myfidelity_re, &fidelity_re, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&myfidelity_im, &fidelity_im, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&myconstraint, &constraint, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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
  obj_regul = gamma_tik / 2. * pow(x_alpha_norm,2.0);

  /* Sum, store and return objective value */
  objective = obj_cost + obj_regul + obj_penal + obj_penal_dpdm + obj_penal_energy + constraint;

  /* Output */
  if (mpirank_world == 0) {
    std::cout<< "Objective = " << std::scientific<<std::setprecision(14) << obj_cost << " + " << obj_regul << " + " << obj_penal << " + " << obj_penal_dpdm << " + " << obj_penal_energy << " + " << constraint << std::endl;
    if (nwindows == 1) // Fidelity only makes sense with one window
      std::cout<< "Fidelity = " << fidelity  << std::endl;
    std::cout<< "Discontinuities = " << interm_discontinuity << std::endl;
  }


  return objective;
}



void OptimProblem::evalGradF(const Vec x, const Vec lambda_, Vec G){

  MasterEq* mastereq = timestepper->mastereq;

  if (mpirank_world == 0 && !quietmode) std::cout<< "EVAL GRAD F... " << std::endl;
  Vec finalstate = NULL;
  Vec adjoint_ic = NULL;

  /* Pass design vector x to oscillators */
  // x_alpha is set only on first processor. Need to communicate x_alpha to all processors here. 
  VecScatterBegin(scatter_alpha, x, x_alpha, INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(scatter_alpha, x, x_alpha, INSERT_VALUES, SCATTER_FORWARD);
  mastereq->setControlAmplitudes(x_alpha); 

  /* Reset Gradient */
  VecZeroEntries(G);

  /* Derivative of regulatization term gamma / 2 ||x||^2 */
  VecScale(x_alpha, gamma_tik);
  VecScatterBegin(scatter_alpha, x_alpha, G, INSERT_VALUES, SCATTER_REVERSE);
  VecScatterEnd(scatter_alpha, x_alpha, G, INSERT_VALUES, SCATTER_REVERSE);

  /*  Iterate over initial condition */
  obj_cost = 0.0;
  obj_regul = 0.0;
  obj_penal = 0.0;
  obj_penal_dpdm = 0.0;
  obj_penal_energy = 0.0;
  fidelity = 0.0;
  interm_discontinuity = 0.0; // for TaoMonitor
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
        timestepper->mastereq->getRhoT0(iinit_global, ninit, initcond_type, initcond_IDs, rho_t0);
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
        optim_target->prepare(rho_t0);
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
        constraint += 0.5 * mu * qnorm2 - cdot;
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
  double myconstraint = constraint;
  double my_interm_disc = interm_discontinuity;
  double my_frob2 = frob2;
  MPI_Allreduce(&mypen, &obj_penal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mypen_dpdm, &obj_penal_dpdm, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mypenen, &obj_penal_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  // Should be comm_init and also comm_time! 
  MPI_Allreduce(&mycost_re, &obj_cost_re, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mycost_im, &obj_cost_im, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&myfidelity_re, &fidelity_re, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&myfidelity_im, &fidelity_im, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&myconstraint, &constraint, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
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
  VecScale(x_alpha, 1.0/gamma_tik); // revert scaling that was done in the beginning of evalGradF
  if (!gamma_tik_interpolate){  // ||x||^2
    VecNorm(x_alpha, NORM_2, &x_alpha_norm);
  } 
  obj_regul = gamma_tik / 2. * pow(x_alpha_norm,2.0);

  /* Sum, store and return objective value */
  objective = obj_cost + obj_regul + obj_penal + obj_penal_dpdm + obj_penal_energy + constraint;

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
          timestepper->mastereq->getRhoT0(iinit_global, ninit, initcond_type, initcond_IDs, rho_t0);
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
          optim_target->prepare(rho_t0);
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

  /* Compute and store gradient norm */
  VecNorm(G, NORM_2, &(gnorm));

  /* Output */
  if (mpirank_world == 0 && !quietmode) {
    std::cout<< "Objective = " << std::scientific<<std::setprecision(14) << obj_cost << " + " << obj_regul << " + " << obj_penal << " + " << obj_penal_dpdm << " + " << obj_penal_energy << " + " << constraint << std::endl;
    if (nwindows == 1) // Fidelity only makes sense with one window
      std::cout<< "Fidelity = " << fidelity << std::endl;
    std::cout<< "Discontinuities = " << interm_discontinuity << std::endl;
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
    if (mpirank_world == 0) { // only first processor stores x_alpha
      PetscScalar* xptr;
      VecGetArray(xinit, &xptr);
      int shift = 0;
      for (int ioscil = 0; ioscil<mastereq->getNOscillators(); ioscil++){
        mastereq->getOscillator(ioscil)->getParams(xptr + shift);
        shift += mastereq->getOscillator(ioscil)->getNParams();
      }
      VecRestoreArray(xinit, &xptr);
    }
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

  /* Set the initial guess for the intermediate states. Here, roll-out forward propagation. TODO: Read from file*/
  // Note: THIS Currently is entirely serial! No parallel initial conditions, no parallel windows. 
  if (mpirank_world==0) {
    printf(" -> Rollout initialization of intermediate states (sequential in time windows parallel in initial conditions. This might take a while...\n");
  }
  rollOut(xinit);
}

void OptimProblem::rollOut(Vec x){

  /* Roll-out forward propagation. */
  for (int iinit = 0; iinit < ninit_local; iinit++) {
    int iinit_global = mpirank_init * ninit_local + iinit;

    // Get the initial condition
    timestepper->mastereq->getRhoT0(iinit_global, ninit, initcond_type, initcond_IDs, rho_t0);

    for (int iwindow_global=0; iwindow_global<nwindows-1; iwindow_global++){

      // Solve forward from starting point.
      Vec xfinal = timestepper->solveODE(1, rho_t0, iwindow_global * timestepper->ntime); 
      VecCopy(xfinal, rho_t0);

      /* Potentially, store the intermediate results in the given vector */
      int size;
      ISGetLocalSize(IS_interm_states[iwindow_global+1][iinit_global], &size);
      if (size > 0) {
        // printf("%d|%d|%d -> Copying size %d into iwindow %d iinit %d\n", mpirank_world, mpirank_time, mpirank_init, size, iwindow+1, iinit);
        VecISCopy(x, IS_interm_states[iwindow_global+1][iinit_global], SCATTER_FORWARD, rho_t0);
      }

      PetscBarrier((PetscObject) x);
    } // end for iwindow
  } // end for initial condition

  VecView(x, PETSC_VIEWER_STDOUT_WORLD);
}

/* lag += - prev_mu * ( S(u_{i-1}) - u_i ) */
void OptimProblem::updateLagrangian(const double prev_mu, const Vec x, Vec lambda) {

  /* Forward solve to store the final states. This might already always be the case... */
  evalF(x, lambda);

  /* Iterate over local initial conditions */
  for (int iinit = 0; iinit < ninit_local; iinit++) {
    /* Iterate over local time windows */
    for (int iwindow = 0; iwindow < nwindows_local; iwindow++){

      int iinit_global = mpirank_init * ninit_local + iinit;
      int iwindow_global = mpirank_time*nwindows_local + iwindow ; 

      // Exclude the very last time window 
      if (iwindow_global == nwindows-1)
        continue;

      /* Get next time-windows initial state and eval discontinuity */
      VecScatterBegin(scatter_xnext[iwindow][iinit], x, x_next, INSERT_VALUES, SCATTER_FORWARD);
      VecScatterEnd(scatter_xnext[iwindow][iinit], x, x_next, INSERT_VALUES, SCATTER_FORWARD);
      VecCopy(store_finalstates[iwindow][iinit], disc);
      VecAXPY(disc, -1.0, x_next);  // disc = S(u_{i-1}) - u_i

      /* lag += - prev_mu * ( S(u_{i-1}) - u_i ) */
      Vec lag;
      VecGetSubVector(lag, IS_interm_lambda[iwindow_global][iinit_global], &lag);
      VecAXPY(lag, -prev_mu, disc);
      VecRestoreSubVector(lag, IS_interm_lambda[iwindow_global][iinit_global], &lag);

      printf("\n\n TODO: Check the updateLagrangian function!\n");
    }
  }
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

  /* Pass current iteration number to output manager */
  ctx->output->optim_iter = iter;

  /* Grab some output stuff */
  double objective = ctx->getObjective();
  double obj_cost = ctx->getCostT();
  double obj_regul = ctx->getRegul();
  double obj_penal = ctx->getPenalty();
  double obj_penal_dpdm = ctx->getPenaltyDpDm();
  double obj_penal_energy = ctx->getPenaltyEnergy();
  double interm_discontinuity = ctx->getDiscontinuity();
  double F_avg = ctx->getFidelity();

  /* Print to optimization file */
  ctx->output->writeOptimFile(f, gnorm, deltax, F_avg, obj_cost, obj_regul, obj_penal, obj_penal_dpdm, obj_penal_energy, interm_discontinuity);

  /* Print parameters and controls to file */
  // if ( optim_iter % optim_monitor_freq == 0 ) {
  // ctx->output->writeControls(params, ctx->timestepper->mastereq, ctx->timestepper->ntime, ctx->timestepper->dt);
  // }

  /* Screen output */
  if (ctx->getMPIrank_world() == 0 && iter == 0) {
    std::cout<<  "    Objective             Tikhonov                Penalty-Leakage        Penalty-StateVar       Penalty-TotalEnergy " << std::endl;
  }

  /* Additional Stopping criteria */
  bool lastIter = false;
  std::string finalReason_str = "";
  if ((1.0 - F_avg <= ctx->getInfTol()) && (interm_discontinuity < ctx->getIntermTol())) {
    finalReason_str = "Optimization converged to a continuous trajectory with small infidelity.";
    TaoSetConvergedReason(tao, TAO_CONVERGED_USER);
    lastIter = true;
  } else if ((obj_cost <= ctx->getFaTol()) && (interm_discontinuity < ctx->getIntermTol())) {
    finalReason_str = "Optimization converged to a continuous trajectory with small final time cost.";
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

  if (ctx->getMPIrank_world() == 0 && (iter == ctx->getMaxIter() || lastIter || iter % ctx->output->optim_monitor_freq == 0)) {
    std::cout<< iter <<  "  " << std::scientific<<std::setprecision(14) << obj_cost << " + " << obj_regul << " + " << obj_penal << " + " << obj_penal_dpdm << " + " << obj_penal_energy;
    if (ctx->getNwindows() == 1) // Fidelity only makes sense with one window
      std::cout<< "  Fidelity = " << F_avg;
    std::cout<< "  ||Grad|| = " << gnorm;
    std::cout<< std::endl;
  }

  if (ctx->getMPIrank_world() == 0 && lastIter){
    std::cout<< finalReason_str << std::endl;
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
  VecView(ctx->lambda, PETSC_VIEWER_STDOUT_WORLD);
  exit(1);
  *f = ctx->evalF(x, ctx->lambda);
  
  return 0;
}


PetscErrorCode TaoEvalGradient(Tao tao, Vec x, Vec G, void*ptr){

  OptimProblem* ctx = (OptimProblem*) ptr;
  assert(ctx->lambda);
  ctx->evalGradF(x, ctx->lambda, G);
  
  return 0;
}
