#include "optimproblem.hpp"

OptimProblem::OptimProblem(MapParam config, TimeStepper* timestepper_, MPI_Comm comm_init_, MPI_Comm comm_time_, int ninit_, int nwindows_, double total_time_, std::vector<double> gate_rot_freq, Output* output_, bool quietmode_){

  timestepper = timestepper_;
  ninit = ninit_;
  nwindows = nwindows_;
  output = output_;
  quietmode = quietmode_;
  /* Reset */
  objective = 0.0;
  lambda = NULL;

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

  /* NOTE(kevin): store intermediate final states regardless of master equation type. */
  store_interm_states.resize(ninit_local);
  for (int i = 0; i < ninit_local; i++) {
    /*
      store_interm_states[i][index] is the final timestep of index-th local time window, for i-th initial condition.
      (index = 0, 1, ..., nwindows_local-1)
      For mpirank_time = mpisize_time - 1,
        store_interm_states[i] has a size of (nwindows_local - 1), excluding final state.
    */
    store_interm_states[i].clear();
    for (int iwindow = 0; iwindow < nwindows_local; iwindow++) {
      if (mpirank_time == mpisize_time-1 && iwindow == nwindows_local-1)
        break;
      
      Vec state;
      VecCreate(PETSC_COMM_WORLD, &state);
      VecSetSizes(state, PETSC_DECIDE, 2*timestepper->mastereq->getDim());
      VecSetFromOptions(state);
      store_interm_states[i].push_back(state);
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

  /* allocate reduced gradient of timestepper */
  timestepper->allocateReducedGradient(ndesign + nstate);

  /* Create vector strides to access the control vector and intermediate states, and intermediate lagrange multipliers */
  ISCreateStride(PETSC_COMM_WORLD, ndesign, 0, 1, &IS_alpha);
  int skip = 0;
  int every = 1;
  for (int ic=0; ic<timestepper->mastereq->getDimRho(); ic++){
    for (int m=0; m<nwindows-1; m++) {
      IS state_m_ic;
      IS lambda_m_ic;
      int xdim = timestepper->mastereq->getDimRho()*2; // state dimension. x2 for real and imag. 
      ISCreateStride(PETSC_COMM_WORLD, xdim, ndesign+ skip, every, &state_m_ic);
      ISCreateStride(PETSC_COMM_WORLD, xdim, skip, every, &lambda_m_ic);
      IS_interm_states.push_back(state_m_ic);
      IS_interm_lambda.push_back(lambda_m_ic);
      skip += xdim; 
    }
  } 
  // printf("state strides: %d\n", IS_interm_states.size());

  /* Store other optimization parameters */
  gamma_tik = config.GetDoubleParam("optim_regul", 1e-4);
  gamma_tik_interpolate = config.GetBoolParam("optim_regul_interpolate", false, false);
  gatol = config.GetDoubleParam("optim_atol", 1e-8);
  fatol = config.GetDoubleParam("optim_ftol", 1e-8);
  inftol = config.GetDoubleParam("optim_inftol", 1e-5);
  grtol = config.GetDoubleParam("optim_rtol", 1e-4);
  interm_tol = config.GetDoubleParam("optim_interm_tol", 1e-4);
  maxiter = config.GetIntParam("optim_maxiter", 200);
  mu = config.GetDoubleParam("optim_mu", 0.0);
  
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
  VecCreate(PETSC_COMM_WORLD, &rho_t0); 
  VecSetSizes(rho_t0,PETSC_DECIDE,2*timestepper->mastereq->getDim());
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
  VecCreateSeq(PETSC_COMM_SELF, ndesign+nstate, &xlower);
  VecSetFromOptions(xlower);
  VecDuplicate(xlower, &xupper);
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
  double very_large = 1.0e20;
  for (int k = getNdesign(); k < getNoptimvars(); k++) {
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
  mygrad = new double[getNoptimvars()];

  // /* Allocat xprev, xinit, xtmp */
  // VecCreateSeq(PETSC_COMM_SELF, ndesign, &xprev);
  // VecSetFromOptions(xprev);
  // VecZeroEntries(xprev);

  /* Allocate temporary storage of a state discontinuity */
  VecCreate(PETSC_COMM_WORLD, &disc);
  VecSetSizes(disc, PETSC_DECIDE, 2*timestepper->mastereq->getDim());
  VecSetFromOptions(disc);
  /* Allocate temporary storage of a lagrange multiplier update */
  VecCreate(PETSC_COMM_WORLD, &lambda_incre);
  VecSetSizes(lambda_incre, PETSC_DECIDE, getNstate());
  VecSetFromOptions(lambda_incre);
  VecSet(lambda_incre, 0.0);
  VecAssemblyBegin(lambda_incre);
  VecAssemblyEnd(lambda_incre);


  if (gamma_tik_interpolate) {
    // DISABLE FOR NOW
    printf("Warning: Disabling gamma_tik_interpolate for multiple shooting.\n");
    gamma_tik_interpolate = false;
    // VecCreateSeq(PETSC_COMM_SELF, ndesign, &xinit);
    // VecSetFromOptions(xinit);
    // VecZeroEntries(xinit);
  }

  // VecCreateSeq(PETSC_COMM_SELF, ndesign, &xtmp);
  // VecSetFromOptions(xtmp);
  // VecZeroEntries(xtmp);
}


OptimProblem::~OptimProblem() {
  delete [] mygrad;
  delete optim_target;
  VecDestroy(&rho_t0);
  VecDestroy(&rho_t0_bar);

  VecDestroy(&xlower);
  VecDestroy(&xupper);
  // VecDestroy(&xprev);
  if (gamma_tik_interpolate) {
    // VecDestroy(&xinit);
  }
  VecDestroy(&disc);
  VecDestroy(&lambda_incre);

  for (int i = 0; i < store_finalstates.size(); i++) {
    VecDestroy(&(store_finalstates[i]));

    for (int k = 0; k < store_interm_states[i].size(); k++)
      VecDestroy(&(store_interm_states[i][k]));
  }

  for (int m=0; m<IS_interm_states.size(); m++){
    ISDestroy(&(IS_interm_states[m]));
  }
  ISDestroy(&IS_alpha);

  TaoDestroy(&tao);
}



double OptimProblem::evalF(const Vec x, const Vec lambda_, const bool store_interm) {   // x = (alpha, interm.states), lambda = lagrange multipliers: dim(lambda) = dim(x_intermediatestates)

  MasterEq* mastereq = timestepper->mastereq;

  if (mpirank_world == 0 && !quietmode) printf("EVAL F... \n");
  Vec finalstate = NULL;

  /* Pass control vector to oscillators */
  Vec x_alpha;
  VecGetSubVector(x, IS_alpha, &x_alpha);
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
      
    /* Prepare the initial condition in [rank * ninit_local, ... , (rank+1) * ninit_local - 1] */
    int iinit_global = mpirank_init * ninit_local + iinit;
    if ( mpirank_time == 0 || mpirank_time == mpisize_time -1 ) {
      // Note: first rank needs it as initial condition. last rank needs it to prepare the target state.
      timestepper->mastereq->getRhoT0(iinit_global, ninit, initcond_type, initcond_IDs, rho_t0);
    }
    // if (mpirank_time == 0 && !quietmode) printf("%d: Initial condition id=%d ...\n", mpirank_init, initid);

    /* If gate optimiztion, compute the target state rho^target = Vrho(0)V^dagger */
    if (mpirank_time == mpisize_time-1) {
      optim_target->prepare(rho_t0);
    }

    /* Iterate over local time windows */
    for (int iwindow=0; iwindow<nwindows_local; iwindow++){
      // Solve forward from starting point.
      int index = mpirank_time*nwindows_local + iwindow ; 
      int n0 = index * timestepper->ntime; // First time-step index for this window.
      // printf("%d: Local window %d , n0=%d\n", mpirank_time, iwindow, n0);
      Vec x0;
      if (mpirank_time == 0 && iwindow == 0) {
        x0 = rho_t0; 
      } else {
        int id = iinit_global*(nwindows-1) + index-1;
        // printf("time rank %d, init rank %d: iinit %d, iwindow %d, starting from global id = %d\n", mpirank_time, mpirank_init, iinit, iwindow, id);
        VecGetSubVector(x, IS_interm_states[id], &x0);
        // VecView(x0, NULL);
      }
      // TODO (SG): Fix timestepper output (-> initid, windowid.)
      finalstate = timestepper->solveODE(1, x0, n0);

      /* Add to integral penalty term */
      obj_penal += obj_weights[iinit] * gamma_penalty * timestepper->penalty_integral;

      /* Add to second derivative penalty term */
      obj_penal_dpdm += obj_weights[iinit] * gamma_penalty_dpdm * timestepper->penalty_dpdm;
    
      /* Add to energy integral penalty term */
      obj_penal_energy += obj_weights[iinit] * gamma_penalty_energy* timestepper->energy_penalty_integral;

      /* Evaluate J(finalstate) and add to final-time cost */
      if (mpirank_time == mpisize_time-1 && iwindow == nwindows_local-1){
        double obj_iinit_re = 0.0;
        double obj_iinit_im = 0.0;
        double frob2_iinit; // For generalized infidelity 
        // Local contribution to the Hilbert-Schmidt overlap between target and final states (S_T)
        optim_target->evalJ(finalstate,  &obj_iinit_re, &obj_iinit_im, &frob2_iinit);
        
        obj_cost_re += obj_weights[iinit] * obj_iinit_re; // For Schroedinger, weights = 1.0/ninit
        obj_cost_im += obj_weights[iinit] * obj_iinit_im;
        frob2 += frob2_iinit / ninit;

        /* Contributions to final-time (regular) fidelity */
        double fidelity_iinit_re = 0.0;
        double fidelity_iinit_im = 0.0;
        // NOTE: scalebypurity = false. TODO: Check.
        optim_target->HilbertSchmidtOverlap(finalstate, false, &fidelity_iinit_re, &fidelity_iinit_im);
        fidelity_re += fidelity_iinit_re / ninit; // Scale by 1/N
        fidelity_im += fidelity_iinit_im / ninit;
    
        // printf("%d, %d: iinit %d, iwindow %d, add to objective obj_cost_re = %1.8e\n", mpirank_time, mpirank_init, iinit, iwindow, obj_cost_re);
      }
      /* Else, add to constraint. */
      else {
        if (store_interm)
          VecCopy(finalstate, store_interm_states[iinit][iwindow]);

        int id = iinit_global*(nwindows-1) + index;
        Vec xnext, lag;
        VecGetSubVector(x, IS_interm_states[id], &xnext);
        VecGetSubVector(lambda_, IS_interm_lambda[id], &lag);
        VecAXPY(finalstate, -1.0, xnext);  // finalstate = S(u_{i-1}) - u_i

        // TODO(kevin): templatize this for various penalty functionals.
        double cdot, qnorm2;
        VecDot(finalstate, finalstate, &qnorm2); // q = || (Su - u) ||^2
        VecDot(finalstate, lag, &cdot);   // c = lambda^T (Su - u)
        interm_discontinuity += qnorm2;
        constraint += 0.5 * mu * qnorm2 - cdot;
        // printf("%d: Window %d, add to constraint taking from id=%d. c=%f\n", mpirank_time, iwindow, id, cdot);
      }
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
  double xnorm;
  if (!gamma_tik_interpolate){  // ||x_alpha||^2
    VecNorm(x_alpha, NORM_2, &xnorm);
  } 
  // else {
    // VecCopy(x, xtmp);
    // VecAXPY(xtmp, -1.0, xinit);    // xtmp =  x - x_0
    // VecNorm(xtmp, NORM_2, &xnorm);
  // }
  obj_regul = gamma_tik / 2. * pow(xnorm,2.0);

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
  Vec x_alpha;
  VecGetSubVector(x, IS_alpha, &x_alpha);
  mastereq->setControlAmplitudes(x_alpha); 

  /* Reset Gradient */
  VecZeroEntries(G);

  /* Derivative of regulatization term gamma / 2 ||x||^2 Note: currently optim variable is global. do this on only one rank in global communicator. */
  if (mpirank_world == 0 && mpirank_time == 0) {
    VecISAXPY(G, IS_alpha, gamma_tik, x_alpha);   // + gamma_tik * x
    if (gamma_tik_interpolate){
      // VecAXPY(G, -1.0*gamma_tik, xinit); // -gamma_tik * xinit
    }
  }

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
  double frob2 = 0.0; // Neede for generalized infidelity
  for (int iinit = 0; iinit < ninit_local; iinit++) {

    /* Prepare the initial condition */
    int iinit_global = mpirank_init * ninit_local + iinit;
    int initid;
    if ( mpirank_time == 0 || mpirank_time == mpisize_time -1 ) {
      // Note: first rank needs it as initial condition. last rank needs it to prepare the target state.
      initid = timestepper->mastereq->getRhoT0(iinit_global, ninit, initcond_type, initcond_IDs, rho_t0);
    }

    /* If gate optimiztion, compute the target state rho^target = Vrho(0)V^dagger */
    if (mpirank_time == mpisize_time-1) {
      optim_target->prepare(rho_t0);
    }

    /* Iterate over local time windows */
    for (int iwindow=0; iwindow<nwindows_local; iwindow++){
      // Solve forward from starting point.
      int index = mpirank_time * nwindows_local + iwindow ;
      int n0 = index * timestepper->ntime; // First time-step index for this window.
      // printf("%d: Local window %d , n0=%d\n", mpirank_time, iwindow, n0);
      Vec x0;
      if (mpirank_time == 0 && iwindow == 0) {
        x0 = rho_t0; 
      } else {
        int id = iinit_global*(nwindows-1) + index-1;
        // printf("%d, %d: iinit %d, iwindow %d, starting from global id = %d\n", mpirank_time, mpirank_init, iinit, iwindow, id);
        VecGetSubVector(x, IS_interm_states[id], &x0);
        // VecView(x0, NULL);
      }

      /* --- Solve primal --- */
      // if (mpirank_time == 0) printf("%d: %d FWD. ", mpirank_init, initid);

      /* Run forward with initial condition rho_t0 */
      // TODO (SG): Fix timestepper output (-> initid, windowid)
      finalstate = timestepper->solveODE(initid, x0, n0);

      /* Add to integral penalty term */
      obj_penal += obj_weights[iinit] * gamma_penalty * timestepper->penalty_integral;

      /* Add to second derivative dpdm integral penalty term */
      obj_penal_dpdm += obj_weights[iinit] * gamma_penalty_dpdm * timestepper->penalty_dpdm;
      /* Add to energy integral penalty term */
      obj_penal_energy += obj_weights[iinit] * gamma_penalty_energy * timestepper->energy_penalty_integral;

      /* Evaluate J(finalstate) and add to final-time cost */
      if (mpirank_time == mpisize_time-1 && iwindow == nwindows_local-1) {
        /* Store the final state for the Schroedinger solver */
        if (timestepper->mastereq->lindbladtype == LindbladType::NONE)
          VecCopy(finalstate, store_finalstates[iinit]);

        /* Evaluate J(finalstate) and add to final-time cost */
        double obj_iinit_re = 0.0;
        double obj_iinit_im = 0.0;
        double frob2_iinit;    // new term needed for the generalized infidelity
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
        /* Store the intermediate state for the Schroedinger solver */
        if (timestepper->mastereq->lindbladtype == LindbladType::NONE)
          VecCopy(finalstate, store_interm_states[iinit][iwindow]);

        int id = iinit_global*(nwindows-1) + index;
        Vec xnext, lag;
        VecGetSubVector(x, IS_interm_states[id], &xnext);
        VecGetSubVector(lambda_, IS_interm_lambda[id], &lag);
        VecAXPY(finalstate, -1.0, xnext);  // finalstate = S(u_{i-1}) - u_i

        // TODO(kevin): templatize this for various penalty functionals.
        double cdot, qnorm2;
        VecDot(finalstate, finalstate, &qnorm2); // q = || (Su - u) ||^2
        VecDot(finalstate, lag, &cdot);   // c = lambda^T (Su - u)
        interm_discontinuity += qnorm2;
        constraint += 0.5 * mu * qnorm2 - cdot;
        // printf("%d: Window %d, add to constraint taking from id=%d. c=%f\n", mpirank_time, iwindow, id, cdot);
      }

      /* If Lindblas solver, compute adjoint for this initial condition. Otherwise (Schroedinger solver), compute adjoint only after all initial conditions have been propagated through (separate loop below) */
      if (timestepper->mastereq->lindbladtype != LindbladType::NONE) {
        if (mpirank_time == 0)
          printf("WARNING: Multiple shooting adjoint is not yet tested for Lindblas solver!\n");
        // if (mpirank_time == 0) printf("%d: %d BWD.", mpirank_init, initid);

        /* Reset adjoint */
        VecZeroEntries(rho_t0_bar);

        /* Terminal condition for adjoint variable: Derivative of final time objective J */
        double obj_cost_re_bar, obj_cost_im_bar, frob2_bar;
        optim_target->finalizeJ_diff(obj_cost_re, obj_cost_im, &obj_cost_re_bar, &obj_cost_im_bar, &frob2_bar);
        optim_target->evalJ_diff(finalstate, rho_t0_bar, obj_weights[iinit]*obj_cost_re_bar, obj_weights[iinit]*obj_cost_im_bar, 1./ninit*frob2_bar);

        /* Derivative of time-stepping */
        adjoint_ic = timestepper->solveAdjointODE(initid, rho_t0_bar, finalstate, obj_weights[iinit] * gamma_penalty, obj_weights[iinit]*gamma_penalty_dpdm, obj_weights[iinit]*gamma_penalty_energy, n0);

        if (!(mpirank_time == 0 && iwindow == 0)) {
          int id = iinit_global*(nwindows-1) + index-1;
          VecISAXPY(G, IS_interm_states[id], 1.0, adjoint_ic);
        }

        /* Add to optimizers's gradient */
        VecAXPY(G, 1.0, timestepper->redgrad);
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
  double xnorm;
  if (!gamma_tik_interpolate){  // ||x||^2
    VecNorm(x_alpha, NORM_2, &xnorm);
  } 
  // else {
    // VecCopy(x, xtmp);
    // VecAXPY(xtmp, -1.0, xinit);    // xtmp =  x_k - x_0
    // VecNorm(xtmp, NORM_2, &xnorm);
  // }
  obj_regul = gamma_tik / 2. * pow(xnorm,2.0);

  /* Sum, store and return objective value */
  objective = obj_cost + obj_regul + obj_penal + obj_penal_dpdm + obj_penal_energy + constraint;

  /* For Schroedinger solver: Solve adjoint equations for all initial conditions here. */
  if (timestepper->mastereq->lindbladtype == LindbladType::NONE) {

    // Iterate over all initial conditions 
    for (int iinit = 0; iinit < ninit_local; iinit++) {
      int iinit_global = mpirank_init * ninit_local + iinit;

      /* Recompute the initial state and target */
      int initid;
      if ( mpirank_time == 0 || mpirank_time == mpisize_time -1 ) {
        initid = timestepper->mastereq->getRhoT0(iinit_global, ninit, initcond_type, initcond_IDs, rho_t0);
      }
      if (mpirank_time == mpisize_time-1) {
        optim_target->prepare(rho_t0);
      }
      
      /* Iterate over local time windows */
      for (int iwindow=0; iwindow<nwindows_local; iwindow++) {
        int index = mpirank_time*nwindows_local + iwindow ; 
        int n0 = index * timestepper->ntime; // First time-step index for this window.

        /* Reset adjoint */
        VecZeroEntries(rho_t0_bar);

        /* Get final primal state and adjoint terminal condition at each local time window */
        if (mpirank_time == mpisize_time-1 && iwindow == nwindows_local-1) {
          /* Get the last time step (finalstate) */
          finalstate = store_finalstates[iinit];

          /* Terminal condition for adjoint variable: Derivative of final time objective J */
          double obj_cost_re_bar, obj_cost_im_bar, frob2_bar;
          optim_target->finalizeJ_diff(obj_cost_re, obj_cost_im, &obj_cost_re_bar, &obj_cost_im_bar, &frob2_bar);
          optim_target->evalJ_diff(finalstate, rho_t0_bar, obj_weights[iinit]*obj_cost_re_bar, obj_weights[iinit]*obj_cost_im_bar, frob2_bar/ninit);
        }
        else {
          finalstate = store_interm_states[iinit][iwindow];
          VecCopy(finalstate, disc);

          int id = iinit_global*(nwindows-1) + index;
          Vec xnext, lag;
          VecGetSubVector(x, IS_interm_states[id], &xnext);
          VecGetSubVector(lambda_, IS_interm_lambda[id], &lag);
          VecAXPY(disc, -1.0, xnext);  // finalstate = S(u_{i-1}) - u_i

          // TODO(kevin): templatize this for various penalty functionals.
          VecAXPY(rho_t0_bar, mu, disc);  // d q / d Su
          VecAXPY(rho_t0_bar, -1.0, lag); // - d c / d Su

          /* add immediate gradient w.r.t the xnext. */
          VecISAXPY(G, IS_interm_states[id], -mu, disc);
          VecISAXPY(G, IS_interm_states[id], 1.0, lag);
        }

        /* Derivative of time-stepping */
        adjoint_ic = timestepper->solveAdjointODE(initid, rho_t0_bar, finalstate, obj_weights[iinit] * gamma_penalty, obj_weights[iinit]*gamma_penalty_dpdm, obj_weights[iinit]*gamma_penalty_energy, n0);

        if (!(mpirank_time == 0 && iwindow == 0)) {
          int id = iinit_global*(nwindows-1) + index-1;
          VecISAXPY(G, IS_interm_states[id], 1.0, adjoint_ic);
        }

        /* Add to optimizers's gradient */
        VecAXPY(G, 1.0, timestepper->redgrad);
      } // for (int iwindow=0; iwindow<nwindows_local; iwindow++)
    } // end of initial condition loop 
  } // end of adjoint for Schroedinger

  /* Sum up the gradient from all initial condition processors */
  PetscScalar* grad; 
  VecGetArray(G, &grad);
  for (int i=0; i<getNoptimvars(); i++) {
    mygrad[i] = grad[i];
  }
  MPI_Allreduce(mygrad, grad, getNoptimvars(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); // This is comm_init AND comm_time.
  VecRestoreArray(G, &grad);

  Vec g_alpha;
  VecGetSubVector(G, IS_alpha, &g_alpha);
  mastereq->setControlAmplitudes_diff(g_alpha);

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

  /* Pass control alphas to the oscillator */
  Vec x_alpha;
  VecGetSubVector(xinit, IS_alpha, &x_alpha);
  timestepper->mastereq->setControlAmplitudes(x_alpha);
  
  /* Write initial control functions to file TODO: Multiple time windows */
  // output->writeControls(xinit, timestepper->mastereq, timestepper->ntime, timestepper->dt);

  /* Set the initial guess for the intermediate states. Here, roll-out forward propagation. TODO: Read from file*/
  // Note: THIS Currently is entirely serial! No parallel initial conditions, no parallel windows. 
  if (mpirank_world==0) {
    printf(" -> Rollout initialization of intermediate states (entirely sequential). This might take a while...\n");
  }
  rollOut(xinit);
  // VecView(xinit, NULL);

}

void OptimProblem::rollOut(Vec x){

  /* Roll-out forward propagation. */
  for (int iinit = 0; iinit < ninit; iinit++) {
    // printf("Initial condition %d\n", iinit);
    // int iinit_global = mpirank_init * ninit_local + iinit;
    int initid = timestepper->mastereq->getRhoT0(iinit, ninit, initcond_type, initcond_IDs, rho_t0);
    Vec x0 = rho_t0; 

    for (int iwindow=0; iwindow<nwindows; iwindow++){
      // Solve forward from starting point.
      int n0 = iwindow * timestepper->ntime; // First time-step index for this window.
      // printf(" Solve in window %d, n0=%d\n", iwindow, n0);
      x0 = timestepper->solveODE(initid, x0, n0);

      /* Potentially, store the intermediate results in the given vector */
      if (x != NULL && iwindow < nwindows-1) {
        int id = iinit*(nwindows-1) + iwindow;
        // printf(" Storing into id=%d\n", id);
        VecISCopy(x, IS_interm_states[id], SCATTER_FORWARD, x0); 
      }
    }
  }

}

/* lag += - prev_mu * ( S(u_{i-1}) - u_i ) */
void OptimProblem::updateLagrangian(const double prev_mu, const Vec x, Vec lambda) {

  /* Forward solve to store intermediate discontinuities */
  evalF(x, lambda, true);

  /* Iterate over local initial conditions */
  for (int iinit = 0; iinit < ninit_local; iinit++) {
    int iinit_global = mpirank_init * ninit_local + iinit;

    /* Iterate over local time windows */
    for (int iwindow = 0; iwindow < nwindows_local; iwindow++){
      // Exclude the very last time window 
      if (mpirank_time == mpisize_time-1 && iwindow == nwindows_local-1)
        continue;

      VecCopy(store_interm_states[iinit][iwindow], disc);

      int index = mpirank_time * nwindows_local + iwindow ;
      int id = iinit_global*(nwindows-1) + index;
      Vec xnext;
      VecGetSubVector(x, IS_interm_states[id], &xnext);
      VecAXPY(disc, -1.0, xnext);  // finalstate = S(u_{i-1}) - u_i

      /* lag += - prev_mu * ( S(u_{i-1}) - u_i ) */
      VecISAXPY(lambda_incre, IS_interm_lambda[id], -prev_mu, disc);
    }
  }

  /* Sum up the increment from all comm_init AND comm_time processors */
  // Note: Reusing allocated temporary storage 'mygrad' here, since its size is already larger than getNstate().
  PetscScalar* lambda_incre_ptr; 
  VecGetArray(lambda_incre, &lambda_incre_ptr);
  for (int i=0; i<getNstate(); i++) {
    mygrad[i] = lambda_incre_ptr[i];
  }
  MPI_Allreduce(mygrad, lambda_incre_ptr, getNstate(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  VecRestoreArray(lambda_incre, &lambda_incre_ptr);

  /* update global Lagrangian */
  VecAXPY(lambda, 1.0, lambda_incre);

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
  *f = ctx->evalF(x, *(ctx->lambda));
  
  return 0;
}


PetscErrorCode TaoEvalGradient(Tao tao, Vec x, Vec G, void*ptr){

  OptimProblem* ctx = (OptimProblem*) ptr;
  assert(ctx->lambda);
  ctx->evalGradF(x, *(ctx->lambda), G);
  
  return 0;
}
