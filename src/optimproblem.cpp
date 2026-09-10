#include "optimproblem.hpp"

OptimProblem::OptimProblem(const Config& config, OptimTarget* optim_target_, TimeStepper* timestepper_, MasterEq* mastereq_, MPI_Comm comm_init_, MPI_Comm comm_optim_, Output* output_, bool quietmode_){

  optim_target = optim_target_;
  timestepper = timestepper_;
  mastereq = mastereq_;
  ninit = config.getNInitialConditions();
  output = output_;
  quietmode = quietmode_;
  output_optimization_stride = config.getOutputOptimizationStride();

  /* Reset */
  objective = 0.0;

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

  
  // Allocate storage for final-time unitary if new objective function is used
  optim_penalty_riemannian = config.getOptimPenaltyRiemannian();
  phase_invariant = config.getOptimPenaltyRiemannianPhaseFree();
  if (optim_penalty_riemannian > 0.0) {
    PetscInt globalsize_rows = mastereq->getDim();
    PetscInt globalsize_cols = ninit;;
    PetscInt localsize_rows = globalsize_rows / mpisize_petsc;
    PetscInt localsize_cols = globalsize_cols / mpisize_petsc;
    MatCreateDense(PETSC_COMM_WORLD, localsize_rows, localsize_cols, globalsize_rows, globalsize_cols, NULL, &U_final_re);
    MatCreateDense(PETSC_COMM_WORLD, localsize_rows, localsize_cols, globalsize_rows, globalsize_cols, NULL, &U_final_im);
    MatCreateDense(PETSC_COMM_WORLD, localsize_rows, localsize_cols, globalsize_rows, globalsize_cols, NULL, &U_final_re_bar);
    MatCreateDense(PETSC_COMM_WORLD, localsize_rows, localsize_cols, globalsize_rows, globalsize_cols, NULL, &U_final_im_bar);
    MatSetUp(U_final_re);
    MatSetUp(U_final_im);
    MatSetUp(U_final_re_bar);
    MatSetUp(U_final_im_bar);
    MatZeroEntries(U_final_re);
    MatZeroEntries(U_final_im);
    MatZeroEntries(U_final_re_bar);
    MatZeroEntries(U_final_im_bar);
    MatAssemblyBegin(U_final_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(U_final_im, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(U_final_re_bar, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(U_final_im_bar, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(U_final_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(U_final_im, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(U_final_re_bar, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(U_final_im_bar, MAT_FINAL_ASSEMBLY);
  }


  /* Store number of design parameters */
  int n = 0;
  for (size_t ioscil = 0; ioscil < mastereq->getNOscillators(); ioscil++) {
      n += mastereq->getOscillator(ioscil)->getNParams(); 
  }
  ndesign = n;
  if (mpirank_world == 0 && !quietmode) std::cout<< "Number of control parameters: " << ndesign << std::endl;

  /* Allocate adjoint terminal state */
  VecDuplicate(optim_target->getInitialState(), &rho_t0_bar);
  VecZeroEntries(rho_t0_bar);
  VecAssemblyBegin(rho_t0_bar); VecAssemblyEnd(rho_t0_bar);

  /* Get weights for the objective function (weighting the different initial conditions */
  obj_weights = config.getOptimWeights();

  /* Store other optimization parameters */
  gamma_tikhonov = config.getOptimTikhonovCoeff();
  tikhonov_use_x0 = config.getOptimTikhonovUseX0();

  // Get tolerance settings
  tol_grad_abs = config.getOptimTolGradAbs();
  tol_final_cost = config.getOptimTolFinalCost();
  tol_infidelity = config.getOptimTolInfidelity();
  tol_grad_rel = config.getOptimTolGradRel();
  maxiter = config.getOptimMaxiter();

  // Get penalty settings
  gamma_penalty_leakage = config.getOptimPenaltyLeakage();
  gamma_penalty_weightedcost = config.getOptimPenaltyWeightedCost();
  gamma_penalty_energy = config.getOptimPenaltyEnergy();
  gamma_penalty_dpdm = config.getOptimPenaltyDpdm();
  gamma_penalty_variation = config.getOptimPenaltyVariation();

  if (gamma_penalty_dpdm > 1e-13 && mastereq->decoherence_type != DecoherenceType::NONE){
    if (mpirank_world == 0 && !quietmode) {
      printf("Warning: Disabling DpDm penalty term because it is not implemented for the Lindblad solver.\n");
    }
    gamma_penalty_dpdm = 0.0;
  }

  /* Store optimization bounds */
  VecCreateSeq(PETSC_COMM_SELF, ndesign, &xlower);
  VecSetFromOptions(xlower);
  VecDuplicate(xlower, &xupper);
  int col = 0;
  for (size_t iosc = 0; iosc < mastereq->getNOscillators(); iosc++){
    // Drive bounds (existing p/q controls)
    double drive_bound = config.getControlAmplitudeBound(iosc);
    drive_bound = drive_bound / (sqrt(2) * mastereq->getOscillator(iosc)->getNCarrierfrequencies());
    drive_bound = drive_bound * 2.0 * M_PI;
    for (size_t i = 0; i < mastereq->getOscillator(iosc)->getNDriveParams(); i++) {
      VecSetValue(xupper, col + i, drive_bound, INSERT_VALUES);
      VecSetValue(xlower, col + i, -1.0 * drive_bound, INSERT_VALUES);
    }
    col += mastereq->getOscillator(iosc)->getNDriveParams();

    // Flux bounds (independent f controls)
    double flux_bound = config.getControlFluxAmplitudeBound(iosc) * 2.0 * M_PI;
    for (size_t i = 0; i < mastereq->getOscillator(iosc)->getNFluxParams(); i++) {
      VecSetValue(xupper, col + i, flux_bound, INSERT_VALUES);
      VecSetValue(xlower, col + i, -1.0 * flux_bound, INSERT_VALUES);
    }
    col += mastereq->getOscillator(iosc)->getNFluxParams();
  }
  VecAssemblyBegin(xlower); VecAssemblyEnd(xlower);
  VecAssemblyBegin(xupper); VecAssemblyEnd(xupper);

  /* Create Petsc's optimization solver */
  TaoCreate(PETSC_COMM_SELF, &tao);
  /* Set optimization type and parameters */
  TaoSetType(tao,TAOBQNLS);         // Optim type: taoblmvm vs BQNLS ??
  TaoSetMaximumIterations(tao, maxiter);
  TaoSetTolerances(tao, tol_grad_abs, PETSC_DEFAULT, tol_grad_rel);
  TaoMonitorSet(tao, TaoMonitor, (void*)this, NULL);
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

  /* Create Geope MatShell for A=L^*L */
  MatCreateShell(PETSC_COMM_SELF, PETSC_DECIDE, PETSC_DECIDE, ndesign, ndesign, this, &A_Geope);
  MatShellSetOperation(A_Geope, MATOP_MULT, (void(*) (void)) applyAGeope);
  VecDuplicate(xinit, &x_for_AGeope);
  VecZeroEntries(x_for_AGeope);
  VecAssemblyBegin(x_for_AGeope); VecAssemblyEnd(x_for_AGeope);

}


OptimProblem::~OptimProblem() {
  delete [] mygrad;
  VecDestroy(&rho_t0_bar);

  VecDestroy(&xlower);
  VecDestroy(&xupper);
  VecDestroy(&xinit);
  VecDestroy(&xtmp);

  if (optim_penalty_riemannian > 0.0) {
    MatDestroy(&U_final_re);
    MatDestroy(&U_final_im);
    MatDestroy(&U_final_re_bar);
    MatDestroy(&U_final_im_bar);
  }
  MatDestroy(&A_Geope);
  VecDestroy(&x_for_AGeope);

  TaoDestroy(&tao);
}



double OptimProblem::evalF(const Vec x, bool writeTrajectoryDataFiles) {
  if (mpirank_world == 0 && !quietmode) printf("EVAL F... \n");

  /* Pass design vector x to oscillators */
  mastereq->setControlAmplitudes(x); 

  // Reset U_final
  if (optim_penalty_riemannian > 0.0) {
    MatZeroEntries(U_final_re);
    MatZeroEntries(U_final_im);
    MatAssemblyBegin(U_final_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(U_final_im, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(U_final_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(U_final_im, MAT_FINAL_ASSEMBLY);
  }

  /*  Iterate over initial condition */
  obj_cost  = 0.0;
  obj_riemann = 0.0;
  obj_regul = 0.0;
  obj_penal_leakage = 0.0;
  obj_penal_weightedcost = 0.0;
  obj_penal_dpdm = 0.0;
  obj_penal_energy = 0.0;
  obj_penal_variation = 0.0;
  fidelity = 0.0;
  double obj_cost_re = 0.0;
  double obj_cost_im = 0.0;
  double fidelity_re = 0.0;
  double fidelity_im = 0.0;
  for (int iinit = 0; iinit < ninit_local; iinit++) {
    int iinit_global = mpirank_init * ninit_local + iinit;
      
    /* Prepare the initial condition in [rank * ninit_local, ... , (rank+1) * ninit_local - 1] */
    int initid = optim_target->prepareInitialAndTargetState(iinit_global, ninit, mastereq->nlevels, mastereq->nessential);

    /* Run forward with initial condition initid */
    if (mpirank_optim == 0 && !quietmode) printf("%d: Initial condition id=%d ...\n", mpirank_init, initid);
    Vec finalstate = timestepper->solveODE(initid, iinit, optim_target->getInitialState(), writeTrajectoryDataFiles, false);

    /* Store the final state for Riemannian objective function */
    if (optim_penalty_riemannian > 0.0) {
      const PetscScalar *finalstate_array;
      VecGetArrayRead(finalstate, &finalstate_array);
      for (size_t row = 0; row < mastereq->getDim(); row++) {
        int id_re = row;
        int id_im = row + mastereq->getDim();
        MatSetValue(U_final_re, row, iinit_global, finalstate_array[id_re], INSERT_VALUES);
        MatSetValue(U_final_im, row, iinit_global, finalstate_array[id_im], INSERT_VALUES);
      }
      VecRestoreArrayRead(finalstate, &finalstate_array);
    }

    /* Add to leakage penalty term */
    obj_penal_leakage += obj_weights[iinit_global] * gamma_penalty_leakage * timestepper->getLeakageIntegral();

    /* Add to running cost penalty term */
    obj_penal_weightedcost += obj_weights[iinit_global] * gamma_penalty_weightedcost * timestepper->getWeightedCostIntegral();

    /* Add to second derivative penalty term */
    obj_penal_dpdm += obj_weights[iinit_global] * gamma_penalty_dpdm * timestepper->getDPDMIntegral();
    
    /* Add to energy integral penalty term */
    obj_penal_energy += obj_weights[iinit_global] * gamma_penalty_energy* timestepper->getEnergyIntegral();

    /* Evaluate J(finalstate) and add to final-time cost */
    double obj_iinit_re = 0.0;
    double obj_iinit_im = 0.0;
    optim_target->evalJ(finalstate,  &obj_iinit_re, &obj_iinit_im);
    obj_cost_re += obj_weights[iinit_global] * obj_iinit_re;
    obj_cost_im += obj_weights[iinit_global] * obj_iinit_im;

    /* Add to final-time fidelity */
    double fidelity_iinit_re = 0.0;
    double fidelity_iinit_im = 0.0;
    optim_target->HilbertSchmidtOverlap(finalstate, false, &fidelity_iinit_re, &fidelity_iinit_im);
    fidelity_re += 1./ ninit * fidelity_iinit_re;
    fidelity_im += 1./ ninit * fidelity_iinit_im;

    // printf("%d, %d: iinit obj_iinit: %f * (%1.14e + i %1.14e, Overlap=%1.14e + i %1.14e\n", mpirank_world, mpirank_init, obj_weights[iinit_global], obj_iinit_re, obj_iinit_im, fidelity_iinit_re, fidelity_iinit_im);
  }
  if (optim_penalty_riemannian > 0.0) {
    MatAssemblyBegin(U_final_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(U_final_im, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(U_final_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(U_final_im, MAT_FINAL_ASSEMBLY);
  }

  /* Sum up from initial conditions processors */
  double mypen_leak = obj_penal_leakage;
  double mypen_wcost = obj_penal_weightedcost;
  double mypen_dpdm = obj_penal_dpdm;
  double mypenen = obj_penal_energy;
  double mycost_re = obj_cost_re;
  double mycost_im = obj_cost_im;
  double myfidelity_re = fidelity_re;
  double myfidelity_im = fidelity_im;
  MPI_Allreduce(&mypen_leak, &obj_penal_leakage, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mypen_wcost, &obj_penal_weightedcost, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mypen_dpdm, &obj_penal_dpdm, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mypenen, &obj_penal_energy, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mycost_re, &obj_cost_re, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mycost_im, &obj_cost_im, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&myfidelity_re, &fidelity_re, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&myfidelity_im, &fidelity_im, 1, MPI_DOUBLE, MPI_SUM, comm_init);

  /* Set the fidelity: If Schroedinger, need to compute the absolute value: Fid= |\sum_i \phi^\dagger \phi_target|^2 */
  if (mastereq->decoherence_type == DecoherenceType::NONE) {
    fidelity = pow(fidelity_re, 2.0) + pow(fidelity_im, 2.0);
  } else {
    fidelity = fidelity_re; 
  }
 
  /* Finalize the objective function */
  obj_cost = optim_target->finalizeJ(obj_cost_re, obj_cost_im);

  /* Penalty: Riemannian distance */
  if (optim_penalty_riemannian > 0.0) {

    MPI_Barrier(MPI_COMM_WORLD);

    /* allreduce the U_final matrix */
    PetscScalar *data;
    MatDenseGetArray(U_final_re, &data);
    int size = mastereq->getDim();
    MPI_Allreduce(MPI_IN_PLACE, data, size * size, MPIU_SCALAR, MPI_SUM, comm_init);
    MatDenseRestoreArray(U_final_re, &data);

    MatDenseGetArray(U_final_im, &data);
    MPI_Allreduce(MPI_IN_PLACE, data, size * size, MPIU_SCALAR, MPI_SUM, comm_init);
    MatDenseRestoreArray(U_final_im, &data);

    double obj_riemannian = optim_target->RiemannianDistance(U_final_re, U_final_im, phase_invariant);

    // if (mpirank_world == 0) printf("\nRiemannian distance objective: %1.14e\n\n", obj_riemannian);
    // obj_cost = obj_riemannian;
    obj_riemann = optim_penalty_riemannian * obj_riemannian;
    obj_cost = 0.0; // Set to zero, so that riemannian objective is used. Choose optim_penalty_riemannian =1.0. 
    obj_cost_re = 0.0;
    obj_cost_im = 0.0;
  }

  /* Evaluate Tikhonov regularization term: gamma/2 * ||x-x0||^2*/
  double xnorm;
  if (!tikhonov_use_x0){  // ||x||^2
    VecNorm(x, NORM_2, &xnorm);
  } else {
    VecCopy(x, xtmp);
    VecAXPY(xtmp, -1.0, xinit);    // xtmp =  x - x_0
    VecNorm(xtmp, NORM_2, &xnorm);
  }
  obj_regul = gamma_tikhonov / 2. * pow(xnorm,2.0);

  /* Evaluate penality term for control variation */
  double var_reg = 0.0;
  for (size_t iosc = 0; iosc < mastereq->getNOscillators(); iosc++){
    var_reg += mastereq->getOscillator(iosc)->evalControlVariation(); // uses Oscillator::params instead of 'x'
  }
  obj_penal_variation = 0.5*gamma_penalty_variation*var_reg; 

  /* Sum, store and return objective value */
  objective = obj_cost + obj_regul + obj_penal_leakage + obj_penal_dpdm + obj_penal_energy + obj_penal_variation + obj_penal_weightedcost + obj_riemann;

  /* Output */
  if (mpirank_world == 0 && !quietmode) {
    std::cout<< "Objective = " << std::scientific<<std::setprecision(14) << obj_cost << " + " << obj_regul << " + " << obj_penal_leakage << " + " << obj_penal_dpdm << " + " << obj_penal_energy << " + " << obj_penal_variation << " + " << obj_penal_weightedcost << " + " << obj_riemann << std::endl;
    std::cout<< "Fidelity = " << fidelity  << std::endl;
  }

  return objective;
}



void OptimProblem::evalGradF(const Vec x, Vec G, bool writeTrajectoryDataFiles){
  if (mpirank_world == 0 && !quietmode) std::cout<< "EVAL GRAD F... " << std::endl;

  /* Pass design vector x to oscillators */
  mastereq->setControlAmplitudes(x); 

  // DEBUG
  // output->writeControl(x, mastereq, timestepper->ntime, timestepper->dt);
  output->writeControlParams(x);

  /* Reset Gradient */
  VecZeroEntries(G);

  // Reset U_final
  if (optim_penalty_riemannian > 0.0) {
    MatZeroEntries(U_final_re);
    MatZeroEntries(U_final_im);
    MatZeroEntries(U_final_re_bar);
    MatZeroEntries(U_final_im_bar);
    MatAssemblyBegin(U_final_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(U_final_im, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(U_final_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(U_final_im, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(U_final_re_bar, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(U_final_im_bar, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(U_final_re_bar, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(U_final_im_bar, MAT_FINAL_ASSEMBLY);
  }

  /* Derivative of regulatization terms (ADD ON ONE PROC ONLY!) */
  // if (mpirank_init == 0 && mpirank_optim == 0) { // TODO: Which one?? 
  if (mpirank_init == 0 ) {

    // Derivative of Tikhonov 0.5 * gamma * ||x||^2 
    VecAXPY(G, gamma_tikhonov, x);   // + gamma * x
    if (tikhonov_use_x0){
      VecAXPY(G, -1.0*gamma_tikhonov, xinit); // -gamma * xinit
    }

    // Derivative of penalization of control variation 
    double var_reg_bar = 0.5*gamma_penalty_variation;
    int skip_to_oscillator = 0;
    for (size_t iosc = 0; iosc < mastereq->getNOscillators(); iosc++){
      Oscillator* osc = mastereq->getOscillator(iosc);
      osc->evalControlVariationDiff(G, var_reg_bar, skip_to_oscillator);
      skip_to_oscillator += osc->getNParams();
    }
  }

  /*  Iterate over initial condition */
  obj_cost = 0.0;
  obj_riemann = 0.0;
  obj_regul = 0.0;
  obj_penal_leakage = 0.0;
  obj_penal_weightedcost = 0.0;
  obj_penal_dpdm = 0.0;
  obj_penal_energy = 0.0;
  obj_penal_variation = 0.0;
  fidelity = 0.0;
  double obj_cost_re = 0.0;
  double obj_cost_im = 0.0;
  double fidelity_re = 0.0;
  double fidelity_im = 0.0;
  for (int iinit = 0; iinit < ninit_local; iinit++) {
    int iinit_global = mpirank_init * ninit_local + iinit;
    // printf("%d: Initial condition id=%d ...\n", mpirank_init, iinit_global);

    /* Prepare the initial and target state */
    int initid = optim_target->prepareInitialAndTargetState(iinit_global, ninit, mastereq->nlevels, mastereq->nessential);

    /* --- Solve primal --- */
    // if (mpirank_optim == 0) printf("%d: %d FWD. ", mpirank_init, initid);

    /* Run forward with initial condition */
    Vec finalstate = timestepper->solveODE(initid, iinit, optim_target->getInitialState(), writeTrajectoryDataFiles, true);

    /* Store the final state for Riemannian objective function */
    if (optim_penalty_riemannian > 0.0) {
      const PetscScalar *finalstate_array;
      VecGetArrayRead(finalstate, &finalstate_array);
      for (size_t row = 0; row < mastereq->getDim(); row++) {
        int id_re = row;
        int id_im = row + mastereq->getDim();
        MatSetValue(U_final_re, row, iinit_global, finalstate_array[id_re], INSERT_VALUES);
        MatSetValue(U_final_im, row, iinit_global, finalstate_array[id_im], INSERT_VALUES);
      }
      VecRestoreArrayRead(finalstate, &finalstate_array);
      
      // Assembly here, or could be after the loop over iinit. 
      MatAssemblyBegin(U_final_re, MAT_FINAL_ASSEMBLY);
      MatAssemblyBegin(U_final_im, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(U_final_re, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(U_final_im, MAT_FINAL_ASSEMBLY);
    }

    /* Add to leakage penalty term */
    obj_penal_leakage += obj_weights[iinit_global] * gamma_penalty_leakage * timestepper->getLeakageIntegral();

    /* Add to running cost penalty term */
    obj_penal_weightedcost += obj_weights[iinit_global] * gamma_penalty_weightedcost * timestepper->getWeightedCostIntegral();

    /* Add to second derivative dpdm integral penalty term */
    obj_penal_dpdm += obj_weights[iinit_global] * gamma_penalty_dpdm * timestepper->getDPDMIntegral();
    /* Add to energy integral penalty term */
    obj_penal_energy += obj_weights[iinit_global] * gamma_penalty_energy * timestepper->getEnergyIntegral();

    /* Evaluate J(finalstate) and add to final-time cost */
    double obj_iinit_re = 0.0;
    double obj_iinit_im = 0.0;
    optim_target->evalJ(finalstate,  &obj_iinit_re, &obj_iinit_im);
    obj_cost_re += obj_weights[iinit_global] * obj_iinit_re;
    obj_cost_im += obj_weights[iinit_global] * obj_iinit_im;

    /* Add to final-time fidelity */
    double fidelity_iinit_re = 0.0;
    double fidelity_iinit_im = 0.0;
    optim_target->HilbertSchmidtOverlap(finalstate, false, &fidelity_iinit_re, &fidelity_iinit_im);
    fidelity_re += 1./ ninit * fidelity_iinit_re;
    fidelity_im += 1./ ninit * fidelity_iinit_im;
  }

  /* Sum up from initial conditions processors */
  double mypen_leak = obj_penal_leakage;
  double mypen_wcost = obj_penal_weightedcost;
  double mypen_dpdm = obj_penal_dpdm;
  double mypenen = obj_penal_energy;
  double mycost_re = obj_cost_re;
  double mycost_im = obj_cost_im;
  double myfidelity_re = fidelity_re;
  double myfidelity_im = fidelity_im;
  MPI_Allreduce(&mypen_leak, &obj_penal_leakage, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mypen_wcost, &obj_penal_weightedcost, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mypen_dpdm, &obj_penal_dpdm, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mypenen, &obj_penal_energy, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mycost_re, &obj_cost_re, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&mycost_im, &obj_cost_im, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&myfidelity_re, &fidelity_re, 1, MPI_DOUBLE, MPI_SUM, comm_init);
  MPI_Allreduce(&myfidelity_im, &fidelity_im, 1, MPI_DOUBLE, MPI_SUM, comm_init);

  /* Set the fidelity: If Schroedinger, need to compute the absolute value: Fid= |\sum_i \phi^\dagger \phi_target|^2 */
  if (mastereq->decoherence_type == DecoherenceType::NONE) {
    fidelity = pow(fidelity_re, 2.0) + pow(fidelity_im, 2.0);
  } else {
    fidelity = fidelity_re; 
  }
 
  /* Finalize the objective function Jtrace to get the infidelity. 
     If Schroedingers solver, need to take the absolute value */
  obj_cost = optim_target->finalizeJ(obj_cost_re, obj_cost_im);

  /* Penalty: Riemannian distance */
  if (optim_penalty_riemannian > 0.0) {

    MPI_Barrier(MPI_COMM_WORLD);

    /* allreduce the U_final matrix */
    PetscScalar *data;
    MatDenseGetArray(U_final_re, &data);
    int size = mastereq->getDim();
    MPI_Allreduce(MPI_IN_PLACE, data, size * size, MPIU_SCALAR, MPI_SUM, comm_init);
    MatDenseRestoreArray(U_final_re, &data);

    MatDenseGetArray(U_final_im, &data);
    MPI_Allreduce(MPI_IN_PLACE, data, size * size, MPIU_SCALAR, MPI_SUM, comm_init);
    MatDenseRestoreArray(U_final_im, &data);

    double obj_riemannian = optim_target->RiemannianDistance(U_final_re, U_final_im, phase_invariant);

    // if (mpirank_world == 0) printf("\nRiemannian distance objective: %1.14e\n\n", obj_riemannian);
    // obj_cost = obj_riemannian;
    obj_riemann = optim_penalty_riemannian * obj_riemannian;
    obj_cost = 0.0; 
    obj_cost_re = 0.0;
    obj_cost_im = 0.0;
  }

  /* Evaluate Tikhonov regularization term += gamma/2 * ||x||^2*/
  double xnorm;
  if (!tikhonov_use_x0){  // ||x||^2
    VecNorm(x, NORM_2, &xnorm);
  } else {
    VecCopy(x, xtmp);
    VecAXPY(xtmp, -1.0, xinit);    // xtmp =  x_k - x_0
    VecNorm(xtmp, NORM_2, &xnorm);
  }
  obj_regul = gamma_tikhonov / 2. * pow(xnorm,2.0);

  /* Evaluate penalty term for control parameter variation */
  double var_reg = 0.0;
  for (size_t iosc = 0; iosc < mastereq->getNOscillators(); iosc++){
    var_reg += mastereq->getOscillator(iosc)->evalControlVariation(); // uses Oscillator::params instead of 'x'
  }
  obj_penal_variation = 0.5*gamma_penalty_variation*var_reg; 

  /* Sum, store and return objective value */
  objective = obj_cost + obj_regul + obj_penal_leakage + obj_penal_dpdm + obj_penal_energy + obj_penal_variation + obj_penal_weightedcost + obj_riemann;

  /* Derivative of new objective function */
  if (optim_penalty_riemannian > 0.0) {
    optim_target->RiemannianDistance_diff(U_final_re, U_final_im, U_final_re_bar, U_final_im_bar, phase_invariant);
    MatScale(U_final_re_bar, optim_penalty_riemannian);
    MatScale(U_final_im_bar, optim_penalty_riemannian);
  }


  /* Solve adjoint equations for all initial conditions . */
  for (int iinit = 0; iinit < ninit_local; iinit++) {
    int iinit_global = mpirank_init * ninit_local + iinit;

    /* Recompute the initial state and target */
    optim_target->prepareInitialAndTargetState(iinit_global, ninit, mastereq->nlevels, mastereq->nessential);
   
    /* Reset adjoint */
    VecZeroEntries(rho_t0_bar);

    /* Terminal condition for adjoint variable: Derivative of final time objective J */
    double obj_cost_re_bar, obj_cost_im_bar;
    optim_target->finalizeJ_diff(obj_cost_re, obj_cost_im, &obj_cost_re_bar, &obj_cost_im_bar);
    optim_target->evalJ_diff(timestepper->getFinalState(iinit), rho_t0_bar, obj_weights[iinit_global]*obj_cost_re_bar, obj_weights[iinit_global]*obj_cost_im_bar);

    // Derivative of Riemannian penalty
    if (optim_penalty_riemannian > 0.0) {

      // NOT SURE IF THIS IS RIGHT. 
      // VecZeroEntries(rho_t0_bar);

      // Pass i-th column of U_final_bar into rho_t0_bar
      for (size_t row = 0; row < mastereq->getDim(); row++) {
        int id_re = row;
        int id_im = row + mastereq->getDim();
        double val_ufinal_re_bar, val_ufinal_im_bar;
        MatGetValue(U_final_re_bar, row, iinit_global, &val_ufinal_re_bar);
        MatGetValue(U_final_im_bar, row, iinit_global, &val_ufinal_im_bar);
        VecSetValue(rho_t0_bar, id_re, val_ufinal_re_bar, ADD_VALUES);
        VecSetValue(rho_t0_bar, id_im, val_ufinal_im_bar, ADD_VALUES);
      }
      VecAssemblyBegin(rho_t0_bar);
      VecAssemblyEnd(rho_t0_bar);
    }

    /* Derivative of time-stepping */
    timestepper->solveAdjointODE(iinit, rho_t0_bar, obj_weights[iinit_global] * gamma_penalty_leakage, obj_weights[iinit_global]*gamma_penalty_weightedcost, obj_weights[iinit_global]*gamma_penalty_dpdm, obj_weights[iinit_global]*gamma_penalty_energy);

    /* Add to optimizers's gradient */
    VecAXPY(G, 1.0, timestepper->getReducedGradient());
  } // end of initial condition loop 

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
  //   std::cout<< "Objective = " << std::scientific<<std::setprecision(14) << obj_cost << " + " << obj_regul << " + " << obj_penal_leakage << " + " << obj_penal_dpdm << " + " << obj_penal_energy << " + " << obj_penal_variation << " + " << obj_penal_weightedcost <<  std::endl;
  //   std::cout<< "Fidelity = " << fidelity << std::endl;
  // }
}



void OptimProblem::evalLinearizedForward(const Vec x, const Vec v){
  // if (mpirank_world == 0 && !quietmode) std::cout<< "EVAL LINEARIZED FWD ... " << std::endl;

  /* Pass design vector x to oscillators */
  mastereq->setControlAmplitudes(x); 
 
  /* Solve ODE and linearized ODE forward in time */
  for (int iinit = 0; iinit < ninit_local; iinit++) {
    int iinit_global = mpirank_init * ninit_local + iinit;
    // printf("Solving ODE for initial condition %d (global index %d), total ninit_local = %d\n", iinit, iinit_global, ninit_local);

    int initid = optim_target->prepareInitialAndTargetState(iinit_global, ninit, mastereq->nlevels, mastereq->nessential);

    // Solve Forward ODE while storing trajectory states
    bool writeTrajectoryDataFiles = false;
    bool storeStates = true;
    timestepper->solveODE(initid, iinit, optim_target->getInitialState(), writeTrajectoryDataFiles, storeStates);


    // Solve linearized forward ODE in direction v while storing linearized states
    bool storeLinearizedStates = true;
    timestepper->solveLinearizedODE(iinit, v, storeLinearizedStates); 
  }
}

void OptimProblem::applyAGeope(Mat A, const Vec v, Vec Av){
  OptimProblem *self;
  MatShellGetContext(A, (void**)&self);
  if (self->mpirank_world == 0) printf("APPLYING A_GEOPE...\n");

  Vec x = self->x_for_AGeope;

  //  Reset output 
  VecZeroEntries(Av);
  
  // Apply linearized forward to get U(t) and dU/dalpha x v
  // fills the timesteppers trajectory_states and lin_trajectory_states.
  self->evalLinearizedForward(x, v);

  // For each final linearized state, solve the adjoint ODE

  for (int iinit = 0; iinit < self->ninit_local; iinit++) {

    // Set terminal condition for adjoint
    Vec lin_final_state = self->timestepper->getLinearizedFinalState(iinit);
    VecCopy(lin_final_state, self->rho_t0_bar);

    // Solve adjoint backward ODE
    self->timestepper->solveAdjointODE(iinit, self->rho_t0_bar, 0.0, 0.0, 0.0, 0.0);

    // Add gradient to output
    VecAXPY(Av, 1.0, self->timestepper->getReducedGradient());
  }

  /* Sum up the gradient from all initial condition processors */
  PetscScalar* Av_data; 
  VecGetArray(Av, &Av_data);
  MPI_Allreduce(MPI_IN_PLACE, Av_data, self->ndesign, MPIU_SCALAR, MPI_SUM, self->comm_init);
  VecRestoreArray(Av, &Av_data);
}


std::vector<double> OptimProblem::computeGeopeEvals(Vec xinit){

  // Store xinit so the MatShell can use it as point of evaluation.
  VecCopy(xinit, x_for_AGeope);

  // Set the number of evals requested
  // int neigvals = ndesign; 
  int neigvals = mastereq->getDim()*mastereq->getDim();  // N^2

  EPS eps;
  EPSCreate(PETSC_COMM_SELF, &eps);
  EPSSetOperators(eps, A_Geope, NULL);
  EPSSetProblemType(eps, EPS_HEP); // Hermitian 
  EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL); // largest eigenvalues
  // int ncv = 2*neigvals; // Dimension of the subspace (?): 2*nev is recommended by SLEPc documentation
  // EPSSetDimensions(eps, neigvals, ncv, PETSC_DEFAULT);
  EPSSetDimensions(eps, neigvals, PETSC_DEFAULT, PETSC_DEFAULT);
  EPSSetTolerances(eps, 1e-3, 10);
  EPSSetFromOptions(eps);

  EPSSolve(eps);
  PetscInt numConv;
  PetscInt iters_taken;
  EPSGetConverged(eps, &numConv);
  EPSGetIterationNumber(eps,&iters_taken);
  if (numConv < neigvals) {
      if (mpirank_world==0) printf("WARNING: Only %d eigenvalues out of %d eigenvalues converged.\n", numConv, neigvals);
  }

  // Set up storage for eigenvalues and eigenvectors (should be real!)
  std::vector<double> evals_re(neigvals);
  std::vector<Vec> evec_re(neigvals);
  for (int ix = 0; ix < neigvals; ix++) {
    MatCreateVecs(A_Geope, &evec_re[ix], NULL);
  }

  // Retrieve eigenpairs of M and compute error. 
  for (PetscInt i = 0; i < numConv && i < neigvals; i++) {

    // Retrieve the eigenvalue (is real) and eigenvector
    EPSGetEigenpair(eps, i, &evals_re[i], NULL, evec_re[i], NULL);
    // EPSGetEigenvalue(eps, i, &evals_re[i], NULL);

    // Estimate the errror (needs one more application of A)
    // double error = 0.0;
    // EPSComputeError(eps,i,EPS_ERROR_RELATIVE,&error);
    // if (error > 1e-12) {
    //     if (mpirank_world==0) printf("WARNING: Relative error of eigenpair %d is large (error=%1.4e)\n", i, error);
    // }
  }

  // Resize to the number of converged eigenvalues. 
  evals_re.resize(std::min(numConv, neigvals));

  // Cleanup
  for (int ix = 0; ix < neigvals; ix++) {
    VecDestroy(&evec_re[ix]);
  }
  EPSDestroy(&eps);

  return evals_re;
}


void OptimProblem::solve(Vec xinit) {
  TaoSetSolution(tao, xinit);
  TaoSolve(tao);
}

void OptimProblem::getStartingPoint(Vec xinit){

  // Grab parameters from oscillators
  PetscScalar* xptr;
  VecGetArray(xinit, &xptr);
  int shift = 0;
  for (size_t ioscil = 0; ioscil<mastereq->getNOscillators(); ioscil++){
    mastereq->getOscillator(ioscil)->getControlParams(xptr + shift);
    shift += mastereq->getOscillator(ioscil)->getNParams();
  }
  VecRestoreArray(xinit, &xptr);
  
  /* Assemble initial guess */
  VecAssemblyBegin(xinit);
  VecAssemblyEnd(xinit);

  /* Pass to oscillator */
  mastereq->setControlAmplitudes(xinit);
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
  double obj_riemann = ctx->getRiemannDistance();
  double obj_regul = ctx->getRegul();
  double obj_penal_leakage = ctx->getPenaltyLeakage();
  double obj_penal_weightedcost = ctx->getPenaltyWeightedCost();
  double obj_penal_dpdm = ctx->getPenaltyDpDm();
  double obj_penal_energy = ctx->getPenaltyEnergy();
  double obj_penal_variation= ctx->getPenaltyVariation();
  double F_avg = ctx->getFidelity();

  // Switch objective functions
  // if (1.0 - F_avg < 0.73) {
  // if (iter > 50) {
  //   printf("Switching to infidelity measure.\n");
  //   ctx->setRiemannianDistance(false);
  // }

  // // Freeze theta_avg if fidelity is sufficiently high
  // if (F_avg > 0.80) {
  //   ctx->getOptimTarget()->freeze_theta_avg = true;
  // }

  /* Additional Stopping criteria */
  bool lastIter = false;
  std::string finalReason_str = "";
  if (1.0 - F_avg <= ctx->getTolInfidelity()) {
    finalReason_str = "Optimization converged with small infidelity.";
    TaoSetConvergedReason(tao, TAO_CONVERGED_USER);
    lastIter = true;
  // } else if (obj_cost <= ctx->getTolFinalCost()) {
  //   finalReason_str = "Optimization converged with small final time cost.";
  //   TaoSetConvergedReason(tao, TAO_CONVERGED_USER);
  //   lastIter = true;
  } else if (iter == ctx->getMaxIter()) {
    finalReason_str = "Optimization stopped at maximum number of iterations.";
    lastIter = true;
  } else if (gnorm < ctx->getTolGradAbs()) {
    finalReason_str = "OPtimization converged with small gradient norm.";
    lastIter=true;
  }

  /* First iteration: Header for screen output of optimization history */
  if (iter == 0 && ctx->getMPIrank_world() == 0) {
    std::cout<<  "    Objective             Tikhonov               Penalty-Leakage        Penalty-StateVar       Penalty-TotalEnergy    Penalty-CtrlVar        Penalty-WeightedCost" << std::endl;
  }

  /* Every <output_optimization_stride> iterations: Output of optimization history */
  if (iter % ctx->getOutputOptimizationStride() == 0 ||lastIter) {
    // Add to optimization history file 
    ctx->getOutput()->writeOptimFile(iter, f, gnorm, deltax, F_avg, obj_cost, obj_riemann, obj_regul, obj_penal_leakage, obj_penal_dpdm, obj_penal_energy, obj_penal_variation, obj_penal_weightedcost);
    // Screen output 
    if (ctx->getMPIrank_world() == 0) {
      std::cout<< iter <<  "  " << std::scientific<<std::setprecision(14) << obj_cost << " + " << obj_regul << " + " << obj_penal_leakage << " + " << obj_penal_dpdm << " + " << obj_penal_energy << " + " << obj_penal_variation << " + " << obj_penal_weightedcost << " + " << obj_riemann;
      std::cout<< "  Fidelity = " << F_avg;
      std::cout<< "  ||Grad|| = " << gnorm;
      std::cout<< std::endl;
    }
  }

  /* Print last iteration stopping reason */
  if (lastIter && ctx->getMPIrank_world() == 0) {
    std::cout<< finalReason_str << std::endl;
  }


  return 0;
}


PetscErrorCode TaoEvalObjectiveAndGradient(Tao tao, Vec x, PetscReal *f, Vec G, void*ptr){

  TaoEvalGradient(tao, x, G, ptr);
  OptimProblem* ctx = (OptimProblem*) ptr;
  *f = ctx->getObjective();
  // *f = 1.0 - ctx->getFidelity();

  return 0;
}

PetscErrorCode TaoEvalObjective(Tao /*tao*/, Vec x, PetscReal *f, void*ptr){

  OptimProblem* ctx = (OptimProblem*) ptr;
  *f = ctx->evalF(x, false);
  
  return 0;
}


PetscErrorCode TaoEvalGradient(Tao /*tao*/, Vec x, Vec G, void*ptr){

  OptimProblem* ctx = (OptimProblem*) ptr;
  ctx->evalGradF(x, G, false);
  
  return 0;
}
