#include "timestepper.hpp"
#include "petscvec.h"

TimeStepper::TimeStepper() {
  dim = 0;
  mastereq = NULL;
  ntime = 0;
  total_time = 0.0;
  dt = 0.0;
  storeFWD = false;
}

TimeStepper::TimeStepper(MasterEq* mastereq_, int ntime_, double total_time_, Output* output_, bool storeFWD_) : TimeStepper() {
  mastereq = mastereq_;
  dim = 2*mastereq->getDim();
  ntime = ntime_;
  total_time = total_time_;
  output = output_;
  storeFWD = storeFWD_;

  /* Set the time-step size */
  dt = total_time / ntime;

  /* Allocate storage of primal state */
  if (storeFWD) { 
    for (int n = 0; n <=ntime; n++) {
      Vec state;
      VecCreate(PETSC_COMM_WORLD, &state);
      VecSetSizes(state, PETSC_DECIDE, dim);
      VecSetFromOptions(state);
      store_states.push_back(state);
    }
  }

  /* Allocate auxiliary state vector */
  VecCreate(PETSC_COMM_WORLD, &x);
  VecSetSizes(x, PETSC_DECIDE, dim);
  VecSetFromOptions(x);
  VecZeroEntries(x);

  /* Allocate the reduced gradient */
  int ndesign = 0;
  for (int ioscil = 0; ioscil < mastereq->getNOscillators(); ioscil++) {
      ndesign += mastereq->getOscillator(ioscil)->getNParams(); 
  }
  VecCreateSeq(PETSC_COMM_SELF, ndesign, &redgrad);
  VecSetFromOptions(redgrad);
  VecAssemblyBegin(redgrad);
  VecAssemblyEnd(redgrad);

}


TimeStepper::~TimeStepper() {
  for (int n = 0; n < store_states.size(); n++) {
    VecDestroy(&(store_states[n]));
  }
  VecDestroy(&x);
  VecDestroy(&redgrad);
}



Vec TimeStepper::getState(int tindex){
  
  if (tindex >= store_states.size()) {
    printf("ERROR: Time-stepper requested state at time index %d, but didn't store it.\n", tindex);
    exit(1);
  }

  return store_states[tindex];
}

Vec TimeStepper::solveODE(int initid, Vec rho_t0){

  /* Open output files */
  output->openDataFiles("rho", initid);

  /* Set initial condition  */
  VecCopy(rho_t0, x);

  /* --- Loop over time interval --- */
  penalty_integral = 0.0;
  for (int n = 0; n < ntime; n++){

    /* current time */
    double tstart = n * dt;
    double tstop  = (n+1) * dt;

    /* store and write current state. */
    if (storeFWD) VecCopy(x, store_states[n]);
    output->writeDataFiles(n, tstart, x, mastereq);

    /* Take one time step */
    evolveFWD(tstart, tstop, x);

    /* Add to penalty objective term */
    if (gamma_penalty > 1e-13) penalty_integral += penaltyIntegral(tstop, x);

#ifdef SANITY_CHECK
    SanityTests(x, tstart);
#endif
  }

  /* Store last time step */
  if (storeFWD) VecCopy(x, store_states[ntime]);

  /* Write last time step and close files */
  output->writeDataFiles(ntime, ntime*dt, x, mastereq);
  output->closeDataFiles();
  

  return x;
}


void TimeStepper::solveAdjointODE(int initid, Vec rho_t0_bar, double Jbar) {

  /* Reset gradient */
  VecZeroEntries(redgrad);

  /* Set terminal condition */
  VecCopy(rho_t0_bar, x);

  /* Loop over time interval */
  for (int n = ntime; n > 0; n--){
    double tstop  = n * dt;
    double tstart = (n-1) * dt;

    /* Derivative of penalty objective term */
    if (gamma_penalty > 1e-13) penaltyIntegral_diff(tstop, getState(n), x, Jbar);

    /* Take one time step backwards */
    evolveBWD(tstop, tstart, getState(n-1), x, redgrad, true);

  }
}


double TimeStepper::penaltyIntegral(double time, const Vec x){
  double penalty = 0.0;
  int dim_rho = (int)sqrt(dim/2);  // dim = 2*N^2 vectorized system. dim_rho = N = dimension of matrix system
  double x_re, x_im;

  /* weighted integral of the objective function */
  if (penalty_param > 1e-13) {
    double weight = 1./penalty_param * exp(- pow((time - total_time)/penalty_param, 2));
    double obj = optim_target->evalJ(x);
    penalty = weight * obj * dt;
  }

  /* If gate optimization: Add guard-level occupation to prevent leakage. A guard level is the LAST NON-ESSENTIAL energy level of an oscillator */
  if (optim_target->getType() == TargetType::GATE) { 
      PetscInt ilow, iupp;
      VecGetOwnershipRange(x, &ilow, &iupp);
      /* Sum over all diagonal elements that correspond to a non-essential guard level. */
      for (int i=0; i<dim_rho; i++) {
        if ( isGuardLevel(i, mastereq->nlevels, mastereq->nessential) ) {
          // printf("isGuard: %d / %d\n", i, dim_rho);
          PetscInt vecID_re = getIndexReal(getVecID(i,i,dim_rho));
          PetscInt vecID_im = getIndexImag(getVecID(i,i,dim_rho));
          x_re = 0.0; x_im = 0.0;
          if (ilow <= vecID_re && vecID_re < iupp) VecGetValues(x, 1, &vecID_re, &x_re);
          if (ilow <= vecID_im && vecID_im < iupp) VecGetValues(x, 1, &vecID_im, &x_im);  // those should be zero!? 
          penalty += dt * (x_re * x_re + x_im * x_im);
        }
      }
      double mine = penalty;
      MPI_Allreduce(&mine, &penalty, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
  }

  return penalty;
}

void TimeStepper::penaltyIntegral_diff(double time, const Vec x, Vec xbar, double penaltybar){
  int dim_rho = (int)sqrt(dim/2);  // dim = 2*N^2 vectorized system. dim_rho = N = dimension of matrix system

  /* Derivative of weighted integral of the objective function */
  if (penalty_param > 1e-13){
    double weight = 1./penalty_param * exp(- pow((time - total_time)/penalty_param, 2));
    optim_target->evalJ_diff(x, xbar, weight*penaltybar*dt);
  }

  /* If gate optimization: Derivative of adding guard-level occupation */
  if (optim_target->getType() == TargetType::GATE) { 
    PetscInt ilow, iupp;
    VecGetOwnershipRange(x, &ilow, &iupp);
    double x_re, x_im;
    for (int i=0; i<dim_rho; i++) {
      if ( isGuardLevel(i, mastereq->nlevels, mastereq->nessential) ) {
        PetscInt vecID_re = getIndexReal(getVecID(i,i,dim_rho));
        PetscInt vecID_im = getIndexImag(getVecID(i,i,dim_rho));
        x_re = 0.0; x_im = 0.0;
        if (ilow <= vecID_re && vecID_re < iupp) VecGetValues(x, 1, &vecID_re, &x_re);
        if (ilow <= vecID_im && vecID_im < iupp) VecGetValues(x, 1, &vecID_im, &x_im);
        // Derivative: 2 * rho(i,i) * weights * penalbar * dt
        if (ilow <= vecID_re && vecID_re < iupp) VecSetValue(xbar, vecID_re, 2.*x_re*dt*penaltybar, ADD_VALUES);
        if (ilow <= vecID_im && vecID_im < iupp) VecSetValue(xbar, vecID_im, 2.*x_im*dt*penaltybar, ADD_VALUES);
      }
      VecAssemblyBegin(xbar);
      VecAssemblyEnd(xbar);
    }
  }
}

void TimeStepper::evolveBWD(const double tstart, const double tstop, const Vec x_stop, Vec x_adj, Vec grad, bool compute_gradient){}

ExplEuler::ExplEuler(MasterEq* mastereq_, int ntime_, double total_time_, Output* output_, bool storeFWD_) : TimeStepper(mastereq_, ntime_, total_time_, output_, storeFWD_) {
  MatCreateVecs(mastereq->getRHS(), &stage, NULL);
  VecZeroEntries(stage);
}

ExplEuler::~ExplEuler() {
  VecDestroy(&stage);
}

void ExplEuler::evolveFWD(const double tstart,const  double tstop, Vec x) {

  double dt = fabs(tstop - tstart);

   /* Compute A(tstart) */
  mastereq->assemble_RHS(tstart);
  Mat A = mastereq->getRHS(); 

  /* update x = x + hAx */
  MatMult(A, x, stage);
  VecAXPY(x, dt, stage);

}

void ExplEuler::evolveBWD(const double tstop,const  double tstart,const  Vec x, Vec x_adj, Vec grad, bool compute_gradient){
  double dt = fabs(tstop - tstart);

  /* Add to reduced gradient */
  if (compute_gradient) {
    mastereq->computedRHSdp(tstop, x, x_adj, dt, grad);
  }

  /* update x_adj = x_adj + hA^Tx_adj */
  mastereq->assemble_RHS(tstop);
  Mat A = mastereq->getRHS(); 
  MatMultTranspose(A, x_adj, stage);
  VecAXPY(x_adj, dt, stage);

}

ImplMidpoint::ImplMidpoint(MasterEq* mastereq_, int ntime_, double total_time_, LinearSolverType linsolve_type_, int linsolve_maxiter_, Output* output_, bool storeFWD_) : TimeStepper(mastereq_, ntime_, total_time_, output_, storeFWD_) {

  /* Create and reset the intermediate vectors */
  MatCreateVecs(mastereq->getRHS(), &stage, NULL);
  VecDuplicate(stage, &stage_adj);
  VecDuplicate(stage, &rhs);
  VecDuplicate(stage, &rhs_adj);
  VecZeroEntries(stage);
  VecZeroEntries(stage_adj);
  VecZeroEntries(rhs);
  VecZeroEntries(rhs_adj);
  linsolve_type = linsolve_type_;
  linsolve_maxiter = linsolve_maxiter_;
  linsolve_reltol = 1.e-20;
  linsolve_abstol = 1.e-10;
  linsolve_iterstaken_avg = 0;
  linsolve_counter = 0;
  linsolve_error_avg = 0.0;

  if (linsolve_type == LinearSolverType::GMRES) {
    /* Create Petsc's linear solver */
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPGetPC(ksp, &preconditioner);
    PCSetType(preconditioner, PCNONE);
    KSPSetTolerances(ksp, linsolve_reltol, linsolve_abstol, PETSC_DEFAULT, linsolve_maxiter);
    KSPSetType(ksp, KSPGMRES);
    KSPSetOperators(ksp, mastereq->getRHS(), mastereq->getRHS());
    KSPSetFromOptions(ksp);
  }
  else {
    /* For Neumann iterations, allocate a temporary vector */
    MatCreateVecs(mastereq->getRHS(), &tmp, NULL);
    MatCreateVecs(mastereq->getRHS(), &err, NULL);
  }
}


ImplMidpoint::~ImplMidpoint(){

  /* Print linear solver statistics */
  linsolve_iterstaken_avg = (int) linsolve_iterstaken_avg / linsolve_counter;
  linsolve_error_avg = linsolve_error_avg / linsolve_counter;
  int myrank;
  // MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  // if (myrank == 0) printf("Linear solver type %d: Average iterations = %d, average error = %1.2e\n", linsolve_type, linsolve_iterstaken_avg, linsolve_error_avg);

  /* Free up Petsc's linear solver */
  if (linsolve_type == LinearSolverType::GMRES) {
    KSPDestroy(&ksp);
  } else {
    VecDestroy(&tmp);
    VecDestroy(&err);
  }

  /* Free up intermediate vectors */
  VecDestroy(&stage_adj);
  VecDestroy(&stage);
  VecDestroy(&rhs_adj);
  VecDestroy(&rhs);

}

void ImplMidpoint::evolveFWD(const double tstart,const  double tstop, Vec x) {

  /* Compute time step size */
  double dt = fabs(tstop - tstart); // absolute values needed in case this runs backwards! 

  /* Compute A(t_n+h/2) */
  mastereq->assemble_RHS( (tstart + tstop) / 2.0);
  Mat A = mastereq->getRHS(); 

  /* Compute rhs = A x */
  MatMult(A, x, rhs);

  /* Solve for the stage variable (I-dt/2 A) k1 = Ax */
  switch (linsolve_type) {
    case LinearSolverType::GMRES:
      /* Set up I-dt/2 A, then solve */
      MatScale(A, - dt/2.0);
      MatShift(A, 1.0);  
      KSPSolve(ksp, rhs, stage);

      /* Monitor error */
      double rnorm;
      PetscInt iters_taken;
      KSPGetResidualNorm(ksp, &rnorm);
      KSPGetIterationNumber(ksp, &iters_taken);
      //printf("Residual norm %d: %1.5e\n", iters_taken, rnorm);
      linsolve_iterstaken_avg += iters_taken;
      linsolve_error_avg += rnorm;

      /* Revert the scaling and shifting if gmres solver */
      MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
      break;

    case LinearSolverType::NEUMANN:
      linsolve_iterstaken_avg += NeumannSolve(A, rhs, stage, dt/2.0, false);
      break;
  }
  linsolve_counter++;

  /* --- Update state x += dt * stage --- */
  VecAXPY(x, dt, stage);
}

void ImplMidpoint::evolveBWD(const double tstop, const double tstart, const Vec x, Vec x_adj, Vec grad, bool compute_gradient){
  Mat A;

  /* Compute time step size */
  double dt = fabs(tstop - tstart); // absolute values needed in case this runs backwards! 
  double thalf = (tstart + tstop) / 2.0;

  /* Assemble RHS(t_1/2) */
  mastereq->assemble_RHS( (tstart + tstop) / 2.0);
  A = mastereq->getRHS();

  /* Get Ax_n for use in gradient */
  if (compute_gradient) {
    MatMult(A, x, rhs);
  }

  /* Solve for adjoint stage variable */
  switch (linsolve_type) {
    case LinearSolverType::GMRES:
      MatScale(A, - dt/2.0);
      MatShift(A, 1.0);  // WARNING: this can be very slow if some diagonal elements are missing.
      KSPSolveTranspose(ksp, x_adj, stage_adj);
      double rnorm;
      KSPGetResidualNorm(ksp, &rnorm);
      if (rnorm > 1e-3)  printf("Residual norm: %1.5e\n", rnorm);
      break;

    case LinearSolverType::NEUMANN: 
      NeumannSolve(A, x_adj, stage_adj, dt/2.0, true);
      break;
  }

  // k_bar = h*k_bar 
  VecScale(stage_adj, dt);

  /* Add to reduced gradient */
  if (compute_gradient) {
    switch (linsolve_type) {
      case LinearSolverType::GMRES: 
        KSPSolve(ksp, rhs, stage);
        break;
      case LinearSolverType::NEUMANN:
        NeumannSolve(A, rhs, stage, dt/2.0, false);
        break;
    }
    VecAYPX(stage, dt / 2.0, x);
    mastereq->computedRHSdp(thalf, stage, stage_adj, 1.0, grad);
  }

  /* Revert changes to RHS from above, if gmres solver */
  A = mastereq->getRHS();
  if (linsolve_type == LinearSolverType::GMRES) {
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  }

  /* Update adjoint state x_adj += dt * A^Tstage_adj --- */
  MatMultTransposeAdd(A, stage_adj, x_adj, x_adj);

}


int ImplMidpoint::NeumannSolve(Mat A, Vec b, Vec y, double alpha, bool transpose){

  double errnorm, errnorm0;

  // Initialize y = b
  VecCopy(b, y);

  int iter;
  for (iter = 0; iter < linsolve_maxiter; iter++) {
    VecCopy(y, err);

    // y = b + alpha * A *  y
    if (!transpose) MatMult(A, y, tmp);
    else            MatMultTranspose(A, y, tmp);
    VecAXPBYPCZ(y, 1.0, alpha, 0.0, b, tmp);

    /* Error approximation  */
    VecAXPY(err, -1.0, y); // err = yprev - y 
    VecNorm(err, NORM_2, &errnorm);

    /* Stopping criteria */
    if (iter == 0) errnorm0 = errnorm;
    if (errnorm < linsolve_abstol) break;
    if (errnorm / errnorm0 < linsolve_reltol) break;
  }

  // printf("Neumann error: %1.14e\n", errnorm);
  linsolve_error_avg += errnorm;

  return iter;
}

PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec u,Mat M,Mat P,void *ctx){

  /* Cast ctx to equation pointer */
  MasterEq *mastereq = (MasterEq*) ctx;

  /* Assembling the Hamiltonian will set the matrix RHS from Re, Im */
  mastereq->assemble_RHS(t);

  /* Set the hamiltonian system matrix */
  M = mastereq->getRHS();

  return 0;
}

PetscErrorCode RHSJacobianP(TS ts, PetscReal t, Vec y, Mat A, void *ctx){

  /* Cast ctx to Hamiltonian pointer */
  MasterEq *mastereq= (MasterEq*) ctx;

  /* Assembling the derivative of RHS with respect to the control parameters */
  printf("THIS IS NOT IMPLEMENTED \n");
  exit(1);
  // mastereq->assemble_dRHSdp(t, y);

  /* Set the derivative */
  // A = mastereq->getdRHSdp();

  return 0;
}


PetscErrorCode TSInit(TS ts, MasterEq* mastereq, PetscInt NSteps, PetscReal Dt, PetscReal Tfinal, Vec x, bool monitor){
  int ierr;

  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts, TSTHETA); CHKERRQ(ierr);
  ierr = TSThetaSetTheta(ts, 0.5); CHKERRQ(ierr);   // midpoint rule
  ierr = TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,mastereq);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,mastereq->getRHS(),mastereq->getRHS(),RHSJacobian,mastereq);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,Dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,NSteps);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,Tfinal);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  if (monitor) {
    ierr = TSMonitorSet(ts, Monitor, NULL, NULL); CHKERRQ(ierr);
  }
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetSolution(ts, x); CHKERRQ(ierr);


  return ierr;
}



PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec x,void *ctx) {
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

    // Vec *stage;
    // int nr;
    // TSGetStages(ts, &nr, &stage);

  const PetscScalar *x_ptr;
  ierr = VecGetArrayRead(x, &x_ptr);
  printf("Monitor ->%d,%f, x[1]=%1.14e\n", step, t, x_ptr[1]);
  ierr = VecRestoreArrayRead(x, &x_ptr);

  PetscFunctionReturn(0);
}

PetscErrorCode AdjointMonitor(TS ts,PetscInt step,PetscReal t,Vec x, PetscInt numcost, Vec* lambda, Vec* mu, void *ctx) {
  PetscErrorCode ierr;

    // Vec *stage;
    // int nr;
    // TSGetStages(ts, &nr, &stage);

  const PetscScalar *x_ptr;
  const PetscScalar *lambda_ptr;
  const PetscScalar *mu_ptr;
  ierr = VecGetArrayRead(x, &x_ptr);
  ierr = VecGetArrayRead(lambda[0], &lambda_ptr);
  ierr = VecGetArrayRead(mu[0], &mu_ptr);
  // ierr = VecGetArrayRead(stage[0], &x_ptr);
  printf("AdjointMonitor %d: %f->, dt=%f  x[1]=%1.14e, lambda[0]=%1.14e, mu[0]=%1.14e\n", step, t, ts->time_step, x_ptr[1], lambda_ptr[0], mu_ptr[0] );
  ierr = VecRestoreArrayRead(x, &x_ptr);
  ierr = VecRestoreArrayRead(lambda[0], &lambda_ptr);
  ierr = VecRestoreArrayRead(mu[0], &mu_ptr);

  PetscFunctionReturn(0);
}



PetscErrorCode TSPreSolve(TS ts, bool tj_store){
  int ierr; 

  ierr = TSSetUp(ts); CHKERRQ(ierr);
  if (tj_store) ierr = TSTrajectorySetUp(ts->trajectory,ts);CHKERRQ(ierr);

  /* reset time step and iteration counters */
  if (!ts->steps) {
    ts->ksp_its           = 0;
    ts->snes_its          = 0;
    ts->num_snes_failures = 0;
    ts->reject            = 0;
    ts->steprestart       = PETSC_TRUE;
    ts->steprollback      = PETSC_FALSE;
  }
  ts->reason = TS_CONVERGED_ITERATING;
  if (ts->steps >= ts->max_steps) ts->reason = TS_CONVERGED_ITS;
  else if (ts->ptime >= ts->max_time) ts->reason = TS_CONVERGED_TIME;

  if (!ts->steps && tj_store) {
    ierr = TSTrajectorySet(ts->trajectory,ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
  }
  // ierr = TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);

  return ierr;
}


PetscErrorCode TSStepMod(TS ts, bool tj_store){
  int ierr; 

  ierr = TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
  ierr = TSPreStep(ts);CHKERRQ(ierr);
  ierr = TSStep(ts);CHKERRQ(ierr);
  ierr = TSPostEvaluate(ts);CHKERRQ(ierr);
  ierr = TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);

  if (tj_store) ierr = TSTrajectorySet(ts->trajectory,ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
  ierr = TSPostStep(ts);CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode TSPostSolve(TS ts){
  int ierr;
  ts->solvetime = ts->ptime;
  return ierr;
}


PetscErrorCode TSAdjointPreSolve(TS ts){
  int ierr;

  /* reset time step and iteration counters */
  ts->adjoint_steps     = 0;
  ts->ksp_its           = 0;
  ts->snes_its          = 0;
  ts->num_snes_failures = 0;
  ts->reject            = 0;
  ts->reason            = TS_CONVERGED_ITERATING;

  if (!ts->adjoint_max_steps) ts->adjoint_max_steps = ts->steps;
  if (ts->adjoint_steps >= ts->adjoint_max_steps) ts->reason = TS_CONVERGED_ITS;


  return ierr;
}


PetscErrorCode TSAdjointStepMod(TS ts, bool tj_save) {
  int ierr;


  if (tj_save) ierr = TSTrajectoryGet(ts->trajectory,ts,ts->steps,&ts->ptime);CHKERRQ(ierr);

  ierr = TSAdjointMonitor(ts,ts->steps,ts->ptime,ts->vec_sol,ts->numcost,ts->vecs_sensi,ts->vecs_sensip);CHKERRQ(ierr);

  ierr = TSAdjointStep(ts);CHKERRQ(ierr);
  if (ts->vec_costintegral && !ts->costintegralfwd) {
    ierr = TSAdjointCostIntegral(ts);CHKERRQ(ierr);
  }
  ierr = TSAdjointMonitor(ts,ts->steps,ts->ptime,ts->vec_sol,ts->numcost,ts->vecs_sensi,ts->vecs_sensip);CHKERRQ(ierr);

  return ierr;
}

PetscErrorCode TSAdjointPostSolve(TS ts, bool tj_save){
  int ierr; 
  if (tj_save) ierr = TSTrajectoryGet(ts->trajectory,ts,ts->steps,&ts->ptime);CHKERRQ(ierr);
  ierr = TSAdjointMonitor(ts,ts->steps,ts->ptime,ts->vec_sol,ts->numcost,ts->vecs_sensi,ts->vecs_sensip);CHKERRQ(ierr);
  ts->solvetime = ts->ptime;
  if (tj_save) ierr = TSTrajectoryViewFromOptions(ts->trajectory,NULL,"-ts_trajectory_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(ts->vecs_sensi[0],(PetscObject) ts, "-ts_adjoint_view_solution");CHKERRQ(ierr);
  ts->adjoint_max_steps = 0;

  return ierr;
}


PetscErrorCode  TSSetAdjointSolution(TS ts,Vec lambda, Vec mu)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ts,TS_CLASSID,1);
  PetscValidHeaderSpecific(lambda,VEC_CLASSID,3);

  ierr = PetscObjectReference((PetscObject)lambda);CHKERRQ(ierr);
  if (ts->vecs_sensi[0]) ierr = VecDestroy(&ts->vecs_sensi[0]);CHKERRQ(ierr);
  ts->vecs_sensi[0] = lambda;
  // if (ts->vecs_sensip[0]) ierr = VecDestroy(&ts->vecs_sensip[0]);CHKERRQ(ierr);
  ts->vecs_sensip[0] = mu;

  PetscFunctionReturn(0);
}

