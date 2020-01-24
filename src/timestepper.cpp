#include "timestepper.hpp"

TimeStepper::TimeStepper() {
  dim = 0;
  hamiltonian = NULL;
}

TimeStepper::TimeStepper(Hamiltonian* hamiltonian_) {
  hamiltonian = hamiltonian_;
  dim = 2*hamiltonian->getDim();
}

TimeStepper::~TimeStepper() {}


ExplEuler::ExplEuler(Hamiltonian* hamiltonian_) : TimeStepper(hamiltonian_) {
  VecCreateSeq(PETSC_COMM_WORLD, dim, &stage);
  VecZeroEntries(stage);
}

ExplEuler::~ExplEuler() {
  VecDestroy(&stage);
}

void ExplEuler::evolve(Mode direction, double tstart, double tstop, Vec x) {

  double dt = fabs(tstop - tstart);

   /* Compute A(tstart) */
  hamiltonian->assemble_RHS(tstart);

  /* Decide for forward mode (A) or backward mode (A^T)*/
  Mat A = hamiltonian->getRHS(); 
  switch(direction)
  {
    case FWD :  // forward stepping. Use A. Do nothing.
      break;
    case BWD :  // backward stepping. Use A^T.
      MatTranspose(A, MAT_INPLACE_MATRIX, &A);
      break;
    default  : 
      printf("ERROR: Wrong timestepping mode!\n"); exit(1);
      break;
  } 

  /* update x = x + hAx */
  MatMult(A, x, stage);
  VecAXPY(x, dt, stage);
}

void ExplEuler::evolveBWD(double tstart, double tstop, Vec x, Vec x_adj, Vec grad){
  double dt = fabs(tstop - tstart);

  hamiltonian->assemble_dRHSdp(tstart, x);
  MatScale(hamiltonian->getdRHSdp(), dt);
  MatMultTransposeAdd(hamiltonian->getdRHSdp(), x_adj, grad, grad); 


   /* Compute A(tstart) */
  hamiltonian->assemble_RHS(tstart);

  /* Decide for forward mode (A) or backward mode (A^T)*/
  Mat A = hamiltonian->getRHS(); 
  MatTranspose(A, MAT_INPLACE_MATRIX, &A);

  /* update x_adj = x_adj + hAx_adj */
  MatMult(A, x_adj, stage);
  VecAXPY(x_adj, dt, stage);

}

ImplMidpoint::ImplMidpoint(Hamiltonian* hamiltonian_) : TimeStepper(hamiltonian_) {

  /* Create and reset the intermediate vectors */
  VecCreateSeq(PETSC_COMM_WORLD, dim, &stage);
  VecCreateSeq(PETSC_COMM_WORLD, dim, &rhs);
  VecZeroEntries(stage);
  VecZeroEntries(rhs);

  /* Create linear solver */
  KSPCreate(PETSC_COMM_WORLD, &linearsolver);

  /* Set options */
  // KSPGetPC(linearsolver, &preconditioner);
  // PCSetType(preconditioner, PCJACOBI);
  double reltol = 1.e-5;
  double abstol = 1.e-10;
  KSPSetTolerances(linearsolver, reltol, abstol, PETSC_DEFAULT, PETSC_DEFAULT);
  KSPSetType(linearsolver, KSPGMRES);
  /* Set runtime options */
  KSPSetFromOptions(linearsolver);

}


ImplMidpoint::~ImplMidpoint(){
  /* Free up intermediate vectors */
  VecDestroy(&stage);
  VecDestroy(&rhs);

  /* Free up linear solver */
  KSPDestroy(&linearsolver);
}

void ImplMidpoint::evolve(Mode direction, double tstart, double tstop, Vec x) {

  /* Compute time step size */
  double dt = fabs(tstop - tstart); // absolute values needed in case this runs backwards! 

  /* Compute A(t_n+h/2) */
  hamiltonian->assemble_RHS( (tstart + tstop) / 2.0);

  /* Decide for forward mode (A) or backward mode (A^T)*/
  Mat A = hamiltonian->getRHS(); 
  switch(direction)
  {
    case FWD :  // forward stepping. System matrix uses A(t_n+h/2). Do nothing.
      break;
    case BWD :  // backward stepping. System matrix uses A(t_n+h/2)^T
      MatTranspose(A, MAT_INPLACE_MATRIX, &A);
      break;
    default  : 
      printf("ERROR: Wrong timestepping mode!\n"); exit(1);
      break;
  }

  /* --- Solve for stage variable */

  /* Compute rhs = A x */
  MatMult(A, x, rhs);

  /* Build system matrix I-h/2 A. This modifies the hamiltonians RHS matrix! Make sure to call assemble_RHS before the next use! */
  MatScale(A, - dt/2.0);
  MatShift(A, 1.0);  // WARNING: this can be very slow if some diagonal elements are missing.
  KSPSetOperators(linearsolver, A, A);// TODO: Do we have to do this in each time step?? 
  
  /* solve nonlinear equation */
  KSPSolve(linearsolver, rhs, stage);

  /* Monitor error */
  double rnorm;
  int iters_taken;
  KSPGetResidualNorm(linearsolver, &rnorm);
  KSPGetIterationNumber(linearsolver, &iters_taken);
  // printf("Residual norm %d: %1.5e\n", iters_taken, rnorm);

  /* --- Update state x += dt * stage --- */
  VecAXPY(x, dt, stage);

}

void ImplMidpoint::evolveBWD(double tstart, double tstop, Vec x, Vec x_adj, Vec grad){}


PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec u,Mat M,Mat P,void *ctx){

  /* Cast ctx to Hamiltonian pointer */
  Hamiltonian *hamiltonian = (Hamiltonian*) ctx;

  /* Assembling the Hamiltonian will set the matrix RHS from Re, Im */
  hamiltonian->assemble_RHS(t);

  /* Set the hamiltonian system matrix */
  M = hamiltonian->getRHS();

  return 0;
}

PetscErrorCode RHSJacobianP(TS ts, PetscReal t, Vec y, Mat A, void *ctx){

  /* Cast ctx to Hamiltonian pointer */
  Hamiltonian *hamiltonian = (Hamiltonian*) ctx;

  /* Assembling the derivative of RHS with respect to the control parameters */
  hamiltonian->assemble_dRHSdp(t, y);

  /* Set the derivative */
  A = hamiltonian->getdRHSdp();

  return 0;
}


PetscErrorCode TSInit(TS ts, Hamiltonian* hamiltonian, PetscInt NSteps, PetscReal Dt, PetscReal Tfinal, Vec x, Vec *lambda, Vec *mu, bool monitor){
  int ierr;

  ierr = TSSetProblemType(ts,TS_LINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts, TSTHETA); CHKERRQ(ierr);
  ierr = TSThetaSetTheta(ts, 0.5); CHKERRQ(ierr);   // midpoint rule
  ierr = TSSetRHSFunction(ts,NULL,TSComputeRHSFunctionLinear,hamiltonian);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,hamiltonian->getRHS(),hamiltonian->getRHS(),RHSJacobian,hamiltonian);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,Dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(ts,NSteps);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,Tfinal);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  if (monitor) {
    ierr = TSMonitorSet(ts, Monitor, NULL, NULL); CHKERRQ(ierr);
    ierr = TSAdjointMonitorSet(ts, AdjointMonitor, NULL, NULL); CHKERRQ(ierr);
  }
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetSolution(ts, x); CHKERRQ(ierr);

  /* Set the derivatives for TS */
  ierr = TSSetCostGradients(ts, 1, lambda, mu); CHKERRQ(ierr);
  ierr = TSSetRHSJacobianP(ts,hamiltonian->getdRHSdp(), RHSJacobianP, hamiltonian); CHKERRQ(ierr);


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