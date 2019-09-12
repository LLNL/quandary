#include "timestepper.hpp"


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


PetscErrorCode TSInit(TS ts, Hamiltonian* hamiltonian, PetscInt NSteps, PetscReal Dt, PetscReal Tfinal, bool monitor){
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

  return ierr;
}



PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec x,void *ctx) {
  PetscErrorCode ierr;
  PetscFunctionBeginUser;

  const PetscScalar *x_ptr;
  double tprev;
  ierr = TSGetPrevTime(ts, &tprev); CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &x_ptr);
  printf("Step %d: %f -> %f, x[1]=%1.14e\n", step, tprev, t, x_ptr[1]);
  ierr = VecRestoreArrayRead(x, &x_ptr);

  PetscFunctionReturn(0);
}

PetscErrorCode AdjointMonitor(TS ts,PetscInt step,PetscReal t,Vec x, PetscInt numcost, Vec* lambda, Vec* mu, void *ctx) {
  PetscErrorCode ierr;

  const PetscScalar *x_ptr;
  const PetscScalar *lambda_ptr;
  const PetscScalar *mu_ptr;
  double tprev;
  ierr = TSGetPrevTime(ts, &tprev); CHKERRQ(ierr);
  ierr = VecGetArrayRead(x, &x_ptr);
  ierr = VecGetArrayRead(lambda[0], &lambda_ptr);
  ierr = VecGetArrayRead(mu[0], &mu_ptr);
  ierr = VecGetArrayRead(x, &x_ptr);
  printf("AdjointStep %d: %f -> %f, x[1]=%1.14e, lambda[0]=%1.14e, mu[0]=%1.14e\n", step, tprev, t, x_ptr[1], lambda_ptr[0], mu_ptr[0] );
  ierr = VecRestoreArrayRead(x, &x_ptr);
  ierr = VecRestoreArrayRead(lambda[0], &lambda_ptr);
  ierr = VecRestoreArrayRead(mu[0], &mu_ptr);

  PetscFunctionReturn(0);
}



  int ierr; 

  ierr = TSSetUp(ts); CHKERRQ(ierr);
  ierr = TSTrajectorySetUp(ts->trajectory,ts);CHKERRQ(ierr);
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

  if (!ts->steps) {
    ierr = TSTrajectorySet(ts->trajectory,ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
  }

  return ierr;
}


PetscErrorCode TSStepMod(TS ts){
  int ierr; 

  ierr = TSMonitor(ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
  ierr = TSPreStep(ts);CHKERRQ(ierr);
  ierr = TSStep(ts);CHKERRQ(ierr);

  ierr = TSPostEvaluate(ts);CHKERRQ(ierr);

  ierr = TSTrajectorySet(ts->trajectory,ts,ts->steps,ts->ptime,ts->vec_sol);CHKERRQ(ierr);
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


PetscErrorCode TSAdjointStepMod(TS ts) {
  int ierr;
  ierr = TSTrajectoryGet(ts->trajectory,ts,ts->steps,&ts->ptime);CHKERRQ(ierr);
  ierr = TSAdjointMonitor(ts,ts->steps,ts->ptime,ts->vec_sol,ts->numcost,ts->vecs_sensi,ts->vecs_sensip);CHKERRQ(ierr);
  ierr = TSAdjointStep(ts);CHKERRQ(ierr);
  if (ts->vec_costintegral && !ts->costintegralfwd) {
    ierr = TSAdjointCostIntegral(ts);CHKERRQ(ierr);
  }

  return ierr;
}

PetscErrorCode TSAdjointPostSolve(TS ts){
  int ierr; 
  ierr = TSTrajectoryGet(ts->trajectory,ts,ts->steps,&ts->ptime);CHKERRQ(ierr);
  ierr = TSAdjointMonitor(ts,ts->steps,ts->ptime,ts->vec_sol,ts->numcost,ts->vecs_sensi,ts->vecs_sensip);CHKERRQ(ierr);
  ts->solvetime = ts->ptime;
  ierr = TSTrajectoryViewFromOptions(ts->trajectory,NULL,"-ts_trajectory_view");CHKERRQ(ierr);
  ierr = VecViewFromOptions(ts->vecs_sensi[0],(PetscObject) ts, "-ts_adjoint_view_solution");CHKERRQ(ierr);
  ts->adjoint_max_steps = 0;

  return ierr;
}