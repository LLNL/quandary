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


PetscErrorCode BuildTimeStepper(TS* ts, Hamiltonian* hamiltonian, PetscInt NSteps, PetscReal Dt, PetscReal Tfinal){
  int ierr;

  ierr = TSCreate(PETSC_COMM_SELF,ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(*ts,TS_LINEAR);CHKERRQ(ierr);
  ierr = TSSetType(*ts, TSTHETA); CHKERRQ(ierr);
  ierr = TSThetaSetTheta(*ts, 0.5); CHKERRQ(ierr);   // midpoint rule
  ierr = TSSetRHSFunction(*ts,NULL,TSComputeRHSFunctionLinear,hamiltonian);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(*ts,hamiltonian->getRHS(),hamiltonian->getRHS(),RHSJacobian,hamiltonian);CHKERRQ(ierr);
  ierr = TSSetTimeStep(*ts,Dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(*ts,NSteps);CHKERRQ(ierr);
  ierr = TSSetMaxTime(*ts,Tfinal);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(*ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(*ts);CHKERRQ(ierr);

  return 0;
}

