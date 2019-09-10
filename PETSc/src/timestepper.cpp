#include "timestepper.hpp"


PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec u,Mat M,Mat P,void *ctx){

  /* Cast ctx to Hamiltonian pointer */
  Hamiltonian *hamiltonian = (Hamiltonian*) ctx;

  /* buildRHSing the Hamiltonian will set the matrix H */
  hamiltonian->buildRHS(t);

  /* Get the hamiltonian system matrix H*/
  M = hamiltonian->getM();

  return 0;
}



PetscErrorCode BuildTimeStepper(TS* ts, Hamiltonian* hamiltonian, PetscInt NSteps, PetscReal Dt, PetscReal Tfinal){
  int ierr;

  ierr = TSCreate(PETSC_COMM_SELF,ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(*ts,TS_LINEAR);CHKERRQ(ierr);
  ierr = TSSetType(*ts, TSTHETA); CHKERRQ(ierr);
  ierr = TSThetaSetTheta(*ts, 0.5); CHKERRQ(ierr);   // midpoint rule
  ierr = TSSetRHSFunction(*ts,NULL,TSComputeRHSFunctionLinear,hamiltonian);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(*ts,hamiltonian->getM(),hamiltonian->getM(),RHSJacobian,hamiltonian);CHKERRQ(ierr);
  ierr = TSSetTimeStep(*ts,Dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(*ts,NSteps);CHKERRQ(ierr);
  ierr = TSSetMaxTime(*ts,Tfinal);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(*ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetFromOptions(*ts);CHKERRQ(ierr);

  return 0;
}

