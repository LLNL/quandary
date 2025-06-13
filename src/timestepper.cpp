#include "timestepper.hpp"
#include "petscvec.h"

TimeStepper::TimeStepper() {
  dim = 0;
  mastereq = NULL;
  ntime = 0;
  total_time = 0.0;
  dt = 0.0;
  storeFWD = false;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
  writeTrajectoryDataFiles = false;
}

TimeStepper::TimeStepper(MasterEq* mastereq_, int ntime_, double total_time_, Output* output_, bool storeFWD_) : TimeStepper() {
  mastereq = mastereq_;
  dim = 2*mastereq->getDim(); // will be either N^2 (Lindblad) or N (Schroedinger)
  ntime = ntime_;
  total_time = total_time_;
  output = output_;
  storeFWD = storeFWD_;

  /* Check if leakage term is added: Only if nessential is smaller than nlevels for at least one oscillator */
  addLeakagePrevent = false; 
  for (size_t i=0; i<mastereq->getNOscillators(); i++){
    if (mastereq->nessential[i] < mastereq->nlevels[i]) addLeakagePrevent = true;
  }

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
  VecDuplicate(x, &xadj);
  VecDuplicate(x, &xprimal);

  /* Allocate the reduced gradient */
  int ndesign = 0;
  for (size_t ioscil = 0; ioscil < mastereq->getNOscillators(); ioscil++) {
      ndesign += mastereq->getOscillator(ioscil)->getNParams(); 
  }
  VecCreateSeq(PETSC_COMM_SELF, ndesign, &redgrad);
  VecSetFromOptions(redgrad);
  VecAssemblyBegin(redgrad);
  VecAssemblyEnd(redgrad);

}


TimeStepper::~TimeStepper() {
  for (size_t n = 0; n < store_states.size(); n++) {
    VecDestroy(&(store_states[n]));
  }
  VecDestroy(&x);
  VecDestroy(&xadj);
  VecDestroy(&xprimal);
  VecDestroy(&redgrad);
}



Vec TimeStepper::getState(size_t tindex){
  
  if (tindex >= store_states.size()) {
    printf("ERROR: Time-stepper requested state at time index %zu, but didn't store it.\n", tindex);
    exit(1);
  }

  return store_states[tindex];
}

Vec TimeStepper::solveODE(int initid, Vec rho_t0){

  /* Open output files */
  if (writeTrajectoryDataFiles) {
    output->openTrajectoryDataFiles("rho", initid);
  }

  /* Set initial condition  */
  VecCopy(rho_t0, x);


  /* Store initial state for dpdm penalty */
  if (gamma_penalty_dpdm > 1e-13){
    for (int i = 0; i < 2; i++) {
      Vec state;
      VecCreate(PETSC_COMM_WORLD, &state);
      VecSetSizes(state, PETSC_DECIDE, dim);
      VecSetFromOptions(state);
      dpdm_states.push_back(state);
    }
    VecCopy(x, dpdm_states[0]);
  }

  /* --- Loop over time interval --- */
  penalty_integral = 0.0;
  penalty_dpdm = 0.0;
  energy_penalty_integral = 0.0;
  for (int n = 0; n < ntime; n++){

    /* current time */
    double tstart = n * dt;
    double tstop  = (n+1) * dt;

    /* store and write current state. */
    if (storeFWD) VecCopy(x, store_states[n]);
    if (writeTrajectoryDataFiles) {
      output->writeTrajectoryDataFiles(n, tstart, x, mastereq);
    }

    /* Take one time step */
    evolveFWD(tstart, tstop, x);

    /* Add to penalty objective term */
    if (gamma_penalty > 1e-13) penalty_integral += penaltyIntegral(tstop, x);

    /* Add to penalty for second derivative */
    if (gamma_penalty_dpdm > 1e-13) {
      // printf("DPDM Forward, f(%d %d %d) \n", n+1, n, n-1);
      if (n > 0) penalty_dpdm += penaltyDpDm(x, dpdm_states[n%2], dpdm_states[(n+1)%2]);  // uses x, x_n, x_n-1

      // Update storage of primal states. Should build a history of 3 states.
      VecCopy(x, dpdm_states[(n+1)%2]);
      
    }

    /* Add to energy penalty objective term */
    if (gamma_penalty_energy > 1e-13) energy_penalty_integral += energyPenaltyIntegral(tstop);

#ifdef SANITY_CHECK
    SanityTests(x, tstart);
#endif
  }
  penalty_dpdm = penalty_dpdm/ntime;

  /* Store last time step */
  if (storeFWD) VecCopy(x, store_states[ntime]);

  /* Clear out dpdm storage */
  if (gamma_penalty_dpdm > 1e-13) {
    for (size_t i=0; i<dpdm_states.size(); i++) {
      VecDestroy(&(dpdm_states[i]));
    }
    dpdm_states.clear();
  }

  /* Write last time step and close files */
  if (writeTrajectoryDataFiles) {
    output->writeTrajectoryDataFiles(ntime, ntime*dt, x, mastereq);
    output->closeTrajectoryDataFiles();
  }
  

  return x;
}


void TimeStepper::solveAdjointODE(Vec rho_t0_bar, Vec finalstate, double Jbar_penalty, double Jbar_penalty_dpdm, double Jbar_energy_penalty) {

  /* Reset gradient */
  VecZeroEntries(redgrad);

  /* Set terminal adjoint condition */
  VecCopy(rho_t0_bar, xadj);

  /* Set terminal primal state */
  VecCopy(finalstate, xprimal);

  /* Store states at N, N-1, N-2 for dpdm penalty */
  if (gamma_penalty_dpdm > 1e-13){
    for (int i = 0; i < 5; i++) {
      Vec state;
      VecCreate(PETSC_COMM_WORLD, &state);
      VecSetSizes(state, PETSC_DECIDE, dim);
      VecSetFromOptions(state);
      dpdm_states.push_back(state);
    }
    VecCopy(xprimal, dpdm_states[2]);  // x_N
    VecCopy(dpdm_states[2], dpdm_states[1]);
    evolveFWD(ntime*dt, (ntime-1)*dt, dpdm_states[1]);  // x_N-1
    VecCopy(dpdm_states[1], dpdm_states[0]);
    evolveFWD((ntime-1)*dt, (ntime-2)*dt, dpdm_states[0]);  // x_N-2
  }
 

  /* Loop over time interval */
  for (int n = ntime; n > 0; n--){
    double tstop  = n * dt;
    double tstart = (n-1) * dt;
    // printf("Backwards %d -> %d ... \n", n, n-1);

    /* Derivative of energy penalty objective term */
    if (gamma_penalty_energy > 1e-13) energyPenaltyIntegral_diff(tstop, Jbar_energy_penalty, redgrad);

    /* Derivative of penalty term */
    if (gamma_penalty_dpdm > 1e-13) penaltyDpDm_diff(n, xadj, Jbar_penalty_dpdm/ntime);

    /* Derivative of penalty objective term */
    if (gamma_penalty > 1e-13) penaltyIntegral_diff(tstop, xprimal, xadj, Jbar_penalty);

    /* Get the state at n-1. If Schroedinger solver, recompute it by taking a step backwards with the forward solver, otherwise get it from storage. */
    if (storeFWD) VecCopy(getState(n-1), xprimal);
    else evolveFWD(tstop, tstart, xprimal);

    /* Take one time step backwards for the adjoint */
    evolveBWD(tstop, tstart, xprimal, xadj, redgrad, true);

    /* Update dpdm storage */
    if (gamma_penalty_dpdm > 1e-13 ) {
      int k = ntime - n;
      int idx = 4-((k+4)%5);
      int idx1 = 4-(k%5);
      VecCopy(dpdm_states[idx], dpdm_states[idx1]);
      if (n>2) evolveFWD((n-2)*dt, (n-3)*dt, dpdm_states[idx1]);
    }
  }

  /* Clear out dpdm storage */
  if (gamma_penalty_dpdm > 1e-13) {
    for (size_t i=0; i<dpdm_states.size(); i++) {
      VecDestroy(&(dpdm_states[i]));
    }
    dpdm_states.clear();
  }
}


double TimeStepper::penaltyIntegral(double time, const Vec x){
  double penalty = 0.0;
  int dim_rho = mastereq->getDimRho(); // N
  double x_re, x_im;
  PetscInt vecID_re, vecID_im;

  /* weighted integral of the objective function */
  if (penalty_param > 1e-13) {
    double weight = 1./penalty_param * exp(- pow((time - total_time)/penalty_param, 2));

    double obj_re = 0.0;
    double obj_im = 0.0;
    optim_target->evalJ(x, &obj_re, &obj_im);
    double obj_cost = optim_target->finalizeJ(obj_re, obj_im);
    penalty = weight * obj_cost * dt;
  }

  /* Add guard-level occupation to prevent leakage. A guard level is the LAST NON-ESSENTIAL energy level of an oscillator */
  if (addLeakagePrevent) {
    double leakage = 0.0;
    PetscInt ilow, iupp;
    VecGetOwnershipRange(x, &ilow, &iupp);
    /* Sum over all diagonal elements that correspond to a non-essential guard level. */
    for (int i=0; i<dim_rho; i++) {
      if ( isGuardLevel(i, mastereq->nlevels, mastereq->nessential) ) {
          // printf("%f: isGuard: %d / %d\n", time, i, dim_rho);
        if (mastereq->lindbladtype != LindbladType::NONE) {
          vecID_re = getIndexReal(getVecID(i,i,dim_rho));
          vecID_im = getIndexImag(getVecID(i,i,dim_rho));
        } else {
          vecID_re = getIndexReal(i);
          vecID_im = getIndexImag(i);
        }
        x_re = 0.0; x_im = 0.0;
        if (ilow <= vecID_re && vecID_re < iupp) VecGetValues(x, 1, &vecID_re, &x_re);
        if (ilow <= vecID_im && vecID_im < iupp) VecGetValues(x, 1, &vecID_im, &x_im); 
        leakage += (x_re * x_re + x_im * x_im) / (dt*ntime);
      }
    }
    double mine = leakage;
    MPI_Allreduce(&mine, &leakage, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    penalty += dt * leakage;
  }

  return penalty;
}

void TimeStepper::penaltyIntegral_diff(double time, const Vec x, Vec xbar, double penaltybar){
  int dim_rho = mastereq->getDimRho();  // N
  PetscInt vecID_re, vecID_im;

  /* Derivative of weighted integral of the objective function */
  if (penalty_param > 1e-13){
    double weight = 1./penalty_param * exp(- pow((time - total_time)/penalty_param, 2));
    
    double obj_cost_re = 0.0;
    double obj_cost_im = 0.0;
    optim_target->evalJ(x, &obj_cost_re, &obj_cost_im);

    double obj_cost_re_bar = 0.0; 
    double obj_cost_im_bar = 0.0;
    optim_target->finalizeJ_diff(obj_cost_re, obj_cost_im, &obj_cost_re_bar, &obj_cost_im_bar);
    optim_target->evalJ_diff(x, xbar, weight*obj_cost_re_bar*penaltybar*dt, weight*obj_cost_im_bar*penaltybar*dt);
  }

  /* If gate optimization: Derivative of adding guard-level occupation */
  if (addLeakagePrevent) {
    PetscInt ilow, iupp;
    VecGetOwnershipRange(x, &ilow, &iupp);
    double x_re, x_im;
    for (int i=0; i<dim_rho; i++) {
      if ( isGuardLevel(i, mastereq->nlevels, mastereq->nessential) ) {
        if (mastereq->lindbladtype != LindbladType::NONE){ 
          vecID_re = getIndexReal(getVecID(i,i,dim_rho));
          vecID_im = getIndexImag(getVecID(i,i,dim_rho));
        } else {
          vecID_re = getIndexReal(i);
          vecID_im = getIndexImag(i);
        }
        x_re = 0.0; x_im = 0.0;
        if (ilow <= vecID_re && vecID_re < iupp) VecGetValues(x, 1, &vecID_re, &x_re);
        if (ilow <= vecID_im && vecID_im < iupp) VecGetValues(x, 1, &vecID_im, &x_im);
        // Derivative: 2 * rho(i,i) * weights * penalbar * dt
        if (ilow <= vecID_re && vecID_re < iupp) VecSetValue(xbar, vecID_re, 2.*x_re*penaltybar/ntime, ADD_VALUES);
        if (ilow <= vecID_im && vecID_im < iupp) VecSetValue(xbar, vecID_im, 2.*x_im*penaltybar/ntime, ADD_VALUES);
      }
    }
    VecAssemblyBegin(xbar);
    VecAssemblyEnd(xbar);
  }
}


double TimeStepper::penaltyDpDm(Vec x, Vec xm1, Vec xm2){
    int dim_rho = mastereq->getDimRho(); // N
    int vecID_re, vecID_im;

    double tmp1, tmp2;

    const PetscScalar *xptr, *xm1ptr, *xm2ptr;
    VecGetArrayRead(x, &xptr);
    VecGetArrayRead(xm1, &xm1ptr);
    VecGetArrayRead(xm2, &xm2ptr);
     
    PetscInt ilow, iupp;

    /* precompute 1/dt^4 */
    double dtinv = 1.0 / (dt*dt*dt*dt);

    double dpdm = 0.0;
    VecGetOwnershipRange(x, &ilow, &iupp);
    for (int i=0; i<dim_rho; i++) {  
        if (mastereq->lindbladtype != LindbladType::NONE) { 
          vecID_re = getIndexReal(getVecID(i,i,dim_rho));
          vecID_im = getIndexImag(getVecID(i,i,dim_rho));
        } else {
          vecID_re = getIndexReal(i);
          vecID_im = getIndexImag(i);
        }

        if (ilow <= vecID_re && vecID_re < iupp) tmp1 = xptr[vecID_re]*xptr[vecID_re] - 2.0*xm1ptr[vecID_re]*xm1ptr[vecID_re] + xm2ptr[vecID_re]*xm2ptr[vecID_re];
        if (ilow <= vecID_im && vecID_im < iupp) tmp2 = xptr[vecID_im]*xptr[vecID_im] - 2.0*xm1ptr[vecID_im]*xm1ptr[vecID_im] + xm2ptr[vecID_im]*xm2ptr[vecID_im];
        dpdm += dtinv * (tmp1 + tmp2)*(tmp1 + tmp2);
    }

    VecRestoreArrayRead(x, &xptr);
    VecRestoreArrayRead(xm1, &xm1ptr);
    VecRestoreArrayRead(xm2, &xm2ptr);

    return dpdm;
}


void TimeStepper::penaltyDpDm_diff(int n, Vec xbar, double Jbar){
    int dim_rho = mastereq->getDimRho(); // N
    int vecID_re, vecID_im;

    const PetscScalar *xptr, *xm1ptr, *xm2ptr, *xp1ptr, *xp2ptr;
    Vec x, xm1, xm2, xp1, xp2;

    int k = ntime - n;
    int idx = 4-(k+4)%5;
    if (n > 1)         xm2 = dpdm_states[idx];
    if (n > 0 )        xm1 = dpdm_states[(idx+1)%5];
    x = dpdm_states[(idx+2)%5];
    if (n < ntime)     xp1 = dpdm_states[(idx+3)%5];
    if (n < ntime-1)   xp2 = dpdm_states[(idx+4)%5];

    VecGetArrayRead(x, &xptr);
    if (n > 0)        VecGetArrayRead(xm1, &xm1ptr);
    if (n > 1)        VecGetArrayRead(xm2, &xm2ptr);
    if (n < ntime)    VecGetArrayRead(xp1, &xp1ptr);
    if (n < ntime-1)  VecGetArrayRead(xp2, &xp2ptr);

     
    PetscInt ilow, iupp;

    /* precompute 1/dt^4 */
    double dtinv = 1.0 / (dt*dt*dt*dt);

    VecGetOwnershipRange(x, &ilow, &iupp);
    for (int i=0; i<dim_rho; i++) {  
        if (mastereq->lindbladtype != LindbladType::NONE) { 
          vecID_re = getIndexReal(getVecID(i,i,dim_rho));
          vecID_im = getIndexImag(getVecID(i,i,dim_rho));
        } else {
          vecID_re = getIndexReal(i);
          vecID_im = getIndexImag(i);
        }

        // first term
        if (n > 1) {
          // if (i==0) printf("DPDM BWD, update %d from on first f(%d %d %d) \n", n, n-2, n-1, n);

          if (ilow <= vecID_re && vecID_re < iupp) { // should be same as im. 
            double tmp1 = xm2ptr[vecID_re]*xm2ptr[vecID_re] - 2.0*xm1ptr[vecID_re]*xm1ptr[vecID_re] + xptr[vecID_re]*xptr[vecID_re];
            double tmp2 = xm2ptr[vecID_im]*xm2ptr[vecID_im] - 2.0*xm1ptr[vecID_im]*xm1ptr[vecID_im] + xptr[vecID_im]*xptr[vecID_im];
            double pop = tmp1+tmp2;
            double dpdphi_re = 2.0*xptr[vecID_re];
            double dpdphi_im = 2.0*xptr[vecID_im];
            VecSetValue(xbar,vecID_re,  2.0 * pop * dpdphi_re * dtinv * Jbar, ADD_VALUES);
            VecSetValue(xbar,vecID_im,  2.0 * pop * dpdphi_im * dtinv * Jbar, ADD_VALUES);
          }
        }

        // center term
        if (n > 0 && n < ntime) {
            // if (i==0) printf("DPDM BWD, update %d from on second f(%d %d %d) \n", n, n-1, n, n+1);
            if (ilow <= vecID_re && vecID_re < iupp) {
              double tmp1 = xm1ptr[vecID_re]*xm1ptr[vecID_re] - 2.0*xptr[vecID_re]*xptr[vecID_re] + xp1ptr[vecID_re]*xp1ptr[vecID_re];
              double tmp2 = xm1ptr[vecID_im]*xm1ptr[vecID_im] - 2.0*xptr[vecID_im]*xptr[vecID_im] + xp1ptr[vecID_im]*xp1ptr[vecID_im];
              double pop = tmp1 + tmp2;
              double dpdphi_re = 2.0*xptr[vecID_re];
              double dpdphi_im = 2.0*xptr[vecID_im];
              VecSetValue(xbar, vecID_re, - 4.0 * pop * dpdphi_re * dtinv * Jbar, ADD_VALUES);
              VecSetValue(xbar, vecID_im, - 4.0 * pop * dpdphi_im * dtinv * Jbar, ADD_VALUES);
            }
        }
        
        // last term 
        if (n < ntime-1) {
            // if (i==0) printf("DPDM BWD, update %d from on third f(%d %d %d) \n", n, n, n+1, n+2);
            if (ilow <= vecID_re && vecID_re < iupp) {
              double tmp1 = xptr[vecID_re]*xptr[vecID_re] - 2.0*xp1ptr[vecID_re]*xp1ptr[vecID_re] + xp2ptr[vecID_re]*xp2ptr[vecID_re];
              double tmp2 = xptr[vecID_im]*xptr[vecID_im] - 2.0*xp1ptr[vecID_im]*xp1ptr[vecID_im] + xp2ptr[vecID_im]*xp2ptr[vecID_im];
              double pop = tmp1 + tmp2;
              double dpdphi_re = 2.0*xptr[vecID_re];
              double dpdphi_im = 2.0*xptr[vecID_im];
              VecSetValue(xbar, vecID_re, 2.0* pop * dpdphi_re * dtinv * Jbar, ADD_VALUES);
              VecSetValue(xbar, vecID_im, 2.0* pop * dpdphi_im * dtinv * Jbar, ADD_VALUES);
            }
        }
    }

    VecRestoreArrayRead(x, &xptr);
    if (n > 0 )       VecRestoreArrayRead(xm1, &xm1ptr);
    if (n > 1)        VecRestoreArrayRead(xm2, &xm2ptr);
    if (n < ntime)    VecRestoreArrayRead(xp1, &xp1ptr);
    if (n < ntime-1)  VecRestoreArrayRead(xp2, &xp2ptr);
 
}

double TimeStepper::energyPenaltyIntegral(double time){
  double pen = 0.0;

  /* Loop over oscillators */
  for (size_t iosc = 0; iosc < mastereq->getNOscillators(); iosc++) {
    double p,q;
    mastereq->getOscillator(iosc)->evalControl(time, &p, &q); 
    pen += (p*p + q*q) / ntime;
  }

  return pen;
}


void TimeStepper::energyPenaltyIntegral_diff(double time, double penaltybar, Vec redgrad){

  int col_shift = 0;
  double* grad_ptr;
  VecGetArray(redgrad, &grad_ptr);

  for (size_t iosc = 0; iosc < mastereq->getNOscillators(); iosc++){

    /* Reevaluate the controls to set pbar, qbar */
    double p,q;
    mastereq->getOscillator(iosc)->evalControl(time, &p, &q); 
    double pbar = penaltybar/ntime * 2.0 * p;
    double qbar = penaltybar/ntime * 2.0 * q;

    /* Derivative of evalControls */
    double* grad_for_this_oscillator = grad_ptr + col_shift;
    mastereq->getOscillator(iosc)->evalControl_diff(time, grad_for_this_oscillator, pbar, qbar);

    // Skip in gradient for next oscillator
    col_shift += mastereq->getOscillator(iosc)->getNParams();
  } 
  VecRestoreArray(redgrad, &grad_ptr);
}

void TimeStepper::evolveBWD(const double /*tstart*/, const double /*tstop*/, const Vec /*x_stop*/, Vec /*x_adj*/, Vec /*grad*/, bool /*compute_gradient*/){}

ExplEuler::ExplEuler(MasterEq* mastereq_, int ntime_, double total_time_, Output* output_, bool storeFWD_) : TimeStepper(mastereq_, ntime_, total_time_, output_, storeFWD_) {
  MatCreateVecs(mastereq->getRHS(), &stage, NULL);
  VecZeroEntries(stage);
}

ExplEuler::~ExplEuler() {
  VecDestroy(&stage);
}

void ExplEuler::evolveFWD(const double tstart,const  double tstop, Vec x) {

  double dt = tstop - tstart;

   /* Compute A(tstart) */
  mastereq->assemble_RHS(tstart);
  Mat A = mastereq->getRHS(); 

  /* update x = x + hAx */
  MatMult(A, x, stage);
  VecAXPY(x, dt, stage);
}

void ExplEuler::evolveBWD(const double tstop,const  double tstart,const  Vec x, Vec x_adj, Vec grad, bool compute_gradient){
  double dt = tstop - tstart;

  /* Add to reduced gradient */
  if (compute_gradient) {
    mastereq->compute_dRHS_dParams(tstop, x, x_adj, dt, grad);
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
  if (linsolve_counter <= 0) linsolve_counter = 1;
  linsolve_iterstaken_avg = (int) linsolve_iterstaken_avg / linsolve_counter;
  linsolve_error_avg = linsolve_error_avg / linsolve_counter;
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
  double dt = tstop - tstart;  

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
      // printf("Residual norm %d: %1.5e\n", iters_taken, rnorm);
      linsolve_iterstaken_avg += iters_taken;
      linsolve_error_avg += rnorm;
      if (rnorm > 1e-3)  {
        printf("WARNING: Linear solver residual norm: %1.5e\n", rnorm);
      }
 
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
  double dt = tstop - tstart;
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
      PetscInt iters_taken;
      KSPGetIterationNumber(ksp, &iters_taken);
      if (rnorm > 1e-3)  {
        printf("WARNING: Linear solver residual norm: %1.5e\n", rnorm);
      }
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
    mastereq->compute_dRHS_dParams(thalf, stage, stage_adj, 1.0, grad);
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



CompositionalImplMidpoint::CompositionalImplMidpoint(int order_, MasterEq* mastereq_, int ntime_, double total_time_, LinearSolverType linsolve_type_, int linsolve_maxiter_, Output* output_, bool storeFWD_): ImplMidpoint(mastereq_, ntime_, total_time_, linsolve_type_, linsolve_maxiter_, output_, storeFWD_) {

  order = order_;

  // coefficients for order 8, stages s=15
  gamma.clear();
  if (order == 8){
    gamma.push_back(0.74167036435061295344822780);
    gamma.push_back(-0.40910082580003159399730010);
    gamma.push_back(0.19075471029623837995387626);
    gamma.push_back(-0.57386247111608226665638773);
    gamma.push_back(0.29906418130365592384446354);
    gamma.push_back(0.33462491824529818378495798);
    gamma.push_back(0.31529309239676659663205666);
    gamma.push_back(-0.79688793935291635401978884); // 8
    gamma.push_back(0.31529309239676659663205666);
    gamma.push_back(0.33462491824529818378495798);
    gamma.push_back(0.29906418130365592384446354);
    gamma.push_back(-0.57386247111608226665638773);
    gamma.push_back(0.19075471029623837995387626);
    gamma.push_back(-0.40910082580003159399730010);
    gamma.push_back(0.74167036435061295344822780);
  } else if (order == 4) {
    gamma.push_back(1./(2. - pow(2., 1./3.)));
    gamma.push_back(- pow(2., 1./3.)*gamma[0] );
    gamma.push_back(1./(2. - pow(2., 1./3.)));
  }

  if (mpirank_world == 0) printf("Timestepper: Compositional Impl. Midpoint, order %d, %lu stages\n", order, gamma.size());

  // Allocate storage of stages for backward process 
  for (size_t i = 0; i <gamma.size(); i++) {
    Vec state;
    VecCreate(PETSC_COMM_WORLD, &state);
    VecSetSizes(state, PETSC_DECIDE, dim);
    VecSetFromOptions(state);
    x_stage.push_back(state);
  }
  VecCreate(PETSC_COMM_WORLD, &aux);
  VecSetSizes(aux, PETSC_DECIDE, dim);
  VecSetFromOptions(aux);
}

CompositionalImplMidpoint::~CompositionalImplMidpoint(){
  for (size_t i = 0; i <gamma.size(); i++) {
    VecDestroy(&(x_stage[i]));
  }
  VecDestroy(&aux);
}


void CompositionalImplMidpoint::evolveFWD(const double tstart,const  double tstop, Vec x) {

  double dt = tstop - tstart;
  double tcurr = tstart;

  // Loop over stages
  for (size_t istage = 0; istage < gamma.size(); istage++) {
    // time-step size and tstart,tstop for compositional step
    double dt_stage = gamma[istage] * dt;

    // Evolve 'tcurr -> tcurr + gamma*dt' using ImpliMidpointrule
    ImplMidpoint::evolveFWD(tcurr, tcurr + dt_stage, x);

    // Update current time
    tcurr = tcurr + dt_stage;
  }
  assert(fabs(tcurr - tstop) < 1e-12);

}

void CompositionalImplMidpoint::evolveBWD(const double tstop, const double tstart, const Vec x, Vec x_adj, Vec grad, bool compute_gradient){
  
  double dt = tstop - tstart;

  // Run forward again to store the (primal) stages
  double tcurr = tstart;
  VecCopy(x, aux);
  for (size_t istage = 0; istage < gamma.size(); istage++) {
    VecCopy(aux, x_stage[istage]);
    double dt_stage = gamma[istage] * dt;
    ImplMidpoint::evolveFWD(tcurr, tcurr + dt_stage, aux);
    tcurr = tcurr + dt_stage;
  }
  assert(fabs(tcurr - tstop) < 1e-12);

  // Run backwards while updating adjoint and gradient
  for (int istage = gamma.size()-1; istage >=0; istage--){
    double dt_stage = gamma[istage] * dt;
    ImplMidpoint::evolveBWD(tcurr, tcurr-dt_stage, x_stage[istage], x_adj, grad, compute_gradient);
    tcurr = tcurr - gamma[istage]*dt;
  }
  assert(fabs(tcurr - tstart) < 1e-12);
}
