#include "optimtarget.hpp"


OptimTarget::OptimTarget(int purestateID_, TargetType target_type_, ObjectiveType objective_type_, Gate* targetgate_){

  // initialize
  target_type = target_type_;
  objective_type = objective_type_;
  targetgate = targetgate_;
  purestateID = purestateID_;

  /* Allocate gate specific vars */
  if (targetgate != NULL) {
    int dim_rho = targetgate->getDimRho();
    VecCreate(PETSC_COMM_WORLD, &aux); 
    VecSetSizes(aux,PETSC_DECIDE, 2*dim_rho*dim_rho);
    VecSetFromOptions(aux);

    /* Allocate target state VrhoV */
    VecCreate(PETSC_COMM_WORLD, &targetstate); 
    VecSetSizes(targetstate,PETSC_DECIDE, 2*dim_rho*dim_rho);
    VecSetFromOptions(targetstate);
  }
}

OptimTarget::~OptimTarget(){
  if (targetgate != NULL) {
    VecDestroy(&aux);
    VecDestroy(&targetstate);
  }
}

double OptimTarget::FrobeniusDistance(const Vec state){
  // Frobenius distance F = 1/2 || targetstate - state ||^2_F  = 1/2 || vec(targetstate-state)||^2_2
  double norm;
  VecAYPX(aux, 0.0, targetstate);    // aux = targetstate
  VecAXPY(aux, -1.0, state);   // aux = targetstate - state
  VecNorm(aux, NORM_2, &norm);
  double J = norm * norm;

  return J;
}

void OptimTarget::FrobeniusDistance_diff(const Vec state, Vec statebar, const double Jbar){
  
  // Derivative of frobenius distance : statebar += 2 * (targetstate - state) * (-1) * Jbar 
  VecAXPY(statebar,  2.0*Jbar, state);
  VecAXPY(statebar, -2.0*Jbar, targetstate);  
}

double OptimTarget::HilbertSchmidtOverlap(const Vec state, bool scalebypurity) {
  // Tr(targetstate*state) = vec(targetstate)^dagger vec(state) 
  double J = 0.0;
  VecTDot(targetstate, state, &J);
  // scale by purity Tr(targetstate^2) = || vec(targetstate)||^2_2
  if (scalebypurity){
    double dot;
    VecNorm(targetstate, NORM_2, &dot);
    J = J / (dot*dot);
  }
  return J;
}

void OptimTarget::HilbertSchmidtOverlap_diff(const Vec state, Vec statebar, const double Jbar, bool scalebypurity){
  // Derivative of Trace: statebar += targetstate^\dagger Jbar / scale
  double scale = 1.0;
  if (scalebypurity){
    double dot;
    VecNorm(targetstate, NORM_2, &dot);
    scale = dot*dot;
  }
  VecAXPY(statebar, Jbar/scale, targetstate);
}


void OptimTarget::prepare(const Vec rho_t0){
  // If gate optimization, apply the gate and store targetstate for later use. Else, do nothing.
  if (target_type == GATE) targetgate->applyGate(rho_t0, targetstate);
}



double OptimTarget::evalJ(const Vec state){
  double objective = 0.0;
  int diagID;
  double sum, mine, rhoii, lambdai, norm;
  int ilo, ihi;

  switch (target_type) {
    case GATE: /* target state \rho_target = Vrho(0)V^\dagger is stored in targetstate */
      
      switch(objective_type) {
        case JFROBENIUS:
          /* J_T = 1/2 * || rho_target - rho(T)||^2_F  */
          objective = FrobeniusDistance(state) / 2.0;
          break;
        case JHS:
          /* J_T = 1 - 1/purity * Tr(rho_target^\dagger * rho(T)) */
          objective = 1.0 - HilbertSchmidtOverlap(state, true);
          break;
        case JMEASURE: // JMEASURE is only for pure-state preparation!
          printf("ERROR: Check settings for optim_target and optim_objective.\n");
          exit(1);
          break;
      }
      break; // case gate

    case PUREM:

      int dim;
      VecGetSize(state, &dim);
      dim = (int) sqrt(dim/2.0);  // dim = N with \rho \in C^{N\times N}
      VecGetOwnershipRange(state, &ilo, &ihi);

      switch(objective_type) {

        case JMEASURE:
          /* J_T = Tr(O_m rho(T)) = \sum_i |i-m| rho_ii(T) */
          // iterate over diagonal elements 
          sum = 0.0;
          for (int i=0; i<dim; i++){
            diagID = getIndexReal(getVecID(i,i,dim));
            rhoii = 0.0;
            if (ilo <= diagID && diagID < ihi) VecGetValues(state, 1, &diagID, &rhoii);
            lambdai = fabs(i - purestateID);
            sum += lambdai * rhoii;
          }
          MPI_Allreduce(&sum, &objective, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
          break;
          
        case JFROBENIUS:
          /* J_T = 1/2 * || rho(T) - e_m e_m^\dagger||_F^2 */
          // substract 1.0 from m-th diagonal element then take the vector norm 
          diagID = getIndexReal(getVecID(purestateID,purestateID,dim));
          if (ilo <= diagID && diagID < ihi) VecSetValue(state, diagID, -1.0, ADD_VALUES);
          VecAssemblyBegin(state); VecAssemblyEnd(state);
          norm = 0.0;
          VecNorm(state, NORM_2, &norm);
          objective = pow(norm, 2.0) / 2.0;
          if (ilo <= diagID && diagID < ihi) VecSetValue(state, diagID, +1.0, ADD_VALUES); // restore original state!
          VecAssemblyBegin(state); VecAssemblyEnd(state);
          break;
          
        case JHS:
          /* J_T = 1 - Tr(e_m e_m^\dagger \rho(T)) = 1 - rho_mm(T) */
          diagID = getIndexReal(getVecID(purestateID,purestateID,dim));
          rhoii = 0.0;
          if (ilo <= diagID && diagID < ihi) VecGetValues(state, 1, &diagID, &rhoii);
          mine = 1. - rhoii;
          MPI_Allreduce(&mine, &objective, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
          break;
      } 
    break; // break pure1
  }
  return objective;
}


void OptimTarget::evalJ_diff(const Vec state, Vec statebar, const double Jbar){
  int ilo, ihi;
  double lambdai, val;
  int diagID;

  switch (target_type) {
    case GATE:
      switch (objective_type) {
        case JFROBENIUS:
          FrobeniusDistance_diff(state, statebar, Jbar/ 2.0);
          break;
        case JHS:
          HilbertSchmidtOverlap_diff(state, statebar, -1.0 * Jbar, true);
          break;
        case JMEASURE: // Will never happen
          printf("ERROR: Check settings for optim_target and optim_objective.\n");
          exit(1);
          break;
      }
      break; // case gate

    case PUREM:
      int dim;
      VecGetSize(state, &dim);
      dim = (int) sqrt(dim/2.0);  // dim = N with \rho \in C^{N\times N}
      VecGetOwnershipRange(state, &ilo, &ihi);

      switch (objective_type) {

        case JMEASURE:
          // iterate over diagonal elements 
          for (int i=0; i<dim; i++){
            lambdai = fabs(i - purestateID);
            diagID = getIndexReal(getVecID(i,i,dim));
            val = lambdai * Jbar;
            if (ilo <= diagID && diagID < ihi) VecSetValue(statebar, diagID, val, ADD_VALUES);
          }
          break;

        case JFROBENIUS:
          // Derivative of J = 1/2||x||^2 is xbar += x * Jbar, where x = rho(t) - E_mm
          VecAXPY(statebar, Jbar, state);
          // now substract 1.0*Jbar from m-th diagonal element
          diagID = getIndexReal(getVecID(purestateID,purestateID,dim));
          if (ilo <= diagID && diagID < ihi) VecSetValue(statebar, diagID, -1.0*Jbar, ADD_VALUES);
          break;

        case JHS:
          diagID = getIndexReal(getVecID(purestateID,purestateID,dim));
          val = -1. * Jbar;
          if (ilo <= diagID && diagID < ihi) VecSetValue(statebar, diagID, val, ADD_VALUES);
          break;
      }
      break; // case pure1
  }

  VecAssemblyBegin(statebar); VecAssemblyEnd(statebar);
}


double OptimTarget::evalFidelity(const Vec state){
  int dim;
  VecGetSize(state, &dim);
  dim = (int) sqrt(dim/2.0);  // dim = N with \rho \in C^{N\times N}

  double fidel = 0.0;

  int vecID, ihi, ilo;
  double rho_mm;

  switch(target_type){
    case PUREM: // fidelity = rho(T)_mm
      vecID = getIndexReal(getVecID(purestateID, purestateID, dim));
      VecGetOwnershipRange(state, &ilo, &ihi);
      rho_mm = 0.0;
      if (ilo <= vecID && vecID < ihi) VecGetValues(state, 1, &vecID, &rho_mm); // local!
      // Communicate over all petsc processors.
      MPI_Allreduce(&rho_mm, &fidel, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
      break;

    case GATE: // fidelity = Tr(Vrho(0)V^\dagger \rho)
      fidel = HilbertSchmidtOverlap(state, false);
      break;
  }
 
  return fidel;

}