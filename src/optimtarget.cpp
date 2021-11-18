#include "optimtarget.hpp"


OptimTarget::OptimTarget(int dim, int purestateID_, TargetType target_type_, ObjectiveType objective_type_, Gate* targetgate_, std::string target_filename_){

  // initialize
  target_type = target_type_;
  objective_type = objective_type_;
  targetgate = targetgate_;
  purestateID = purestateID_;
  target_filename = target_filename_;

  /* Allocate target state, if it is read from file, of if target is a gate transformation VrhoV */
  if (target_type == TargetType::GATE || target_type == TargetType::FROMFILE) {
    VecCreate(PETSC_COMM_WORLD, &targetstate); 
    VecSetSizes(targetstate,PETSC_DECIDE, 2*dim);   // input dim is the dimension of the vectorized system: dim=N^2
    VecSetFromOptions(targetstate);
  }

  if (target_type == TargetType::FROMFILE) {
    // Read the target state from file into vec
    int mpirank_world;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank_world);
    double* vec = new double[2*dim];
    if (mpirank_world == 0) read_vector(target_filename.c_str(), vec, 2*dim);
    MPI_Bcast(vec, 2*dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // pass vec into the targetstate
    PetscInt ilow, iupp;
    VecGetOwnershipRange(targetstate, &ilow, &iupp);
    for (int i = 0; i < dim; i++) { // iterates up to N^2
      int elemid_re = getIndexReal(i);
      int elemid_im = getIndexImag(i);
      if (ilow <= elemid_re && elemid_re < iupp) VecSetValue(targetstate, elemid_re, vec[i],       INSERT_VALUES); // RealPart
      if (ilow <= elemid_im && elemid_im < iupp) VecSetValue(targetstate, elemid_im, vec[i + dim], INSERT_VALUES); // Imaginary Part
    }
    VecAssemblyBegin(targetstate); VecAssemblyEnd(targetstate);
    delete [] vec;
    // VecView(targetstate, NULL);
  }

  /* Allocate an auxiliary vec needed for evaluating the frobenius norm */
  if (objective_type == ObjectiveType::JFROBENIUS) {
    VecCreate(PETSC_COMM_WORLD, &aux); 
    VecSetSizes(aux,PETSC_DECIDE, 2*dim);
    VecSetFromOptions(aux);
  }
}

OptimTarget::~OptimTarget(){
  if (objective_type == ObjectiveType::JFROBENIUS) VecDestroy(&aux);
  if (target_type == TargetType::GATE || target_type == TargetType::FROMFILE)  VecDestroy(&targetstate);
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
  if (target_type == TargetType::GATE) targetgate->applyGate(rho_t0, targetstate);
}



double OptimTarget::evalJ(const Vec state){
  double objective = 0.0;
  PetscInt diagID;
  double sum, mine, rhoii, lambdai, norm;
  PetscInt ilo, ihi;

  PetscInt dim;
  VecGetSize(state, &dim);
  dim = (int) sqrt(dim/2.0);  // dim = N with \rho \in C^{N\times N}


  switch(objective_type) {

    /* J_Frob = 1/2 * || rho_target - rho(T)||^2_F  */
    case ObjectiveType::JFROBENIUS:

      if (target_type == TargetType::GATE || target_type == TargetType::FROMFILE ) {
        // target state is already set. Either \rho_target = Vrho(0)V^\dagger or read from file. Just eval norm.
        objective = FrobeniusDistance(state) / 2.0;
      } 
      else {  // target = e_me_m^\dagger
        assert(target_type == TargetType::PURE);
        // substract 1.0 from m-th diagonal element then take the vector norm 
        diagID = getIndexReal(getVecID(purestateID,purestateID,dim));
        VecGetOwnershipRange(state, &ilo, &ihi);
        if (ilo <= diagID && diagID < ihi) VecSetValue(state, diagID, -1.0, ADD_VALUES);
        VecAssemblyBegin(state); VecAssemblyEnd(state);
        norm = 0.0;
        VecNorm(state, NORM_2, &norm);
        objective = pow(norm, 2.0) / 2.0;
        if (ilo <= diagID && diagID < ihi) VecSetValue(state, diagID, +1.0, ADD_VALUES); // restore original state!
        VecAssemblyBegin(state); VecAssemblyEnd(state);
      }
      break;  // case Frobenius

    /* J_HS = 1 - 1/purity * Tr(rho_target^\dagger * rho(T)) */
    case ObjectiveType::JHS:

      if (target_type == TargetType::GATE || target_type == TargetType::FROMFILE ) {
        // target state is already set. Either \rho_target = Vrho(0)V^\dagger or read from file. Just eval Trace.
        objective = 1.0 - HilbertSchmidtOverlap(state, true);
      }
      else { // target = e_m e_m^\dagger
        /* -> J_HS = 1 - Tr(e_m e_m^\dagger \rho(T)) = 1 - rho_mm(T) */
        assert(target_type == TargetType::PURE);
        diagID = getIndexReal(getVecID(purestateID,purestateID,dim));
        rhoii = 0.0;
        VecGetOwnershipRange(state, &ilo, &ihi);
        if (ilo <= diagID && diagID < ihi) VecGetValues(state, 1, &diagID, &rhoii);
        mine = 1. - rhoii;
        MPI_Allreduce(&mine, &objective, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
      }
      break; // case J_HS

    /* J_T = Tr(O_m rho(T)) = \sum_i |i-m| rho_ii(T) */
    case ObjectiveType::JMEASURE:
      if (target_type == TargetType::GATE || target_type == TargetType::FROMFILE ) {
        printf("ERROR: Check settings for optim_target and optim_objective.\n");
        exit(1);
      }
      else {
        assert(target_type == TargetType::PURE);
        // iterate over diagonal elements 
        sum = 0.0;
        for (int i=0; i<dim; i++){
          diagID = getIndexReal(getVecID(i,i,dim));
          rhoii = 0.0;
          VecGetOwnershipRange(state, &ilo, &ihi);
          if (ilo <= diagID && diagID < ihi) VecGetValues(state, 1, &diagID, &rhoii);
          lambdai = fabs(i - purestateID);
          sum += lambdai * rhoii;
        }
        MPI_Allreduce(&sum, &objective, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
      }
      break; // case J_MEASURE
  }

  return objective;
}


void OptimTarget::evalJ_diff(const Vec state, Vec statebar, const double Jbar){
  PetscInt ilo, ihi;
  double lambdai, val;
  PetscInt diagID;

  PetscInt dim;
  VecGetSize(state, &dim);
  dim = (int) sqrt(dim/2.0);  // dim = N with \rho \in C^{N\times N}

  switch (objective_type) {

    case ObjectiveType::JFROBENIUS:

      if (target_type == TargetType::GATE || target_type == TargetType::FROMFILE ) {
        FrobeniusDistance_diff(state, statebar, Jbar/ 2.0);
      } else {
        assert(target_type == TargetType::PURE);         
        // Derivative of J = 1/2||x||^2 is xbar += x * Jbar, where x = rho(t) - E_mm
        VecAXPY(statebar, Jbar, state);
        // now substract 1.0*Jbar from m-th diagonal element
        diagID = getIndexReal(getVecID(purestateID,purestateID,dim));
        VecGetOwnershipRange(state, &ilo, &ihi);
        if (ilo <= diagID && diagID < ihi) VecSetValue(statebar, diagID, -1.0*Jbar, ADD_VALUES);
      }
      break; // case JFROBENIUS

    case ObjectiveType::JHS:
      if (target_type == TargetType::GATE || target_type == TargetType::FROMFILE ) {
          HilbertSchmidtOverlap_diff(state, statebar, -1.0 * Jbar, true);
      } else {
        assert(target_type == TargetType::PURE);         
        diagID = getIndexReal(getVecID(purestateID,purestateID,dim));
        val = -1. * Jbar;
        VecGetOwnershipRange(state, &ilo, &ihi);
        if (ilo <= diagID && diagID < ihi) VecSetValue(statebar, diagID, val, ADD_VALUES);
      }
    break;

    case ObjectiveType::JMEASURE:
      if (target_type == TargetType::GATE || target_type == TargetType::FROMFILE ) {
        printf("ERROR: Check settings for optim_target and optim_objective.\n");
        exit(1);
      } else {
        assert(target_type == TargetType::PURE);         
        // iterate over diagonal elements 
        for (int i=0; i<dim; i++){
          lambdai = fabs(i - purestateID);
          diagID = getIndexReal(getVecID(i,i,dim));
          val = lambdai * Jbar;
          VecGetOwnershipRange(state, &ilo, &ihi);
          if (ilo <= diagID && diagID < ihi) VecSetValue(statebar, diagID, val, ADD_VALUES);
        }
      }
    break;
  }
  VecAssemblyBegin(statebar); VecAssemblyEnd(statebar);
}


double OptimTarget::evalFidelity(const Vec state){
  PetscInt dim;
  VecGetSize(state, &dim);
  dim = (int) sqrt(dim/2.0);  // dim = N with \rho \in C^{N\times N}

  PetscInt vecID, ihi, ilo;
  double rho_mm;

  /* Evaluate the Fidelity = Tr(targetstate^\dagger \rho) */
  double fidel = 0.0;
  if (target_type == TargetType::PURE) {
    // if Pure target, then fidelity = rho(T)_mm
      vecID = getIndexReal(getVecID(purestateID, purestateID, dim));
      VecGetOwnershipRange(state, &ilo, &ihi);
      rho_mm = 0.0;
      if (ilo <= vecID && vecID < ihi) VecGetValues(state, 1, &vecID, &rho_mm); // local!
      // Communicate over all petsc processors.
      MPI_Allreduce(&rho_mm, &fidel, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
  } else {
    assert(target_type == TargetType::FROMFILE || target_type == TargetType::GATE);
    fidel = HilbertSchmidtOverlap(state, true);   // scale by purity.
  }

  return fidel;

}