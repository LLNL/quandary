#include "optimtarget.hpp"


OptimTarget::OptimTarget(int dim_, int purestateID_, TargetType target_type_, ObjectiveType objective_type_, Gate* targetgate_, std::string target_filename_, LindbladType lindbladtype_){

  // initialize
  dim = dim_;
  target_type = target_type_;
  objective_type = objective_type_;
  targetgate = targetgate_;
  purestateID = purestateID_;
  target_filename = target_filename_;
  lindbladtype = lindbladtype_;

  /* Allocate target state, if it is read from file, of if target is a gate transformation VrhoV */
  if (target_type == TargetType::GATE || target_type == TargetType::FROMFILE) {
    VecCreate(PETSC_COMM_WORLD, &targetstate); 
    VecSetSizes(targetstate,PETSC_DECIDE, 2*dim);   // input dim is either N^2 (lindblad eq) or N (schroedinger eq)
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
    for (int i = 0; i < dim; i++) { 
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
  /* Lindblas solver: Tr(targetstate*state) = vec(targetstate)^dagger vec(state), will be real!
   * Schroedinger:    | targetstate^\dagger state |^2  */
  double J = 0.0;
  if (lindbladtype != LindbladType::NONE) // Lindblad solver. Tr(target*state).
    VecTDot(targetstate, state, &J);
  else {  // Schroedinger solver. |target^dagger state|^2
    const PetscScalar* target_ptr;
    const PetscScalar* state_ptr;
    VecGetArrayRead(targetstate, &target_ptr); // these should be local vectors.
    VecGetArrayRead(state, &state_ptr);
    int ilo, ihi;
    VecGetOwnershipRange(state, &ilo, &ihi);
    double u=0.0;
    double v=0.0;
    for (int i=0; i<dim; i++){
      int ia = getIndexReal(i);
      int ib = getIndexImag(i);
      if (ilo <= ia && ia < ihi) {
        int idre = ia - ilo;
        int idim = ib - ilo;
        u +=  target_ptr[idre]*state_ptr[idre] + target_ptr[idim]*state_ptr[idim];
        v += -target_ptr[idim]*state_ptr[idre] + target_ptr[idre]*state_ptr[idim];
      }
    } 
    VecRestoreArrayRead(targetstate, &target_ptr);
    VecRestoreArrayRead(state, &state_ptr);
    // The above computation was local, so have to sum up here.
    double Jre=0.0;
    double Jim=0.0;
    MPI_Allreduce(&u, &Jre, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    MPI_Allreduce(&v, &Jim, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    J = pow(Jre, 2.0) + pow(Jim, 2.0);
  }

  // scale by purity Tr(targetstate^2) = || vec(targetstate)||^2_2. Will be 1.0 in Schroedinger case.
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
  if (lindbladtype != LindbladType::NONE)
    VecAXPY(statebar, Jbar/scale, targetstate);
  else {
    const PetscScalar* target_ptr;
    const PetscScalar* state_ptr;
    PetscScalar* statebar_ptr;
    VecGetArrayRead(targetstate, &target_ptr); 
    VecGetArrayRead(state, &state_ptr);
    VecGetArray(statebar, &statebar_ptr);
    int ilo, ihi;
    VecGetOwnershipRange(state, &ilo, &ihi);
    // First recompute Jre, Jim
    double u = 0.0;
    double v = 0.0;
    for (int i=0; i<dim; i++){
      int ia = getIndexReal(i);
      int ib = getIndexImag(i);
      if (ilo <= ia && ia < ihi) {
        int idre = ia - ilo;
        int idim = ib - ilo;
        u +=  target_ptr[idre]*state_ptr[idre] + target_ptr[idim]*state_ptr[idim];
        v += -target_ptr[idim]*state_ptr[idre] + target_ptr[idre]*state_ptr[idim];
      }
    } 
    double Jre=0.0;
    double Jim=0.0;
    MPI_Allreduce(&u, &Jre, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    MPI_Allreduce(&v, &Jim, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    // Then update adjoint variable 
    for (int i=0; i<dim; i++){
      int ia = getIndexReal(i);
      int ib = getIndexImag(i);
      if (ilo <= ia && ia < ihi) {
        int idre = ia - ilo;
        int idim = ib - ilo;
        statebar_ptr[idre] += 2.0*Jbar/scale * ( target_ptr[idre] * Jre  - target_ptr[idim] * Jim );
        statebar_ptr[idim] += 2.0*Jbar/scale * ( target_ptr[idim] * Jre  + target_ptr[idre] * Jim );
      }
    }
    VecRestoreArrayRead(targetstate, &target_ptr);
    VecRestoreArrayRead(state, &state_ptr);
    VecRestoreArray(statebar, &statebar_ptr);
  }
}


void OptimTarget::prepare(const Vec rho_t0){
  // If gate optimization, apply the gate and store targetstate for later use. Else, do nothing.
  if (target_type == TargetType::GATE) targetgate->applyGate(rho_t0, targetstate);
}



double OptimTarget::evalJ(const Vec state){
  double objective = 0.0;
  PetscInt diagID, diagID_re, diagID_im;
  double sum, mine, rhoii, rhoii_re, rhoii_im, lambdai, norm;
  PetscInt ilo, ihi;
  int dimsq;

  switch(objective_type) {

    /* J_Frob = 1/2 * || rho_target - rho(T)||^2_F  */
    case ObjectiveType::JFROBENIUS:

      if (target_type == TargetType::GATE || target_type == TargetType::FROMFILE ) {
        // target state is already set. Either \rho_target = Vrho(0)V^\dagger or read from file. Just eval norm.
        objective = FrobeniusDistance(state) / 2.0;
      } 
      else {  // target = e_me_m^\dagger ( or target = e_m for Schroedinger)
        assert(target_type == TargetType::PURE);
        // substract 1.0 from m-th diagonal element then take the vector norm 
        if (lindbladtype != LindbladType::NONE) diagID = getIndexReal(getVecID(purestateID,purestateID,(int)sqrt(dim)));
        else diagID = getIndexReal(purestateID);
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

    /* J_Trace = 1 - 1/purity * Tr(rho_target^\dagger * rho(T)) */
    case ObjectiveType::JTRACE:

      if (target_type == TargetType::GATE || target_type == TargetType::FROMFILE ) {
        // target state is already set. Either \rho_target = Vrho(0)V^\dagger or read from file. Just eval Trace.
        objective = 1.0 - HilbertSchmidtOverlap(state, true);
      }
      else { // target = e_m e_m^\dagger
        /* -> J_Trace = 1 - Tr(e_m e_m^\dagger \rho(T)) = 1 - rho_mm(T) or 1 - |phi_m|^2 */
        assert(target_type == TargetType::PURE);
        VecGetOwnershipRange(state, &ilo, &ihi);
        if (lindbladtype != LindbladType::NONE) { // Lindblad
          diagID = getIndexReal(getVecID(purestateID,purestateID,(int)sqrt(dim)));
          rhoii = 0.0;
          if (ilo <= diagID && diagID < ihi) VecGetValues(state, 1, &diagID, &rhoii);
          mine = 1. - rhoii;
        } else { // Schroedinger 
          diagID_re = getIndexReal(purestateID);
          diagID_im = getIndexImag(purestateID);
          rhoii_re = 0.0;
          rhoii_im = 0.0;
          if (ilo <= diagID_re && diagID_re < ihi) VecGetValues(state, 1, &diagID_re, &rhoii_re);
          if (ilo <= diagID_im && diagID_im < ihi) VecGetValues(state, 1, &diagID_im, &rhoii_im);
          mine = 1. - ( pow(rhoii_re, 2.0) + pow(rhoii_im, 2.0));
        }
        MPI_Allreduce(&mine, &objective, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
      }
      break; // case J_Trace

    /* J_T = Tr(O_m rho(T)) = \sum_i |i-m| rho_ii(T) */
    case ObjectiveType::JMEASURE:
      assert(target_type == TargetType::PURE);

      if (lindbladtype != LindbladType::NONE) dimsq = (int)sqrt(dim); // Lindblad solver: dim = N^2
      else dimsq = dim;   // Schroedinger solver: dim = N

      VecGetOwnershipRange(state, &ilo, &ihi);
      // iterate over diagonal elements 
      sum = 0.0;
      for (int i=0; i<dimsq; i++){
        if (lindbladtype != LindbladType::NONE) {
          diagID = getIndexReal(getVecID(i,i,dimsq));
          rhoii = 0.0;
          if (ilo <= diagID && diagID < ihi) VecGetValues(state, 1, &diagID, &rhoii);
        } else  {
          diagID_re = getIndexReal(i);
          diagID_im = getIndexImag(i);
          rhoii_re = 0.0;
          rhoii_im = 0.0;
          if (ilo <= diagID_re && diagID_re < ihi) VecGetValues(state, 1, &diagID_re, &rhoii_re);
          if (ilo <= diagID_im && diagID_im < ihi) VecGetValues(state, 1, &diagID_im, &rhoii_im);
          rhoii = pow(rhoii_re, 2.0) + pow(rhoii_im, 2.0);
        }
        lambdai = fabs(i - purestateID);
        sum += lambdai * rhoii;
      }
      MPI_Allreduce(&sum, &objective, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
      break; // case J_MEASURE
  }

  return objective;
}


void OptimTarget::evalJ_diff(const Vec state, Vec statebar, const double Jbar){
  PetscInt ilo, ihi;
  double lambdai, val, val_re, val_im, rhoii_re, rhoii_im;
  PetscInt diagID, diagID_re, diagID_im, dimsq;

  switch (objective_type) {

    case ObjectiveType::JFROBENIUS:

      if (target_type == TargetType::GATE || target_type == TargetType::FROMFILE ) {
        FrobeniusDistance_diff(state, statebar, Jbar/ 2.0);
      } else {
        assert(target_type == TargetType::PURE);         
        // Derivative of J = 1/2||x||^2 is xbar += x * Jbar, where x = rho(t) - E_mm
        VecAXPY(statebar, Jbar, state);
        // now substract 1.0*Jbar from m-th diagonal element
        if (lindbladtype != LindbladType::NONE) diagID = getIndexReal(getVecID(purestateID,purestateID,(int)sqrt(dim)));
        else diagID = getIndexReal(purestateID);
        VecGetOwnershipRange(state, &ilo, &ihi);
        if (ilo <= diagID && diagID < ihi) VecSetValue(statebar, diagID, -1.0*Jbar, ADD_VALUES);
      }
      break; // case JFROBENIUS

    case ObjectiveType::JTRACE:
      if (target_type == TargetType::GATE || target_type == TargetType::FROMFILE ) {
          HilbertSchmidtOverlap_diff(state, statebar, -1.0 * Jbar, true);
      } else {
        assert(target_type == TargetType::PURE);         
        VecGetOwnershipRange(state, &ilo, &ihi);
        if (lindbladtype != LindbladType::NONE) { // Lindblad
          diagID = getIndexReal(getVecID(purestateID,purestateID,(int)sqrt(dim)));
          val = -1. * Jbar;
          if (ilo <= diagID && diagID < ihi) VecSetValue(statebar, diagID, val, ADD_VALUES);
        } else { // Schroedinger
          diagID_re = getIndexReal(purestateID);
          diagID_im = getIndexImag(purestateID);
          rhoii_re = 0.0;
          rhoii_im = 0.0;
          if (ilo <= diagID_re && diagID_re < ihi) VecGetValues(state, 1, &diagID_re, &rhoii_re);
          if (ilo <= diagID_im && diagID_im < ihi) VecGetValues(state, 1, &diagID_im, &rhoii_im);
          if (ilo <= diagID_re && diagID_re < ihi) VecSetValue(statebar, diagID_re, -2.*Jbar*rhoii_re, ADD_VALUES);
          if (ilo <= diagID_im && diagID_im < ihi) VecSetValue(statebar, diagID_im, -2.*Jbar*rhoii_im, ADD_VALUES);
        }
      }
    break;

    case ObjectiveType::JMEASURE:
      assert(target_type == TargetType::PURE);         

      if (lindbladtype != LindbladType::NONE) dimsq = (int)sqrt(dim); // Lindblad solver: dim = N^2
      else dimsq = dim;   // Schroedinger solver: dim = N

      // iterate over diagonal elements 
      for (int i=0; i<dimsq; i++){
        lambdai = fabs(i - purestateID);
        if (lindbladtype != LindbladType::NONE) {
          diagID = getIndexReal(getVecID(i,i,dimsq));
          val = lambdai * Jbar;
          VecGetOwnershipRange(state, &ilo, &ihi);
          if (ilo <= diagID && diagID < ihi) VecSetValue(statebar, diagID, val, ADD_VALUES);
        } else {
          diagID_re = getIndexReal(i);
          diagID_im = getIndexImag(i);
          rhoii_re = 0.0;
          rhoii_im = 0.0;
          if (ilo <= diagID_re && diagID_re < ihi) VecGetValues(state, 1, &diagID_re, &rhoii_re);
          if (ilo <= diagID_im && diagID_im < ihi) VecGetValues(state, 1, &diagID_im, &rhoii_im);
          if (ilo <= diagID_re && diagID_re < ihi) VecSetValue(statebar, diagID_re, 2.*Jbar*lambdai*rhoii_re, ADD_VALUES);
          if (ilo <= diagID_im && diagID_im < ihi) VecSetValue(statebar, diagID_im, 2.*Jbar*lambdai*rhoii_im, ADD_VALUES);
        }
      }
    break;
  }
  VecAssemblyBegin(statebar); VecAssemblyEnd(statebar);
}


double OptimTarget::evalFidelity(const Vec state){

  PetscInt vecID_re, vecID_im, ihi, ilo;
  double rho_mm_re, rho_mm_im;

  /* Evaluate the Fidelity  
   * Lindblad:     Fidelity = Tr(targetstate^\dagger \rho) 
   * Schroedinger: Fidelity = |phi_target^\dagger phi|^2  */
  double fidel = 0.0;
  double myfidel = 0.0;
  if (target_type == TargetType::PURE) {
  // if Pure target, then fidelity = rho(T)_mm, or |phi_m|^2
      VecGetOwnershipRange(state, &ilo, &ihi);
      rho_mm_re = 0.0;
      rho_mm_im = 0.0;

      if (lindbladtype != LindbladType::NONE) { // Lindblad solver
        vecID_re = getIndexReal(getVecID(purestateID, purestateID, (int)sqrt(dim)));
        if (ilo <= vecID_re && vecID_re < ihi) VecGetValues(state, 1, &vecID_re, &rho_mm_re); // local!
        myfidel = rho_mm_re; // rho_mm is real
      }
      else { // Schroedinger solver
        vecID_re = getIndexReal(purestateID);
        vecID_im = getIndexImag(purestateID);
        if (ilo <= vecID_re && vecID_re < ihi) VecGetValues(state, 1, &vecID_re, &rho_mm_re); // local!
        if (ilo <= vecID_im && vecID_im < ihi) VecGetValues(state, 1, &vecID_im, &rho_mm_im); // local!
        myfidel = pow(rho_mm_re,2.0) + pow(rho_mm_im, 2.0); // |phi_m|^2
      }
      // Communicate over all petsc processors.
      MPI_Allreduce(&myfidel, &fidel, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
  } else {
    assert(target_type == TargetType::FROMFILE || target_type == TargetType::GATE);
    fidel = HilbertSchmidtOverlap(state, true);   // scale by purity.
  }

  return fidel;

}