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
  purity_rho0 = 1.0;
 

  /* Allocate target state, if it is read from file, of if target is a gate transformation VrhoV. If pure target, only store the ID. */
  if (target_type == TargetType::GATE || target_type == TargetType::FROMFILE) {
    VecCreate(PETSC_COMM_WORLD, &targetstate); 
    VecSetSizes(targetstate,PETSC_DECIDE, 2*dim);   // input dim is either N^2 (lindblad eq) or N (schroedinger eq)
    VecSetFromOptions(targetstate);
  }

  /* Read the target state from file into vec */
  if (target_type == TargetType::FROMFILE) {
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

void OptimTarget::HilbertSchmidtOverlap(const Vec state, const bool scalebypurity, double* HS_re_ptr, double* HS_im_ptr ){
  /* Lindblas solver: Tr(state * target^\dagger) = vec(target)^dagger * vec(state), will be real!
   * Schroedinger:    Tr(state * target^\dagger) = target^\dag * state, will be complex!*/
  double HS_re = 0.0;
  double HS_im = 0.0;

  /* Simplify computation if the target is PURE, i.e. target = e_m or e_m * e_m^\dag */
  /* Tr(...) = phi_m if Schroedinger, or \rho_mm if Lindblad */
  if (target_type == TargetType::PURE){
    PetscInt ilo, ihi;
    VecGetOwnershipRange(state, &ilo, &ihi);

    int idm = purestateID;
    if (lindbladtype != LindbladType::NONE) idm = getVecID(purestateID, purestateID, (int)sqrt(dim));
    int idm_re = getIndexReal(idm);
    int idm_im = getIndexImag(idm);
    if (ilo <= idm_re && idm_re < ihi) VecGetValues(state, 1, &idm_re, &HS_re); // local!
    if (ilo <= idm_im && idm_im < ihi) VecGetValues(state, 1, &idm_im, &HS_im); // local! Should be 0.0 if Lindblad!
    if (lindbladtype != LindbladType::NONE) assert(fabs(HS_im) <= 1e-14);

    // Communicate over all petsc processors.
    double myre = HS_re;
    double myim = HS_im;
    MPI_Allreduce(&myre, &HS_re, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    MPI_Allreduce(&myim, &HS_im, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

  } else { // Target is not of the form e_m (schroedinger) or e_m e_m^\dagger (lindblad).

    if (lindbladtype != LindbladType::NONE) // Lindblad solver. HS overlap is real!
      VecTDot(targetstate, state, &HS_re);  
    else {  // Schroedinger solver. target^\dagger * state
      const PetscScalar* target_ptr;
      const PetscScalar* state_ptr;
      VecGetArrayRead(targetstate, &target_ptr); // these are local vectors
      VecGetArrayRead(state, &state_ptr);
      int ilo, ihi;
      VecGetOwnershipRange(state, &ilo, &ihi);
      for (int i=0; i<dim; i++){
        int ia = getIndexReal(i);
        int ib = getIndexImag(i);
        if (ilo <= ia && ia < ihi) {
          int idre = ia - ilo;
          int idim = ib - ilo;
          HS_re +=  target_ptr[idre]*state_ptr[idre] + target_ptr[idim]*state_ptr[idim];
          HS_im += -target_ptr[idim]*state_ptr[idre] + target_ptr[idre]*state_ptr[idim];
        }
      } 
      VecRestoreArrayRead(targetstate, &target_ptr);
      VecRestoreArrayRead(state, &state_ptr);
      // The above computation was local, so have to sum up here.
      double re=HS_re;
      double im=HS_im;
      MPI_Allreduce(&re, &HS_re, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
      MPI_Allreduce(&im, &HS_im, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
    }
  }

  // scale by purity Tr(rho(0)^2). 
  if (scalebypurity){ 
    HS_re = HS_re / purity_rho0;
  }

  // return
  *HS_re_ptr = HS_re;
  *HS_im_ptr = HS_im;
}

void OptimTarget::HilbertSchmidtOverlap_diff(const Vec state, Vec statebar, bool scalebypurity, const double HS_re_bar, const double HS_im_bar){

  double scale = 1.0;
  if (scalebypurity){ 
    scale = 1./purity_rho0;
  }

  // Simplified computation if target is pure 
  if (target_type == TargetType::PURE){
    PetscInt ilo, ihi;
    VecGetOwnershipRange(state, &ilo, &ihi);
    int idm = purestateID;
    if (lindbladtype != LindbladType::NONE) idm = getVecID(purestateID, purestateID, (int)sqrt(dim));
    int idm_re = getIndexReal(idm);
    int idm_im = getIndexImag(idm);
    if (ilo <= idm_re && idm_re < ihi) VecSetValue(statebar, idm_re, HS_re_bar*scale, ADD_VALUES);
    if (ilo <= idm_im && idm_im < ihi) VecSetValue(statebar, idm_im, HS_im_bar, ADD_VALUES);

  } else { // Target is not of the form e_m or e_m*e_m^\dagger 

    if (lindbladtype != LindbladType::NONE)
      VecAXPY(statebar, HS_re_bar*scale, targetstate);
    else {
      const PetscScalar* target_ptr;
      PetscScalar* statebar_ptr;
      VecGetArrayRead(targetstate, &target_ptr); 
      VecGetArray(statebar, &statebar_ptr);
      int ilo, ihi;
      VecGetOwnershipRange(state, &ilo, &ihi);
      for (int i=0; i<dim; i++){
        int ia = getIndexReal(i);
        int ib = getIndexImag(i);
        if (ilo <= ia && ia < ihi) {
          int idre = ia - ilo;
          int idim = ib - ilo;
          statebar_ptr[idre] += target_ptr[idre] * HS_re_bar*scale  - target_ptr[idim] * HS_im_bar;
          statebar_ptr[idim] += target_ptr[idim] * HS_re_bar*scale  + target_ptr[idre] * HS_im_bar;
        }
      }
      VecRestoreArrayRead(targetstate, &target_ptr);
      VecRestoreArray(statebar, &statebar_ptr);
    }
  }
}


void OptimTarget::prepare(const Vec rho_t0){
  // If gate optimization, apply the gate and store targetstate for later use. Else, do nothing.
  if (target_type == TargetType::GATE) targetgate->applyGate(rho_t0, targetstate);

  /* Compute and store the purity of rho(0), Tr(rho(0)^2), so that it can be used by JTrace (HS overlap) */
  VecNorm(rho_t0, NORM_2, &purity_rho0);
  purity_rho0 = purity_rho0 * purity_rho0;
}



void OptimTarget::evalJ(const Vec state, double* J_re_ptr, double* J_im_ptr){
  double J_re = 0.0;
  double J_im = 0.0;
  PetscInt diagID, diagID_re, diagID_im;
  double sum, mine, rhoii, rhoii_re, rhoii_im, lambdai, norm;
  PetscInt ilo, ihi;
  int dimsq;

  switch(objective_type) {

    /* J_Frob = 1/2 * || rho_target - rho(T)||^2_F  */
    case ObjectiveType::JFROBENIUS:

      if (target_type == TargetType::GATE || target_type == TargetType::FROMFILE ) {
        // target state is already set. Either \rho_target = Vrho(0)V^\dagger or read from file. Just eval norm.
        J_re = FrobeniusDistance(state) / 2.0;
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
        J_re = pow(norm, 2.0) / 2.0;
        if (ilo <= diagID && diagID < ihi) VecSetValue(state, diagID, +1.0, ADD_VALUES); // restore original state!
        VecAssemblyBegin(state); VecAssemblyEnd(state);
      }
      break;  // case Frobenius

    /* J_Trace:  1 / purity * Tr(state * target^\dagger)  =  HilbertSchmidtOverlap(target, state) is real if Lindblad, and complex if Schroedinger! */
    case ObjectiveType::JTRACE:

      HilbertSchmidtOverlap(state, true, &J_re, &J_im); // is real if Lindblad solver. 
      break; // case J_Trace

    /* J_Measure = Tr(O_m rho(T)) = \sum_i |i-m| rho_ii(T) if Lindblad and \sum_i |i-m| |phi_i(T)|^2  if Schroedinger */
    case ObjectiveType::JMEASURE:
      // Sanity check
      if (target_type != TargetType::PURE) {
        printf("Error: Wrong setting for objective function. Jmeasure can only be used for 'pure' targets.\n");
        exit(1);
      }

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
      MPI_Allreduce(&sum, &J_re, 1, MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);
      break; // case J_MEASURE
  }

  // return
  *J_re_ptr = J_re;
  *J_im_ptr = J_im;
}


void OptimTarget::evalJ_diff(const Vec state, Vec statebar, const double J_re_bar, const double J_im_bar){
  PetscInt ilo, ihi;
  double lambdai, val, val_re, val_im, rhoii_re, rhoii_im;
  PetscInt diagID, diagID_re, diagID_im, dimsq;

  switch (objective_type) {

    case ObjectiveType::JFROBENIUS:

      if (target_type == TargetType::GATE || target_type == TargetType::FROMFILE ) {
        FrobeniusDistance_diff(state, statebar, J_re_bar/ 2.0);
      } else {
        assert(target_type == TargetType::PURE);         
        // Derivative of J = 1/2||x||^2 is xbar += x * Jbar, where x = rho(t) - E_mm
        VecAXPY(statebar, J_re_bar, state);
        // now substract 1.0*Jbar from m-th diagonal element
        if (lindbladtype != LindbladType::NONE) diagID = getIndexReal(getVecID(purestateID,purestateID,(int)sqrt(dim)));
        else diagID = getIndexReal(purestateID);
        VecGetOwnershipRange(state, &ilo, &ihi);
        if (ilo <= diagID && diagID < ihi) VecSetValue(statebar, diagID, -1.0*J_re_bar, ADD_VALUES);
      }
      break; // case JFROBENIUS

    case ObjectiveType::JTRACE:
      HilbertSchmidtOverlap_diff(state, statebar, true, J_re_bar, J_im_bar);
    break;

    case ObjectiveType::JMEASURE:
      assert(target_type == TargetType::PURE);         

      if (lindbladtype != LindbladType::NONE) dimsq = (int)sqrt(dim); // Lindblad solver: dim = N^2
      else dimsq = dim;   // Schroedinger solver: dim = N

      // iterate over diagonal elements 
      for (int i=0; i<dimsq; i++){
        lambdai = fabs(i - purestateID);
        VecGetOwnershipRange(state, &ilo, &ihi);
        if (lindbladtype != LindbladType::NONE) {
          diagID = getIndexReal(getVecID(i,i,dimsq));
          val = lambdai * J_re_bar;
          if (ilo <= diagID && diagID < ihi) VecSetValue(statebar, diagID, val, ADD_VALUES);
        } else {
          diagID_re = getIndexReal(i);
          diagID_im = getIndexImag(i);
          rhoii_re = 0.0;
          rhoii_im = 0.0;
          if (ilo <= diagID_re && diagID_re < ihi) VecGetValues(state, 1, &diagID_re, &rhoii_re);
          if (ilo <= diagID_im && diagID_im < ihi) VecGetValues(state, 1, &diagID_im, &rhoii_im);
          if (ilo <= diagID_re && diagID_re < ihi) VecSetValue(statebar, diagID_re, 2.*J_re_bar*lambdai*rhoii_re, ADD_VALUES);
          if (ilo <= diagID_im && diagID_im < ihi) VecSetValue(statebar, diagID_im, 2.*J_re_bar*lambdai*rhoii_im, ADD_VALUES);
        }
      }
    break;
  }
  VecAssemblyBegin(statebar); VecAssemblyEnd(statebar);
}

double OptimTarget::finalizeJ(const double obj_cost_re, const double obj_cost_im) {
  double obj_cost = 0.0;

  if (objective_type == ObjectiveType::JTRACE) {
    if (lindbladtype == LindbladType::NONE) {
      obj_cost = 1.0 - (pow(obj_cost_re,2.0) + pow(obj_cost_im, 2.0));
    } else {
      obj_cost = 1.0 - obj_cost_re;
    }
  } else {
    obj_cost = obj_cost_re;
    assert(obj_cost_im <= 1e-14);
  }

  return obj_cost;
}


void OptimTarget::finalizeJ_diff(const double obj_cost_re, const double obj_cost_im, double* obj_cost_re_bar, double* obj_cost_im_bar){

  if (objective_type == ObjectiveType::JTRACE) {
    if (lindbladtype == LindbladType::NONE) {
      // obj_cost = 1.0 - (pow(obj_cost_re,2.0) + pow(obj_cost_im, 2.0));
      *obj_cost_re_bar = -2.*obj_cost_re;
      *obj_cost_im_bar = -2.*obj_cost_im;
    } else {
      *obj_cost_re_bar = -1.0;
      *obj_cost_im_bar = 0.0;
    }
  } else {
    *obj_cost_re_bar = 1.0;
    *obj_cost_im_bar = 0.0;
  }
}
