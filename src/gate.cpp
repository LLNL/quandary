#include "gate.hpp"

Gate::Gate(){
  dim_ess = 0;
  dim_rho = 0;
}

Gate::Gate(std::vector<int> nlevels_, std::vector<int> nessential_){

  MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank_petsc);

  nessential = nessential_;
  nlevels = nlevels_;

  /* Dimension of gate = \prod_j nessential_j */
  dim_ess = 1;
  for (int i=0; i<nessential.size(); i++) {
    dim_ess *= nessential[i];
  }

  /* Dimension of system matrix rho */
  dim_rho = 1;
  for (int i=0; i<nlevels.size(); i++) {
    dim_rho *= nlevels[i];
  }

  /* Allocate input Gate in essential level dimension, sequential matrix (real and imaginary parts), copied on all processors */
  MatCreateSeqDense(PETSC_COMM_SELF, dim_ess, dim_ess, NULL, &V_re);
  MatCreateSeqDense(PETSC_COMM_SELF, dim_ess, dim_ess, NULL, &V_im);
  MatSetUp(V_re);
  MatSetUp(V_im);
  MatAssemblyBegin(V_re, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(V_im, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(V_re, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(V_im, MAT_FINAL_ASSEMBLY);

  // /* TODO: Test rotation! Allocate real and imaginary rotation */
  // MatCreateVecs(V_re, &rotA, NULL);
  // MatCreateVecs(V_im, &rotB, NULL);
  // // default: rotA = I, rotB = 0
  // int nrot;
  // VecGetSize(rotA, &nrot);
  // for (int irow =0; irow<nrot; irow++){
  //   VecSetValue(rotA, irow, 1.0, INSERT_VALUES);
  // }
  // VecAssemblyBegin(rotA); VecAssemblyEnd(rotA);
  // VecAssemblyBegin(rotB); VecAssemblyEnd(rotB);

  /* Allocate vectorized Gate in full dimensions G = VxV, where V is the full-dimension gate (inserting zero rows and colums for all non-essential levels) */ 
  // parallel matrix, essential levels dimension TODO: PREALLOCATE!
  MatCreate(PETSC_COMM_WORLD, &VxV_re);
  MatCreate(PETSC_COMM_WORLD, &VxV_im);
  MatSetSizes(VxV_re, PETSC_DECIDE, PETSC_DECIDE, dim_rho*dim_rho, dim_rho*dim_rho);
  MatSetSizes(VxV_im, PETSC_DECIDE, PETSC_DECIDE, dim_rho*dim_rho, dim_rho*dim_rho);
  MatSetUp(VxV_re);
  MatSetUp(VxV_im);
  MatAssemblyBegin(VxV_re, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(VxV_re, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(VxV_im, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(VxV_im, MAT_FINAL_ASSEMBLY);

  /* Allocate auxiliare vectors */
  MatCreateVecs(VxV_re, &x, NULL);
}

Gate::~Gate(){
  if (dim_rho == 0) return;
  MatDestroy(&VxV_re);
  MatDestroy(&VxV_im);
  MatDestroy(&V_re);
  MatDestroy(&V_im);
  VecDestroy(&x);
}


void Gate::assembleGateRotation2x2(double T, std::vector<double>gate_rot_freq){

  /* Get frequencies */
  double f0 = 0.0;
  double f1 = 0.0;
  if (gate_rot_freq.size() >= 2) {
    f0 = gate_rot_freq[0];
    f1 = gate_rot_freq[1];
  }

  /* Assemble diagonal rotation matrices, stored as vectors */
  // RotA = Real(R^1\otimes R^2),  RotB = Imag(R^1\otimes R^2)
  VecSetValue(rotA, 0, 1.0, INSERT_VALUES);
  VecSetValue(rotB, 0, 0.0, INSERT_VALUES);
  VecSetValue(rotA, 1, cos(2.*M_PI*f1*T), INSERT_VALUES);
  VecSetValue(rotB, 1, sin(2.*M_PI*f1*T), INSERT_VALUES);
  VecSetValue(rotA, 2, cos(2.*M_PI*f0*T), INSERT_VALUES);
  VecSetValue(rotB, 2, sin(2.*M_PI*f0*T), INSERT_VALUES);
  VecSetValue(rotA, 3, cos(2.*M_PI*(f0+f1)*T), INSERT_VALUES);
  VecSetValue(rotB, 3, sin(2.*M_PI*(f0+f1)*T), INSERT_VALUES);
  VecAssemblyBegin(rotA); VecAssemblyEnd(rotA);
  VecAssemblyBegin(rotB); VecAssemblyEnd(rotB);
}


void Gate::assembleGateRotation1x2(double T, std::vector<double>gate_rot_freq){
  // RotA = Real(R^1)
  // RotB = Imag(R^1)
  // diagonal matrix, stored as vectors
  assert(gate_rot_freq.size() >= 1);
  VecSetValue(rotA, 0, 1.0, INSERT_VALUES);
  VecSetValue(rotB, 0, 0.0, INSERT_VALUES);
  VecSetValue(rotA, 1, cos(2.*M_PI*gate_rot_freq[0]*T), INSERT_VALUES);
  VecSetValue(rotB, 1, sin(2.*M_PI*gate_rot_freq[0]*T), INSERT_VALUES);
  VecAssemblyBegin(rotA); VecAssemblyEnd(rotA);
  VecAssemblyBegin(rotB); VecAssemblyEnd(rotB);

}

void Gate::assembleGate(){

  int ilow, iupp;
  double val;
  MatGetOwnershipRange(VxV_re, &ilow, &iupp);

  /* Assemble vectorized gate V\kron V from essential level gate input. */
  /* Each element in V\kron V is a product V(r1,c1)*V(r2,c2), for rows and columns r1,r2,c1,c2. */
  // iterate over local rows in VxV_re, VxV_im
  for (int i = ilow; i<iupp; i++) {
    // Rows for the first and second factors
    const int r1 = (int) i / dim_rho;
    const int r2 = i % dim_rho;
    // iterate over columns in G
    for (int j=0; j<dim_rho*dim_rho; j++){
      // Columns for the first and second factors
      const int c1 = (int) j / dim_rho;
      const int c2 = j % dim_rho;
      // Map rows and columns from full to essential dimension
      int r1e = mapFullToEss(r1, nlevels, nessential);
      int r2e = mapFullToEss(r2, nlevels, nessential);
      int c1e = mapFullToEss(c1, nlevels, nessential);
      int c2e = mapFullToEss(c2, nlevels, nessential);
      double va1=0.0;
      double va2=0.0;
      double vb1=0.0;
      double vb2=0.0;
      // Get values V(r1,c1), V(r2,c2) from essential-level gate
      MatGetValues(V_re, 1, &r1e, 1, &c1e, &va1); 
      MatGetValues(V_re, 1, &r2e, 1, &c2e, &va2); 
      MatGetValues(V_im, 1, &r1e, 1, &c1e, &vb1); 
      MatGetValues(V_im, 1, &r2e, 1, &c2e, &vb2); 
      // Kronecker product for real and imaginar part
      val = va1*va2 + vb1*vb2;
      if (fabs(val) > 1e-14) MatSetValue(VxV_re, i,j, val, INSERT_VALUES);
      val = va1*vb2 - vb1*va2;
      if (fabs(val) > 1e-14) MatSetValue(VxV_im, i,j, val, INSERT_VALUES);
    }
  }

  MatAssemblyBegin(VxV_re, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(VxV_re, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(VxV_im, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(VxV_im, MAT_FINAL_ASSEMBLY);
}


void Gate::compare_frobenius(const Vec finalstate, const Vec rho0, double& frob){
  frob = 0.0;

  /* Exit, if this is a dummy gate */
  if (dim_rho == 0) {
    return;
  }

  /* First, project full dimension state to essential levels only by zero'ing out rows and columns */
  /* iterate over rows of system matrix, check if it corresponds to an essential level, and if not, set this row and colum to zero */
  int reID, imID;
  int ilow, iupp;
  VecGetOwnershipRange(finalstate, &ilow, &iupp);
  for (int i=0; i<dim_rho; i++) {
    if (!isEssential(i, nlevels, nessential)) {
      for (int j=0; j<dim_rho; j++) {
        // zero out row
        reID = getIndexReal(getVecID(i,j,dim_rho));
        imID = getIndexImag(getVecID(i,j,dim_rho));
        if (ilow <= reID && reID < iupp) VecSetValue(finalstate, reID, 0.0, INSERT_VALUES);
        if (ilow <= imID && imID < iupp) VecSetValue(finalstate, imID, 0.0, INSERT_VALUES);
        // zero out colum
        reID = getIndexReal(getVecID(j,i,dim_rho));
        imID = getIndexImag(getVecID(j,i,dim_rho));
        if (ilow <= reID && reID < iupp) VecSetValue(finalstate, reID, 0.0, INSERT_VALUES);
        if (ilow <= imID && imID < iupp) VecSetValue(finalstate, imID, 0.0, INSERT_VALUES);
      }
    } 
  }
  VecAssemblyBegin(finalstate);
  VecAssemblyEnd(finalstate);

  /* Create vector strides for accessing real and imaginary part of co-located state */
  int dimis = (iupp - ilow)/2;
  IS isu, isv;
  ISCreateStride(PETSC_COMM_WORLD, dimis, ilow, 2, &isu);
  ISCreateStride(PETSC_COMM_WORLD, dimis, ilow+1, 2, &isv);

  /* Get real and imag part of final state and initial state */
  Vec ufinal, vfinal, u0, v0;
  VecGetSubVector(finalstate, isu, &ufinal);
  VecGetSubVector(finalstate, isv, &vfinal);
  VecGetSubVector(rho0, isu, &u0);
  VecGetSubVector(rho0, isv, &v0);

  /* Add real part of frobenius norm || u - VxV_re*u0 + VxV_im*v0 ||^2 */
  double norm;
  MatMult(VxV_re, u0, x);            // x = VxV_re*u0
  VecAYPX(x, -1.0, ufinal);       // x = ufinal - VxV_re*u0 
  MatMultAdd(VxV_im, v0, x, x);      // x = ufinal - VxV_re*u0 + VxV_im*v0
  VecNorm(x, NORM_2, &norm);
  frob = pow(norm,2.0);           // frob = || x ||^2

  /* Add imaginary part of frobenius norm || v - VxV_re*v0 - VxV_im*u0 ||^2 */
  MatMult(VxV_re, v0, x);         // x = VxV_re*v0
  MatMultAdd(VxV_im, u0, x, x);   // x = VxV_re*v0 + VxV_im*u0
  VecAYPX(x, -1.0, vfinal);     // x = vfinal - (VxV_re*v0 + VxV_im*u0)
  VecNorm(x, NORM_2, &norm);
  frob += pow(norm, 2.0);      // frob += ||x||^2

  /* obj = 1/2 * || finalstate - gate*rho(0) ||^2 */
  frob *= 1./2.;

  /* Restore vectors from index set */
  VecRestoreSubVector(finalstate, isu, &ufinal);
  VecRestoreSubVector(finalstate, isv, &vfinal);
  VecRestoreSubVector(rho0, isu, &u0);
  VecRestoreSubVector(rho0, isv, &v0);

  /* Free index strides */
  ISDestroy(&isu);
  ISDestroy(&isv);
}

void Gate::compare_frobenius_diff(const Vec finalstate, const Vec rho0, Vec rho0_bar, const double frob_bar){

  // /* Exit, if this is a dummy gate */
  // if (dim_rho == 0) {
  //   return;
  // }

  // /* Create vector strides for accessing real and imaginary part of co-located x */
  // int ilow, iupp;
  // VecGetOwnershipRange(finalstate, &ilow, &iupp);
  // int dimis = (iupp - ilow)/2;
  // IS isu, isv;
  // ISCreateStride(PETSC_COMM_WORLD, dimis, ilow, 2, &isu);
  // ISCreateStride(PETSC_COMM_WORLD, dimis, ilow+1, 2, &isv);


  // /* Get real and imag part of final state, initial state, and adjoint */
  // Vec ufinal_full, vfinal_full, u0_full, v0_full;
  // VecGetSubVector(finalstate, isu, &ufinal_full);
  // VecGetSubVector(finalstate, isv, &vfinal_full);
  // VecGetSubVector(rho0, isu, &u0_full);
  // VecGetSubVector(rho0, isv, &v0_full);

  // /* Project final and initial states onto essential levels */
  // MatMult(PxP, ufinal_full, ufinal_e);
  // MatMult(PxP, vfinal_full, vfinal_e);
  // MatMult(PxP, u0_full, u0_e);
  // MatMult(PxP, v0_full, v0_e);

  // /* Derivative of 1/2 * J */
  // double dfb = 1./2. * frob_bar;

  // /* Derivative of real part of frobenius norm: 2 * (u - VxV_re*u0 + VxV_im*v0) * dfb */
  // MatMult(VxV_re, u0_e, x_e);            // x = VxV_re*u0
  // VecAYPX(x_e, -1.0, ufinal_e);       // x = ufinal - VxV_re*u0 
  // MatMultAdd(VxV_im, v0_e, x_e, x_e);      // x = ufinal - VxV_re*u0 + VxV_im*v0
  // VecScale(x_e, 2*dfb);             // x = 2*(ufinal - VxV_re*u0 + VxV_im*v0)*dfb
  // // Project essential to full state 
  // MatMultTranspose(PxP, x_e, x_full);
  // // set real part in rho0bar
  // VecISCopy(rho0_bar, isu, SCATTER_FORWARD, x_full);

  // /* Derivative of imaginary part of frobenius norm 2 * (v - VxV_re*v0 - VxV_im*u0) * dfb */
  // MatMult(VxV_re, v0_e, x_e);         // x = VxV_re*v0
  // MatMultAdd(VxV_im, u0_e, x_e, x_e);   // x = VxV_re*v0 + VxV_im*u0
  // VecAYPX(x_e, -1.0, vfinal_e);     // x = vfinal - (VxV_re*v0 + VxV_im*u0)
  // VecScale(x_e, 2*dfb);          // x = 2*(vfinal - (VxV_re*v0 + VxV_im*u0)*dfb
  // // Project essential to full state 
  // MatMultTranspose(PxP, x_e, x_full);
  // VecISCopy(rho0_bar, isv, SCATTER_FORWARD, x_full);  // set imaginary part in rho0bar

  // /* Restore final, initial and adjoint state */
  // VecRestoreSubVector(finalstate, isu, &ufinal_full);
  // VecRestoreSubVector(finalstate, isv, &vfinal_full);
  // VecRestoreSubVector(rho0, isu, &u0_full);
  // VecRestoreSubVector(rho0, isv, &v0_full);

  // /* Free vindex strides */
  // ISDestroy(&isu);
  // ISDestroy(&isv);
}

void Gate::compare_trace(const Vec finalstate, const Vec rho0, double& obj){
  obj = 0.0;

//   /* Exit, if this is a dummy gate */
//   if (dim_rho== 0) {
//     return;
//   }

//   /* Create vector strides for accessing real and imaginary part of co-located x */
//   int ilow, iupp;
//   VecGetOwnershipRange(finalstate, &ilow, &iupp);
//   int dimis = (iupp - ilow)/2;
//   IS isu, isv;
//   ISCreateStride(PETSC_COMM_WORLD, dimis, ilow, 2, &isu);
//   ISCreateStride(PETSC_COMM_WORLD, dimis, ilow+1, 2, &isv);

//   /* Get real and imag part of final state and initial state */
//   Vec ufinal_full, vfinal_full, u0_full, v0_full;
//   VecGetSubVector(finalstate, isu, &ufinal_full);
//   VecGetSubVector(finalstate, isv, &vfinal_full);
//   VecGetSubVector(rho0, isu, &u0_full);
//   VecGetSubVector(rho0, isv, &v0_full);

//   /* Project final and initial states onto essential levels */
//   MatMult(PxP, ufinal_full, ufinal_e);
//   MatMult(PxP, vfinal_full, vfinal_e);
//   MatMult(PxP, u0_full, u0_e);
//   MatMult(PxP, v0_full, v0_e);


//   /* trace overlap: (VxV_re*u0 - VxV_im*v0)^T u + (VxV_re*v0 + VxV_im*u0)^Tv
//               [ + i (VxV_re*u0 - VxV_im*v0)^T v - (VxV_re*v0 + VxV_im*u0)^Tu ]   <- this should be zero!
//   */
//   double dot;
//   double trace = 0.0;

//   // first term: (VxV_re*u0 - VxV_im*v0)^T u
//   MatMult(VxV_im, v0_e, x_e);      
//   VecScale(x_e, -1.0);                // x = - VxV_im*v0
//   MatMultAdd(VxV_re, u0_e, x_e, x_e);  // x = VxV_re*u0 - VxV_im*v0
//   VecTDot(x_e, ufinal_e, &dot);       // dot = (VxV_re*u0 - VxV_im*v0)^T u    
//   trace += dot;
  
//   // second term: (VxV_re*v0 + VxV_im*u0)^Tv
//   MatMult(VxV_im, u0_e, x_e);         // x = VxV_im*u0
//   MatMultAdd(VxV_re, v0_e, x_e, x_e); // x = VxV_re*v0 + VxV_im*u0
//   VecTDot(x_e, vfinal_e, &dot);      // dot = (VxV_re*v0 + VxV_im*u0)^T v    
//   trace += dot;

//   /* Objective J = 1.0 - Trace(...) */
//   obj = 1.0 - trace;
//   // obj = - trace;
 
//   // // Test: compute purity of rho(T): 1/2*Tr(rho^2)
//   // double purity_rhoT = 0.0;
//   // VecNorm(ufinal, NORM_2, &dot);
//   // purity_rhoT += dot*dot;
//   // VecNorm(vfinal, NORM_2, &dot);
//   // purity_rhoT += dot*dot;
//   // // Test: compute constant term  1/2*Tr((Vrho0V^dag)^2)
//   // double purity_VrhoV = 0.0;
//   // MatMult(VxV_im, v0, x);      
//   // VecScale(x, -1.0);           // x = - VxV_im*v0
//   // MatMultAdd(VxV_re, u0, x, x);   // x = VxV_re*u0 - VxV_im*v0
//   // VecNorm(x, NORM_2, &dot);
//   // purity_VrhoV += dot*dot;
//   // MatMult(VxV_im, u0, x);         // x = VxV_im*u0
//   // MatMultAdd(VxV_re, v0, x, x);   // x = VxV_re*v0 + VxV_im*u0
//   // VecNorm(x, NORM_2, &dot);
//   // purity_VrhoV += dot*dot;
//   // double J_dist = purity_rhoT/2. - trace + purity_VrhoV/2.;
//   // printf("J_dist = 1/2 * %f - %f + 1/2 * %f = %1.14e\n", purity_rhoT, trace, purity_VrhoV, J_dist);

//   // // obj = obj + purity_rhoT / 2. - 0.5;

//   /* Restore vectors from index set */
//   VecRestoreSubVector(finalstate, isu, &ufinal_full);
//   VecRestoreSubVector(finalstate, isv, &vfinal_full);
//   VecRestoreSubVector(rho0, isu, &u0_full);
//   VecRestoreSubVector(rho0, isv, &v0_full);

//   /* Free index strides */
//   ISDestroy(&isu);
//   ISDestroy(&isv);

//   // /* Verify trace overlap */
//   // double Jdist = 0.0;
//   // compare_frobenius(finalstate, rho0, Jdist);
//   // test = test + 1. + Jdist;

//   // printf("\n");
//   // printf(" J_T:   %1.14e\n", obj);
//   // printf(" test:  %1.14e\n", test);

}


void Gate::compare_trace_diff(const Vec finalstate, const Vec rho0, Vec rho0_bar, const double obj_bar){

  // /* Exit, if this is a dummy gate */
  // if (dim_rho== 0) {
  //   return;
  // }

  // /* Create vector strides for accessing real and imaginary part of co-located x */
  // int ilow, iupp;
  // VecGetOwnershipRange(finalstate, &ilow, &iupp);
  // int dimis = (iupp - ilow)/2;
  // IS isu, isv;
  // ISCreateStride(PETSC_COMM_WORLD, dimis, ilow, 2, &isu);
  // ISCreateStride(PETSC_COMM_WORLD, dimis, ilow+1, 2, &isv);

  // /* Get real and imag part of final state and initial state */
  // Vec ufinal_full, vfinal_full, u0_full, v0_full;
  // VecGetSubVector(finalstate, isu, &ufinal_full);
  // VecGetSubVector(finalstate, isv, &vfinal_full);
  // VecGetSubVector(rho0, isu, &u0_full);
  // VecGetSubVector(rho0, isv, &v0_full);

  // /* Project final and initial states onto essential levels */
  // MatMult(PxP, ufinal_full, ufinal_e);
  // MatMult(PxP, vfinal_full, vfinal_e);
  // MatMult(PxP, u0_full, u0_e);
  // MatMult(PxP, v0_full, v0_e);

  // /* Derivative of 1-trace */
  // double dfb = -1.0 * obj_bar;

  // // Derivative of first term: -(VxV_re*u0 - VxV_im*v0)*obj_bar
  // MatMult(VxV_im, v0_e, x_e);      
  // VecScale(x_e, -1.0);              // x = - VxV_im*v0
  // MatMultAdd(VxV_re, u0_e, x_e, x_e);  // x = VxV_re*u0 - VxV_im*v0
  // VecScale(x_e, dfb);                 // x = -(VxV_re*u0 - VxV_im*v0)*obj_bar

  // /* Derivative of purity */
  // // VecAXPY(x, obj_bar, ufinal);

  // MatMultTranspose(PxP, x_e, x_full);
  // VecISCopy(rho0_bar, isu, SCATTER_FORWARD, x_full);  // set real part in rho0bar
  
  // // Derivative of second term: -(VxV_re*v0 + VxV_im*u0)*obj_bar
  // MatMult(VxV_im, u0_e, x_e);         // x = VxV_im*u0
  // MatMultAdd(VxV_re, v0_e, x_e, x_e); // x = VxV_re*v0 + VxV_im*u0
  // VecScale(x_e, dfb);               // x = -(VxV_re*v0 + VxV_im*u0)*obj_bar

  // /* Derivative of purity */
  // // VecAXPY(x, obj_bar, vfinal);

  // MatMultTranspose(PxP, x_e, x_full);
  // VecISCopy(rho0_bar, isv, SCATTER_FORWARD, x_full);  // set imaginary part in rho0bar


  // /* Restore final, initial and adjoint state */
  // VecRestoreSubVector(finalstate, isu, &ufinal_full);
  // VecRestoreSubVector(finalstate, isv, &vfinal_full);
  // VecRestoreSubVector(rho0, isu, &u0_full);
  // VecRestoreSubVector(rho0, isv, &v0_full);

  // /* Free vindex strides */
  // ISDestroy(&isu);
  // ISDestroy(&isv);
}


  XGate::XGate(std::vector<int> nlevels, std::vector<int> nessential, double time, std::vector<double> gate_rot_freq) : Gate(nlevels, nessential) {

  assert(dim_ess == 2);

  /* Fill V_re = Re(V) and V_im = Im(V), V = V_re + iVb */
  /* V_re = 0 1    V_im = 0 0
   *      1 0         0 0
   */
  if (mpirank_petsc == 0) {
    MatSetValue(V_re, 0, 1, 1.0, INSERT_VALUES);
    MatSetValue(V_re, 1, 0, 1.0, INSERT_VALUES);
    MatAssemblyBegin(V_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(V_re, MAT_FINAL_ASSEMBLY);
  }

  /* Set up gate rotation rotA, rotB */
  assembleGateRotation1x2(time, gate_rot_freq);

  /* Assemble vectorized target gate \bar VP \kron VP from  V = V_re + i V_im */
  assembleGate();
}

XGate::~XGate() {}

YGate::YGate(std::vector<int> nlevels, std::vector<int> nessential, double time, std::vector<double> gate_rot_freq ) : Gate(nlevels, nessential) {

  assert(dim_ess == 2);
  
  /* Fill A = Re(V) and B = Im(V), V = A + iB */
  /* A = 0 0    B = 0 -1
   *     0 0        1  0
   */
  if (mpirank_petsc == 0) {
    MatSetValue(V_im, 0, 1, -1.0, INSERT_VALUES);
    MatSetValue(V_im, 1, 0,  1.0, INSERT_VALUES);
    MatAssemblyBegin(V_im, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(V_im, MAT_FINAL_ASSEMBLY);
  }

  /* Set up gate rotation rotA, rotB */
  assembleGateRotation1x2(time, gate_rot_freq);

  /* Assemble vectorized target gate \bar VP \kron VP from  V = V_re + i V_im*/
  assembleGate();
}
YGate::~YGate() {}

ZGate::ZGate(std::vector<int> nlevels, std::vector<int> nessential, double time, std::vector<double> gate_rot_freq ) : Gate(nlevels, nessential) {

  assert(dim_ess == 2);

  /* Fill A = Re(V) and B = Im(V), V = A + iB */
  /* A =  1  0     B = 0 0
   *      0 -1         0 0
   */
  if (mpirank_petsc == 0) {
    MatSetValue(V_im, 0, 0,  1.0, INSERT_VALUES);
    MatSetValue(V_im, 1, 1, -1.0, INSERT_VALUES);
    MatAssemblyBegin(V_im, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(V_im, MAT_FINAL_ASSEMBLY);
  }

  /* Set up gate rotation rotA, rotB */
  assembleGateRotation1x2(time, gate_rot_freq);

  /* Assemble target gate \bar V \kron V from  V = V_re + i V_im*/
  assembleGate();
}

ZGate::~ZGate() {}

HadamardGate::HadamardGate(std::vector<int> nlevels, std::vector<int> nessential, double time, std::vector<double> gate_rot_freq ) : Gate(nlevels, nessential) {

  assert(dim_ess == 2);

  /* Fill A = Re(V) and B = Im(V), V = A + iB */
  /* A =  1  0     B = 0 0
   *      0 -1         0 0
   */
  if (mpirank_petsc == 0) {
    double val = 1./sqrt(2);
    MatSetValue(V_re, 0, 0,  val, INSERT_VALUES);
    MatSetValue(V_re, 0, 1,  val, INSERT_VALUES);
    MatSetValue(V_re, 1, 0,  val, INSERT_VALUES);
    MatSetValue(V_re, 1, 1, -val, INSERT_VALUES);
    MatAssemblyBegin(V_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(V_re, MAT_FINAL_ASSEMBLY);
  }

  /* Set up gate rotation rotA, rotB */
  assembleGateRotation1x2(time, gate_rot_freq);

  /* Assemble target gate \bar V \kron V from  V = V_re + i V_im*/
  assembleGate();
}
HadamardGate::~HadamardGate() {}



CNOT::CNOT(std::vector<int> nlevels, std::vector<int> nessential, double time, std::vector<double> gate_rot_freq) : Gate(nlevels, nessential) {

  assert(dim_ess == 4);

  /* Fill A = Re(V) and B = Im(V), V = A + iB */
  /* A =  1 0 0 0   B = 0 0 0 0
   *      0 1 0 0       0 0 0 0
   *      0 0 0 1       0 0 0 0
   *      0 0 1 0       0 0 0 0
   */  if (mpirank_petsc == 0) {
    MatSetValue(V_re, 0, 0, 1.0, INSERT_VALUES);
    MatSetValue(V_re, 1, 1, 1.0, INSERT_VALUES);
    MatSetValue(V_re, 2, 3, 1.0, INSERT_VALUES);
    MatSetValue(V_re, 3, 2, 1.0, INSERT_VALUES);
    MatAssemblyBegin(V_re, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(V_re, MAT_FINAL_ASSEMBLY);
  }

  /* Set up gate rotation rotA, rotB */
  assembleGateRotation2x2(time, gate_rot_freq);

  /* assemble V = \bar V \kron V */
  assembleGate();
}

CNOT::~CNOT(){}


SWAP::SWAP(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> gate_rot_freq) : Gate(nlevels_, nessential_) {
  assert(dim_ess == 4);

  /* Fill lab-frame swap gate in essential dimension system V_re = Re(V), V_im = Im(V) = 0 */
  MatSetValue(V_re, 0, 0, 1.0, INSERT_VALUES);
  MatSetValue(V_re, 1, 2, 1.0, INSERT_VALUES);
  MatSetValue(V_re, 2, 1, 1.0, INSERT_VALUES);
  MatSetValue(V_re, 3, 3, 1.0, INSERT_VALUES);

  MatAssemblyBegin(V_re, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(V_im, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(V_re, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(V_im, MAT_FINAL_ASSEMBLY);

  /* Set up gate rotation rotA, rotB */
  // TODO: ROTATE!
  // assembleGateRotation2x2(time, gate_rot_freq);

  /* assemble V = \bar V \kron V */
  assembleGate();

}

SWAP::~SWAP(){}



