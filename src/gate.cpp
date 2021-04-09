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

  /* Allocate Va, Vb, in essential level dimension, sequential matrix, copied on all processors */
  MatCreateSeqDense(PETSC_COMM_SELF, dim_ess, dim_ess, NULL, &Va);
  MatCreateSeqDense(PETSC_COMM_SELF, dim_ess, dim_ess, NULL, &Vb);
  MatSetUp(Va);
  MatSetUp(Vb);
  MatAssemblyBegin(Va, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(Vb, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Va, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Vb, MAT_FINAL_ASSEMBLY);

  // /* TODO: Test rotation! Allocate real and imaginary rotation */
  // MatCreateVecs(Va, &rotA, NULL);
  // MatCreateVecs(Vb, &rotB, NULL);
  // // default: rotA = I, rotB = 0
  // int nrot;
  // VecGetSize(rotA, &nrot);
  // for (int irow =0; irow<nrot; irow++){
  //   VecSetValue(rotA, irow, 1.0, INSERT_VALUES);
  // }
  // VecAssemblyBegin(rotA); VecAssemblyEnd(rotA);
  // VecAssemblyBegin(rotB); VecAssemblyEnd(rotB);

  /* Allocate ReG (FULL) = Re(\bar V \kron V), ImG = Im(\bar V \kron V), parallel, essential levels dimension */
  MatCreate(PETSC_COMM_WORLD, &ReG);
  MatCreate(PETSC_COMM_WORLD, &ImG);
  MatSetSizes(ReG, PETSC_DECIDE, PETSC_DECIDE, dim_rho*dim_rho, dim_rho*dim_rho);
  MatSetSizes(ImG, PETSC_DECIDE, PETSC_DECIDE, dim_rho*dim_rho, dim_rho*dim_rho);
  MatSetUp(ReG);
  MatSetUp(ImG);
  MatAssemblyBegin(ReG, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ReG, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(ImG, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ImG, MAT_FINAL_ASSEMBLY);

  MatCreateVecs(ReG, &x, NULL);
}

Gate::~Gate(){
  if (dim_rho == 0) return;
  MatDestroy(&ReG);
  MatDestroy(&ImG);
  MatDestroy(&Va);
  MatDestroy(&Vb);
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

/************************************************/
// Compute ReG_f, ImG_f =  V\kron V. Parallel matrix 

  // MatView(Va, NULL);
  
  int ilow, iupp;
  double val;
  MatGetOwnershipRange(ReG, &ilow, &iupp);
  // iterate over local rows in ReG, ImGf
  for (int i = ilow; i<iupp; i++) {
    const int r1 = (int) i / dim_rho;
    const int r2 = i % dim_rho;
    // iterate over columns in G
    for (int j=0; j<dim_rho*dim_rho; j++){
      const int c1 = (int) j / dim_rho;
      const int c2 = j % dim_rho;
      // Get values Va_r1c1, Va_r2C2, Vb_r1c1, Vbr2c2
      // first map from full to essential dimension
      int r1e = mapFullToEss(r1, nlevels, nessential);
      int r2e = mapFullToEss(r2, nlevels, nessential);
      int c1e = mapFullToEss(c1, nlevels, nessential);
      int c2e = mapFullToEss(c2, nlevels, nessential);
      double va1=0.0;
      double va2=0.0;
      double vb1=0.0;
      double vb2=0.0;
      int ierr;
      ierr = MatGetValues(Va, 1, &r1e, 1, &c1e, &va1); 
      ierr = MatGetValues(Va, 1, &r2e, 1, &c2e, &va2); 
      ierr = MatGetValues(Vb, 1, &r1e, 1, &c1e, &vb1); 
      ierr = MatGetValues(Vb, 1, &r2e, 1, &c2e, &vb2); 
      val = va1*va2 + vb1*vb2;
      if (fabs(val) > 1e-14) MatSetValue(ReG, i,j, val, INSERT_VALUES);
      val = va1*vb2 - vb1*va2;
      if (fabs(val) > 1e-14) MatSetValue(ImG, i,j, val, INSERT_VALUES);
    }
  }

  MatAssemblyBegin(ReG, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ReG, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(ImG, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ImG, MAT_FINAL_ASSEMBLY);

  // std::cout<<"ReG new \n";
  // MatView(ReG, NULL);


/************************************************/


// /**************************************/
//   /* PRoject A and B onto full levels */
//   Mat Af, Bf;
//   int ncols, nrowsA;
//   const int *cols;
//   const double *vals;
//   MatCreate(PETSC_COMM_WORLD, &Af);
//   MatCreate(PETSC_COMM_WORLD, &Bf);
//   MatSetSizes(Af, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
//   MatSetSizes(Bf, PETSC_DECIDE, PETSC_DECIDE, dim_rho, dim_rho);
//   MatSetUp(Af);
//   MatSetUp(Bf);
//   MatGetSize(A, &nrowsA, NULL);
//   for (int i = 0; i < nrowsA; i++){
//     MatGetRow(A, i, &ncols, &cols, &vals);
//     for (int j=0; j<ncols; j++){
//       int rowF = mapEssToFull(i, nlevels, nessential);
//       int colF = mapEssToFull(j, nlevels, nessential);
//       MatSetValue(Af, rowF, colF, vals[j], INSERT_VALUES);  // Todo: locally!
//     }
//     MatGetRow(B, i, &ncols, &cols, &vals);
//     for (int j=0; j<ncols; j++){
//       int rowF = mapEssToFull(i, nlevels, nessential);
//       int colF = mapEssToFull(j, nlevels, nessential);
//       MatSetValue(Bf, rowF, colF, vals[j], INSERT_VALUES);  // Todo: locally!
//     }
//   }

//   /* Compute ReImGf = V \kron V  */
//   AkronB(Af, Af,  1.0, &ReGf, ADD_VALUES);
//   AkronB(Bf, Bf,  1.0, &ReGf, ADD_VALUES);
//   AkronB(Af, Bf,  1.0, &ImGf, ADD_VALUES);
//   AkronB(Bf, Af, -1.0, &ImGf, ADD_VALUES);

//   MatAssemblyBegin(ReGf, MAT_FINAL_ASSEMBLY);
//   MatAssemblyEnd(ReGf, MAT_FINAL_ASSEMBLY);
//   MatAssemblyBegin(ImGf, MAT_FINAL_ASSEMBLY);
//   MatAssemblyEnd(ImGf, MAT_FINAL_ASSEMBLY);

//   MatView(ReG, NULL);
//   MatView(ReGf, NULL);

// /**************************************/


  // MatDestroy(&tmp);
  // MatDestroy(&A);
  // MatDestroy(&B);
}


void Gate::compare_frobenius(const Vec finalstate, const Vec rho0, double& frob){
  frob = 0.0;


  /* Exit, if this is a dummy gate */
  if (dim_rho == 0) {
    return;
  }

  /* Create vector strides for accessing real and imaginary part of co-located state */
  int ilow, iupp;
  VecGetOwnershipRange(finalstate, &ilow, &iupp);
  int dimis = (iupp - ilow)/2;
  IS isu, isv;
  ISCreateStride(PETSC_COMM_WORLD, dimis, ilow, 2, &isu);
  ISCreateStride(PETSC_COMM_WORLD, dimis, ilow+1, 2, &isv);

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

//   /* Add real part of frobenius norm || u - ReG*u0 + ImG*v0 ||^2 */
//   MatMult(ReG, u0_e, x_e);            // x = ReG*u0
//   VecAYPX(x_e, -1.0, ufinal_e);       // x = ufinal - ReG*u0 
//   MatMultAdd(ImG, v0_e, x_e, x_e);      // x = ufinal - ReG*u0 + ImG*v0
//   double norm;
//   VecNorm(x_e, NORM_2, &norm);
//   frob = pow(norm,2.0);           // frob = || x ||^2

//   /* Add imaginary part of frobenius norm || v - ReG*v0 - ImG*u0 ||^2 */
//   MatMult(ReG, v0_e, x_e);         // x = ReG*v0
//   MatMultAdd(ImG, u0_e, x_e, x_e);   // x = ReG*v0 + ImG*u0
//   VecAYPX(x_e, -1.0, vfinal_e);     // x = vfinal - (ReG*v0 + ImG*u0)
//   VecNorm(x_e, NORM_2, &norm);
//   frob += pow(norm, 2.0);      // frob += ||x||^2

//   /* obj = 1/2 * || finalstate - gate*rho(0) ||^2 */
//   frob *= 1./2.;

//   /* Restore vectors from index set */
//   VecRestoreSubVector(finalstate, isu, &ufinal_full);
//   VecRestoreSubVector(finalstate, isv, &vfinal_full);
//   VecRestoreSubVector(rho0, isu, &u0_full);
//   VecRestoreSubVector(rho0, isv, &v0_full);

// /* ---------------------------------------*/
//   printf("Frobenius previous: %1.14e\n", frob);
//   // MatView(ReG, NULL);


  /* First, project full dimension system to essential levels only: */
  /* iterate over rows of system matrix, check if it corresponds to an essential level, and if not, set this row and colum to zero */
  int reID, imID;
  for (int i=0; i<dim_rho; i++) {
    if (!isEssential(i, nlevels, nessential)) { // if not essential, zero out row and colum
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

  /* Get real and imag part of final state and initial state */
  Vec ufinal, vfinal, u0, v0;
  VecGetSubVector(finalstate, isu, &ufinal);
  VecGetSubVector(finalstate, isv, &vfinal);
  VecGetSubVector(rho0, isu, &u0);
  VecGetSubVector(rho0, isv, &v0);

  /* Add real part of frobenius norm || u - ReG*u0 + ImG*v0 ||^2 */
  double norm;
  MatMult(ReG, u0, x);            // x = ReG*u0
  VecAYPX(x, -1.0, ufinal);       // x = ufinal - ReG*u0 
  MatMultAdd(ImG, v0, x, x);      // x = ufinal - ReG*u0 + ImG*v0
  VecNorm(x, NORM_2, &norm);
  frob = pow(norm,2.0);           // frob = || x ||^2

  /* Add imaginary part of frobenius norm || v - ReG*v0 - ImG*u0 ||^2 */
  MatMult(ReG, v0, x);         // x = ReG*v0
  MatMultAdd(ImG, u0, x, x);   // x = ReG*v0 + ImG*u0
  VecAYPX(x, -1.0, vfinal);     // x = vfinal - (ReG*v0 + ImG*u0)
  VecNorm(x, NORM_2, &norm);
  frob += pow(norm, 2.0);      // frob += ||x||^2

  /* obj = 1/2 * || finalstate - gate*rho(0) ||^2 */
  frob *= 1./2.;

  /* Restore vectors from index set */
  VecRestoreSubVector(finalstate, isu, &ufinal);
  VecRestoreSubVector(finalstate, isv, &vfinal);
  VecRestoreSubVector(rho0, isu, &u0);
  VecRestoreSubVector(rho0, isv, &v0);

/* ---------------------------------------*/

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

  // /* Derivative of real part of frobenius norm: 2 * (u - ReG*u0 + ImG*v0) * dfb */
  // MatMult(ReG, u0_e, x_e);            // x = ReG*u0
  // VecAYPX(x_e, -1.0, ufinal_e);       // x = ufinal - ReG*u0 
  // MatMultAdd(ImG, v0_e, x_e, x_e);      // x = ufinal - ReG*u0 + ImG*v0
  // VecScale(x_e, 2*dfb);             // x = 2*(ufinal - ReG*u0 + ImG*v0)*dfb
  // // Project essential to full state 
  // MatMultTranspose(PxP, x_e, x_full);
  // // set real part in rho0bar
  // VecISCopy(rho0_bar, isu, SCATTER_FORWARD, x_full);

  // /* Derivative of imaginary part of frobenius norm 2 * (v - ReG*v0 - ImG*u0) * dfb */
  // MatMult(ReG, v0_e, x_e);         // x = ReG*v0
  // MatMultAdd(ImG, u0_e, x_e, x_e);   // x = ReG*v0 + ImG*u0
  // VecAYPX(x_e, -1.0, vfinal_e);     // x = vfinal - (ReG*v0 + ImG*u0)
  // VecScale(x_e, 2*dfb);          // x = 2*(vfinal - (ReG*v0 + ImG*u0)*dfb
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


//   /* trace overlap: (ReG*u0 - ImG*v0)^T u + (ReG*v0 + ImG*u0)^Tv
//               [ + i (ReG*u0 - ImG*v0)^T v - (ReG*v0 + ImG*u0)^Tu ]   <- this should be zero!
//   */
//   double dot;
//   double trace = 0.0;

//   // first term: (ReG*u0 - ImG*v0)^T u
//   MatMult(ImG, v0_e, x_e);      
//   VecScale(x_e, -1.0);                // x = - ImG*v0
//   MatMultAdd(ReG, u0_e, x_e, x_e);  // x = ReG*u0 - ImG*v0
//   VecTDot(x_e, ufinal_e, &dot);       // dot = (ReG*u0 - ImG*v0)^T u    
//   trace += dot;
  
//   // second term: (ReG*v0 + ImG*u0)^Tv
//   MatMult(ImG, u0_e, x_e);         // x = ImG*u0
//   MatMultAdd(ReG, v0_e, x_e, x_e); // x = ReG*v0 + ImG*u0
//   VecTDot(x_e, vfinal_e, &dot);      // dot = (ReG*v0 + ImG*u0)^T v    
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
//   // MatMult(ImG, v0, x);      
//   // VecScale(x, -1.0);           // x = - ImG*v0
//   // MatMultAdd(ReG, u0, x, x);   // x = ReG*u0 - ImG*v0
//   // VecNorm(x, NORM_2, &dot);
//   // purity_VrhoV += dot*dot;
//   // MatMult(ImG, u0, x);         // x = ImG*u0
//   // MatMultAdd(ReG, v0, x, x);   // x = ReG*v0 + ImG*u0
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

  // // Derivative of first term: -(ReG*u0 - ImG*v0)*obj_bar
  // MatMult(ImG, v0_e, x_e);      
  // VecScale(x_e, -1.0);              // x = - ImG*v0
  // MatMultAdd(ReG, u0_e, x_e, x_e);  // x = ReG*u0 - ImG*v0
  // VecScale(x_e, dfb);                 // x = -(ReG*u0 - ImG*v0)*obj_bar

  // /* Derivative of purity */
  // // VecAXPY(x, obj_bar, ufinal);

  // MatMultTranspose(PxP, x_e, x_full);
  // VecISCopy(rho0_bar, isu, SCATTER_FORWARD, x_full);  // set real part in rho0bar
  
  // // Derivative of second term: -(ReG*v0 + ImG*u0)*obj_bar
  // MatMult(ImG, u0_e, x_e);         // x = ImG*u0
  // MatMultAdd(ReG, v0_e, x_e, x_e); // x = ReG*v0 + ImG*u0
  // VecScale(x_e, dfb);               // x = -(ReG*v0 + ImG*u0)*obj_bar

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

  /* Fill Va = Re(V) and Vb = Im(V), V = Va + iVb */
  /* Va = 0 1    Vb = 0 0
   *      1 0         0 0
   */
  if (mpirank_petsc == 0) {
    MatSetValue(Va, 0, 1, 1.0, INSERT_VALUES);
    MatSetValue(Va, 1, 0, 1.0, INSERT_VALUES);
    MatAssemblyBegin(Va, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Va, MAT_FINAL_ASSEMBLY);
  }

  /* Set up gate rotation rotA, rotB */
  assembleGateRotation1x2(time, gate_rot_freq);

  /* Assemble vectorized target gate \bar VP \kron VP from  V = Va + i Vb */
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
    MatSetValue(Vb, 0, 1, -1.0, INSERT_VALUES);
    MatSetValue(Vb, 1, 0,  1.0, INSERT_VALUES);
    MatAssemblyBegin(Vb, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Vb, MAT_FINAL_ASSEMBLY);
  }

  /* Set up gate rotation rotA, rotB */
  assembleGateRotation1x2(time, gate_rot_freq);

  /* Assemble vectorized target gate \bar VP \kron VP from  V = Va + i Vb*/
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
    MatSetValue(Vb, 0, 0,  1.0, INSERT_VALUES);
    MatSetValue(Vb, 1, 1, -1.0, INSERT_VALUES);
    MatAssemblyBegin(Vb, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Vb, MAT_FINAL_ASSEMBLY);
  }

  /* Set up gate rotation rotA, rotB */
  assembleGateRotation1x2(time, gate_rot_freq);

  /* Assemble target gate \bar V \kron V from  V = Va + i Vb*/
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
    MatSetValue(Va, 0, 0,  val, INSERT_VALUES);
    MatSetValue(Va, 0, 1,  val, INSERT_VALUES);
    MatSetValue(Va, 1, 0,  val, INSERT_VALUES);
    MatSetValue(Va, 1, 1, -val, INSERT_VALUES);
    MatAssemblyBegin(Va, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Va, MAT_FINAL_ASSEMBLY);
  }

  /* Set up gate rotation rotA, rotB */
  assembleGateRotation1x2(time, gate_rot_freq);

  /* Assemble target gate \bar V \kron V from  V = Va + i Vb*/
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
    MatSetValue(Va, 0, 0, 1.0, INSERT_VALUES);
    MatSetValue(Va, 1, 1, 1.0, INSERT_VALUES);
    MatSetValue(Va, 2, 3, 1.0, INSERT_VALUES);
    MatSetValue(Va, 3, 2, 1.0, INSERT_VALUES);
    MatAssemblyBegin(Va, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Va, MAT_FINAL_ASSEMBLY);
  }

  /* Set up gate rotation rotA, rotB */
  assembleGateRotation2x2(time, gate_rot_freq);

  /* assemble V = \bar V \kron V */
  assembleGate();
}

CNOT::~CNOT(){}


SWAP::SWAP(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> gate_rot_freq) : Gate(nlevels_, nessential_) {
  assert(dim_ess == 4);

/************************************/
  /* Fill lab-frame swap gate in essential dimension system Va = Re(V), Vb = Im(V) = 0 */

  // copy on all processors //

  int row, col;
  // if (mpirank_petsc == 0) {
    MatSetValue(Va, 0, 0, 1.0, INSERT_VALUES);
    MatSetValue(Va, 1, 2, 1.0, INSERT_VALUES);
    MatSetValue(Va, 2, 1, 1.0, INSERT_VALUES);
    MatSetValue(Va, 3, 3, 1.0, INSERT_VALUES);
  // }
  MatAssemblyBegin(Va, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(Vb, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Va, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Vb, MAT_FINAL_ASSEMBLY);
/************************************/


  // printf("Fuck!\n");
  // MatView(Va, NULL);
  // MatView(Vb, NULL);
  // exit(1);


  /* Rotate... */

  /* assemble V = \bar V \kron V */
  assembleGate();

}

SWAP::~SWAP(){}



