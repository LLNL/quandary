#include "gate.hpp"

Gate::Gate(){
  dim_ess = 0;
  // dim_vec = 0;
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


  /* Allocate Va, Vb, sequential, only on proc 0 */
  MatCreateSeqDense(PETSC_COMM_SELF, dim_ess, dim_ess, NULL, &Va);
  MatCreateSeqDense(PETSC_COMM_SELF, dim_ess, dim_ess, NULL, &Vb);
  MatSetUp(Va);
  MatSetUp(Vb);
  MatAssemblyBegin(Va, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(Vb, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Va, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Vb, MAT_FINAL_ASSEMBLY);

  /* Set up projection matrix to map ful system to essential levels */
  Mat P;
  MatCreate(PETSC_COMM_WORLD, &P);
  MatSetSizes(P, PETSC_DECIDE, PETSC_DECIDE, dim_ess, dim_rho);
  MatSetUp(P);
  if (dim_ess < dim_rho && nlevels.size() > 2) {
    printf("\n ERROR: Gate objective for essential levels with noscillators > 2 not implemented yet. \n");
    exit(1);
  }
  for (int i=0; i<nessential[0]; i++) {
    // Place identity of size n_e^B \times n_e^B at position (i*n_e^B, i*n^B)
    for (int j=0; j<nessential[1]; j++) {        
      int row = i * nessential[1] + j;
      int col = i * nlevels[1] + j;
      MatSetValue(P, row, col,  1.0, INSERT_VALUES);
    }
  }
  MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY);
  /* Set up vectorized projection P\kron P */
  MatCreate(PETSC_COMM_WORLD, &PxP);
  MatSetSizes(PxP, PETSC_DECIDE, PETSC_DECIDE, dim_ess*dim_ess, dim_rho*dim_rho);
  MatSetUp(PxP);
  AkronB(P, P, 1.0, &PxP, INSERT_VALUES);
  MatAssemblyBegin(PxP, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(PxP, MAT_FINAL_ASSEMBLY);
  MatDestroy(&P);

  /* Allocate final and initial states, projected onto essential levels */
  MatCreateVecs(PxP, NULL, &ufinal_e);
  MatCreateVecs(PxP, NULL, &vfinal_e);
  MatCreateVecs(PxP, NULL, &u0_e);
  MatCreateVecs(PxP, NULL, &v0_e);


  /* Allocate ReG = Re(\bar V \kron V), ImG = Im(\bar V \kron V), parallel */
  MatCreate(PETSC_COMM_WORLD, &ReG);
  MatCreate(PETSC_COMM_WORLD, &ImG);
  MatSetSizes(ReG, PETSC_DECIDE, PETSC_DECIDE, dim_ess*dim_ess, dim_ess*dim_ess);
  MatSetSizes(ImG, PETSC_DECIDE, PETSC_DECIDE, dim_ess*dim_ess, dim_ess*dim_ess);
  // MatSetSizes(ReG, PETSC_DECIDE, PETSC_DECIDE, dim_rho*dim_rho, dim_rho*dim_rho);
  // MatSetSizes(ImG, PETSC_DECIDE, PETSC_DECIDE, dim_rho*dim_rho, dim_rho*dim_rho);
  MatSetUp(ReG);
  MatSetUp(ImG);
  MatAssemblyBegin(ReG, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ReG, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(ImG, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ImG, MAT_FINAL_ASSEMBLY);


  /* Create auxiliary vectors */
  MatCreateVecs(PxP, &x_full, NULL);   // full state dimension
  MatCreateVecs(PxP, NULL, &x_e);      // essential levels only 

}

Gate::~Gate(){
  if (dim_rho == 0) return;
  MatDestroy(&ReG);
  MatDestroy(&ImG);
  MatDestroy(&Va);
  MatDestroy(&Vb);
  MatDestroy(&PxP);
  VecDestroy(&x_full);
  VecDestroy(&x_e);
  VecDestroy(&ufinal_e);
  VecDestroy(&vfinal_e);
  VecDestroy(&u0_e);
  VecDestroy(&v0_e);
}


void Gate::assembleGate(){
  
  /* Compute ReG = Re(\bar V \kron V) = Va\kron Va + Vb\kron Vb  */
  AkronB(Va, Va,  1.0, &ReG, ADD_VALUES);
  AkronB(Vb, Vb,  1.0, &ReG, ADD_VALUES);
  /* Compute ImG = Im(\bar V\kron V) = A\kron B - B\kron A */
  AkronB(Va, Vb,  1.0, &ImG, ADD_VALUES);
  AkronB(Vb, Va, -1.0, &ImG, ADD_VALUES);

  MatAssemblyBegin(ReG, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ReG, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(ImG, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ImG, MAT_FINAL_ASSEMBLY);

  // printf("ReG:");
  // MatView(ReG, PETSC_VIEWER_STDOUT_WORLD);
  // printf("ImG:");
  // MatView(ImG, PETSC_VIEWER_STDOUT_WORLD);
  // exit(1);

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

  /* Get real and imag part of final state and initial state */
  Vec ufinal_full, vfinal_full, u0_full, v0_full;
  VecGetSubVector(finalstate, isu, &ufinal_full);
  VecGetSubVector(finalstate, isv, &vfinal_full);
  VecGetSubVector(rho0, isu, &u0_full);
  VecGetSubVector(rho0, isv, &v0_full);

  /* Project final and initial states onto essential levels */
  MatMult(PxP, ufinal_full, ufinal_e);
  MatMult(PxP, vfinal_full, vfinal_e);
  MatMult(PxP, u0_full, u0_e);
  MatMult(PxP, v0_full, v0_e);


  /* Add real part of frobenius norm || u - ReG*u0 + ImG*v0 ||^2 */
  MatMult(ReG, u0_e, x_e);            // x = ReG*u0
  VecAYPX(x_e, -1.0, ufinal_e);       // x = ufinal - ReG*u0 
  MatMultAdd(ImG, v0_e, x_e, x_e);      // x = ufinal - ReG*u0 + ImG*v0
  double norm;
  VecNorm(x_e, NORM_2, &norm);
  frob = pow(norm,2.0);           // frob = || x ||^2


  /* Add imaginary part of frobenius norm || v - ReG*v0 - ImG*u0 ||^2 */
  MatMult(ReG, v0_e, x_e);         // x = ReG*v0
  MatMultAdd(ImG, u0_e, x_e, x_e);   // x = ReG*v0 + ImG*u0
  VecAYPX(x_e, -1.0, vfinal_e);     // x = vfinal - (ReG*v0 + ImG*u0)
  VecNorm(x_e, NORM_2, &norm);
  frob += pow(norm, 2.0);      // frob += ||x||^2

  /* obj = 1/2 * || finalstate - gate*rho(0) ||^2 */
  frob *= 1./2.;

  /* Restore vectors from index set */
  VecRestoreSubVector(finalstate, isu, &ufinal_full);
  VecRestoreSubVector(finalstate, isv, &vfinal_full);
  VecRestoreSubVector(rho0, isu, &u0_full);
  VecRestoreSubVector(rho0, isv, &v0_full);

  /* Free index strides */
  ISDestroy(&isu);
  ISDestroy(&isv);
}

void Gate::compare_frobenius_diff(const Vec finalstate, const Vec rho0, Vec rho0_bar, const double frob_bar){

  /* Exit, if this is a dummy gate */
  if (dim_rho == 0) {
    return;
  }

  /* Create vector strides for accessing real and imaginary part of co-located x */
  int ilow, iupp;
  VecGetOwnershipRange(finalstate, &ilow, &iupp);
  int dimis = (iupp - ilow)/2;
  IS isu, isv;
  ISCreateStride(PETSC_COMM_WORLD, dimis, ilow, 2, &isu);
  ISCreateStride(PETSC_COMM_WORLD, dimis, ilow+1, 2, &isv);


  /* Get real and imag part of final state, initial state, and adjoint */
  Vec ufinal_full, vfinal_full, u0_full, v0_full;
  VecGetSubVector(finalstate, isu, &ufinal_full);
  VecGetSubVector(finalstate, isv, &vfinal_full);
  VecGetSubVector(rho0, isu, &u0_full);
  VecGetSubVector(rho0, isv, &v0_full);

  /* Project final and initial states onto essential levels */
  MatMult(PxP, ufinal_full, ufinal_e);
  MatMult(PxP, vfinal_full, vfinal_e);
  MatMult(PxP, u0_full, u0_e);
  MatMult(PxP, v0_full, v0_e);

  /* Derivative of 1/2 * J */
  double dfb = 1./2. * frob_bar;

  /* Derivative of real part of frobenius norm: 2 * (u - ReG*u0 + ImG*v0) * dfb */
  MatMult(ReG, u0_e, x_e);            // x = ReG*u0
  VecAYPX(x_e, -1.0, ufinal_e);       // x = ufinal - ReG*u0 
  MatMultAdd(ImG, v0_e, x_e, x_e);      // x = ufinal - ReG*u0 + ImG*v0
  VecScale(x_e, 2*dfb);             // x = 2*(ufinal - ReG*u0 + ImG*v0)*dfb
  // Project essential to full state 
  MatMultTranspose(PxP, x_e, x_full);
  // set real part in rho0bar
  VecISCopy(rho0_bar, isu, SCATTER_FORWARD, x_full);

  /* Derivative of imaginary part of frobenius norm 2 * (v - ReG*v0 - ImG*u0) * dfb */
  MatMult(ReG, v0_e, x_e);         // x = ReG*v0
  MatMultAdd(ImG, u0_e, x_e, x_e);   // x = ReG*v0 + ImG*u0
  VecAYPX(x_e, -1.0, vfinal_e);     // x = vfinal - (ReG*v0 + ImG*u0)
  VecScale(x_e, 2*dfb);          // x = 2*(vfinal - (ReG*v0 + ImG*u0)*dfb
  // Project essential to full state 
  MatMultTranspose(PxP, x_e, x_full);
  VecISCopy(rho0_bar, isv, SCATTER_FORWARD, x_full);  // set imaginary part in rho0bar

  /* Restore final, initial and adjoint state */
  VecRestoreSubVector(finalstate, isu, &ufinal_full);
  VecRestoreSubVector(finalstate, isv, &vfinal_full);
  VecRestoreSubVector(rho0, isu, &u0_full);
  VecRestoreSubVector(rho0, isv, &v0_full);

  /* Free vindex strides */
  ISDestroy(&isu);
  ISDestroy(&isv);
}

void Gate::compare_trace(const Vec finalstate, const Vec rho0, double& obj){
  obj = 0.0;

  /* Exit, if this is a dummy gate */
  if (dim_rho== 0) {
    return;
  }

  /* Create vector strides for accessing real and imaginary part of co-located x */
  int ilow, iupp;
  VecGetOwnershipRange(finalstate, &ilow, &iupp);
  int dimis = (iupp - ilow)/2;
  IS isu, isv;
  ISCreateStride(PETSC_COMM_WORLD, dimis, ilow, 2, &isu);
  ISCreateStride(PETSC_COMM_WORLD, dimis, ilow+1, 2, &isv);

  /* Get real and imag part of final state and initial state */
  Vec ufinal_full, vfinal_full, u0_full, v0_full;
  VecGetSubVector(finalstate, isu, &ufinal_full);
  VecGetSubVector(finalstate, isv, &vfinal_full);
  VecGetSubVector(rho0, isu, &u0_full);
  VecGetSubVector(rho0, isv, &v0_full);

  /* Project final and initial states onto essential levels */
  MatMult(PxP, ufinal_full, ufinal_e);
  MatMult(PxP, vfinal_full, vfinal_e);
  MatMult(PxP, u0_full, u0_e);
  MatMult(PxP, v0_full, v0_e);


  /* trace overlap: (ReG*u0 - ImG*v0)^T u + (ReG*v0 + ImG*u0)^Tv
              [ + i (ReG*u0 - ImG*v0)^T v - (ReG*v0 + ImG*u0)^Tu ]   <- this should be zero!
  */
  double dot;
  double trace = 0.0;

  // first term: (ReG*u0 - ImG*v0)^T u
  MatMult(ImG, v0_e, x_e);      
  VecScale(x_e, -1.0);                // x = - ImG*v0
  MatMultAdd(ReG, u0_e, x_e, x_e);  // x = ReG*u0 - ImG*v0
  VecTDot(x_e, ufinal_e, &dot);       // dot = (ReG*u0 - ImG*v0)^T u    
  trace += dot;
  
  // second term: (ReG*v0 + ImG*u0)^Tv
  MatMult(ImG, u0_e, x_e);         // x = ImG*u0
  MatMultAdd(ReG, v0_e, x_e, x_e); // x = ReG*v0 + ImG*u0
  VecTDot(x_e, vfinal_e, &dot);      // dot = (ReG*v0 + ImG*u0)^T v    
  trace += dot;

  /* Objective J = 1.0 - Trace(...) */
  obj = 1.0 - trace;
  // obj = - trace;
 
  // // Test: compute purity of rho(T): 1/2*Tr(rho^2)
  // double purity_rhoT = 0.0;
  // VecNorm(ufinal, NORM_2, &dot);
  // purity_rhoT += dot*dot;
  // VecNorm(vfinal, NORM_2, &dot);
  // purity_rhoT += dot*dot;
  // // Test: compute constant term  1/2*Tr((Vrho0V^dag)^2)
  // double purity_VrhoV = 0.0;
  // MatMult(ImG, v0, x);      
  // VecScale(x, -1.0);           // x = - ImG*v0
  // MatMultAdd(ReG, u0, x, x);   // x = ReG*u0 - ImG*v0
  // VecNorm(x, NORM_2, &dot);
  // purity_VrhoV += dot*dot;
  // MatMult(ImG, u0, x);         // x = ImG*u0
  // MatMultAdd(ReG, v0, x, x);   // x = ReG*v0 + ImG*u0
  // VecNorm(x, NORM_2, &dot);
  // purity_VrhoV += dot*dot;
  // double J_dist = purity_rhoT/2. - trace + purity_VrhoV/2.;
  // printf("J_dist = 1/2 * %f - %f + 1/2 * %f = %1.14e\n", purity_rhoT, trace, purity_VrhoV, J_dist);

  // // obj = obj + purity_rhoT / 2. - 0.5;

  /* Restore vectors from index set */
  VecRestoreSubVector(finalstate, isu, &ufinal_full);
  VecRestoreSubVector(finalstate, isv, &vfinal_full);
  VecRestoreSubVector(rho0, isu, &u0_full);
  VecRestoreSubVector(rho0, isv, &v0_full);

  /* Free index strides */
  ISDestroy(&isu);
  ISDestroy(&isv);

  // /* Verify trace overlap */
  // double Jdist = 0.0;
  // compare_frobenius(finalstate, rho0, Jdist);
  // test = test + 1. + Jdist;

  // printf("\n");
  // printf(" J_T:   %1.14e\n", obj);
  // printf(" test:  %1.14e\n", test);

}


void Gate::compare_trace_diff(const Vec finalstate, const Vec rho0, Vec rho0_bar, const double obj_bar){

  /* Exit, if this is a dummy gate */
  if (dim_rho== 0) {
    return;
  }

  /* Create vector strides for accessing real and imaginary part of co-located x */
  int ilow, iupp;
  VecGetOwnershipRange(finalstate, &ilow, &iupp);
  int dimis = (iupp - ilow)/2;
  IS isu, isv;
  ISCreateStride(PETSC_COMM_WORLD, dimis, ilow, 2, &isu);
  ISCreateStride(PETSC_COMM_WORLD, dimis, ilow+1, 2, &isv);

  /* Get real and imag part of final state and initial state */
  Vec ufinal_full, vfinal_full, u0_full, v0_full;
  VecGetSubVector(finalstate, isu, &ufinal_full);
  VecGetSubVector(finalstate, isv, &vfinal_full);
  VecGetSubVector(rho0, isu, &u0_full);
  VecGetSubVector(rho0, isv, &v0_full);

  /* Project final and initial states onto essential levels */
  MatMult(PxP, ufinal_full, ufinal_e);
  MatMult(PxP, vfinal_full, vfinal_e);
  MatMult(PxP, u0_full, u0_e);
  MatMult(PxP, v0_full, v0_e);

  /* Derivative of 1-trace */
  double dfb = -1.0 * obj_bar;

  // Derivative of first term: -(ReG*u0 - ImG*v0)*obj_bar
  MatMult(ImG, v0_e, x_e);      
  VecScale(x_e, -1.0);              // x = - ImG*v0
  MatMultAdd(ReG, u0_e, x_e, x_e);  // x = ReG*u0 - ImG*v0
  VecScale(x_e, dfb);                 // x = -(ReG*u0 - ImG*v0)*obj_bar

  /* Derivative of purity */
  // VecAXPY(x, obj_bar, ufinal);

  MatMultTranspose(PxP, x_e, x_full);
  VecISCopy(rho0_bar, isu, SCATTER_FORWARD, x_full);  // set real part in rho0bar
  
  // Derivative of second term: -(ReG*v0 + ImG*u0)*obj_bar
  MatMult(ImG, u0_e, x_e);         // x = ImG*u0
  MatMultAdd(ReG, v0_e, x_e, x_e); // x = ReG*v0 + ImG*u0
  VecScale(x_e, dfb);               // x = -(ReG*v0 + ImG*u0)*obj_bar

  /* Derivative of purity */
  // VecAXPY(x, obj_bar, vfinal);

  MatMultTranspose(PxP, x_e, x_full);
  VecISCopy(rho0_bar, isv, SCATTER_FORWARD, x_full);  // set imaginary part in rho0bar


  /* Restore final, initial and adjoint state */
  VecRestoreSubVector(finalstate, isu, &ufinal_full);
  VecRestoreSubVector(finalstate, isv, &vfinal_full);
  VecRestoreSubVector(rho0, isu, &u0_full);
  VecRestoreSubVector(rho0, isv, &v0_full);

  /* Free vindex strides */
  ISDestroy(&isu);
  ISDestroy(&isv);
}


  XGate::XGate(std::vector<int> nlevels, std::vector<int> nessential) : Gate(nlevels, nessential) {

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

  /* Assemble vectorized target gate \bar VP \kron VP from  V = Va + i Vb */
  assembleGate();
}

XGate::~XGate() {}

YGate::YGate(std::vector<int> nlevels, std::vector<int> nessential) : Gate(nlevels, nessential) {

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

  /* Assemble vectorized target gate \bar VP \kron VP from  V = Va + i Vb*/
  assembleGate();
}
YGate::~YGate() {}

ZGate::ZGate(std::vector<int> nlevels, std::vector<int> nessential) : Gate(nlevels, nessential) {

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

  /* Assemble target gate \bar V \kron V from  V = Va + i Vb*/
  assembleGate();
}

ZGate::~ZGate() {}

HadamardGate::HadamardGate(std::vector<int> nlevels, std::vector<int> nessential) : Gate(nlevels, nessential) {

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

  /* Assemble target gate \bar V \kron V from  V = Va + i Vb*/
  assembleGate();
}
HadamardGate::~HadamardGate() {}



CNOT::CNOT(std::vector<int> nlevels, std::vector<int> nessential) : Gate(nlevels, nessential) {

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

  /* assemble V = \bar V \kron V */
  assembleGate();
}

CNOT::~CNOT(){}


SWAP::SWAP(std::vector<int> nlevels, std::vector<int> nessential) : Gate(nlevels, nessential) {

  assert(dim_ess == 4);

  /* Fill Va = Re(V) = V, Vb = Im(V) = 0 */
  if (mpirank_petsc == 0) {
    MatSetValue(Va, 0, 0, 1.0, INSERT_VALUES);
    MatSetValue(Va, 1, 2, 1.0, INSERT_VALUES);
    MatSetValue(Va, 2, 1, 1.0, INSERT_VALUES);
    MatSetValue(Va, 3, 3, 1.0, INSERT_VALUES);
    MatAssemblyBegin(Va, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Va, MAT_FINAL_ASSEMBLY);
  }

  /* assemble V = \bar V \kron V */
  assembleGate();
}

SWAP::~SWAP(){}

