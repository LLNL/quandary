#include "gate.hpp"

Gate::Gate(){
  dim_v   = 0;
  dim_vec = 0;
}

Gate::Gate(int dim_v_) {
  dim_v = dim_v_; 
  dim_vec = (int) pow(dim_v,2);      // vectorized version squares dimensions.

  MPI_Comm_rank(PETSC_COMM_WORLD, &mpirank_petsc);

  // /* Set the frequencies */
  // assert(freq_.size() >= noscillators);
  // for (int i=0; i<noscillators; i++) {
  //   omega.push_back(2.*M_PI * freq_[i]);
  // }

  /* Allocate Va, Vb, sequential, only on proc 0 */
  MatCreateSeqDense(PETSC_COMM_SELF, dim_v, dim_v, NULL, &Va);
  MatCreateSeqDense(PETSC_COMM_SELF, dim_v, dim_v, NULL, &Vb);
  MatSetUp(Va);
  MatSetUp(Vb);
  MatAssemblyBegin(Va, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(Vb, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Va, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Vb, MAT_FINAL_ASSEMBLY);

  /* Allocate ReG = Re(\bar V \kron V), ImG = Im(\bar V \kron V), parallel */
  MatCreate(PETSC_COMM_WORLD, &ReG);
  MatCreate(PETSC_COMM_WORLD, &ImG);
  MatSetSizes(ReG, PETSC_DECIDE, PETSC_DECIDE, dim_vec, dim_vec);
  MatSetSizes(ImG, PETSC_DECIDE, PETSC_DECIDE, dim_vec, dim_vec);
  MatSetUp(ReG);
  MatSetUp(ImG);
  MatAssemblyBegin(ReG, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ReG, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(ImG, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ImG, MAT_FINAL_ASSEMBLY);

  /* Create auxiliary vectors */
  MatCreateVecs(ReG, &x, NULL);

}

Gate::~Gate(){
  if (dim_vec == 0) return;
  MatDestroy(&ReG);
  MatDestroy(&ImG);
  MatDestroy(&Va);
  MatDestroy(&Vb);
  VecDestroy(&x);
}


void Gate::assembleGate(){
  /* Compute ReG = Re(\bar V \kron V) = A\kron A + B\kron B  */
  AkronB(dim_v, Va, Va,  1.0, &ReG, ADD_VALUES);
  AkronB(dim_v, Vb, Vb,  1.0, &ReG, ADD_VALUES);
  /* Compute ImG = Im(\bar V\kron V) = A\kron B - B\kron A */
  AkronB(dim_v, Va, Vb,  1.0, &ImG, ADD_VALUES);
  AkronB(dim_v, Vb, Va, -1.0, &ImG, ADD_VALUES);

  MatAssemblyBegin(ReG, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ReG, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(ImG, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ImG, MAT_FINAL_ASSEMBLY);

  // MatView(ReG, PETSC_VIEWER_STDOUT_WORLD);
  // MatView(ImG, PETSC_VIEWER_STDOUT_WORLD);
  // exit(1);
}


void Gate::compare(const Vec finalstate, const Vec rho0, double& frob){
  frob = 0.0;

  /* Exit, if this is a dummy gate */
  if (dim_vec == 0) {
    return;
  }

  /* Create vector strides for accessing real and imaginary part of co-located x */
  int ilow, iupp;
  VecGetOwnershipRange(finalstate, &ilow, &iupp);
  int dimis = (iupp - ilow)/2;
  IS isu, isv;
  ISCreateStride(PETSC_COMM_WORLD, dimis, ilow, 2, &isu);
  ISCreateStride(PETSC_COMM_WORLD, dimis, ilow+1, 2, &isv);

  /* Get real and imag part of final state, x = [u,v] */
  Vec ufinal, vfinal, u0, v0;
  VecGetSubVector(finalstate, isu, &ufinal);
  VecGetSubVector(finalstate, isv, &vfinal);
  VecGetSubVector(rho0, isu, &u0);
  VecGetSubVector(rho0, isv, &v0);


  /* Make sure that state dimensions match the gate dimension */
  int dimstate, dimG;
  VecGetSize(finalstate, &dimstate); 
  VecGetSize(x, &dimG);
  if (dimstate/2 != dimG) {
    printf("\n ERROR: Target gate dimension %d doesn't match system dimension %u\n", dimG, dimstate/2);
    exit(1);
  }

  /* Add real part of frobenius norm || u - ReG*u0 + ImG*v0 ||^2 */
  MatMult(ReG, u0, x);            // x = ReG*u0
  VecAYPX(x, -1.0, ufinal);       // x = ufinal - ReG*u0 
  MatMultAdd(ImG, v0, x, x);      // x = ufinal - ReG*u0 + ImG*v0
  double norm;
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

  /* Free index strides */
  ISDestroy(&isu);
  ISDestroy(&isv);
}


void Gate::compare_diff(const Vec finalstate, const Vec rho0, Vec rho0_bar, const double frob_bar){

  /* Exit, if this is a dummy gate */
  if (dim_vec == 0) {
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
  Vec ufinal, vfinal, u0, v0;
  VecGetSubVector(finalstate, isu, &ufinal);
  VecGetSubVector(finalstate, isv, &vfinal);
  VecGetSubVector(rho0, isu, &u0);
  VecGetSubVector(rho0, isv, &v0);

  /* Derivative of 1/2 * J */
  double dfb = 1./2. * frob_bar;

  /* Derivative of read part of frobenius norm: 2 * (u - ReG*u0 + ImG*v0) * dfb */
  MatMult(ReG, u0, x);            // x = ReG*u0
  VecAYPX(x, -1.0, ufinal);       // x = ufinal - ReG*u0 
  MatMultAdd(ImG, v0, x, x);      // x = ufinal - ReG*u0 + ImG*v0
  VecScale(x, 2*dfb);             // x = 2*(ufinal - ReG*u0 + ImG*v0)*dfb
  VecISCopy(rho0_bar, isu, SCATTER_FORWARD, x);  // set real part in rho0bar

  /* Derivative of imaginary part of frobenius norm 2 * (v - ReG*v0 - ImG*u0) * dfb */
  MatMult(ReG, v0, x);         // x = ReG*v0
  MatMultAdd(ImG, u0, x, x);   // x = ReG*v0 + ImG*u0
  VecAYPX(x, -1.0, vfinal);     // x = vfinal - (ReG*v0 + ImG*u0)
  VecScale(x, 2*dfb);          // x = 2*(vfinal - (ReG*v0 + ImG*u0)*dfb
  VecISCopy(rho0_bar, isv, SCATTER_FORWARD, x);  // set imaginary part in rho0bar

  /* Restore final, initial and adjoint state */
  VecRestoreSubVector(finalstate, isu, &ufinal);
  VecRestoreSubVector(finalstate, isv, &vfinal);
  VecRestoreSubVector(rho0, isu, &u0);
  VecRestoreSubVector(rho0, isv, &v0);

  /* Free vindex strides */
  ISDestroy(&isu);
  ISDestroy(&isv);
}

XGate::XGate() : Gate(2) {

  /* Fill Va = Re(V) and Vb = Im(V), V = Va + iVb */
  /* Va = 0 1    Vb = 0
   *      1 0 
   */
  if (mpirank_petsc == 0) {
    MatSetValue(Va, 0, 1, 1.0, INSERT_VALUES);
    MatSetValue(Va, 1, 0, 1.0, INSERT_VALUES);
    MatAssemblyBegin(Va, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Va, MAT_FINAL_ASSEMBLY);
  }

  /* Assemble vectorized target gate \bar V \kron V from  V = Va + i Vb*/
  assembleGate();
}

XGate::~XGate() {}

YGate::YGate() : Gate(2) { 
  
  /* Fill A = Re(V) and B = Im(V), V = A + iB */
  /* A = 0     B = 0 -1
   *               1  0
   */
  if (mpirank_petsc == 0) {
    MatSetValue(Vb, 0, 1, -1.0, INSERT_VALUES);
    MatSetValue(Vb, 1, 0,  1.0, INSERT_VALUES);
    MatAssemblyBegin(Vb, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Vb, MAT_FINAL_ASSEMBLY);
  }

  /* Assemble vectorized target gate \bar V \kron V from  V = Va + i Vb*/
  assembleGate();
}
YGate::~YGate() {}

ZGate::ZGate() : Gate(2) { 

  /* Fill A = Re(V) and B = Im(V), V = A + iB */
  /* A =  1  0     B = 0
   *      0 -1
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

HadamardGate::HadamardGate() : Gate(2) { 

  /* Fill A = Re(V) and B = Im(V), V = A + iB */
  /* A =  1  0     B = 0
   *      0 -1
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



CNOT::CNOT() : Gate(4) {

  /* Fill Va = Re(V) = V, Vb = Im(V) = 0 */
  if (mpirank_petsc == 0) {
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

