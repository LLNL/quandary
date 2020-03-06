#include "gate.hpp"

Gate::Gate(){
  nqubits = 0; 
  dim = 0;
  time = 0.0;
}

// Gate::Gate(int nqubits_){
Gate::Gate(int nqubits_, const std::vector<double> freq_, double time_) {
  nqubits = nqubits_;
  time = time_;
  dim = pow(2 * nqubits,2);      // 2-levels per qubit, vectorized version squares it.

  /* Set the frequencies */
  assert(freq_.size() >= nqubits);
  for (int i=0; i<nqubits; i++) {
    omega.push_back(2.*M_PI * freq_[i]);
  }

  /* Allocate Re(\bar V \kron V), Im(\bar V \kron V) */
  MatCreate(PETSC_COMM_WORLD, &ReG);
  MatCreate(PETSC_COMM_WORLD, &ImG);
  MatSetSizes(ReG, PETSC_DECIDE, PETSC_DECIDE, dim, dim);
  MatSetSizes(ImG, PETSC_DECIDE, PETSC_DECIDE, dim, dim);
  MatSetUp(ReG);
  MatSetUp(ImG);

  /* Create vector strides for later use */
  ISCreateStride(PETSC_COMM_WORLD, dim, 0, 1, &isu);
  ISCreateStride(PETSC_COMM_WORLD, dim, dim, 1, &isv);
  

}

Gate::~Gate(){
  ISDestroy(&isu);
  ISDestroy(&isv);
  MatDestroy(&ReG);
  MatDestroy(&ImG);
  MatDestroy(&RVa);
  MatDestroy(&RVb);
}

void Gate::apply(int i, Vec state, double& obj_Re, double& obj_Im){
  obj_Re = 0.0;
  obj_Im = 0.0;

  /* Exit, if this is a dummy gate */
  if (dim == 0) {
    return;
  }
}

void Gate::compare(int i, Vec state, double& delta){
  delta = 0.0;

  /* Exit, if this is a dummy gate */
  if (dim == 0) {
    return;
  }

  /* Get the i-th column of the gate matrix  \bar V\kron V */
  /* TODO: This might be slow! Find an alternative!  */
  Vec ReG_col, ImG_col;
  MatCreateVecs(ReG, &ReG_col, NULL);
  MatCreateVecs(ReG, &ImG_col, NULL);
  MatGetColumnVector(ReG, ReG_col, i);
  MatGetColumnVector(ImG, ImG_col, i);

  /* first term state^\dag state */
  double norm;
  VecNorm(state, NORM_2, &norm);
  delta += norm*norm;

  /* Second term gate_i^\dag gate_i */
  VecNorm(ReG_col, NORM_2, &norm); 
  delta += norm*norm;
  VecNorm(ImG_col, NORM_2, &norm);
  delta += norm*norm;

  /* third term -2*Re(state^\dag gate_i) */
  Vec u, v;
  VecGetSubVector(state, isu, &u);
  VecGetSubVector(state, isv, &v);
  VecDot(u, ReG_col, &norm);
  delta -= 2. * norm;
  VecDot(v, ImG_col, &norm);
  delta -= 2. * norm;
  VecRestoreSubVector(state, isu, &u);
  VecRestoreSubVector(state, isv, &v);

}

void Gate::compare_diff(int i, const Vec state, Vec state_bar, const double delta_bar){

  /* Exit, if this is a dummy gate */
  if (dim == 0) {
    return;
  }

  /* Get the i-th column of the gate matrix  \bar V\kron V */
  /* TODO: This might be slow! Find an alternative!  */
  Vec ReG_col, ImG_col;
  MatCreateVecs(ReG, &ReG_col, NULL);
  MatCreateVecs(ImG, &ImG_col, NULL);
  MatGetColumnVector(ReG, ReG_col, i);
  MatGetColumnVector(ImG, ImG_col, i);

  /* Get Real and imaginary adjoint state_bar = u+iv */
  Vec ubar, vbar;
  VecGetSubVector(state_bar, isu, &ubar);
  VecGetSubVector(state_bar, isv, &vbar);

  /* Derivative of third term: -2*Gate_k*delta_bar */
  VecAXPY(ubar, -2.*delta_bar, ReG_col);
  VecAXPY(vbar, -2.*delta_bar, ImG_col);

  /* Derivative of second term: 0.0 */

  /* Derivative of first term: xbar += 2*objbar*x  */
  VecAXPY(state_bar, 2.*delta_bar, state);

  /* Restore state_bar */
  VecRestoreSubVector(state_bar, isu, &ubar);
  VecRestoreSubVector(state_bar, isv, &vbar);
}

void Gate::fidelity(int i, Vec state, double& fid_re, double& fid_im){
  fid_re = 0.0;
  fid_im = 0.0;
  
  /* Exit, if this is a dummy gate */
  if (dim == 0) {
    return;
  }

  /* Get the i-th column of RVa, RVb */
  /* TODO: This might be slow! Find an alternative!  */
  Vec RVa_col, RVb_col;
  int colid = i % ((int) sqrt(dim));
  MatCreateVecs(RVa, &RVa_col, NULL);
  MatCreateVecs(RVb, &RVb_col, NULL);
  MatGetColumnVector(RVa, RVa_col, colid);
  MatGetColumnVector(RVb, RVb_col, colid);

  /* Get real (u) and imaginary (v) part of x = [u v] */
  Vec u, v;
  VecGetSubVector(state, isu, &u);
  VecGetSubVector(state, isv, &v);
  const PetscScalar *uptr, *vptr, *rvaptr, *rvbptr;
  VecGetArrayRead(u, &uptr);
  VecGetArrayRead(v, &vptr);
  VecGetArrayRead(RVa_col, &rvaptr);
  VecGetArrayRead(RVb_col, &rvbptr);

  // VecView(u, PETSC_VIEWER_STDOUT_WORLD);
  // VecView(v, PETSC_VIEWER_STDOUT_WORLD);
  // VecView(RVa_col, PETSC_VIEWER_STDOUT_WORLD);
  // VecView(RVb_col, PETSC_VIEWER_STDOUT_WORLD);

  /* compute (real) u^T RVa_k + v^T RVb_k and (imag) u^T RVb_k - v^T RVa_k */
  int size;
  VecGetSize(u, &size);
  int k = 0; 
  for (int j=0; j<dim; j++){
    if (j % ((int)sqrt(dim)+1) == 0 ) { // this hits the diagonal elements
    printf("\nAdding %d %d ure %f rvare %f", i, j, uptr[j], rvaptr[k]);
      fid_re += uptr[j] * rvaptr[k] + vptr[j] * rvbptr[k];
      fid_im += uptr[j] * rvbptr[k] - vptr[j] * rvaptr[k];
      k++;
    }
  }

  /* Restore state */
  VecRestoreArrayRead(u, &uptr);
  VecRestoreArrayRead(v, &uptr);
  VecRestoreArrayRead(RVa_col, &rvaptr);
  VecRestoreArrayRead(RVb_col, &rvbptr);
  VecRestoreSubVector(state, isu, &u);
  VecRestoreSubVector(state, isv, &v);
}


XGate::XGate(const std::vector<double>f, double time) : Gate(1, f, time) { // XGate spans one qubit
  assert(dim == 4);
  assert(f.size() >= 1);

  int Vdim = 2*nqubits;
 
  Mat A, B; 
  MatCreate(PETSC_COMM_WORLD, &A);
  MatCreate(PETSC_COMM_WORLD, &B);
  MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, Vdim, Vdim);
  MatSetSizes(B, PETSC_DECIDE, PETSC_DECIDE, Vdim, Vdim);
  MatSetUp(A);
  MatSetUp(B);

  /* Fill A = Re(V) and B = Im(V), V = A + iB */
  /* A = 0 1    B = 0
   *     1 0 
   */
  MatSetValue(A, 0, 1, 1.0, INSERT_VALUES);
  MatSetValue(A, 1, 0, 1.0, INSERT_VALUES);

  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);

  /* Compute Re(\bar V \kron V) = A\kron A + B\kron B  */
  AkronB(Vdim, A, A,  1.0, &ReG, ADD_VALUES);
  AkronB(Vdim, B, B,  1.0, &ReG, ADD_VALUES);
  /* Compute Im(\bar V\kron V) = A\kron B - B\kron A */
  AkronB(Vdim, A, B,  1.0, &ImG, ADD_VALUES);
  AkronB(Vdim, B, A, -1.0, &ImG, ADD_VALUES);

  MatAssemblyBegin(ReG, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ReG, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(ImG, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ImG, MAT_FINAL_ASSEMBLY);

  MatView(ReG, PETSC_VIEWER_STDOUT_WORLD);
  MatView(ImG, PETSC_VIEWER_STDOUT_WORLD);
  exit(1);

  MatDestroy(&A);
  MatDestroy(&B);
}

XGate::~XGate() {}


CNOT::CNOT(const std::vector<double> f, double time) : Gate(2, f, time) { // CNOT spans two qubits
  assert(dim == 16);
  assert(f.size() >= 2);

   /* Fill the CNOT lookup table V\kron V! */
  lookup = new int[dim];
  lookup[0] = 0;
  lookup[1] = 1;
  lookup[2] = 3;
  lookup[3] = 2;
  lookup[4] = 4;
  lookup[5] = 5;
  lookup[6] = 7;
  lookup[7] = 6;

  lookup[8] = 12;
  lookup[9] = 13;
  lookup[10] = 15;
  lookup[11] = 14;
  lookup[12] = 8;
  lookup[13] = 9;
  lookup[14] = 11;
  lookup[15] = 10;

  /* Create the rotation matrix (diagonal!) */
  rotation_Re = new double[dim];
  rotation_Im = new double[dim];

  double o1t = omega[0] * time;
  double o2t = omega[1] * time;

  rotation_Re[0] = 1.0;            rotation_Im[0] = 0.0;
  rotation_Re[1] = cos(o2t);       rotation_Im[1] = sin(o2t);
  rotation_Re[2] = cos(o1t);       rotation_Im[2] = sin(o1t);
  rotation_Re[3] = cos(o1t+o2t);   rotation_Im[3] = sin(o1t+o2t);
  rotation_Re[4] = cos(o2t);       rotation_Im[4] = -sin(o2t);
  rotation_Re[5] = 1.0;            rotation_Im[5] = 0.0;
  rotation_Re[6] = cos(o1t-o2t);   rotation_Im[6] = sin(o1t-o2t);
  rotation_Re[7] = cos(o1t);       rotation_Im[7] = sin(o1t);
  rotation_Re[8] = cos(o1t);       rotation_Im[8] = -sin(o1t);
  rotation_Re[9] = cos(o2t-o1t);   rotation_Im[9] = sin(o2t-o1t);
  rotation_Re[10] = 1.0;           rotation_Im[10] = 0.0;
  rotation_Re[11] = cos(o2t);      rotation_Im[11] = sin(o2t);
  rotation_Re[12] = cos(o1t+o2t);  rotation_Im[12] = -sin(o1t+o2t);
  rotation_Re[13] = cos(o1t);      rotation_Im[13] = -sin(o1t);
  rotation_Re[14] = cos(o2t);      rotation_Im[14] = -sin(o2t);
  rotation_Re[15] = 1.0;           rotation_Im[15] = 0.0;

  int Vdim = 2*nqubits;

  /* Compute rotation R = rotRe + i rotIm*/
  double *rotRe = new double[4];
  double *rotIm = new double[4];
  rotRe[0] = 1.0;
  rotRe[1] = cos(o2t);
  rotRe[2] = cos(o1t);
  rotRe[3] = cos(o1t+o2t);
  rotIm[0] =  0.0;
  rotIm[1] =  sin(o2t);
  rotIm[2] =  sin(o1t);
  rotIm[3] =  sin(o1t+o2t);
 
  MatCreate(PETSC_COMM_WORLD, &RVa);
  MatCreate(PETSC_COMM_WORLD, &RVb);
  MatSetSizes(RVa, PETSC_DECIDE, PETSC_DECIDE, Vdim, Vdim);
  MatSetSizes(RVb, PETSC_DECIDE, PETSC_DECIDE, Vdim, Vdim);
  MatSetUp(RVa);
  MatSetUp(RVb);

  // /* Fill A = Re(Rot*V) = Re(Rot)*Re(V) - Im(Rot)*Im(V) */
  // MatSetValue(RVa, 0, 0, 1.0 * rotRe[0] - 0.0 * rotIm[0], INSERT_VALUES);
  // MatSetValue(RVa, 1, 1, 1.0 * rotRe[1] - 0.0 * rotIm[1], INSERT_VALUES);
  // MatSetValue(RVa, 2, 3, 1.0 * rotRe[2] - 0.0 * rotIm[2], INSERT_VALUES);
  // MatSetValue(RVa, 3, 2, 1.0 * rotRe[3] - 0.0 * rotIm[3], INSERT_VALUES);
  // /* Fill B = Im(Rot*V) = Im(Rot)*Re(V) + Re(Rot)*Im(V) */
  // MatSetValue(RVb, 0, 0, 1.0 * rotIm[0] + 0.0 * rotRe[0], INSERT_VALUES);
  // MatSetValue(RVb, 1, 1, 1.0 * rotIm[1] + 0.0 * rotRe[1], INSERT_VALUES);
  // MatSetValue(RVb, 2, 3, 1.0 * rotIm[2] + 0.0 * rotRe[2], INSERT_VALUES);
  // MatSetValue(RVb, 3, 2, 1.0 * rotIm[3] + 0.0 * rotRe[3], INSERT_VALUES);

  /* Fill A = Re(V)  */
  MatSetValue(RVa, 0, 0, 1.0, INSERT_VALUES);
  MatSetValue(RVa, 1, 1, 1.0, INSERT_VALUES);
  MatSetValue(RVa, 2, 3, 1.0, INSERT_VALUES);
  MatSetValue(RVa, 3, 2, 1.0, INSERT_VALUES);

  MatAssemblyBegin(RVa, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(RVa, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(RVb, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(RVb, MAT_FINAL_ASSEMBLY);

  /* Compute Re(\bar V \kron V) = A\kron A + B\kron B  */
  AkronB(Vdim, RVa, RVa,  1.0, &ReG, ADD_VALUES);
  AkronB(Vdim, RVb, RVb,  1.0, &ReG, ADD_VALUES);
  /* Compute Im(\bar V\kron V) = A\kron B - B\kron A */
  AkronB(Vdim, RVa, RVb,  1.0, &ImG, ADD_VALUES);
  AkronB(Vdim, RVb, RVa, -1.0, &ImG, ADD_VALUES);

  MatAssemblyBegin(ReG, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ReG, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(ImG, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(ImG, MAT_FINAL_ASSEMBLY);

  delete[] rotRe;
  delete[] rotIm;
}

int CNOT::getIndex(int i) { return lookup[i]; }

CNOT::~CNOT(){
  delete [] lookup;
  delete [] rotation_Re;
  delete [] rotation_Im;
}


// void CNOT::apply(int i, Vec state, double& obj_re, double& obj_im){

//   Vec u, v;
//   const PetscScalar *re_ptr, *im_ptr;

//   /* Get real (u) and imaginary (v) part of the state x = [u v] */
//   VecGetSubVector(state, isu, &u);
//   VecGetSubVector(state, isv, &v);
//   VecGetArrayRead(u, &re_ptr);
//   VecGetArrayRead(v, &im_ptr);

//   int id = getIndex(i);

//   /* obj_re = Re(x)^T Re(g_i) + Im(x)^T Im(g_i) */
//   printf("\nAdd %d -> %d ure %f, rotRe %f\n", i, id, re_ptr[id], rotation_Re[id]);
//   obj_re =  re_ptr[id] * rotation_Re[id] \
//           + im_ptr[id] * rotation_Im[id];

//   /* obj_im = Re(x)^T Im(g_i) - Im(x)^T Re(g_i) */
//   obj_im =  re_ptr[id] * rotation_Im[id] \
//           - im_ptr[id] * rotation_Re[id];

//   /* Restore state */
//   VecRestoreArrayRead(u, &re_ptr);
//   VecRestoreArrayRead(v, &im_ptr);
//   VecRestoreSubVector(state, isu, &u);
//   VecRestoreSubVector(state, isv, &v);
// }



// void CNOT::compare(int i, Vec state, double& delta) {

//   delta = 0.0;

//   /* first term state^dag state */
//   double t1;
//   VecNorm(state, NORM_2, &t1);
//   delta += t1*t1;

//   /* second term gate_i^\dag gate_i */
//   int id = getIndex(i);
//   delta += pow(rotation_Re[id],2) + pow(rotation_Im[id],2); // this should be one always !?
   
//   /* Third term 2*Re(state^dag gate_i) */
//   Vec u, v;
//   const PetscScalar *re_ptr, *im_ptr;
//   VecGetSubVector(state, isu, &u);
//   VecGetSubVector(state, isv, &v);
//   VecGetArrayRead(u, &re_ptr);
//   VecGetArrayRead(v, &im_ptr);

//   delta -= 2. * ( re_ptr[id] * rotation_Re[id] + im_ptr[id] * rotation_Im[id] );

//   VecRestoreArrayRead(u, &re_ptr);
//   VecRestoreArrayRead(v, &im_ptr);
//   VecRestoreSubVector(state, isu, &u);
//   VecRestoreSubVector(state, isv, &v);

// }


// void CNOT::compare_diff(int i, const Vec state, Vec state_bar, const double delta_bar){

//   Vec u, v;
//   PetscScalar *re_ptr, *im_ptr;

//   /* Derivative of third term: -2*Rot[id]*objbar */
//   VecGetSubVector(state_bar, isu, &u);
//   VecGetSubVector(state_bar, isv, &v);
//   VecGetArray(u, &re_ptr);
//   VecGetArray(v, &im_ptr);
//   int id = getIndex(i);
  
//   re_ptr[id] = - 2.*rotation_Re[id]*delta_bar;
//   im_ptr[id] = - 2.*rotation_Im[id]*delta_bar;

//   VecRestoreArray(u, &re_ptr);
//   VecRestoreArray(v, &im_ptr);
//   VecRestoreSubVector(state_bar, isu, &u);
//   VecRestoreSubVector(state_bar, isv, &v);

//   /* Derivative of second term: 0.0 */

//   /* Derivative of first term: xbar += 2*objbar*x  */
//   VecAXPY(state_bar, 2.*delta_bar, state);



// }
