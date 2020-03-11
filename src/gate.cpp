#include "gate.hpp"

Gate::Gate(){
  nqubits = 0; 
  dim_vec = 0;
  dim_v   = 0;
  time = 0.0;
}

Gate::Gate(int nqubits_, const std::vector<double> freq_, double time_) {
  nqubits = nqubits_;
  time = time_;
  dim_v = 2*nqubits;           // 2-levels per qubit
  dim_vec = (int) pow(dim_v,2);      // vectorized version squares it.

  /* Set the frequencies */
  assert(freq_.size() >= nqubits);
  for (int i=0; i<nqubits; i++) {
    omega.push_back(2.*M_PI * freq_[i]);
  }

  /* Allocate Va, Vb */
  MatCreate(PETSC_COMM_WORLD, &Va);
  MatCreate(PETSC_COMM_WORLD, &Vb);
  MatSetSizes(Va, PETSC_DECIDE, PETSC_DECIDE, dim_v, dim_v);
  MatSetSizes(Vb, PETSC_DECIDE, PETSC_DECIDE, dim_v, dim_v);
  MatSetUp(Va);
  MatSetUp(Vb);
  MatAssemblyBegin(Va, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(Vb, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Va, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Vb, MAT_FINAL_ASSEMBLY);

  /* Allocate ReG = Re(\bar V \kron V), ImG = Im(\bar V \kron V) */
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


  /* Create vector strides for later use */
  ISCreateStride(PETSC_COMM_WORLD, dim_vec, 0, 1, &isu);
  ISCreateStride(PETSC_COMM_WORLD, dim_vec, dim_vec, 1, &isv);

  /* Create auxiliary vectors */
  MatCreateVecs(ReG, &ReG_col, NULL);
  MatCreateVecs(ReG, &ImG_col, NULL);

}

Gate::~Gate(){
  if (dim_vec == 0) return;
  ISDestroy(&isu);
  ISDestroy(&isv);
  MatDestroy(&ReG);
  MatDestroy(&ImG);
  MatDestroy(&Va);
  MatDestroy(&Vb);
  VecDestroy(&ReG_col);
  VecDestroy(&ImG_col);
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

void Gate::compare(int i, Vec state, double& delta){
  delta = 0.0;

  /* Exit, if this is a dummy gate */
  if (dim_vec == 0) {
    return;
  }

  int size;
  Vec u, v;
  const PetscScalar *uptr, *vptr, *ReGptr, *ImGptr;

  /* Get the i-th column of the gate matrix  \bar V\kron V */
  /* TODO: This might be slow! Find an alternative!  */
  MatGetColumnVector(ReG, ReG_col, i);
  MatGetColumnVector(ImG, ImG_col, i);
  VecGetArrayRead(ReG_col, &ReGptr);
  VecGetArrayRead(ImG_col, &ImGptr);

  /* Get u and v from state=[u,v] */
  VecGetSubVector(state, isu, &u);
  VecGetSubVector(state, isv, &v);
  VecGetArrayRead(u, &uptr);
  VecGetArrayRead(v, &vptr);

  /* Compute ||state - Gcolumn||^2 */
  VecGetSize(ReG_col, &size);
  for (int j=0; j<size; j++) {
    delta += pow(uptr[j] - ReGptr[j],2);
    delta += pow(vptr[j] - ImGptr[j],2);
  }

  /* Restore */
  VecRestoreArrayRead(u, &uptr);
  VecRestoreArrayRead(v, &vptr);
  VecRestoreSubVector(state, isu, &u);
  VecRestoreSubVector(state, isv, &v);
  VecRestoreArrayRead(ReG_col, &ReGptr);
  VecRestoreArrayRead(ImG_col, &ImGptr);
}

void Gate::compare_diff(int i, const Vec state, Vec state_bar, const double delta_bar){

  /* Exit, if this is a dummy gate */
  if (dim_vec == 0) {
    return;
  }
  int size;
  Vec u, v;
  Vec ubar, vbar;
  const PetscScalar *uptr, *vptr, *ReGptr, *ImGptr;
  PetscScalar *ubarptr, *vbarptr;

  /* Get the i-th column of the gate matrix  \bar V\kron V */
  /* TODO: This might be slow! Find an alternative!  */
  MatGetColumnVector(ReG, ReG_col, i);
  MatGetColumnVector(ImG, ImG_col, i);
  VecGetArrayRead(ReG_col, &ReGptr);
  VecGetArrayRead(ImG_col, &ImGptr);

  /* Get Real and imaginary of adjoint state */
  VecGetSubVector(state_bar, isu, &ubar);
  VecGetSubVector(state_bar, isv, &vbar);
  VecGetArray(ubar, &ubarptr);
  VecGetArray(vbar, &vbarptr);
  /* Get Real and imaginary of state */
  VecGetSubVector(state, isu, &u);
  VecGetSubVector(state, isv, &v);
  VecGetArrayRead(u, &uptr);
  VecGetArrayRead(v, &vptr);

  /* Derivative of || state - Gcolumn ||^2 */
  VecGetSize(ReG_col, &size);
  for (int j=0; j<size; j++){
    ubarptr[j] += 2. * (uptr[j] - ReGptr[j]) * delta_bar;
    vbarptr[j] += 2. * (vptr[j] - ImGptr[j]) * delta_bar;
  }

  /* Restore */
  VecRestoreArrayRead(u, &uptr);
  VecRestoreArrayRead(v, &vptr);
  VecRestoreSubVector(state, isu, &u);
  VecRestoreSubVector(state, isv, &v);
  VecRestoreArray(ubar, &ubarptr);
  VecRestoreArray(vbar, &vbarptr);
  VecRestoreSubVector(state_bar, isu, &ubar);
  VecRestoreSubVector(state_bar, isv, &vbar);
  VecRestoreArrayRead(ReG_col, &ReGptr);
  VecRestoreArrayRead(ImG_col, &ImGptr);
}

XGate::XGate(const std::vector<double>f, double time) : Gate(1, f, time) { // XGate spans one qubit
  assert(dim_v   == 2);
  assert(dim_vec == 4);
  assert(f.size() >= 1);

  /* Fill Va = Re(V) and Vb = Im(V), V = Va + iVb */
  /* Va = 0 1    Vb = 0
   *      1 0 
   */
  MatSetValue(Va, 0, 1, 1.0, INSERT_VALUES);
  MatSetValue(Va, 1, 0, 1.0, INSERT_VALUES);
  // Assemble
  MatAssemblyBegin(Va, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Va, MAT_FINAL_ASSEMBLY);


  /* Assemble \bar V \kron V from  V = Va + i Vb*/
  assembleGate();
}

XGate::~XGate() {}

YGate::YGate(const std::vector<double>f, double time) : Gate(1, f, time) { // XGate spans one qubit
  assert(dim_v   == 2);
  assert(dim_vec == 4);
  assert(f.size() >= 1);

  /* Fill A = Re(V) and B = Im(V), V = A + iB */
  /* A = 0     B = 0 -1
   *               1  0
   */
  MatSetValue(Vb, 0, 1, -1.0, INSERT_VALUES);
  MatSetValue(Vb, 1, 0,  1.0, INSERT_VALUES);

  // Assemble
  MatAssemblyBegin(Vb, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Vb, MAT_FINAL_ASSEMBLY);

  /* Assemble \bar V \kron V from  V = Va + i Vb*/
  assembleGate();
}

YGate::~YGate() {}

ZGate::ZGate(const std::vector<double>f, double time) : Gate(1, f, time) { // XGate spans one qubit
  assert(dim_v   == 2);
  assert(dim_vec == 4);
  assert(f.size() >= 1);

  /* Fill A = Re(V) and B = Im(V), V = A + iB */
  /* A =  1  0     B = 0
   *      0 -1
   */
  MatSetValue(Vb, 0, 0,  1.0, INSERT_VALUES);
  MatSetValue(Vb, 1, 1, -1.0, INSERT_VALUES);

  // Assemble
  MatAssemblyBegin(Vb, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Vb, MAT_FINAL_ASSEMBLY);

  /* Assemble \bar V \kron V from  V = Va + i Vb*/
  assembleGate();
}

ZGate::~ZGate() {}

HadamardGate::HadamardGate(const std::vector<double>f, double time) : Gate(1, f, time) { // XGate spans one qubit
  assert(dim_v   == 2);
  assert(dim_vec == 4);
  assert(f.size() >= 1);

  /* Fill A = Re(V) and B = Im(V), V = A + iB */
  /* A =  1  0     B = 0
   *      0 -1
   */
  double val = 1./sqrt(2);
  MatSetValue(Vb, 0, 0,  val, INSERT_VALUES);
  MatSetValue(Vb, 0, 1,  val, INSERT_VALUES);
  MatSetValue(Vb, 1, 0,  val, INSERT_VALUES);
  MatSetValue(Vb, 1, 1, -val, INSERT_VALUES);

  // Assemble
  MatAssemblyBegin(Vb, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Vb, MAT_FINAL_ASSEMBLY);

  /* Assemble \bar V \kron V from  V = Va + i Vb*/
  assembleGate();
}

HadamardGate::~HadamardGate() {}



CNOT::CNOT(const std::vector<double> f, double time) : Gate(2, f, time) { // CNOT spans two qubits
  assert(dim_vec == 16);
  assert(f.size() >= 2);

   /* Fill the CNOT lookup table V\kron V! */
  lookup = new int[dim_vec];
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
  rotation_Re = new double[dim_vec];
  rotation_Im = new double[dim_vec];

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
  rotRe = new double[4];
  rotIm = new double[4];
  rotRe[0] = 1.0;
  rotRe[1] = cos(o2t);
  rotRe[2] = cos(o1t);
  rotRe[3] = cos(o1t+o2t);
  rotIm[0] =  0.0;
  rotIm[1] =  sin(o2t);
  rotIm[2] =  sin(o1t);
  rotIm[3] =  sin(o1t+o2t);
 
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

  /* Fill Va = Re(V) = V, Vb = Im(V) = 0 */
  MatSetValue(Va, 0, 0, 1.0, INSERT_VALUES);
  MatSetValue(Va, 1, 1, 1.0, INSERT_VALUES);
  MatSetValue(Va, 2, 3, 1.0, INSERT_VALUES);
  MatSetValue(Va, 3, 2, 1.0, INSERT_VALUES);

  MatAssemblyBegin(Va, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Va, MAT_FINAL_ASSEMBLY);

  /* assemble V = \bar V \kron V */
  assembleGate();

}

int CNOT::getIndex(int i) { return lookup[i]; }

CNOT::~CNOT(){
  delete [] lookup;
  delete [] rotation_Re;
  delete [] rotation_Im;
  delete [] rotRe;
  delete [] rotIm;
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
