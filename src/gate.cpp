#include "gate.hpp"

Gate::Gate(){
  nqubits = 0; 
  dim = 0;
}

Gate::Gate(int nqubits_){
  nqubits = nqubits_;
  dim = pow(nqubits,4);  


}

Gate::~Gate(){}


CNOT::CNOT(double f1, double f2, double time) : Gate(2) { // CNOT spans two qubits
  assert(dim == 16);

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

  /* Transform frequencies to radian */
  omega1 = 2.*M_PI * f1;
  omega2 = 2.*M_PI * f2; 

  /* Fill the rotation matrix (diagonal!) */
  rotation_Re = new double[dim];
  rotation_Im = new double[dim];

  double o1t = omega1 * time;
  double o2t = omega2 * time;
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


  /* Create vector strides for later use */
  ISCreateStride(PETSC_COMM_WORLD, dim, 0, 1, &isu);
  ISCreateStride(PETSC_COMM_WORLD, dim, dim, 1, &isv);
  
}

int CNOT::getIndex(int i) { return lookup[i]; }

CNOT::~CNOT(){
  delete [] lookup;
  delete [] rotation_Re;
  delete [] rotation_Im;
  ISDestroy(&isu);
  ISDestroy(&isv);
}


void CNOT::apply(int i, Vec state, double& obj_re, double& obj_im){

  Vec u, v;
  const PetscScalar *re_ptr, *im_ptr;

  /* Get real (u) and imaginary (v) part of the state x = [u v] */
  VecGetSubVector(state, isu, &u);
  VecGetSubVector(state, isv, &v);
  VecGetArrayRead(u, &re_ptr);
  VecGetArrayRead(v, &im_ptr);

  int id = getIndex(i);

  /* obj_re = Re(x)^T Re(g_i) + Im(x)^T Im(g_i) */
  obj_re =  re_ptr[id] * rotation_Re[id] \
          + im_ptr[id] * rotation_Im[id];

  /* obj_im = Re(x)^T Im(g_i) - Im(x)^T Re(g_i) */
  obj_im =  re_ptr[id] * rotation_Im[id] \
          - im_ptr[id] * rotation_Re[id];

  /* Restore state */
  VecRestoreArrayRead(u, &re_ptr);
  VecRestoreArrayRead(v, &im_ptr);
  VecRestoreSubVector(state, isu, &u);
  VecRestoreSubVector(state, isv, &v);
}



void CNOT::apply_diff(int i, Vec state_bar, const double obj_re_bar, const double obj_im_bar){

  Vec u, v;
  PetscScalar *re_ptr, *im_ptr;

  /* Get real (u) and imaginary (v) part of the state x = [u v] */
  VecGetSubVector(state_bar, isu, &u);
  VecGetSubVector(state_bar, isv, &v);
  VecGetArray(u, &re_ptr);
  VecGetArray(v, &im_ptr);

  int id = getIndex(i);

  /* Set derivatives */
  re_ptr[id] +=   rotation_Re[id] * obj_re_bar;
  im_ptr[id] +=   rotation_Im[id] * obj_re_bar;
  re_ptr[id] +=   rotation_Im[id] * obj_im_bar;
  im_ptr[id] += - rotation_Re[id] * obj_im_bar;

  /* Restore state_bar */
  VecRestoreArray(u, &re_ptr);
  VecRestoreArray(v, &im_ptr);
  VecRestoreSubVector(state_bar, isu, &u);
  VecRestoreSubVector(state_bar, isv, &v);

}