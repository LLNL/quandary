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



CNOT::CNOT() : Gate(2) { // CNOT spans two qubits
  assert(dim == 16);

  lookup = new int[dim];
  for (int i=0; i<dim; i++) {
    lookup[i] = 0;
  }

  /* Fill the CNOT lookup table V\kron V! */
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
}

int CNOT::getIndex(int i) { return lookup[i]; }

CNOT::~CNOT(){
  delete [] lookup;
}


double CNOT::apply(int i, Vec state){

  const PetscScalar* x_ptr;
  VecGetArrayRead(state, &x_ptr);
  double out = x_ptr[getIndex(i)];
  VecRestoreArrayRead(state, &x_ptr);

  return out;
}



void CNOT::apply_diff(int i, Vec state_bar, double obj_bar){

  PetscScalar* x_ptr;
  VecGetArray(state_bar, &x_ptr);
  x_ptr[getIndex(i)] += 1.0 * obj_bar;
  VecRestoreArray(state_bar, &x_ptr);
}