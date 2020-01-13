#include "gate.hpp"

Gate::Gate(){
  nqubits = 0; 
  dim = 0;
  lookup = NULL;
}

Gate::Gate(int nqubits_){
  nqubits = nqubits_;
  dim = pow(nqubits,4);  

  lookup = new int[dim];
  for (int i=0; i<dim; i++) {
    lookup[i] = 0;
  }
}

Gate::~Gate(){
  delete [] lookup;
}

int Gate::getIndex(int i) { return lookup[i]; }


CNOT::CNOT() : Gate(2) { // CNOT spans two qubits

  assert(dim == 16);

  /* Fill the lookup table ! */
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
  lookup[15] = 12;
}

CNOT::~CNOT(){}
