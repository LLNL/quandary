#include "gate.hpp"

Gate::Gate(){
  nqubits = 0; 
  dim = 0;
  lookup = NULL;
}

Gate::Gate(int nqubits_){
  nqubits = nqubits_;
  dim = pow(nqubits,4);  

  lookup = new double[dim];
  for (int i=0; i<dim; i++) {
    lookup[i] = 0.0;
  }
}

Gate::~Gate(){
  delete [] lookup;
}


CNOT::CNOT() : Gate(2) { // CNOT spans two qubits

  /* TODO: Fill the lookup table ! */
}

CNOT::~CNOT(){}
