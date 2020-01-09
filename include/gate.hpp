#include <stdio.h>
#include <math.h>
#pragma once

// enum GATETYPE = {CNOT};

class Gate {
  protected:
    int nqubits;    // number of qubits the gate spans. 
    int dim;      // dimension of vectorized system (=n^2^2)
    double* lookup; // look-up table of vectorized matrix.

  public:
    Gate();
    Gate(int nbits_);
    virtual ~Gate();

};

class CNOT : public Gate {

  public:
    CNOT();
    ~CNOT();
};