#include <stdio.h>
#include <math.h>
#include <assert.h>
#pragma once

// enum GATETYPE = {CNOT};

class Gate {
  protected:
    int nqubits;    // number of qubits the gate spans. 
    int dim;      // dimension of vectorized system (=n^2^2)
    int* lookup; // look-up table of vectorized matrix.

  public:
    Gate();
    Gate(int nbits_);
    virtual ~Gate();
    
    /* Return the gate's index from the lookup table */
    int getIndex(int i);

};

class CNOT : public Gate {

  public:
    CNOT();
    ~CNOT();
};