#include <stdio.h>
#include <math.h>
#include <assert.h>
#pragma once

// enum GATETYPE = {CNOT};

class Gate {
  protected:
    int nqubits;    // number of qubits the gate spans. 
    int dim;      // dimension of vectorized system (=n^2^2)

  public:
    Gate();
    Gate(int nbits_);
    virtual ~Gate();
    
    /* Apply i-th column of the gate to a state vector */
    virtual double apply(int i, const double* state) = 0;

    /* Derivative of gate application */
    virtual void apply_diff(int i, double* state_bar) = 0;

};

/* CNOT Gate
 * V = 1 0 0 0
 *     0 1 0 0 
 *     0 0 0 1
 *     0 0 1 0 
 */
class CNOT : public Gate {
  protected:
    int* lookup;   /* look-up table for vectorized V\kronV */
    
    /* Return the CNOT lookup index */
    int getIndex(int i);

  public:
    CNOT();
    ~CNOT();

    /* Apply i-th column of the gate to a state vector */
    virtual double apply(int i, const double* state);

    /* Derivative of gate application */
    virtual void apply_diff(int i, double* state_bar);
};