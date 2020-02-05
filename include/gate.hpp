#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <petscmat.h>
#include <vector>
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
    
    /* Apply i-th column of the gate to a state vector.
     * Out: real and imaginary part of state^T Gate_i */
    virtual void apply(int i, Vec state, double& obj_Re, double& obj_Im) = 0;

    /* Derivative of gate application */
    virtual void apply_diff(int i, Vec state_bar, const double obj_Re_bar, const double obj_Im_bar) = 0;

};

/* CNOT Gate, spanning two qubits. This class is mostly hardcoded. TODO: Generalize!
 * V = 1 0 0 0
 *     0 1 0 0 
 *     0 0 0 1
 *     0 0 1 0 
 */
class CNOT : public Gate {
  protected:
    double omega1, omega2;  /* Frequencies for first and second oscillator omega = 2\pi*fa radian */
    int* lookup;   /* look-up table for vectorized V\kronV */
    double* rotation_Re;  /* Transform gate to rotational frame. Real part. */
    double* rotation_Im;  /* Transform gate to rotational frame. Imaginary part. */

    IS isu, isv;
    
    /* Return the CNOT lookup index */
    int getIndex(int i);

  public:
    /* Constructor takes ground frequencies, and time when to apply the gate. */
    CNOT(const std::vector<double> f, double time);
    ~CNOT();

    /* Apply i-th column of the gate to a state vector */
    virtual void apply(int i, Vec state, double& obj_re, double& obj_im);

    /* Derivative of gate application */
    virtual void apply_diff(int i, Vec state_bar, const double obj_re_bar, const double obj_im_bar);
};