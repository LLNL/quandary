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

    /* compare the k-th column of the gate to a state vector */
    /* in Frobenius norm ||w_k - g_k||^2_F = w_k^dag w_k + g_k^dag g_k - 2*Re(w_k^dag g_k) */
    virtual void compare(int i, Vec state, double& delta) = 0;
    virtual void compare_diff(int i, const Vec state, Vec state_bar, const double delta_bar) = 0;

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

    /* compare the k-th column of the gate to a state vector */
    /* in Frobenius norm ||w_k - g_k||^2_F = w_k^dag w_k + g_k^dag g_k - 2*Re(w_k^dag g_k) */
    virtual void compare(int i, Vec state, double& delta);
    virtual void compare_diff(int i, const Vec state, Vec state_bar, const double delta_bar);
};