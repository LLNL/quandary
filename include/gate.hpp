#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <petscmat.h>
#include <vector>
#include "util.hpp"
#pragma once


class Gate {
  protected:
    int nqubits;               /* number of qubits the gate spans.  */
    int dim_vec;                   /* dimension of vectorized system (=n^2^2) */
    double time;               /* Time when to apply the gate */
    std::vector<double> omega; /* Frequencies per oscillator omega = 2\pi*f radian */

    int dim_v;   // dimension of target gate (non-vectorized. dimv = sqrt(dim_vec))
    Mat Va, Vb;  // Input per gate: Real and imaginary part of V_target 

    Mat ReG, ImG;  // Real and imaginary part of \bar RV \kron RV
    IS isu, isv;   // Vector strides for extracting u,v from x = [u,v]

  private:
    Vec ReG_col, ImG_col; // auxiliary vectors for computing frobenius norm 
    Vec Va_col, Vb_col;   // auxiliary vectors for computing fidelity

  public:
    Gate();
    Gate(int nqubits_, const std::vector<double> freq_, double time_);
    virtual ~Gate();

    /* Assemble ReG = Re(\bar V \kron V) and ImG = Im(\bar V \kron V) */
    void assembleGate();
    
    /* Apply i-th column of the gate to a state vector.
     * Out: real and imaginary part of state^T Gate_i */
    // virtual void apply(int i, Vec state, double& obj_Re, double& obj_Im);

    /* compare the k-th column of the gate to a state vector */
    /* in Frobenius norm ||w_k - g_k||^2_F = w_k^dag w_k + g_k^dag g_k - 2*Re(w_k^dag g_k) */
    void compare(int i, Vec state, double& delta);
    void compare_diff(int i, const Vec state, Vec state_bar, const double delta_bar);

    /* Compute real and imaginary part of fidelity trace term */
    /* fid_re = Re(state^\dag RV_i), fid_im = Im(state^\dag RV_i) */
    void fidelity(int i, Vec state, double& fid_re, double& fid_im);

};

/* X Gate, spanning one qubit. 
 * V = 0 1
 *     1 0
 */

class XGate : public Gate {

  public:
    /* Constructor takes ground frequencies, and time when to apply the gate. */
    XGate(const std::vector<double> f, double time);
    ~XGate();
};

/* Y Gate, spanning one qubit. 
 * V = 0 -i
 *     i  0
 */

class YGate : public Gate {

  public:
    /* Constructor takes ground frequencies, and time when to apply the gate. */
    YGate(const std::vector<double> f, double time);
    ~YGate();
};

/* Z Gate, spanning one qubit. 
 * V = 1   0
 *     0  -1
 */
class ZGate : public Gate {

  public:
    /* Constructor takes ground frequencies, and time when to apply the gate. */
    ZGate(const std::vector<double> f, double time);
    ~ZGate();
};

/* Hadamard Gate, spanning one qubit. 
 * V = 1/sqrt(2) * | 1   1 |
 *                 | 1  -1 |
 */
class HadamardGate : public Gate {

  public:
    /* Constructor takes ground frequencies, and time when to apply the gate. */
    HadamardGate(const std::vector<double> f, double time);
    ~HadamardGate();
};




/* CNOT Gate, spanning two qubits. This class is mostly hardcoded. TODO: Generalize!
 * V = 1 0 0 0
 *     0 1 0 0 
 *     0 0 0 1
 *     0 0 1 0 
 */
class CNOT : public Gate {
  protected:
    int* lookup;   /* look-up table for vectorized V\kronV */

    double* rotation_Re;  /* Transform gate to rotational frame. Real part. */
    double* rotation_Im;  /* Transform gate to rotational frame. Imaginary part. */

    double *rotRe, *rotIm;
    
    /* Return the CNOT lookup index */
    int getIndex(int i);

  public:
    /* Constructor takes ground frequencies, and time when to apply the gate. */
    CNOT(const std::vector<double> f, double time);
    ~CNOT();

    // /* Apply i-th column of the gate to a state vector */
    // virtual void apply(int i, Vec state, double& obj_re, double& obj_im);
};