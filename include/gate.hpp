#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <petscmat.h>
#include <vector>
#include "util.hpp"
#pragma once


class Gate {
  protected:
    int dim_v;        /* Input: dimension of target gate (non-vectorized) */
    Mat Va, Vb;       /* Input: Real and imaginary part of V_target, non-vectorized */

  private:
    int dim_vec;      /* dimension of vectorized system dim_vec = dim_v^2 */
    Mat ReG, ImG;     /* Real and imaginary part of \bar V \kron V */
    IS isu, isv;      /* Vector strides for extracting u,v from x = [u,v] */

    Vec ReG_col, ImG_col;   /* auxiliary vectors for computing frobenius norm  */

  public:
    Gate();
    Gate(int dim_V_);
    virtual ~Gate();

    /* Assemble ReG = Re(\bar V \kron V) and ImG = Im(\bar V \kron V) */
    void assembleGate();
    
    /* compare the k-th column of the gate to a state vector */
    /* in Frobenius norm ||w_k - g_k||^2_F = w_k^dag w_k + g_k^dag g_k - 2*Re(w_k^dag g_k) */
    void compare(int i, Vec state, double& delta);
    void compare_diff(int i, const Vec state, Vec state_bar, const double delta_bar);
};

/* X Gate, spanning one qubit. 
 * V = 0 1
 *     1 0
 */
class XGate : public Gate {
  public:
    XGate();
    ~XGate();
};

/* Y Gate, spanning one qubit. 
 * V = 0 -i
 *     i  0
 */
class YGate : public Gate {
  public:
    YGate();
    ~YGate();
};

/* Z Gate, spanning one qubit. 
 * V = 1   0
 *     0  -1
 */
class ZGate : public Gate {
  public:
    ZGate();
    ~ZGate();
};

/* Hadamard Gate 
 * V = 1/sqrt(2) * | 1   1 |
 *                 | 1  -1 |
 */
class HadamardGate : public Gate {
  public:
    HadamardGate();
    ~HadamardGate();
};

/* CNOT Gate, spanning two qubits. This class is mostly hardcoded. TODO: Generalize!
 * V = 1 0 0 0
 *     0 1 0 0 
 *     0 0 0 1
 *     0 0 1 0 
 */
class CNOT : public Gate {
    public:
    CNOT();
    ~CNOT();
};

/* Groundstate Gate pushes every initial condition towards the ground state |0><0|
 * V = | 1        |
 *     |   0      |
 *     |     0    |
 *     |      ... |
 */
class GroundstateGate : public Gate {
  public:
    GroundstateGate(int dim_v_);
    ~GroundstateGate();
};

