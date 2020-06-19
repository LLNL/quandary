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
    Mat ReG, ImG;     /* Real and imaginary part of \bar V \kron V */

    Vec x;             /* auxiliary vectors */
  protected:
    int dim_vec;      /* dimension of vectorized system dim_vec = dim_v^2 */

    int mpirank_petsc;

  public:
    Gate();
    Gate(int dim_V_);
    virtual ~Gate();

    /* Assemble ReG = Re(\bar V \kron V) and ImG = Im(\bar V \kron V) */
    void assembleGate();
    
    /* compare the final state to gate-transformed initialcondition in Frobenius norm 1/2 * || q(T) - V\kronV q(0)||^2 */
    void compare(const Vec finalstate, const Vec rho0, double& frob);
    void compare_diff(const Vec finalstate, const Vec rho0, Vec rho0bar, const double delta_bar);
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

/* CNOT Gate, spanning two qubits. 
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

