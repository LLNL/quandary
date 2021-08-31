#include <stdio.h>
#include <iostream> 
#include <math.h>
#include <assert.h>
#include <petscmat.h>
#include <vector>
#include "util.hpp"
#pragma once


class Gate {
  protected:
    Mat V_re, V_im;    /* Input: Real and imaginary part of V_target, non-vectorized, essential levels only */
    Vec rotA, rotB;    /* Input: Diagonal elements of real and imaginary rotational matrices */

    std::vector<int> nessential;
    std::vector<int> nlevels;
    int mpirank_petsc;

    int dim_ess;   /* Dimension of target Gate matrix (non-vectorized), essential levels only */
    int dim_rho;   /* Dimension of system matrix rho (non-vectorized), all levels, N */

    double final_time;  /* Final time T. Time of gate rotation. */
    std::vector<double> gate_rot_freq; /* Frequencies of gate rotation (rad/time). Often same as rotational frequencies. */

  private:
    Mat VxV_re, VxV_im;     /* Real and imaginary part of vectorized Gate G=\bar V \kron V */
    Vec x;                  /* auxiliary */
    IS isu, isv;            /* Vector strides for accessing real and imaginary part of the state */

  public:
    Gate();
    Gate(std::vector<int> nlevels_, std::vector<int> nessential_, double time_, std::vector<double> gate_rot_freq);
    virtual ~Gate();

    int getDimRho() { return dim_rho; };

    /* Assemble VxV_re = Re(\bar V \kron V) and VxV_im = Im(\bar V \kron V) */
    void assembleGate();

    /* compare the final state to gate-transformed initialcondition in Frobenius norm 1/2 * || q(T) - V\kronV q(0)||^2 */
    void compare_frobenius(const Vec finalstate, const Vec rho0, double& obj);
    void compare_frobenius_diff(const Vec finalstate, const Vec rho0, Vec rho0bar, const double delta_bar);

    /* compare the final state to gate-transformed initialcondition using trace distance overlap 1 - Tr(V\rho(0)V^d \rho(T))  * 1/purity_of_rho(0) */
    void compare_trace(const Vec finalstate, const Vec rho0, double& obj);
    void compare_trace_diff(const Vec finalstate, const Vec rho0, Vec rho0bar, const double delta_bar);
};

/* X Gate, spanning one qubit. 
 * V = 0 1
 *     1 0
 */
class XGate : public Gate {
  public:
    XGate(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> rotation_frequencies_);
    ~XGate();
};

/* Y Gate, spanning one qubit. 
 * V = 0 -i
 *     i  0
 */
class YGate : public Gate {
  public:
    YGate(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> rotation_frequencies_);
    ~YGate();
};

/* Z Gate, spanning one qubit. 
 * V = 1   0
 *     0  -1
 */
class ZGate : public Gate {
  public:
    ZGate(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> rotation_frequencies_);
    ~ZGate();
};

/* Hadamard Gate 
 * V = 1/sqrt(2) * | 1   1 |
 *                 | 1  -1 |
 */
class HadamardGate : public Gate {
  public:
    HadamardGate(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> rotation_frequencies_);
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
    CNOT(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> rotation_frequencies_);
    ~CNOT();
};

/* SWAP Gate, spanning two qubits. 
 * V = 1 0 0 0
 *     0 0 1 0 
 *     0 1 0 1
 *     0 0 0 1 
 */
class SWAP: public Gate {
    public:
    SWAP(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> rotation_frequencies_);
    ~SWAP();
};



/* SWAP for three qubits, swapping 0<->2. 
 * V = 1 0 0 0 0 0 0 0
 *     0 0 0 0 1 0 0 0 
 *     0 0 1 0 0 0 0 0 
 *     0 0 0 0 0 0 1 0 
 *     0 1 0 0 0 0 0 0 
 *     0 0 0 0 0 1 0 0 
 *     0 0 0 1 0 0 0 0 
 *     0 0 0 0 0 0 0 1 
 */
class SWAP_02: public Gate {
    public:
    SWAP_02(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> rotation_frequencies_);
    ~SWAP_02();
};


/* SWAP for four qubits, swapping 0<->3 */
class SWAP_03: public Gate {
    public:
    SWAP_03(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> rotation_frequencies_);
    ~SWAP_03();
};

/* SWAP for general Q qubits, swapping 0 <-> Q-1 while leaving other in their state */
class SWAP_0Q: public Gate {
    public:
    SWAP_0Q(std::vector<int> nlevels_, std::vector<int> nessential_, int Q, double time, std::vector<double> rotation_frequencies_);
    ~SWAP_0Q();
};

/* C7NOT for four 8 qubits NOT on qubit 8, controlled by qubits 1,...,7 */
class C7NOT: public Gate {
    public:
    C7NOT(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> rotation_frequencies_);
    ~C7NOT();
};

