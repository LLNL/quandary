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

    /* apply the gate transformation  VrhoV =  V \rho V^\dagger. The output vector VrhoV must be allocated! */
    void applyGate(const Vec state, Vec VrhoV);
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


/* SWAP gate for Q qubits, swapping qubit 0 <-> Q-1 while leaving all others in their state */
class SWAP_0Q: public Gate {
    public:
    SWAP_0Q(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> rotation_frequencies_);
    ~SWAP_0Q();
};

/* CQNOT gate spanning Q qubits: NOT operation on qubit Q-1 controlled by all other qubits */
class CQNOT: public Gate {
    public:
    CQNOT(std::vector<int> nlevels_, std::vector<int> nessential_, double time, std::vector<double> rotation_frequencies_);
    ~CQNOT();
};

