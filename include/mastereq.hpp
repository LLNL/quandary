#include "oscillator.hpp"
#include "util.hpp"
#include <petscts.h>
#include <vector>
#include <assert.h>
#pragma once

/* Available lindblad types */
enum LindbladType {NONE, DECAY, DEPHASING, BOTH};

/* 
 * Implements the Lindblad master equation
 */
class MasterEq{

  public: 

  protected:
    int dim;                 // Dimension of vectorized system 
    int noscillators;        // Number of oscillators
    Oscillator** oscil_vec;  // Vector storing pointers to the oscillators

    Mat Re, Im;             // Real and imaginary part of system matrix operator
    Mat RHS;                // Realvalued, vectorized systemmatrix
    Mat dRHSdp;             // Derivative of RHS(x) wrt control parameters

    Mat* Ac_vec;  // Vector of constant mats for time-varying Hamiltonian (real) 
    Mat* Bc_vec;  // Vector of constant mats for time-varying Hamiltonian (imag) 
    Mat  Ad, Bd;  // Real and imaginary part of constant system matrix

    std::vector<double> xi;     // Constants for frequencies of drift Hamiltonian
    std::vector<double> gamma;  /* Time-constants for decay and dephasing operator */



  private: 
    PetscInt    *col_idx_shift; // Auxiliary vector: shifted indices
    PetscScalar *negvals;       // Auxiliary vector: negative vals of some matrix

    /* Some auxiliary vectors */
    double *dRedp;
    double *dImdp;
    int *rowid;
    int *rowid_shift;
    IS isu, isv;



  public:
    /* Default constructor sets zero */
    MasterEq();
    /* This constructor sets the variables and allocates Re, Im and M */
    MasterEq(int noscillators_, Oscillator** oscil_vec_, const std::vector<double> xi_, LindbladType lindbladtype, const std::vector<double> gamma_);
    ~MasterEq();

    /* Return the i-th oscillator */
    Oscillator* getOscillator(int i);

    /* Return number of oscillators */
    int getNOscillators();

    /* Return dimension of vectorized system */
    int getDim();

    /* Compute lowering operator a_k = I_n1 \kron ... \kron a^(nk) \kron ... \kron I_nQ */
    int createLoweringOP(int ioscillator, Mat* loweringOP);

    /* Compute number operator N_k = a_k^T a_k */
    int createNumberOP(int ioscillator, Mat* numberOP);

    /* 
     * Evaluate the exact solution at time t.
     * Inheriting classes should store exact solution in x and return true.  
     * Return false, if exact solution is not available.
     */
    bool ExactSolution(double t, Vec x);

    /* 
     * Uses Re and Im to build the vectorized Hamiltonian operator M = vec(-i(Hq-qH)). 
     * M(0, 0) =  Re    M(0,1) = -Im
     * M(1, 0) =  Im    M(1,1) = Re
     * Both Re and Im should be set up in the inherited 'assemble_RHS' routines. 
     */
    int assemble_RHS(double t);

    /* 
     * Assemble the derivative of RHS(x) wrt the controls.
     * dimensions: dRHSdp \in \R^{2*dim \times 2*nparam}
     * dRHSdp =   du+/dparamRe   du+/dparamIm
     *            dv+/dparamRe   dv+/dparamIm
     * where [u+ v+] = RHS(u,v)
     */
    int assemble_dRHSdp(double t, Vec x);

    /* 
     * Set x to the initial condition of index iinit
     */
    int initialCondition(int iinit, Vec x);

    /* Access the right-hand-side and derivative matrix */
    Mat getRHS();
    Mat getdRHSdp();
    
};

