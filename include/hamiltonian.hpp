#include "oscillator.hpp"
#include "util.hpp"
#include <petscts.h>
#include <vector>
#include <assert.h>
#pragma once


/* 
 * Abstract base class for Hamiltonian systems 
 */
class Hamiltonian{

  protected:
    int dim;                 // Dimension of vectorized system 
    int noscillators;        // Number of oscillators
    Oscillator** oscil_vec;  // Vector storing pointers to the oscillators

    Mat Re, Im;             // Real and imaginary part of Hamiltonian operator
    Mat RHS;                // Realvalued, vectorized Hamiltonian operator vec(-i(Hq-qH))
    Mat dRHSdp;             // Derivative of RHS(x) wrt control parameters

  private: 
    PetscInt    *col_idx_shift; // Auxiliary vector holding shifted indices
    PetscScalar *negvals;       // Auxiliary vector holding negative vals of some matrix


  public:
    /* Default constructor sets zero */
    Hamiltonian();
    /* This constructor sets the variables and allocates Re, Im and M */
    Hamiltonian(int noscillators_, Oscillator** oscil_vec_);
    virtual ~Hamiltonian();

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
    virtual bool ExactSolution(double t, Vec x);

    /* 
     * Uses Re and Im to build the vectorized Hamiltonian operator M = vec(-i(Hq-qH)). 
     * M(0, 0) =  Re    M(0,1) = -Im
     * M(1, 0) =  Im    M(1,1) = Re
     * Both Re and Im should be set up in the inherited 'assemble_RHS' routines. 
     */
    virtual int assemble_RHS(double t);

    /* 
     * Assemble the derivative of RHS(x) wrt the controls.
     * dimensions: dRHSdp \in \R^{2*dim \times 2*nparam}
     * dRHSdp =   du+/dparamRe   du+/dparamIm
     *            dv+/dparamRe   dv+/dparamIm
     * where [u+ v+] = RHS(u,v)
     */
    virtual int assemble_dRHSdp(double t, Vec x);

    /* 
     * Set x to the initial condition of index iinit
     */
    virtual int initialCondition(int iinit, Vec x) = 0;

    /* Access the Hamiltonian and derivative matrix */
    Mat getRHS();
    Mat getdRHSdp();
    
};


/*
 * Implements the Liouville-van-Neumann Hamiltonian
 */
class LiouvilleVN : public Hamiltonian {

  Mat* Ac_vec;  // Vector of constant matrices for building time-varying Hamiltonian (real part)
  Mat* Bc_vec;  // Vector of constant matrices for building time-varying Hamiltonian (imaginary part)
  Mat  Ad, Bd;  // Real and imaginary part of constant drift Hamiltonian Hd 

  std::vector<double> xi;  // Constants for frequencies of drift Hamiltonian

  /* Some auxiliary vectors */
  double *dRedp;
  double *dImdp;
  int *rowid;
  int *rowid_shift;
  IS isu, isv;

  public: 
    LiouvilleVN();
    LiouvilleVN(const std::vector<double> xi_, int noscillators_, Oscillator** oscil_vec_);
    virtual ~LiouvilleVN();

    /* Set the initial condition of index iinit */
    virtual int initialCondition(int iinit, Vec x);

    /* Eval Re and Im of vectorized Hamiltonian, and derivative */
    virtual int assemble_RHS(double t);
    virtual int assemble_dRHSdp(double t, Vec x);
};

/*
 * Implements the Lindblad terms
 */

class Lindblad : public LiouvilleVN {

  public: 
    /* Available lindblad terms */
    enum CollapseType {DECAY, DEPHASING}; 
    
    Lindblad();
    Lindblad(CollapseType collapse_type, const std::vector<double> xi_, int noscillators_, Oscillator** oscil_vec_);
    virtual ~Lindblad();

};

/* 
 * Ander's testcase with analytic solution 
 */
class AnalyticHam : public LiouvilleVN {
  
  public:
    AnalyticHam(const std::vector<double> xi_, Oscillator** oscil_vec_);
    virtual ~AnalyticHam();

    /* Evaluate the exact solution at time t. */
    virtual bool ExactSolution(double t, Vec x);

    /* Set the initial condition (exact(0)) */
    int initialCondition(int iinit, Vec x);
};

/* Real part for Oscillator 1 of analytic solution */
PetscScalar F1_analytic(PetscReal t,PetscReal freq);

/* Imaginary part for Oscillator 2 of analytic solution */
PetscScalar G2_analytic(PetscReal t,PetscReal freq);


/* Derivative freq_diff = dF1/dfreq * Fbar */
PetscScalar dF1_analytic(PetscReal t,PetscReal freq, PetscReal Fbar);

/* Derivative freq_diff = dG2/dfreq * Gbar */
PetscScalar dG2_analytic(PetscReal t,PetscReal freq, PetscReal Gbar);
