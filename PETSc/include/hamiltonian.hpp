#include "oscillator.hpp"
#include "util.hpp"
#include <petscts.h>
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

  public:
    /* Default constructor sets zero */
    Hamiltonian();
    /* This constructor sets the variables and allocates Re, Im and M */
    Hamiltonian(int noscillators_, Oscillator** oscil_vec_);
    ~Hamiltonian();

    /* Return dimension of vectorized system */
    int getDim();

    /* Compute lowering operator a_k = I_n1 \kron ... \kron a^(nk) \kron ... \kron I_nQ */
    int createLoweringOP(int ioscillator, Mat* loweringOP);

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
     * Set x to the initial condition 
     */
    virtual int initialCondition(Vec x) = 0;

    /* Access the Hamiltonian and derivative matrix */
    Mat getRHS();
    Mat getdRHSdp();
    
    /* 
     * Evaluate the objective function at time t and current solution x
     * Return objective = F(t,x)
     */
    virtual int evalObjective(double t, Vec x, double *objective_ptr);

    /* 
     * Evaluate the derivative of the objective function wrt x.
     * Return lambda = dFdx(t,x)
     * lambda must be allocated and of size dim (matching RHS)
     * mu must be allocated and of size noscil*nparam*2 (matching dRHSdp)
     */
    virtual int evalObjective_diff(double t, Vec x, Vec *lambda, Vec *mu);
};


/*
 * Implements the Liouville-van-Neumann Hamiltonian
 */
class LiouvilleVN : public Hamiltonian {

  Mat* Ac_vec;  // Vector of constant matrices for building time-varying Hamiltonian (real part)
  Mat* Bc_vec;  // Vector of constant matrices for building time-varying Hamiltonian (imaginary part)
  Mat  Ad, Bd;  // Real and imaginary part of constant drift Hamiltonian Hd 

  double* xi;  // Constants for rotating frame frequencies of drift Hamiltonian

  public: 
    LiouvilleVN();
    LiouvilleVN(double* xi, int noscillators_, Oscillator** oscil_vec_);
    ~LiouvilleVN();

    /* Set the initial condition (zero so far...) */
    virtual int initialCondition(Vec x);

    /* Eval Re and Im of vectorized Hamiltonian, and derivative */
    virtual int assemble_RHS(double t);
    virtual int assemble_dRHSdp(double t, Vec x);
};

/*
 * Hamiltonian with two oscillators 
 */
class TwoOscilHam : public Hamiltonian {

  Mat A1, A2;  // Building blocks for real part of Hamiltonian
  Mat B1, B2;  // Building blocks for imaginary part of Hamiltonian
  Mat Hd;      // Constant part of Hamiltonian matrix ("drift Hamiltonian")

  double* xi;   // xi = xi1, xi2, xi12

  public:
    TwoOscilHam();
    TwoOscilHam(int nlevels_, double* xi, Oscillator** oscil_vec_); 
    ~TwoOscilHam(); 

    /* Helper function for constructing building blocks */
    int BuildingBlock(Mat C, int sign, int k, int m);

    /* Set the initial condition (zero so far...) */
    int initialCondition(Vec x);

    /* Evaluates Re and Im of the Hamiltonian. Then calls the base-class assembleRHS routine to set up M. */
    virtual int assemble_RHS(double t);

    /* 
     * Assemble the derivative of RHS(x) wrt the controls.
     */
    virtual int assemble_dRHSdp(double t, Vec x);

};


/* 
 * Ander's testcase with analytic solution 
 */
class AnalyticHam : public LiouvilleVN {
// class AnalyticHam : public TwoOscilHam{
  
  public:
    AnalyticHam(double* xi_, Oscillator** oscil_vec_);
    ~AnalyticHam();

    /* Evaluate the exact solution at time t. */
    virtual bool ExactSolution(double t, Vec x);

    /* Set the initial condition (exact(0)) */
    int initialCondition(Vec x);
};

/* Real part for Oscillator 1 of analytic solution */
PetscScalar F1_analytic(PetscReal t,PetscReal freq);

/* Imaginary part for Oscillator 2 of analytic solution */
PetscScalar G2_analytic(PetscReal t,PetscReal freq);


/* Derivative freq_diff = dF1/dfreq * Fbar */
PetscScalar dF1_analytic(PetscReal t,PetscReal freq, PetscReal Fbar);

/* Derivative freq_diff = dG2/dfreq * Gbar */
PetscScalar dG2_analytic(PetscReal t,PetscReal freq, PetscReal Gbar);