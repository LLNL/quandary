#include "oscillator.hpp"
#include "util.hpp"
#include <petscts.h>
#include <vector>
#include <assert.h>
#include <iostream> 
#pragma once

/* Available lindblad types */
enum LindbladType {NONE, DECAY, DEPHASE, BOTH};

/* 
 * Implements the Lindblad master equation
 */
class MasterEq{

  public: 

  protected:
    int dim;                 // Dimension of vectorized system  N^2
    int noscillators;        // Number of oscillators
    Oscillator** oscil_vec;  // Vector storing pointers to the oscillators

    Mat Re, Im;             // Real and imaginary part of system matrix operator
    Mat RHS;                // Realvalued, vectorized systemmatrix

    Mat* Ac_vec;  // Vector of constant mats for time-varying Hamiltonian (real) 
    Mat* Bc_vec;  // Vector of constant mats for time-varying Hamiltonian (imag) 
    Mat  Ad, Bd;  // Real and imaginary part of constant system matrix

    std::vector<double> xi;     // Constants for frequencies of drift Hamiltonian
    std::vector<double> collapse_time;  /* Time-constants for decay and dephase operator */



  private: 
    PetscInt    *col_idx_shift; // Auxiliary vector: shifted indices
    PetscScalar *negvals;       // Auxiliary vector: negative vals of some matrix

    /* Some auxiliary vectors */
    double *dRedp;
    double *dImdp;
    int *rowid;
    int *rowid_shift;
    IS isu, isv;
    Vec Acu, Acv, Bcu, Bcv, auxil;
 


  public:
    /* Default constructor sets zero */
    MasterEq();
    /* This constructor sets the variables and allocates Re, Im and M */
    MasterEq(int noscillators_, Oscillator** oscil_vec_, const std::vector<double> xi_, LindbladType lindbladtype, const std::vector<double> collapse_time_);
    ~MasterEq();

    /* Return the i-th oscillator */
    Oscillator* getOscillator(int i);

    /* Return number of oscillators */
    int getNOscillators();

    /* Return dimension of vectorized system */
    int getDim();

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
     * Compute gradient of RHS wrt controls:
     * grad += alpha * RHS(x)^T * x_bar  
     */
    void computedRHSdp(double t, Vec x, Vec x_bar, double alpha, Vec grad);

    /* Access the right-hand-side and derivative matrix */
    Mat getRHS();


    /* Compute reduced density operator for oscillator ID given in the oscilID's vector. 
     * OscilIDs must a consecuitive block (0,1,2 or 4,5 etc.)
     */
    void reducedDensity(Vec fulldensitymatrix, Vec *reduced, int dim_pre, int dim_post, int dim_reduced);
    void reducedDensity_diff(Vec reddens_bar, Vec x0_re_bar, Vec x0_im_bar, int dim_pre, int dim_post, int dim_reduced);
    
};

