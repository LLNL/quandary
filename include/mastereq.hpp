#include "oscillator.hpp"
#include "util.hpp"
#include <petscts.h>
#include <vector>
#include <assert.h>
#include <iostream> 
#pragma once

/* Available lindblad types */
enum LindbladType {NONE, DECAY, DEPHASE, BOTH};
enum InitialConditionType {FROMFILE, PURE, DIAGONAL, BASIS};

/* 
 * Implements the Lindblad master equation
 */
class MasterEq{

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

    InitialConditionType initcond_type; 

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
    int* cols;
    double* vals;
 


  public:
    /* Default constructor sets zero */
    MasterEq();
    /* This constructor sets the variables and allocates Re, Im and M */
    MasterEq(int noscillators_, Oscillator** oscil_vec_, const std::vector<double> xi_, LindbladType lindbladtype_, InitialConditionType initcondtype_, const std::vector<double> collapse_time_);
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


    /* Compute reduced density operator */
    void reducedDensity(Vec fulldensitymatrix, Vec *reduced, int dim_pre, int dim_post, int dim_reduced);
    void reducedDensity_diff(Vec fulldens_bar, Vec reduced_bar, int dim_pre, int dim_post, int dim_reduced);
    

    /* Set the oscillators control function amplitudes from design vector x */
    void setControlAmplitudes(Vec x);

    /* Set initial conditions 
     * In:   iinit -- index in processors range [rank * ninit_local .. (rank+1) * ninit_local - 1]
     *       ninit -- Number of different initial conditions 
     *       oscilIDs -- ID of oscillators that are considered for various initial conditions 
     * Out: initID -- Idenifyier this initial condition: Element number in matrix vectorization. 
     *       rho_0 -- Vector for setting initial condition 
     */
    int getRhoT0(int iinit, std::vector<int> oscilIDs, int ninit, Vec rho0);
};

