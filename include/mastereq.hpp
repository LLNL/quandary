#include "defs.hpp"
#include "oscillator.hpp"
#include "util.hpp"
#include <petscts.h>
#include <vector>
#include <assert.h>
#include <iostream> 
#include "gate.hpp"
#pragma once


/* Define a matshell context containing pointers to data needed for applying the RHS matrix to a vector */
typedef struct {
  std::vector<int> nlevels;
  IS *isu, *isv;
  Oscillator** oscil_vec;
  std::vector<double> crosskerr;
  std::vector<double> Jkl;
  std::vector<double> eta;
  bool addT1, addT2;
  std::vector<double> control_Re, control_Im;
  Mat** Ac_vec;
  Mat** Bc_vec;
  Mat *Ad, *Bd;
  Mat** Ad_vec;
  Mat** Bd_vec;
  Vec *Acu, *Acv, *Bcu, *Bcv;
  Vec *Adklu, *Adklv, *Bdklu, *Bdklv;
  double time;
} MatShellCtx;


/* Define the Matrix-Vector products for the RHS MatShell */
int myMatMult_matfree_2osc(Mat RHS, Vec x, Vec y);              // Matrix free solver, currently only for 2 oscillators 
int myMatMultTranspose_matfree_2Osc(Mat RHS, Vec x, Vec y);
int myMatMult_sparsemat(Mat RHS, Vec x, Vec y);                 // Sparse matrix solver
int myMatMultTranspose_sparsemat(Mat RHS, Vec x, Vec y);


/* 
 * Implements the Lindblad master equation
 */
class MasterEq{

  protected:
    int dim;                   // Dimension of full vectorized system = N^2
    int dim_rho;               // Dimension of full system = N
    int dim_ess;               // Dimension of system of essential levels = N_e
    int noscillators;          // Number of oscillators
    Oscillator** oscil_vec;    // Vector storing pointers to the oscillators

    Mat RHS;                // Realvalued, vectorized systemmatrix (2N^2 x 2N^2)
    MatShellCtx RHSctx;     // MatShell context that contains data needed to apply the RHS

    Mat* Ac_vec;  // Vector of constant mats for time-varying control term (real)
    Mat* Bc_vec;  // Vector of constant mats for time-varying control term (imag)
    Mat  Ad, Bd;  // Real and imaginary part of constant system matrix
    Mat* Ad_vec;  // Vector of constant mats for Jaynes-Cummings coupling term in drift Hamiltonian (real)
    Mat* Bd_vec;  // Vector of constant mats for Jaynes-Cummings coupling term in drift Hamiltonian (imag)

    std::vector<double> crosskerr;    // Cross ker coefficients (rad/time) $\xi_{kl} for zz-coupling ak^d ak al^d al
    std::vector<double> Jkl;          // Jaynes-Cummings coupling coefficient (rad/time), multiplies ak^d al + ak al^d
    std::vector<double> eta;          // Delta in rotational frame frequencies (rad/time). Used for Jaynes-Cummings coupling terms in rotating frame
    bool addT1, addT2;                // flags for including Lindblad collapse operators T1-decay and/or T2-dephasing

    /* Auxiliary stuff */
    int mpirank_petsc;   // Rank of Petsc's communicator
    int mpirank_world;   // Rank of global communicator
    int nparams_max;     // Maximum number of design parameters per oscilator 
    IS isu, isv;         // Vector strides for accessing u=Re(x), v=Im(x) 

    double *dRedp;
    double *dImdp;
    Vec Acu, Acv, Bcu, Bcv;
    Vec Adklu, Adklv, Bdklu, Bdklv;
    int* cols;           // holding columns when evaluating dRHSdp
    PetscScalar* vals;   // holding values when evaluating dRHSdp
 
  public:
    std::vector<int> nlevels;  // Number of levels per oscillator
    std::vector<int> nessential; // Number of essential levels per oscillator
    bool usematfree;  // Flag for using matrix free solver

  public:
    MasterEq();
    MasterEq(std::vector<int> nlevels, std::vector<int> nessential, Oscillator** oscil_vec_, const std::vector<double> crosskerr_, const std::vector<double> Jkl_, const std::vector<double> eta_, LindbladType lindbladtype_, bool usematfree_);
    ~MasterEq();

    /* initialize matrices needed for applying sparse-mat solver */
    void initSparseMatSolver();

    /* Return the i-th oscillator */
    Oscillator* getOscillator(const int i);

    /* Return number of oscillators */
    int getNOscillators();

    /* Return dimension of vectorized system N^2 */
    int getDim();

    /* Return dimension of essential level system: N_e */
    int getDimEss();
    
    /* Return dimension of system matrix rho: N */
    int getDimRho();

    /* 
     * Uses Re and Im to build the vectorized Hamiltonian operator M = vec(-i(Hq-qH)+Lindblad). 
     * This should always be called before applying the RHS matrix.
     */
    int assemble_RHS(const double t);

    /* Access the right-hand-side matrix */
    Mat getRHS();

    /* 
     * Compute gradient of RHS wrt control parameters:
     * grad += alpha * RHS(x)^T * x_bar  
     */
    void computedRHSdp(const double t,const Vec x,const Vec x_bar, const double alpha, Vec grad);

    // /* Compute reduced density operator for a sub-system defined by IDs in the oscilIDs vector */
    // void createReducedDensity(const Vec rho, Vec *reduced, const std::vector<int>& oscilIDs);
    // /* Derivative of reduced density computation */
    // void createReducedDensity_diff(Vec rhobar, const Vec reducedbar, const std::vector<int>& oscilIDs);

    /* Set the oscillators control function parameters from global design vector x */
    void setControlAmplitudes(const Vec x);

    /* Set initial conditions 
     * In:   iinit -- index in processors range [rank * ninit_local .. (rank+1) * ninit_local - 1]
     *       ninit -- number of initial conditions 
     *       initcond_type -- type of initial condition (pure, fromfile, diagona, basis)
     *       oscilIDs -- ID of oscillators defining the subsystem for the initial conditions  
     * Out: initID -- Idenifyier for this initial condition: Element number in matrix vectorization. 
     *       rho0 -- Vector for setting initial condition 
     */
    int getRhoT0(const int iinit, const int ninit, const InitialConditionType initcond_type, const std::vector<int>& oscilIDs, Vec rho0);

};


inline double H_detune(const double detuning0, const double detuning1, const int a, const int b) {
  return detuning0*a + detuning1*b;
};

inline double H_selfkerr(const double xi0, const double xi1, const int a, const int b) {
  return - xi0 / 2.0 * a * (a-1) - xi1 / 2.0 * b * (b-1);
};

inline double H_crosskerr(const double xi01, const int a, const int b) {
  return - xi01 * a * b;
};

inline double L2(double dephase0, double dephase1, const int i0, const int i1, const int i0p, const int i1p){
  return dephase0 * ( i0*i0p - 1./2. * (i0*i0 + i0p*i0p) ) + dephase1 * ( i1*i1p - 1./2. * (i1*i1 + i1p*i1p) );
};



inline double L1diag(double decay0, double decay1, const int i0, const int i1, const int i0p, const int i1p){
  return - decay0 / 2.0 * ( i0 + i0p ) - decay1 / 2.0 * ( i1 + i1p );
};


inline int TensorGetIndex(const int nlevels0, const int nlevels1,const  int i0, const int i1, int i0p, const int i1p){
  return i0*nlevels1 + i1 + (nlevels0 * nlevels1) * ( i0p * nlevels1 + i1p);
};


