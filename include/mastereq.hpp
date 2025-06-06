#include "defs.hpp"
#include "oscillator.hpp"
#include "util.hpp"
#include <petscts.h>
#include <vector>
#include <assert.h>
#include <iostream> 
#include "gate.hpp"
#include "pythoninterface.hpp"

#pragma once

/**
 * @brief MatShell context containing data needed for applying the RHS matrix to a vector.
 *
 * This structure holds all the necessary data for matrix-free operations
 * in the Lindblad master equation solver.
 */
typedef struct {
  std::vector<int> nlevels; ///< Number of levels per oscillator
  IS *isu, *isv; ///< Vector strides for accessing real and imaginary parts
  Oscillator** oscil_vec; ///< Array of pointers to oscillators
  std::vector<double> crosskerr; ///< Cross-Kerr coupling coefficients
  std::vector<double> Jkl; ///< Dipole-dipole coupling coefficients
  std::vector<double> eta; ///< Frequency differences for rotating frame
  LindbladType lindbladtype; ///< Type of Lindblad operators to include
  bool addT1, addT2; ///< Flags for T1 decay and T2 dephasing
  std::vector<std::vector<double>> control_Re; ///< Real parts of control amplitudes
  std::vector<std::vector<double>> control_Im; ///< Imaginary parts of control amplitudes
  std::vector<double> eval_transfer_Hdt_re; ///< Evaluated real transfer functions for time-varying Hamiltonian
  std::vector<double> eval_transfer_Hdt_im; ///< Evaluated imaginary transfer functions for time-varying Hamiltonian
  std::vector<std::vector<Mat>> Ac_vec; ///< Real parts of control matrices
  std::vector<std::vector<Mat>> Bc_vec; ///< Imaginary parts of control matrices
  Mat *Ad, *Bd; ///< Real and imaginary parts of drift Hamiltonian matrices
  std::vector<Mat> Ad_vec; ///< Real parts of dipole-dipole coupling matrices
  std::vector<Mat> Bd_vec; ///< Imaginary parts of dipole-dipole coupling matrices
  Vec *aux; ///< Auxiliary vector for computations
  double time; ///< Current time
} MatShellCtx;


/**
 * @brief Matrix-vector product functions for the RHS MatShell.
 *
 * These functions implement matrix-free solvers for different numbers of oscillators
 * and sparse matrix solvers for larger systems.
 */
int myMatMult_matfree_1Osc(Mat RHS, Vec x, Vec y); ///< Matrix-free solver for 1 oscillator
int myMatMultTranspose_matfree_1Osc(Mat RHS, Vec x, Vec y); ///< Transpose matrix-free solver for 1 oscillator
int myMatMult_matfree_2Osc(Mat RHS, Vec x, Vec y); ///< Matrix-free solver for 2 oscillators
int myMatMultTranspose_matfree_2Osc(Mat RHS, Vec x, Vec y); ///< Transpose matrix-free solver for 2 oscillators
int myMatMult_matfree_3Osc(Mat RHS, Vec x, Vec y); ///< Matrix-free solver for 3 oscillators
int myMatMultTranspose_matfree_3Osc(Mat RHS, Vec x, Vec y); ///< Transpose matrix-free solver for 3 oscillators
int myMatMult_matfree_4Osc(Mat RHS, Vec x, Vec y); ///< Matrix-free solver for 4 oscillators
int myMatMultTranspose_matfree_4Osc(Mat RHS, Vec x, Vec y); ///< Transpose matrix-free solver for 4 oscillators
int myMatMult_matfree_5Osc(Mat RHS, Vec x, Vec y); ///< Matrix-free solver for 5 oscillators
int myMatMultTranspose_matfree_5Osc(Mat RHS, Vec x, Vec y); ///< Transpose matrix-free solver for 5 oscillators
int myMatMult_sparsemat(Mat RHS, Vec x, Vec y); ///< Sparse matrix solver
int myMatMultTranspose_sparsemat(Mat RHS, Vec x, Vec y); ///< Transpose sparse matrix solver


/**
 * @brief Implementation of the Lindblad master equation solver.
 *
 * This class provides functionality for solving both the Lindblad master equation
 * for open quantum systems and the Schroedinger equation for closed systems.
 * It supports matrix-free and sparse matrix solvers for different system sizes.
 */
class MasterEq{

  protected:
    int dim; ///< Dimension of full vectorized system: N^2 if Lindblad, N if Schroedinger
    int dim_rho; ///< Dimension of Hilbert space = N
    int dim_ess; ///< Dimension of essential level system = N_e
    int noscillators; ///< Number of oscillators in the system
    Oscillator** oscil_vec; ///< Array of pointers to oscillator objects

    Mat RHS; ///< Real-valued, vectorized system matrix (2N^2 x 2N^2)
    MatShellCtx RHSctx; ///< MatShell context containing data for RHS operations

    std::vector<std::vector<Mat>> Ac_vec; ///< Real parts of control matrices for each oscillator
    std::vector<std::vector<Mat>> Bc_vec; ///< Imaginary parts of control matrices for each oscillator
    Mat  Ad, Bd; ///< Real and imaginary parts of constant system matrix
    std::vector<Mat> Ad_vec; ///< Real parts of dipole-dipole coupling matrices in drift Hamiltonian
    std::vector<Mat> Bd_vec; ///< Imaginary parts of dipole-dipole coupling matrices in drift Hamiltonian

    std::vector<double> crosskerr; ///< Cross-Kerr coefficients (rad/time) \f$\xi_{kl}\f$ for ZZ-coupling \f$a_k^\dagger a_k a_l^\dagger a_l\f$
    std::vector<double> Jkl; ///< Dipole-dipole coupling coefficients (rad/time), multiplies \f$a_k^\dagger a_l + a_k a_l^\dagger\f$
    std::vector<double> eta; ///< Frequency differences in rotating frame (rad/time) for dipole-dipole coupling
    bool addT1, addT2; ///< Flags for including T1 decay and T2 dephasing Lindblad operators

    int mpirank_petsc; ///< Rank of PETSc's communicator
    int mpirank_world; ///< Rank of global MPI communicator
    int nparams_max; ///< Maximum number of design parameters per oscillator
    IS isu, isv; ///< Vector strides for accessing real and imaginary parts u=Re(x), v=Im(x)

    double *dRedp; ///< Derivative of real part with respect to parameters
    double *dImdp; ///< Derivative of imaginary part with respect to parameters
    Vec aux; ///< Auxiliary vector for computations
    PetscInt* cols; ///< Column indices for evaluating dRHS/dp
    PetscScalar* vals; ///< Matrix values for evaluating dRHS/dp

    bool quietmode; ///< Flag for quiet mode operation

  public:
    std::vector<int> nlevels; ///< Number of levels per oscillator
    std::vector<int> nessential; ///< Number of essential levels per oscillator
    bool usematfree; ///< Flag for using matrix-free solver
    LindbladType lindbladtype; ///< Type of Lindblad operators to include (NONE means Schroedinger equation)

    std::vector<std::vector<TransferFunction*>> transfer_Hc_re; ///< Real transfer functions for control terms per oscillator
    std::vector<std::vector<TransferFunction*>> transfer_Hc_im; ///< Imaginary transfer functions for control terms per oscillator
    std::vector<TransferFunction*> transfer_Hdt_re; ///< Real transfer functions for time-varying system Hamiltonian
    std::vector<TransferFunction*> transfer_Hdt_im; ///< Imaginary transfer functions for time-varying system Hamiltonian
    std::string hamiltonian_file; ///< Filename for Hamiltonian data ('none' if not used)


  public:
    MasterEq();

    /**
     * @brief Constructor with full system specification.
     *
     * @param nlevels Number of levels per oscillator
     * @param nessential Number of essential levels per oscillator
     * @param oscil_vec_ Array of pointers to oscillator objects
     * @param crosskerr_ Cross-Kerr coupling coefficients
     * @param Jkl_ Dipole-dipole coupling coefficients
     * @param eta_ Frequency differences for rotating frame
     * @param lindbladtype_ Type of Lindblad operators to include
     * @param usematfree_ Flag to use matrix-free solver
     * @param hamiltonian_file Filename for Hamiltonian data
     * @param quietmode Flag for quiet operation (default: false)
     */
    MasterEq(std::vector<int> nlevels, std::vector<int> nessential, Oscillator** oscil_vec_, const std::vector<double> crosskerr_, const std::vector<double> Jkl_, const std::vector<double> eta_, LindbladType lindbladtype_, bool usematfree_, std::string hamiltonian_file, bool quietmode=false);

    ~MasterEq();

    /**
     * @brief Initializes matrices needed for the sparse matrix solver.
     */
    void initSparseMatSolver();

    /**
     * @brief Sets time points that determine when transfer functions are active.
     *
     * @param tlist Vector of time points defining active intervals
     */
    void setTransferOnOffTimes(std::vector<double> tlist);

    /**
     * @brief Retrieves the i-th oscillator.
     *
     * @param i Index of the oscillator
     * @return Oscillator* Pointer to the oscillator object
     */
    Oscillator* getOscillator(const size_t i);

    /**
     * @brief Retrieves the number of oscillators in the system.
     *
     * @return size_t Number of oscillators
     */
    size_t getNOscillators();

    /**
     * @brief Retrieves the dimension of the vectorized system.
     *
     * @return int \f$N^2\f$ for Lindblad solver, \f$N\f$ for Schroedinger solver
     */
    int getDim();

    /**
     * @brief Retrieves the dimension of the essential level system.
     *
     * @return int Dimension N_e of essential levels
     */
    int getDimEss();

    /**
     * @brief Retrieves the dimension of the density matrix.
     *
     * @return int Dimension N of the Hilbert space
     */
    int getDimRho();

    /**
     * @brief Assembles the vectorized Hamiltonian operator.
     *
     * Builds the operator \f$M = \text{vec}(-i(H\rho - \rho H) + \text{Lindblad terms})\f$.
     * Must be called before applying the RHS matrix.
     *
     * @param t Current time
     * @return int Error code
     */
    int assemble_RHS(const double t);

    /**
     * @brief Retrieves the right-hand-side matrix.
     *
     * @return Mat PETSc matrix object
     */
    Mat getRHS();

    /**
     * @brief Computes gradient of RHS with respect to control parameters.
     *
     * Computes grad += alpha * RHS(x)^T * x_bar.
     *
     * @param t Current time
     * @param x State vector
     * @param x_bar Adjoint state vector
     * @param alpha Scaling factor
     * @param grad Gradient vector to update
     */
    void computedRHSdp(const double t,const Vec x,const Vec x_bar, const double alpha, Vec grad);

    // /* Compute reduced density operator for a sub-system defined by IDs in the oscilIDs vector */
    // void createReducedDensity(const Vec rho, Vec *reduced, const std::vector<int>& oscilIDs);
    // /* Derivative of reduced density computation */
    // void createReducedDensity_diff(Vec rhobar, const Vec reducedbar, const std::vector<int>& oscilIDs);

    /**
     * @brief Sets control function parameters from global design vector.
     *
     * @param x Global design vector containing control parameters
     */
    void setControlAmplitudes(const Vec x);

    /**
     * @brief Computes expected energy of the full composite system.
     *
     * @param x State vector
     * @return double Expected energy value
     */
    double expectedEnergy(const Vec x);

    /**
     * @brief Computes population of the full composite system.
     *
     * @param x State vector
     * @param population_com Reference to vector to store population values
     */
    void population(const Vec x, std::vector<double> &population_com);
};

// Matrix-free solver inlines for 1 oscillator

/**
 * @brief Computes detuning Hamiltonian term for 1 oscillator.
 *
 * @param detuning0 Detuning frequency for oscillator 0
 * @param a Occupation number
 * @return double Detuning energy contribution
 */
inline double H_detune(const double detuning0, const int a) {
  return detuning0*a;
};

/**
 * @brief Computes self-Kerr nonlinearity term for 1 oscillator.
 *
 * @param xi0 Self-Kerr coefficient for oscillator 0
 * @param a Occupation number
 * @return double Self-Kerr energy contribution
 */
inline double H_selfkerr(const double xi0, const int a) {
  return - xi0 / 2.0 * a * (a-1);
};

/**
 * @brief Computes L2 dephasing Lindblad operator for 1 oscillator.
 *
 * @param dephase0 Dephasing rate for oscillator 0
 * @param i0 Occupation number (bra)
 * @param i0p Occupation number (ket)
 * @return double L2 operator contribution
 */
inline double L2(const double dephase0, const int i0, const int i0p){
  return dephase0 * ( i0*i0p - 1./2. * (i0*i0 + i0p*i0p) );
};

/**
 * @brief Computes L1 decay Lindblad operator diagonal term for 1 oscillator.
 *
 * @param decay0 Decay rate for oscillator 0
 * @param i0 Occupation number (bra)
 * @param i0p Occupation number (ket)
 * @return double L1 diagonal contribution
 */
inline double L1diag(const double decay0, const int i0, const int i0p){
  return - decay0 / 2.0 * ( i0 + i0p ) ;
};

/**
 * @brief Computes tensor product index for 1 oscillator system.
 *
 * @param nlevels0 Number of levels for oscillator 0
 * @param i0 Occupation number (bra)
 * @param i0p Occupation number (ket)
 * @return int Linear index in tensor product space
 */
inline int TensorGetIndex(const int nlevels0, const  int i0, const int i0p){
  return i0 + nlevels0 * i0p;
};


// Matrix-free solver inlines for 2 oscillators

/**
 * @brief Computes detuning Hamiltonian term for 2 oscillators.
 *
 * @param detuning0 Detuning frequency for oscillator 0
 * @param detuning1 Detuning frequency for oscillator 1
 * @param a Occupation number for oscillator 0
 * @param b Occupation number for oscillator 1
 * @return double Total detuning energy contribution
 */
inline double H_detune(const double detuning0, const double detuning1, const int a, const int b) {
  return detuning0*a + detuning1*b;
};

/**
 * @brief Computes self-Kerr nonlinearity terms for 2 oscillators.
 *
 * @param xi0 Self-Kerr coefficient for oscillator 0
 * @param xi1 Self-Kerr coefficient for oscillator 1
 * @param a Occupation number for oscillator 0
 * @param b Occupation number for oscillator 1
 * @return double Total self-Kerr energy contribution
 */
inline double H_selfkerr(const double xi0, const double xi1, const int a, const int b) {
  return - xi0 / 2.0 * a * (a-1) - xi1 / 2.0 * b * (b-1);
};

/**
 * @brief Computes cross-Kerr coupling between 2 oscillators.
 *
 * @param xi01 Cross-Kerr coefficient between oscillators 0 and 1
 * @param a Occupation number for oscillator 0
 * @param b Occupation number for oscillator 1
 * @return double Cross-Kerr energy contribution
 */
inline double H_crosskerr(const double xi01, const int a, const int b) {
  return - xi01 * a * b;
};

/**
 * @brief Computes L2 dephasing Lindblad operator for 2 oscillators.
 *
 * @param dephase0 Dephasing rate for oscillator 0
 * @param dephase1 Dephasing rate for oscillator 1
 * @param i0 Occupation number for oscillator 0 (bra)
 * @param i1 Occupation number for oscillator 1 (bra)
 * @param i0p Occupation number for oscillator 0 (ket)
 * @param i1p Occupation number for oscillator 1 (ket)
 * @return double Total L2 operator contribution
 */
inline double L2(const double dephase0, const double dephase1, const int i0, const int i1, const int i0p, const int i1p){
  return dephase0 * ( i0*i0p - 1./2. * (i0*i0 + i0p*i0p) ) + dephase1 * ( i1*i1p - 1./2. * (i1*i1 + i1p*i1p) );
};

/**
 * @brief Computes L1 decay Lindblad operator diagonal term for 2 oscillators.
 *
 * @param decay0 Decay rate for oscillator 0
 * @param decay1 Decay rate for oscillator 1
 * @param i0 Occupation number for oscillator 0 (bra)
 * @param i1 Occupation number for oscillator 1 (bra)
 * @param i0p Occupation number for oscillator 0 (ket)
 * @param i1p Occupation number for oscillator 1 (ket)
 * @return double Total L1 diagonal contribution
 */
inline double L1diag(const double decay0, const double decay1, const int i0, const int i1, const int i0p, const int i1p){
  return - decay0 / 2.0 * ( i0 + i0p ) - decay1 / 2.0 * ( i1 + i1p );
};

/**
 * @brief Computes tensor product index for 2 oscillator system.
 *
 * @param nlevels0 Number of levels for oscillator 0
 * @param nlevels1 Number of levels for oscillator 1
 * @param i0 Occupation number for oscillator 0 (bra)
 * @param i1 Occupation number for oscillator 1 (bra)
 * @param i0p Occupation number for oscillator 0 (ket)
 * @param i1p Occupation number for oscillator 1 (ket)
 * @return int Linear index in tensor product space
 */
inline int TensorGetIndex(const int nlevels0, const int nlevels1,const  int i0, const int i1, const int i0p, const int i1p){
  return i0*nlevels1 + i1 + (nlevels0 * nlevels1) * ( i0p * nlevels1 + i1p);
};


// Matrix-free solver inlines for 3 oscillators
// See documentation for 1-2 oscillator functions above for parameter descriptions
inline double H_detune(const double detuning0, const double detuning1, const double detuning2, const int i0, const int i1, const int i2) {
  return detuning0*i0 + detuning1*i1 + detuning2*i2;
};
inline double H_selfkerr(const double xi0, const double xi1, const double xi2, const int i0, const int i1, const int i2) {
  return - xi0 / 2.0 * i0 * (i0-1) - xi1 / 2.0 * i1 * (i1-1) - xi2 / 2.0 * i2 * (i2-1);
};
inline double H_crosskerr(const double xi01, const double xi02, const double xi12, const int i0, const int i1, const int i2) {
  return - xi01 * i0 * i1 - xi02 * i0 * i2 - xi12 * i1 * i2;
};
inline double L2(const double dephase0, const double dephase1, const double dephase2, const int i0, const int i1, const int i2, const int i0p, const int i1p, const int i2p){
  return dephase0 * ( i0*i0p - 1./2. * (i0*i0 + i0p*i0p) ) 
       + dephase1 * ( i1*i1p - 1./2. * (i1*i1 + i1p*i1p) )
       + dephase2 * ( i2*i2p - 1./2. * (i2*i2 + i2p*i2p) );
};
inline double L1diag(const double decay0, const double decay1, const double decay2, const int i0, const int i1, const int i2, const int i0p, const int i1p, const int i2p){
  return - decay0 / 2.0 * ( i0 + i0p ) 
         - decay1 / 2.0 * ( i1 + i1p )
         - decay2 / 2.0 * ( i2 + i2p );
};
inline int TensorGetIndex(const int nlevels0, const int nlevels1, const int nlevels2, const  int i0, const int i1, const int i2, const int i0p, const int i1p, const int i2p){
  return i0*nlevels1*nlevels2 + i1*nlevels2 + i2 + (nlevels0 * nlevels1 * nlevels2) * ( i0p * nlevels1*nlevels2 + i1p*nlevels2 + i2p);
};


// Matrix-free solver inlines for 4 oscillators
// See documentation for 1-2 oscillator functions above for parameter descriptions
inline double H_detune(const double detuning0, const double detuning1, const double detuning2, const double detuning3, const int i0, const int i1, const int i2, const int i3) {
  return detuning0*i0 + detuning1*i1 + detuning2*i2 + detuning3*i3;
};
inline double H_selfkerr(const double xi0, const double xi1, const double xi2, const double xi3, const int i0, const int i1, const int i2, const int i3) {
  return - xi0 / 2.0 * i0 * (i0-1) - xi1 / 2.0 * i1 * (i1-1) - xi2 / 2.0 * i2 * (i2-1) - xi3/2.0 * i3 * (i3-1);
};
inline double H_crosskerr(const double xi01, const double xi02, const double xi03, const double xi12, const double xi13, const double xi23, const int i0, const int i1, const int i2, const int i3) {
  return - xi01 * i0 * i1 - xi02 * i0 * i2  - xi03*i0*i3 - xi12 * i1 * i2 - xi13*i1*i3 - xi23*i2*i3;
};
inline double L2(const double dephase0, const double dephase1, const double dephase2, const double dephase3, const int i0, const int i1, const int i2, const int i3, const int i0p, const int i1p, const int i2p, const int i3p){
  return dephase0 * ( i0*i0p - 1./2. * (i0*i0 + i0p*i0p) ) 
       + dephase1 * ( i1*i1p - 1./2. * (i1*i1 + i1p*i1p) )
       + dephase2 * ( i2*i2p - 1./2. * (i2*i2 + i2p*i2p) )
       + dephase3 * ( i3*i3p - 1./2. * (i3*i3 + i3p*i3p) );
};
inline double L1diag(const double decay0, const double decay1, const double decay2, const double decay3, const int i0, const int i1, const int i2, const int i3, const int i0p, const int i1p, const int i2p, const int i3p){
  return - decay0 / 2.0 * ( i0 + i0p ) 
         - decay1 / 2.0 * ( i1 + i1p )
         - decay2 / 2.0 * ( i2 + i2p )
         - decay3 / 2.0 * ( i3 + i3p );
};
inline int TensorGetIndex(const int nlevels0, const int nlevels1, const int nlevels2, const int nlevels3, const  int i0, const int i1, const int i2, const int i3, const int i0p, const int i1p, const int i2p, const int i3p){
  return i0*nlevels1*nlevels2*nlevels3 + i1*nlevels2*nlevels3 + i2*nlevels3 + i3 + (nlevels0 * nlevels1 * nlevels2 * nlevels3) * ( i0p * nlevels1*nlevels2*nlevels3 + i1p*nlevels2*nlevels3 + i2p*nlevels3 + i3p);
}

// Matrix-free solver inlines for 5 oscillators
// See documentation for 1-2 oscillator functions above for parameter descriptions
inline double H_detune(const double detuning0, const double detuning1, const double detuning2, const double detuning3, const double detuning4, const int i0, const int i1, const int i2, const int i3, const int i4) {
  return detuning0*i0 + detuning1*i1 + detuning2*i2 + detuning3*i3 + detuning4*i4;
};
inline double H_selfkerr(const double xi0, const double xi1, const double xi2, const double xi3, const double xi4, const int i0, const int i1, const int i2, const int i3, const int i4) {
  return - xi0 / 2.0 * i0 * (i0-1) - xi1 / 2.0 * i1 * (i1-1) - xi2 / 2.0 * i2 * (i2-1) - xi3/2.0 * i3 * (i3-1) - xi4/2.0 * i4 * (i4-1);
};
inline double H_crosskerr(const double xi01, const double xi02, const double xi03, const double xi04, const double xi12, const double xi13, const double xi14, const double xi23, const double xi24, const double xi34, const int i0, const int i1, const int i2, const int i3, const int i4) {
  return - xi01 * i0 * i1 - xi02 * i0 * i2  - xi03*i0*i3 - xi04*i0*i4 - xi12 * i1 * i2 - xi13*i1*i3 - xi14*i1*i4 - xi23*i2*i3 - xi24*i2*i4 - xi34*i3*i4;
};
inline double L2(const double dephase0, const double dephase1, const double dephase2, const double dephase3, const double dephase4, const int i0, const int i1, const int i2, const int i3, const int i4, const int i0p, const int i1p, const int i2p, const int i3p, const int i4p){
  return dephase0 * ( i0*i0p - 1./2. * (i0*i0 + i0p*i0p) ) 
       + dephase1 * ( i1*i1p - 1./2. * (i1*i1 + i1p*i1p) )
       + dephase2 * ( i2*i2p - 1./2. * (i2*i2 + i2p*i2p) )
       + dephase3 * ( i3*i3p - 1./2. * (i3*i3 + i3p*i3p) )
       + dephase4 * ( i4*i4p - 1./2. * (i4*i4 + i4p*i4p) );
};
inline double L1diag(const double decay0, const double decay1, const double decay2, const double decay3, const double decay4, const int i0, const int i1, const int i2, const int i3, const int i4, const int i0p, const int i1p, const int i2p, const int i3p, const int i4p){
  return - decay0 / 2.0 * ( i0 + i0p ) 
         - decay1 / 2.0 * ( i1 + i1p )
         - decay2 / 2.0 * ( i2 + i2p )
         - decay3 / 2.0 * ( i3 + i3p )
         - decay4 / 2.0 * ( i4 + i4p );
};
inline int TensorGetIndex(const int nlevels0, const int nlevels1, const int nlevels2, const int nlevels3, const int nlevels4, const  int i0, const int i1, const int i2, const int i3, const int i4, const int i0p, const int i1p, const int i2p, const int i3p, const int i4p){
  return i0*nlevels1*nlevels2*nlevels3*nlevels4 + i1*nlevels2*nlevels3*nlevels4 + i2*nlevels3*nlevels4 + i3*nlevels4 + i4 + (nlevels0 * nlevels1 * nlevels2 * nlevels3*nlevels4) * ( i0p * nlevels1*nlevels2*nlevels3*nlevels4 + i1p*nlevels2*nlevels3*nlevels4 + i2p*nlevels3*nlevels4+ i3p*nlevels4 + i4p);
}


/**
 * @brief Matrix-free solver inline for gradient updates for oscillator i.
 *
 * Computes coefficients for gradient computation with respect to control parameters.
 *
 * @param it Current tensor index
 * @param n Number of levels for current oscillator
 * @param np Number of levels for conjugate oscillator
 * @param i Current occupation number
 * @param ip Conjugate occupation number
 * @param stridei Stride for current oscillator
 * @param strideip Stride for conjugate oscillator
 * @param xptr Pointer to state vector data
 * @param res_p_re Pointer to store real part of p result
 * @param res_p_im Pointer to store imaginary part of p result
 * @param res_q_re Pointer to store real part of q result
 * @param res_q_im Pointer to store imaginary part of q result
 */
inline void dRHSdp_getcoeffs(const int it, const int n, const int np, const int i, const int ip, const int stridei, const int strideip, const double* xptr, double* res_p_re, double* res_p_im, double* res_q_re, double* res_q_im) {

  *res_p_re = 0.0;
  *res_p_im = 0.0;
  *res_q_re = 0.0;
  *res_q_im = 0.0;

  /* ik+1..,ik'.. term */
  if (i < n-1) {
    int itx = it + stridei;
    double xre = xptr[2 * itx];
    double xim = xptr[2 * itx + 1];
    double sq = sqrt(i + 1);
    *res_p_re +=   sq * xim;
    *res_p_im += - sq * xre;
    *res_q_re +=   sq * xre;
    *res_q_im +=   sq * xim;
  }
  /* \rho(ik..,ik'+1..) */
  if (ip < np-1) {
    int itx = it + strideip;
    double xre = xptr[2 * itx];
    double xim = xptr[2 * itx + 1];
    double sq = sqrt(ip + 1);
    *res_p_re += - sq * xim;
    *res_p_im += + sq * xre;
    *res_q_re +=   sq * xre;
    *res_q_im +=   sq * xim;
  }
  /* \rho(ik-1..,ik'..) */
  if (i > 0) {
    int itx = it - stridei;
    double xre = xptr[2 * itx];
    double xim = xptr[2 * itx + 1];
    double sq = sqrt(i);
    *res_p_re += + sq * xim;
    *res_p_im += - sq * xre;
    *res_q_re += - sq * xre;
    *res_q_im += - sq * xim;
  }
  /* \rho(ik..,ik'-1..) */
  if (ip > 0) {
    int itx = it - strideip;
    double xre = xptr[2 * itx];
    double xim = xptr[2 * itx + 1];
    double sq = sqrt(ip);
    *res_p_re += - sq * xim;
    *res_p_im += + sq * xre;
    *res_q_re += - sq * xre;
    *res_q_im += - sq * xim;
  }
}

/**
 * @brief Matrix-free solver inline for dipole-dipole coupling between oscillators.
 *
 * Implements J_kl coupling terms in the rotating frame between oscillator i and j.
 *
 * @param it Current tensor index
 * @param ni Number of levels for oscillator i
 * @param nj Number of levels for oscillator j
 * @param nip Number of levels for oscillator i (conjugate)
 * @param njp Number of levels for oscillator j (conjugate)
 * @param i Occupation number for oscillator i
 * @param ip Occupation number for oscillator i (conjugate)
 * @param j Occupation number for oscillator j
 * @param jp Occupation number for oscillator j (conjugate)
 * @param stridei Stride for oscillator i
 * @param strideip Stride for oscillator i (conjugate)
 * @param stridej Stride for oscillator j
 * @param stridejp Stride for oscillator j (conjugate)
 * @param xptr Pointer to state vector data
 * @param Jij Coupling coefficient
 * @param cosij Cosine of frequency difference
 * @param sinij Sine of frequency difference
 * @param yre Pointer to store real part of result
 * @param yim Pointer to store imaginary part of result
 */
inline void Jkl_coupling(const int it, const int ni, const int nj, const int nip, const int njp, const int i, const int ip, const int j, const int jp, const int stridei, const int strideip, const int stridej, const int stridejp, const double* xptr, const double Jij, const double cosij, const double sinij, double* yre, double* yim) {
  if (fabs(Jij)>1e-10) {
    //  1) J_kl (-icos + sin) * ρ_{E−k+l i, i′}
    if (i > 0 && j < nj-1) {
      int itx = it - stridei + stridej;
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(i * (j + 1));
      // sin u + cos v + i ( -cos u + sin v)
      *yre += Jij * sq * (   cosij * xim + sinij * xre);
      *yim += Jij * sq * ( - cosij * xre + sinij * xim);
    }
    // 2) J_kl (−icos − sin)sqrt(il*(ik +1)) ρ_{E+k−li,i′}
    if (i < ni-1 && j > 0) {
      int itx = it + stridei - stridej;  // E+k-l i, i'
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(j * (i + 1)); // sqrt( il*(ik+1))
      // -sin u + cos v + i (-cos u - sin v)
      *yre += Jij * sq * (   cosij * xim - sinij * xre);
      *yim += Jij * sq * ( - cosij * xre - sinij * xim);
    }
    // 3) J_kl ( icos + sin)sqrt(ik'*(il' +1)) ρ_{i,E-k+li'}
    if (ip > 0 && jp < njp-1) {
      int itx = it - strideip + stridejp;  // i, E-k+l i'
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(ip * (jp + 1)); // sqrt( ik'*(il'+1))
      //  sin u - cos v + i ( cos u + sin v)
      *yre += Jij * sq * ( - cosij * xim + sinij * xre);
      *yim += Jij * sq * (   cosij * xre + sinij * xim);
    }
    // 4) J_kl ( icos - sin)sqrt(il'*(ik' +1)) ρ_{i,E+k-li'}
    if (ip < nip-1 && jp > 0) {
      int itx = it + strideip - stridejp;  // i, E+k-l i'
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(jp * (ip + 1)); // sqrt( il'*(ik'+1))
      // - sin u - cos v + i ( cos u - sin v)
      *yre += Jij * sq * ( - cosij * xim - sinij * xre);
      *yim += Jij * sq * (   cosij * xre - sinij * xim);
    }
  }
}

/**
 * @brief Transpose of dipole-dipole coupling for adjoint computations.
 *
 * @param it Current tensor index
 * @param ni Number of levels for oscillator i
 * @param nj Number of levels for oscillator j
 * @param nip Number of levels for oscillator i (conjugate)
 * @param njp Number of levels for oscillator j (conjugate)
 * @param i Occupation number for oscillator i
 * @param ip Occupation number for oscillator i (conjugate)
 * @param j Occupation number for oscillator j
 * @param jp Occupation number for oscillator j (conjugate)
 * @param stridei Stride for oscillator i
 * @param strideip Stride for oscillator i (conjugate)
 * @param stridej Stride for oscillator j
 * @param stridejp Stride for oscillator j (conjugate)
 * @param xptr Pointer to state vector data
 * @param Jij Coupling coefficient
 * @param cosij Cosine of frequency difference
 * @param sinij Sine of frequency difference
 * @param yre Pointer to store real part of result
 * @param yim Pointer to store imaginary part of result
 */
inline void Jkl_coupling_T(const int it, const int ni, const int nj, const int nip, const int njp, const int i, const int ip, const int j, const int jp, const int stridei, const int strideip, const int stridej, const int stridejp, const double* xptr, const double Jij, const double cosij, const double sinij, double* yre, double* yim) {
  if (fabs(Jij)>1e-10) {
    //  1) [...] * \bar y_{E+k-l i, i′}
    if (i < ni-1 && j > 0) {
      int itx = it + stridei - stridej;
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(j * (i + 1));
      *yre += Jij * sq * ( - cosij * xim + sinij * xre);
      *yim += Jij * sq * ( + cosij * xre + sinij * xim);
    }
    // 2) J_kl (−icos − sin)sqrt(ik*(il +1)) \bar y_{E-k+li,i′}
    if (i > 0 && j < nj-1) {
      int itx = it - stridei + stridej;  // E-k+l i, i'
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(i * (j + 1)); // sqrt( ik*(il+1))
      *yre += Jij * sq * ( - cosij * xim - sinij * xre);
      *yim += Jij * sq * ( + cosij * xre - sinij * xim);
    }
    // 3) J_kl ( icos + sin)sqrt(il'*(ik' +1)) \bar y_{i,E+k-li'}
    if (ip < nip-1 && jp > 0) {
      int itx = it + strideip - stridejp;  // i, E+k-l i'
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(jp * (ip + 1)); // sqrt( il'*(ik'+1))
      *yre += Jij * sq * (   cosij * xim + sinij * xre);
      *yim += Jij * sq * ( - cosij * xre + sinij * xim);
    }
    // 4) J_kl ( icos - sin)sqrt(ik'*(il' +1)) \bar y_{i,E-k+li'}
    if (ip > 0 && jp < njp-1) {
      int itx = it - strideip + stridejp;  // i, E-k+l i'
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(ip * (jp + 1)); // sqrt( ik'*(il'+1))
      *yre += Jij * sq * (   cosij * xim - sinij * xre);
      *yim += Jij * sq * ( - cosij * xre - sinij * xim);
    }
  }

}

/**
 * @brief Matrix-free solver inline for off-diagonal L1 decay term.
 *
 * @param it Current tensor index
 * @param n Number of levels
 * @param i Occupation number (bra)
 * @param ip Occupation number (ket)
 * @param stridei Stride for bra index
 * @param strideip Stride for ket index
 * @param xptr Pointer to state vector data
 * @param decayi Decay rate
 * @param yre Pointer to store real part of result
 * @param yim Pointer to store imaginary part of result
 */
inline void L1decay(const int it, const int n, const int i, const int ip, const int stridei, const int strideip, const double* xptr, const double decayi, double* yre, double* yim){
  if  (fabs(decayi) > 1e-12) {
    if (i < n-1 && ip < n-1) {
      double l1off = decayi * sqrt((i+1)*(ip+1));
      int itx = it + stridei + strideip;
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      *yre += l1off * xre;
      *yim += l1off * xim;
    }
  }
}


/**
 * @brief Transpose of off-diagonal L1 decay for adjoint computations.
 *
 * @param it Current tensor index
 * @param i Occupation number (bra)
 * @param ip Occupation number (ket)
 * @param stridei Stride for bra index
 * @param strideip Stride for ket index
 * @param xptr Pointer to state vector data
 * @param decayi Decay rate
 * @param yre Pointer to store real part of result
 * @param yim Pointer to store imaginary part of result
 */
inline void L1decay_T(const int it, const int i, const int ip, const int stridei, const int strideip, const double* xptr, const double decayi, double* yre, double* yim){
  if (fabs(decayi) > 1e-12) {
      if (i > 0 && ip > 0) {
        double l1off = decayi * sqrt(i*ip);
        int itx = it - stridei - strideip;
        double xre = xptr[2 * itx];
        double xim = xptr[2 * itx + 1];
        *yre += l1off * xre;
        *yim += l1off * xim;
      }
    }
}

/**
 * @brief Matrix-free solver inline for control terms.
 *
 * Applies control Hamiltonian terms (ladder operators) to the state.
 *
 * @param it Current tensor index
 * @param n Number of levels
 * @param i Occupation number (bra)
 * @param np Number of levels (conjugate)
 * @param ip Occupation number (ket)
 * @param stridei Stride for bra index
 * @param strideip Stride for ket index
 * @param xptr Pointer to state vector data
 * @param pt Real part of control amplitude
 * @param qt Imaginary part of control amplitude
 * @param yre Pointer to store real part of result
 * @param yim Pointer to store imaginary part of result
 */
inline void control(const int it, const int n, const int i, const int np, const int ip, const int stridei, const int strideip, const double* xptr, const double pt, const double qt, double* yre, double* yim){
  /* \rho(ik+1..,ik'..) term */
  if (i < n-1) {
      int itx = it + stridei;
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(i + 1);
      *yre += sq * (   pt * xim + qt * xre);
      *yim += sq * ( - pt * xre + qt * xim);
    }
    /* \rho(ik..,ik'+1..) */
    if (ip < np-1) {
      int itx = it + strideip;
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(ip + 1);
      *yre += sq * ( -pt * xim + qt * xre);
      *yim += sq * (  pt * xre + qt * xim);
    }
    /* \rho(ik-1..,ik'..) */
    if (i > 0) {
      int itx = it - stridei;
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(i);
      *yre += sq * (  pt * xim - qt * xre);
      *yim += sq * (- pt * xre - qt * xim);
    }
    /* \rho(ik..,ik'-1..) */
    if (ip > 0) {
      int itx = it - strideip;
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(ip);
      *yre += sq * (- pt * xim - qt * xre);
      *yim += sq * (  pt * xre - qt * xim);
    }
}


/**
 * @brief Transpose of control terms for adjoint computations.
 *
 * @param it Current tensor index
 * @param n Number of levels
 * @param i Occupation number (bra)
 * @param np Number of levels (conjugate)
 * @param ip Occupation number (ket)
 * @param stridei Stride for bra index
 * @param strideip Stride for ket index
 * @param xptr Pointer to state vector data
 * @param pt Real part of control amplitude
 * @param qt Imaginary part of control amplitude
 * @param yre Pointer to store real part of result
 * @param yim Pointer to store imaginary part of result
 */
inline void control_T(const int it, const int n, const int i, const int np, const int ip, const int stridei, const int strideip, const double* xptr, const double pt, const double qt, double* yre, double* yim){
  /* \rho(ik+1..,ik'..) term */
  if (i > 0) {
    int itx = it - stridei;
    double xre = xptr[2 * itx];
    double xim = xptr[2 * itx + 1];
    double sq = sqrt(i);
    *yre += sq * ( - pt * xim + qt * xre);
    *yim += sq * (   pt * xre + qt * xim);
  }
  /* \rho(ik..,ik'+1..) */
  if (ip > 0) {
    int itx = it - strideip;
    double xre = xptr[2 * itx];
    double xim = xptr[2 * itx + 1];
    double sq = sqrt(ip);
    *yre += sq * (  pt * xim + qt * xre);
    *yim += sq * ( -pt * xre + qt * xim);
  }
  /* \rho(ik-1..,ik'..) */
  if (i < n-1) {
    int itx = it + stridei;
    double xre = xptr[2 * itx];
    double xim = xptr[2 * itx + 1];
    double sq = sqrt(i+1);
    *yre += sq * (- pt * xim - qt * xre);
    *yim += sq * (  pt * xre - qt * xim);
  }
  /* \rho(ik..,ik'-1..) */
  if (ip < np-1) {
    int itx = it + strideip;
    double xre = xptr[2 * itx];
    double xim = xptr[2 * itx + 1];
    double sq = sqrt(ip+1);
    *yre += sq * (+ pt * xim - qt * xre);
    *yim += sq * (- pt * xre - qt * xim);
  }
}