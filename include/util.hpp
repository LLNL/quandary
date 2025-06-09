#include <petscmat.h>
#include <iostream>
#include <vector>
#ifdef WITH_SLEPC
#include <slepceps.h>
#endif

#pragma once


/**
 * @brief Sigmoid function for smooth transitions.
 *
 * @param width Transition width parameter
 * @param x Input value
 * @return double Sigmoid function value
 */
double sigmoid(double width, double x);

/**
 * @brief Derivative of sigmoid function.
 *
 * @param width Transition width parameter
 * @param x Input value
 * @return double Derivative of sigmoid function
 */
double sigmoid_diff(double width, double x);

/**
 * @brief Computes ramping factor for control pulse shaping.
 *
 * Computes smooth ramping factor for interval [tstart, tstop] using sigmoid
 * transitions with specified width tramp.
 *
 * @param time Current time
 * @param tstart Start time of interval
 * @param tstop Stop time of interval
 * @param tramp Ramping transition width
 * @return double Ramping factor between 0 and 1
 */
double getRampFactor(const double time, const double tstart, const double tstop, const double tramp);

/**
 * @brief Derivative of ramping factor with respect to stop time.
 *
 * @param time Current time
 * @param tstart Start time of interval
 * @param tstop Stop time of interval
 * @param tramp Ramping transition width
 * @return double Derivative with respect to tstop
 */
double getRampFactor_diff(const double time, const double tstart, const double tstop, const double tramp);

/**
 * @brief Returns storage index for real part of a state vector element.
 *
 * @param i Element index
 * @return int Storage index (colocated: x[2*i])
 */
int getIndexReal(const int i);

/**
 * @brief Returns storage index for imaginary part of a state vector element.
 *
 * @param i Element index
 * @return int Storage index (colocated: x[2*i+1])
 */
int getIndexImag(const int i);

/**
 * @brief Returns vectorized index for matrix element (row,col).
 *
 * @param row Matrix row index
 * @param col Matrix column index
 * @param dim Matrix dimension
 * @return int Vectorized index for element (row,col)
 */
int getVecID(const int row, const int col, const int dim);

/**
 * @brief Maps index from essential level system to full-dimension system.
 *
 * @param i Index in essential level system
 * @param nlevels Number of levels per oscillator
 * @param nessential Number of essential levels per oscillator
 * @return int Corresponding index in full-dimension system
 */
int mapEssToFull(const int i, const std::vector<int> &nlevels, const std::vector<int> &nessential);

/**
 * @brief Maps index from full dimension to essential dimension system.
 *
 * @param i Index in full dimension system
 * @param nlevels Number of levels per oscillator
 * @param nessential Number of essential levels per oscillator
 * @return int Corresponding index in essential dimension system
 */
int mapFullToEss(const int i, const std::vector<int> &nlevels, const std::vector<int> &nessential);

/**
 * @brief Tests if density matrix index corresponds to an essential level.
 *
 * @param i Row/column index of density matrix
 * @param nlevels Number of levels per oscillator
 * @param nessential Number of essential levels per oscillator
 * @return int Non-zero if index corresponds to essential level
 */
int isEssential(const int i, const std::vector<int> &nlevels, const std::vector<int> &nessential);

/**
 * @brief Tests if density matrix index corresponds to a guard level.
 *
 * A guard level is the highest energy level of an oscillator, used for
 * leakage detection and prevention.
 *
 * @param i Row/column index of density matrix
 * @param nlevels Number of levels per oscillator
 * @param nessential Number of essential levels per oscillator
 * @return int Non-zero if index corresponds to guard level
 */
int isGuardLevel(const int i, const std::vector<int> &nlevels, const std::vector<int> &nessential);

/**
 * @brief Computes Kronecker product \f$Id \otimes A\f$.
 *
 * Computes the Kronecker product of an identity matrix with matrix A.
 * Output matrix must be pre-allocated with sufficient non-zeros A * dimI.
 *
 * @param[in] A Input matrix
 * @param[in] dimI Dimension of identity matrix
 * @param[in] alpha Scaling factor
 * @param[out] Out Output matrix \f$(Id \otimes A)\f$
 * @param[in] insert_mode INSERT_VALUES or ADD_VALUES
 * @return PetscErrorCode Error code
 */
PetscErrorCode Ikron(const Mat A, const int dimI, const double alpha, Mat *Out, InsertMode insert_mode);

/**
 * @brief Computes Kronecker product \f$A \otimes Id\f$.
 *
 * Computes the Kronecker product of matrix A with an identity matrix.
 * Output matrix must be pre-allocated with sufficient non-zeros A * dimI.
 *
 * @param[in] A Input matrix
 * @param[in] dimI Dimension of identity matrix
 * @param[in] alpha Scaling factor
 * @param[out] Out Output matrix \f$(A \otimes Id)\f$
 * @param[in] insert_mode INSERT_VALUES or ADD_VALUES
 * @return PetscErrorCode Error code
 */
PetscErrorCode kronI(const Mat A, const int dimI, const double alpha, Mat *Out, InsertMode insert_mode);

/**
 * @brief Computes general Kronecker product \f$A \otimes B\f$.
 *
 * Computes the Kronecker product of two arbitrary matrices A and B.
 * Works in PETSc serial mode only. Output matrix must be pre-allocated
 * and should be assembled afterwards.
 *
 * @param A First input matrix
 * @param B Second input matrix
 * @param alpha Scaling factor
 * @param Out Output matrix \f$(A \otimes B)\f$
 * @param insert_mode INSERT_VALUES or ADD_VALUES
 * @return PetscErrorCode Error code
 */
PetscErrorCode AkronB(const Mat A, const Mat B, const double alpha, Mat *Out, InsertMode insert_mode);

/**
 * @brief Tests if matrix A is anti-symmetric (A^T = -A).
 *
 * @param A Input matrix to test
 * @param tol Tolerance for comparison
 * @param flag Output flag indicating anti-symmetry
 * @return PetscErrorCode Error code
 */
PetscErrorCode MatIsAntiSymmetric(Mat A, PetscReal tol, PetscBool *flag);

/**
 * @brief Tests if vectorized state represents a Hermitian matrix.
 *
 * For vectorized state x=[u,v] to represent a Hermitian matrix,
 * u must be symmetric and v must be anti-symmetric.
 *
 * @param x Vectorized state vector
 * @param tol Tolerance for comparison
 * @param flag Output flag indicating Hermiticity
 * @return PetscErrorCode Error code
 */
PetscErrorCode StateIsHermitian(Vec x, PetscReal tol, PetscBool *flag);

/**
 * @brief Tests if vectorized state vector x=[u,v] represents matrix with trace 1.
 *
 * @param x Vectorized state vector
 * @param tol Tolerance for comparison
 * @param flag Output flag indicating unit trace
 * @return PetscErrorCode Error code
 */
PetscErrorCode StateHasTrace1(Vec x, PetscReal tol, PetscBool *flag);

/**
 * @brief Performs all sanity tests on state vector.
 *
 * @param x State vector to test
 * @param time Current time for diagnostic output
 * @return PetscErrorCode Error code
 */
PetscErrorCode SanityTests(Vec x, PetscReal time);

/**
 * @brief Reads data vector from file.
 *
 * @param filename Name of file to read
 * @param var Array to store data
 * @param dim Dimension of data to read
 * @param quietmode Flag for reduced output
 * @param skiplines Number of header lines to skip
 * @param testheader Expected header string for validation
 * @return int Error code
 */
int read_vector(const char *filename, double *var, int dim, bool quietmode=false, int skiplines=0, const std::string testheader="");

/**
 * @brief Computes eigenvalues and eigenvectors of matrix A.
 *
 * Requires compilation with SLEPc for eigenvalue computations.
 *
 * @param A Input matrix
 * @param neigvals Number of eigenvalues to compute
 * @param eigvals Vector to store eigenvalues
 * @param eigvecs Vector to store eigenvectors
 * @return int Error code
 */
int getEigvals(const Mat A, const int neigvals, std::vector<double>& eigvals, std::vector<Vec>& eigvecs);

/**
 * @brief Tests if complex matrix A+iB is unitary.
 *
 * Tests whether (A+iB)(A+iB)^dagger = I for real matrices A and B.
 *
 * @param A Real part of complex matrix
 * @param B Imaginary part of complex matrix
 * @return bool True if matrix is unitary
 */
bool isUnitary(const Mat A, const Mat B);

/**
 * @brief Extends vector by repeating the last element.
 *
 * Template function that fills a vector to the specified size by
 * repeating the last element.
 *
 * @param fillme Vector to extend
 * @param tosize Target size for the vector
 */
template <typename Tval>
void copyLast(std::vector<Tval>& fillme, int tosize){
    // int norg = fillme.size();

    for (int i=fillme.size(); i<tosize; i++) 
      fillme.push_back(fillme[fillme.size()-1]);

    // if (norg < tosize) {
      // std::cout<< "I filled this: ";
      // for (int i=0; i<fillme.size(); i++) std::cout<< " " << fillme[i];
      // std::cout<<std::endl;
    // }
};