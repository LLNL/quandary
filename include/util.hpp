#include <petscmat.h>
#include <vector>
#ifdef WITH_SLEPC
#include <slepceps.h>
#endif

#pragma once

int getIndexReal(const int i); // Return storage index of Re(x[i]) (colocated: x[2*i], blocked: x[i])
int getIndexImag(const int i); // Return storage index of Im(x[i]) (colocated: x[2*i+1], blocked: x[i+dim])

/* Return the index of vectorized matrix element (row,col) with matrix dimension dim x dim */
int getVecID(const int row, const int col, const int dim);

/* Map an index i in essential level system to the corresponding index in full-dimension system */
int mapEssToFull(const int i, const std::vector<int> &nlevels, const std::vector<int> &nessential);
/* Map an index i in full dimension to the corresponding index in essential dimensions */
int mapFullToEss(const int i, const std::vector<int> &nlevels, const std::vector<int> &nessential);

/* Test if a certain row/column i of the full density matrix corresponds to an essential level */
int isEssential(const int i, const std::vector<int> &nlevels, const std::vector<int> &nessential);

/* Test if a certain row/column i of the full density matrix corresponds to a guard level. A Guard level is the LAST energy level of an oscillator */
int isGuardLevel(const int i, const std::vector<int> &nlevels, const std::vector<int> &nessential);

// /* Project a state vector onto essential levels by zero'ing out elements that belong to guard-levels. */
// void projectToEss(Vec state,const std::vector<int> &nlevels, const std::vector<int> &nessential);

/* Kronecker product : Id \kron A, where Id is the Identitymatrix 
 * Mat Out must be allocated with nonzerosA * dimI
 * Input: mat A       Matrix that is multiplied
 *        int dimI    Dimension of the identity
 *        insert_mode  either INSERT_VALUES or ADD_VALUES
 * Output: Mat Out = Id \kron A
 */
PetscErrorCode Ikron(const Mat A, const int dimI, const double alpha, Mat *Out, InsertMode insert_mode);

/* Kronecker product : A \kron Id, where Id is the Identitymatrix 
 * Mat Out must be allocated with nonzerosA * dimI
 * Input: mat A       Matrix that is multiplied
 *        int dimI    Dimension of the identity
 *        insert_mode  either INSERT_VALUES or ADD_VALUES
 * Output: Mat Out = A \kron Id
 */
PetscErrorCode kronI(const Mat A, const int dimI, const double alpha, Mat *Out, InsertMode insert_mode);

/* Computes kronecker product A \kron B for matrices A,B */
/* This works in PETSC SERIAL only. */
/* The output matrix has to be allocated before with matching sizes and should to be assembles afterwards, if neccessary. */
PetscErrorCode AkronB(const Mat A, const Mat B, const double alpha, Mat *Out, InsertMode insert_mode);

/* Tests if a matrix A is anti-symmetric (A^T=-A) */
PetscErrorCode MatIsAntiSymmetric(Mat A, PetscReal tol, PetscBool *flag);


/* Test if the vectorized state vector x=[u,v] represents a hermitian matrix.
 * For this to be true, we need u being symmetric, v being antisymmetric
 */
PetscErrorCode StateIsHermitian(Vec x, PetscReal tol, PetscBool *flag);


/* Test if vectorized state vector x=[u,v] represent matrix with Trace 1 */
PetscErrorCode StateHasTrace1(Vec x, PetscReal tol, PetscBool *flag);

/* All sanity tests */
PetscErrorCode SanityTests(Vec x, PetscReal time);

/**
 * Read data from file
 */
void read_vector(const char *filename, double *var, int dim);


/* 
 * Compute <neigvals> eigenvalues of A
 */
int getEigvals(const Mat A, const int neigvals, std::vector<double>& eigvals, std::vector<Vec>& eigvecs);



/* 
 * Test if A+iB is unitary (A and B should be real-valued!)
 */
bool isUnitary(const Mat A, const Mat B);