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

/* Computes kronecker product A \kron B for SQUARE matrices A,B \in \R^{dim \times dim} */
/* This works in PETSC SERIAL only. */
/* The output matrix has to be allocated before with size dim*dim \times dim*dim */
PetscErrorCode AkronB(const int dim, const Mat A, const Mat B, const double alpha, Mat *Out, InsertMode insert_mode);

/* Tests if a matrix A is anti-symmetric (A^T=-A) */
PetscErrorCode MatIsAntiSymmetric(Mat A, PetscReal tol, PetscBool *flag);


/* Test if the vectorized state vector x=[u,v] represents a hermitian matrix.
 * For this to be true, we need u being symmetric, v being antisymmetric
 */
PetscErrorCode StateIsHermitian(Vec x, PetscReal tol, PetscBool *flag);


/* Test if vectorized state vector x=[u,v] represent matrix with Trace 1 */
PetscErrorCode StateHasTrace1(Vec x, PetscReal tol, PetscBool *flag);


/**
 * Read data from file
 */
void read_vector(const char *filename, double *var, int dim);


/* 
 * Compute <neigvals> eigenvalues of A
 */
int getEigvals(const Mat A, const int neigvals, std::vector<double>& eigvals, std::vector<Vec>& eigvecs);