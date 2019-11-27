#include <petscmat.h>
#pragma once


/* Kronecker product : Id \kron A, where Id is the Identitymatrix 
 * Mat Out must be allocated with nonzerosA * dimI
 * Input: mat A       Matrix that is multiplied
 *        int dimI    Dimension of the identity
 *        insert_mode  either INSERT_VALUES or ADD_VALUES
 * Output: Mat Out = Id \kron A
 */
PetscErrorCode Ikron(Mat A, int dimI, double alpha, Mat *Out, InsertMode insert_mode);

/* Kronecker product : A \kron Id, where Id is the Identitymatrix 
 * Mat Out must be allocated with nonzerosA * dimI
 * Input: mat A       Matrix that is multiplied
 *        int dimI    Dimension of the identity
 *        insert_mode  either INSERT_VALUES or ADD_VALUES
 * Output: Mat Out = A \kron Id
 */
PetscErrorCode kronI(Mat A, int dimI, double alpha, Mat *Out, InsertMode insert_mode);


/* Tests if a matrix A is anti-symmetric (A^T=-A) */
PetscErrorCode MatIsAntiSymmetric(Mat A, PetscReal tol, PetscBool *flag);


/* Test if the vectorized state vector x=[u,v] represents a hermitian matrix.
 * For this to be true, we need u being symmetric, v being antisymmetric
 */
PetscErrorCode StateIsHermitian(Vec x, PetscReal tol, PetscBool *flag);


/* Test if vectorized state vector x=[u,v] represent matrix with Trace 1 */
PetscErrorCode StateHasTrace1(Vec x, PetscReal tol, PetscBool *flag);