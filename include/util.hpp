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
