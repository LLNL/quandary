#include <petscmat.h>
#pragma once


/* Kronecker product : Id \kron A, where Id is the Identitymatrix 
 * Input: mat A       Matrix that is multiplied
 *        int dimI    Dimension of the identity
 * Output: newly created Mat Out = Id \kron A
 */
PetscErrorCode Ikron(Mat A, int dimI, Mat *Out);

/* Kronecker product : A \kron Id, where Id is the Identitymatrix 
 * Input: mat A       Matrix that is multiplied
 *        int dimI    Dimension of the identity
 * Output: newly created Mat Out = A \kron Id
 */
PetscErrorCode kronI(Mat A, int dimI, Mat *Out);
