#include "linalg.hpp"

/* -----------------*/
/* ---- Vectors ----*/
/* -----------------*/

/* Abstract base class */
Vector::Vector() {}
Vector::Vector(int size_) {size = size_;}
Vector::~Vector() {}

/* MPI Petsc vector */
Vec_MPI::Vec_MPI() : Vector() {}

Vec_MPI::Vec_MPI(int size) : Vector(size) {

  // Create the Petsc Vector of given size
  VecCreate(PETSC_COMM_WORLD, &petscvec);
  VecSetSizes(petscvec, PETSC_DECIDE, size);
  VecSetFromOptions(petscvec);
  VecGetOwnershipRange(petscvec, &ilow, &iupp);
}

void Vec_MPI::assemble() {
  VecAssemblyBegin(petscvec); 
  VecAssemblyEnd(petscvec);
}

Vec_MPI::~Vec_MPI() {

  // Destroy the petsc vector
  VecDestroy(&petscvec);
}


/* OpenMP vector a la Bjorn */
Vec_OpenMP::Vec_OpenMP() : Vector() {}
Vec_OpenMP::~Vec_OpenMP() {}


/* ------------------*/
/* ---- Matrices ----*/
/* ------------------*/

SparseMatrix::SparseMatrix() {}
SparseMatrix::~SparseMatrix() {}

Mat_MPI::Mat_MPI() : SparseMatrix() {}
Mat_MPI::~Mat_MPI()  {}

Mat_OpenMP::Mat_OpenMP() : SparseMatrix() {}
Mat_OpenMP::~Mat_OpenMP() {}