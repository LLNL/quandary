#include "linalg.hpp"

/* -----------------*/
/* ---- Vectors ----*/
/* -----------------*/

/* Abstract base class */
Vector::Vector() {}
Vector::Vector(int dim_) {dim= dim_;}
Vector::~Vector() {}

void Vector::getDistribution(int* ilow_ptr, int* iupp_ptr){ 
  *ilow_ptr = 0; 
  *iupp_ptr = dim;
}

/*----------------*/

/* MPI Petsc vector */
Vec_MPI::Vec_MPI() : Vector() {}

Vec_MPI::Vec_MPI(int dim) : Vector(dim) {

  // Create the Petsc Vector of given size
  VecCreate(PETSC_COMM_WORLD, &petscvec);
  VecSetSizes(petscvec, PETSC_DECIDE, dim);
  VecSetFromOptions(petscvec);
  VecGetOwnershipRange(petscvec, &ilow, &iupp);
}

void Vec_MPI::getDistribution(int* ilow_ptr, int* iupp_ptr){ 
  *ilow_ptr = ilow; 
  *iupp_ptr = iupp;
}

void Vec_MPI::setValues(std::vector<int>& indices, std::vector<double>& vals){

  for (int i=0; i<indices.size(); i++){
    if (ilow <= indices[i] && indices[i] < iupp) VecSetValue(petscvec, indices[i], vals[i], INSERT_VALUES);
  }

  VecAssemblyBegin(petscvec); 
  VecAssemblyEnd(petscvec);
}

void Vec_MPI::setZero() {
  VecZeroEntries(petscvec);
}


void Vec_MPI::view() {
  printf("Viewing the Vec_MPI vector:\n");
  VecView(petscvec, NULL);
}

Vec_MPI::~Vec_MPI() {

  // Destroy the petsc vector
  VecDestroy(&petscvec);
}


/*----------------*/

Vec_OpenMP::Vec_OpenMP() : Vector() {}
Vec_OpenMP::~Vec_OpenMP() {}
void Vec_OpenMP::view() {}
void Vec_OpenMP::setZero() {}


/* ------------------*/
/* ---- Matrices ----*/
/* ------------------*/

SparseMatrix::SparseMatrix() {}
SparseMatrix::SparseMatrix(int dim_) {dim = dim_;}
SparseMatrix::~SparseMatrix() {}
void SparseMatrix::getDistribution(int* ilow_ptr, int* iupp_ptr){ 
  *ilow_ptr = 0;
  *iupp_ptr = dim;
}

/*----------------*/

Mat_MPI::Mat_MPI() : SparseMatrix() {}
Mat_MPI::Mat_MPI(int dim, int preallocate_per_row) : SparseMatrix(dim) {

    MatCreate(PETSC_COMM_WORLD, &PetscMat);
    MatSetType(PetscMat, MATMPIAIJ);
    MatSetSizes(PetscMat, PETSC_DECIDE, PETSC_DECIDE, dim, dim);  
    if (preallocate_per_row>0) MatMPIAIJSetPreallocation(PetscMat, preallocate_per_row, NULL, preallocate_per_row, NULL);
    MatSetUp(PetscMat);
    MatSetFromOptions(PetscMat);
    MatGetOwnershipRange(PetscMat, &ilow, &iupp);


}

void Mat_MPI::setValue(int row, int col, double val, bool add){
  InsertMode mode = INSERT_VALUES;
  if (add) mode = ADD_VALUES;
  MatSetValue(PetscMat, row, col, val, mode);
}

void Mat_MPI::setValues(std::vector<int>& rows, std::vector<int>& cols, std::vector<double>& vals){
  // vals have to be row-major!
  MatSetValues(PetscMat, rows.size(), rows.data(), cols.size(), cols.data(), vals.data(), INSERT_VALUES);

  MatAssemblyBegin(PetscMat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(PetscMat, MAT_FINAL_ASSEMBLY);
}

void Mat_MPI::getDistribution(int* ilow_ptr, int* iupp_ptr){ 
  *ilow_ptr = ilow;
  *iupp_ptr = iupp;
}

void Mat_MPI::assemble(){
  MatAssemblyBegin(PetscMat, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(PetscMat, MAT_FINAL_ASSEMBLY);
}

void Mat_MPI::view(){
  printf("The Petsc Matrix is:\n");
  MatView(PetscMat, NULL);
}

Mat_MPI::~Mat_MPI()  {
  MatDestroy(&PetscMat);
}

/*----------------*/

Mat_OpenMP::Mat_OpenMP() : SparseMatrix() {}
Mat_OpenMP::~Mat_OpenMP() {}

void Mat_OpenMP::view() {}

void Mat_OpenMP::setValues(std::vector<int>& rows, std::vector<int>& cols, std::vector<double>& vals){}