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

double Vec_MPI::normsq(){
  double pnorm=0.0;
  VecNorm(petscvec, NORM_2, &pnorm);
  return pow(pnorm, 2.0);
}

void Vec_MPI::view() {
  printf("Viewing the Vec_MPI vector:\n");
  VecView(petscvec, NULL);
}

void Vec_MPI::add(double a, Vector* y){
  Vec_MPI* vecy = (Vec_MPI*) y;
  VecAXPY(petscvec, a, vecy->getData());
}

void Vec_MPI::scale(double a){
  VecScale(petscvec, a);
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
void Vec_OpenMP::add(double a, Vector* y){}
void Vec_OpenMP::scale(double a){}
double Vec_OpenMP::normsq(){return 0.0;}


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

void Mat_MPI::mult(Vector* xin, Vector* xout, bool add){


  Vec_MPI* vecxin = (Vec_MPI*) xin;
  Vec_MPI* vecxout = (Vec_MPI*) xout;
  Vec pin  = vecxin->getData();
  Vec pout = vecxout->getData();
  if (add)  MatMultAdd(PetscMat, pin, pout, pout);
  else MatMult(PetscMat, pin, pout);
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
// TODO 

Mat_OpenMP::Mat_OpenMP() : SparseMatrix() {}
Mat_OpenMP::~Mat_OpenMP() {}
void Mat_OpenMP::view() {}
void Mat_OpenMP::mult(Vector* xin, Vector* xout, bool add){}
void Mat_OpenMP::setValues(std::vector<int>& rows, std::vector<int>& cols, std::vector<double>& vals){}
void Mat_OpenMP::add(double a, Vector* y){}