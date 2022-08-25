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

double* Vec_MPI::c_ptr(){
   double* vptr;
   VecGetArray(petscvec,&vptr);
   return vptr;
}
void Vec_MPI::restore_ptr(double* vptr ){
   VecRestoreArray(petscvec,&vptr);
}

Vec_MPI::~Vec_MPI() {

  // Destroy the petsc vector
  VecDestroy(&petscvec);
}


/*----------------*/

Vec_OpenMP::Vec_OpenMP() : Vector() 
{ 
   ilow=0;
   iupp=-1;
   m_data=0;
}

Vec_OpenMP::Vec_OpenMP( int n ) : Vector(n) 
{ 
   iupp=n-1;
   ilow=0;
   if( iupp-ilow+1 > 0 ) 
      m_data = new double[iupp-ilow+1]; 
   else
      m_data=0;
}

Vec_OpenMP::~Vec_OpenMP() 
{
   if( iupp-ilow+1 > 0 ) 
      delete[] m_data;
}

void Vec_OpenMP::view()
{
   std::cout << "OpenMP-vector: " << std::endl;
   for( int i=0; i<iupp-ilow+1; i++ )
      std::cout << i+ilow << " " << m_data[i] << std::endl;
   std::cout << std::endl;
}

void Vec_OpenMP::setZero() 
{
#pragma omp parallel for
   for( int i=0; i<iupp-ilow+1; i++ )
      m_data[i]=0;
}

void Vec_OpenMP::add(double a, Vector* y)
{
   double* yp = y->c_ptr();
#pragma omp parallel for
   for( int i=0; i<iupp-ilow+1; i++ )
      m_data[i] += a*yp[i];
}

void Vec_OpenMP::scale(double a)
{
#pragma omp parallel for
   for( int i=0; i<iupp-ilow+1; i++ )
      m_data[i] *= a;
}

double Vec_OpenMP::normsq()
{
   double nsq=0;
#pragma omp parallel for reduction(+:nsq)
   for( int i=0; i<iupp-ilow+1; i++ )
      nsq += m_data[i]*m_data[i];
   return nsq;
}

double* Vec_OpenMP::c_ptr() 
{
   return m_data;
}

void Vec_OpenMP::restore_ptr(double* vptr )
{
}

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
Mat_OpenMP::Mat_OpenMP(int dim) : SparseMatrix(dim){ m_nrows=m_ncols=dim;}
Mat_OpenMP::~Mat_OpenMP() {}

//-----------------------------------------------------------------------
void Mat_OpenMP::view()
{
   for( int i=0 ; i<m_nrows ;i++ )
   {
      for( int j=m_rowstarts[i]; j<m_rowstarts[i+1]; j++ )
         std::cout << "a(" << i << ", " << m_cols[j] << ") = " 
                   << m_elements[j] << " ";
      std::cout << std::endl;
   }
}

//-----------------------------------------------------------------------
void Mat_OpenMP::setValues(std::vector<int>& rows, std::vector<int>& cols, std::vector<double>& vals)
{
   std::cout << "Mat_OpenMP::setValues not yet implemented " << std::endl;
}

//-----------------------------------------------------------------------
void Mat_OpenMP::mult(Vector* xin, Vector* xout, bool add)
{
   double* xinp=xin->c_ptr(), *xoutp=xout->c_ptr();
   if( !add )
#pragma omp parallel for
      for( int i=0; i<m_nrows; i++ )
         xoutp[i]=0;

#pragma omp parallel for
      for( int i=0; i<m_nrows; i++ )
         for( int ind=m_rowstarts[i]; ind < m_rowstarts[i+1]; ind++)
         {
            int j=m_cols[ind];
            xoutp[i] += m_elements[ind]*xinp[j];
         }
}

//-----------------------------------------------------------------------
void Mat_OpenMP::setup( int sysnr, int offnr, std::vector<int> n, double* alpha, bool kmat )
{
   int sgn=-1;
   if( kmat )
      sgn = 1;
   m_nrows = 1;
   for( int k=0 ; k<n.size() ;k++)
      m_nrows *= n[k];
   m_ncols = m_nrows;

   m_rowstarts = new int[m_nrows+1];
   std::vector<int> cols;
   std::vector<double> elements;

   int nsys = n.size();
   int nnz  = 0;
   for( int ind = 0 ; ind < m_nrows ;ind++)
   {
      int inr;
      int t0 = ind;
      for( int k=nsys-1 ; k >= sysnr ;k--)
      {
         int t = t0/n[k];
         inr   = t0-n[k]*t;
         t0    = t;
      }
      m_rowstarts[ind] = nnz;
      if( inr > 0 )
      {
         elements.push_back(sgn*alpha[inr-1]);
         cols.push_back(ind-offnr);
         nnz++;
      }
      if( inr < n[sysnr]-1 )
      {
         elements.push_back(alpha[inr]);
         cols.push_back(ind+offnr);
         nnz++;
      }
   }
   m_rowstarts[m_nrows]=nnz;

   m_cols = new int[cols.size()];
   for( int i=0; i < cols.size(); i++ )
      m_cols[i] = cols[i];
   m_elements = new double[elements.size()];
   for( int i=0; i < elements.size(); i++ )
      m_elements[i] = elements[i];
}

//-----------------------------------------------------------------------
void Mat_OpenMP::setupJ( std::vector<int> n, 
                         int k, double* alphak, int ok, 
                         int m, double* alpham, int om, bool re )
{
   int sg=re?1:-1;
   m_nrows = 1;
   for( int kk=0 ; kk<n.size() ;kk++)
      m_nrows *= n[kk];
   m_ncols = m_nrows;
   m_rowstarts = new int[m_nrows+1];

   std::vector<int> cols;
   std::vector<double> elements;

   int nsys = n.size();
   int nnz  = 0;
   for( int ind = 0 ; ind < m_nrows ;ind++)
   {
      int ik;
      int t0 = ind;
      for( int p=nsys-1 ; p >= k ;p--)
      {
         int t = t0/n[p];
         ik   = t0-n[p]*t;
         t0    = t;
      }
      int im;
      t0=ind;
      for( int p=nsys-1 ; p >= m ;p--)
      {
         int t = t0/n[p];
         im    = t0-n[p]*t;
         t0    = t;
      }
      m_rowstarts[ind] = nnz;
      
      if( ik > 0 && im < n[m]-1)
      {
         elements.push_back(sg*alphak[ik-1]*alpham[im]);
         cols.push_back(ind-ok+om);
         nnz++;
      }
      if( ik < n[k]-1 && im > 0 )
      {
         elements.push_back(alphak[ik]*alpham[im-1]);
         cols.push_back(ind+ok-om);
         nnz++;
      }
   }
   m_rowstarts[m_nrows]=nnz;

   m_cols = new int[cols.size()];
   for( int i=0; i < cols.size(); i++ )
      m_cols[i] = cols[i];
   m_elements = new double[elements.size()];
   for( int i=0; i < elements.size(); i++ )
      m_elements[i] = elements[i];
}
