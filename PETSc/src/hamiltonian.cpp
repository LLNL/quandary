#include "hamiltonian.hpp"


Hamiltonian::Hamiltonian(){
  dim = 0;
  nlevels = 0;
  noscillators = 0;
  oscil_vec = NULL;
  Re = NULL;
  Im = NULL;
}

Hamiltonian::~Hamiltonian(){
  if (dim > 0){
    MatDestroy(&Re);
    MatDestroy(&Im);
    MatDestroy(&H);
  }
}

int Hamiltonian::initialize(int nlevels_, int noscillators_, Oscillator** oscil_vec_){
  int ierr;

  /* Set dimensions */
  dim = (int) pow(nlevels_, noscillators_*2); // n^osc : pure states, (n^osc)^2 : Density matrix
  nlevels = nlevels_;
  noscillators = noscillators_;
  /* Set oscillator vector */
  oscil_vec = oscil_vec_;

  /* Allocate Re */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,0,NULL,&Re);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Re);CHKERRQ(ierr);
  ierr = MatSetUp(Re);CHKERRQ(ierr);
  ierr = MatSetOption(Re, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);CHKERRQ(ierr);
  for (int irow = 0; irow < dim; irow++)
  {
    ierr = MatSetValue(Re, irow, irow, 0.0, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(Re,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Re,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);


  /* Allocate Im */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,0,NULL,&Im);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Im);CHKERRQ(ierr);
  ierr = MatSetUp(Im);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Im,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Im,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Allocate H */
  ierr = MatCreate(PETSC_COMM_SELF,&H);CHKERRQ(ierr);
  ierr = MatSetSizes(H, PETSC_DECIDE, PETSC_DECIDE,2*dim,2*dim);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(H, "system");
  ierr = MatSetFromOptions(H);CHKERRQ(ierr);
  ierr = MatSetUp(H);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);


  return 0;
}


int Hamiltonian::apply(double t){
  int ierr;
  int ncol;
  const PetscInt *col_idx;
  const PetscScalar *vals;
  PetscScalar *negvals;
  PetscInt *col_idx_shift;

  /* Allocate tmp vectors */
  ierr = PetscMalloc1(dim, &col_idx_shift);CHKERRQ(ierr);
  ierr = PetscMalloc1(dim, &negvals);CHKERRQ(ierr);

  /* Set up Jacobian M 
   * H(0, 0) =  Re    H(0,1) = Im
   * H(0, 1) = -Im    H(1,1) = Re
   */
  for (int irow = 0; irow < dim; irow++) {
    PetscInt irow_shift = irow + dim;

    /* Get row in Re */
    ierr = MatGetRow(Re, irow, &ncol, &col_idx, &vals);CHKERRQ(ierr);
    for (int icol = 0; icol < ncol; icol++)
    {
      col_idx_shift[icol] = col_idx[icol] + dim;
    }
    // Set A in H: H(0,0) = Re  H(1,1) = Re
    ierr = MatSetValues(H,1,&irow,ncol,col_idx,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(H,1,&irow_shift,ncol,col_idx_shift,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(Re,irow,&ncol,&col_idx,&vals);CHKERRQ(ierr);

    /* Get row in Im */
    ierr = MatGetRow(Im, irow, &ncol, &col_idx, &vals);CHKERRQ(ierr);
    for (int icol = 0; icol < ncol; icol++)
    {
      col_idx_shift[icol] = col_idx[icol] + dim;
      negvals[icol] = -vals[icol];
    }
    // Set Im in H: H(1,0) = Im, H(0,1) = -Im
    ierr = MatSetValues(H,1,&irow,ncol,col_idx_shift,negvals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(H,1,&irow_shift,ncol,col_idx,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(Im,irow,&ncol,&col_idx,&vals);CHKERRQ(ierr);
  }

  /* Assemble M */
  ierr = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  // MatView(M, PETSC_VIEWER_STDOUT_SELF);

  /* Cleanup */
  ierr = PetscFree(col_idx_shift);
  ierr = PetscFree(negvals);

  return 0;
}

Mat Hamiltonian::getH(){
  return H;
}

TwoOscilHam::TwoOscilHam(int nlevels_, Oscillator** oscil_vec_){
  int ierr;

  /* Initialize Hamiltonian */
  Hamiltonian::initialize(nlevels_, 2, oscil_vec_);

  /* Initialize building blocks */
  this->initialize();
}

TwoOscilHam::~TwoOscilHam() {

  MatDestroy(&A1);
  MatDestroy(&A2);
  MatDestroy(&B1);
  MatDestroy(&B2);
  MatDestroy(&Hd);

}

int TwoOscilHam::initialize(){
  int ierr;
  Mat tmp;

  /* Create building blocks */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,1,NULL,&A1);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,1,NULL,&A2);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,1,NULL,&B1);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,1,NULL,&B2);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,1,NULL,&Hd);CHKERRQ(ierr);

  /* --- Set up A1 = C^{-}(n^2, n) - C^-(1,n^3)^T --- */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,1,NULL,&tmp);CHKERRQ(ierr);
  ierr = BuildingBlock(tmp, -1, 1, (int) pow(nlevels, 3));
  ierr = MatTranspose(tmp, MAT_INPLACE_MATRIX, &tmp);
  ierr = BuildingBlock(A1, -1, (int)pow(nlevels, 2), nlevels);
  ierr = MatAXPY(A1, -1., tmp, DIFFERENT_NONZERO_PATTERN);
  ierr = MatDestroy(&tmp);

  /* --- Set up A2 = C^{-}(n^3,1) - C^{-}(n,n^2)^T --- */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,1,NULL,&tmp);CHKERRQ(ierr);
  ierr = BuildingBlock(tmp, -1,nlevels, (int) pow(nlevels, 2));
  ierr = MatTranspose(tmp, MAT_INPLACE_MATRIX, &tmp);
  ierr = BuildingBlock(A2, -1, (int)pow(nlevels, 3), 1);
  ierr = MatAXPY(A2, -1., tmp, DIFFERENT_NONZERO_PATTERN);
  ierr = MatDestroy(&tmp);
  
  /* --- Set up B1 = C^+(1,n^3)^T - C^+(n^2, n) --- */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,1,NULL,&tmp);CHKERRQ(ierr);
  ierr = BuildingBlock(tmp, 1, (int) pow(nlevels, 2), nlevels);
  ierr = BuildingBlock(B1, 1, 1, (int)pow(nlevels, 3));
  ierr = MatTranspose(B1, MAT_INPLACE_MATRIX, &B1);
  ierr = MatAXPY(B1, -1., tmp, DIFFERENT_NONZERO_PATTERN);
  ierr = MatDestroy(&tmp);

  /* --- Set up B2 = C^+(n,n^2)^T - C^+(n^3, 1) --- */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,1,NULL,&tmp);CHKERRQ(ierr);
  ierr = BuildingBlock(tmp, 1, nlevels, (int) pow(nlevels, 2));
  ierr = BuildingBlock(B2, 1, nlevels, (int)pow(nlevels, 2));
  ierr = MatTranspose(B2, MAT_INPLACE_MATRIX, &B2);
  ierr = MatAXPY(B2, -1., tmp, DIFFERENT_NONZERO_PATTERN);
  ierr = MatDestroy(&tmp);

  /* --- Set up Hd --- */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,1,NULL,&Hd);CHKERRQ(ierr);
  ierr = MatZeroEntries(Hd);
  // TODO: Set Hd!

  /* Assemble the matrices */
  ierr = MatAssemblyBegin(A1, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A1, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A2, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A2, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B1, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B1, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B2, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B2, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Hd, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Hd, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  return ierr;
}


int TwoOscilHam::BuildingBlock(Mat C, int sign, int k, int m){
  int i, j;
  int ierr;
  double val;

  // iterate over the blocks
  for(int x = 0; x < k; x++) {
    // iterate over the unique entrys within each block
    for(int y = 0; y < nlevels-1; y++) {
      // iterate over the repetitions of each entry
      for(int z = 0; z < m; z++) {
        // Position of upper element
        i = x*nlevels*m + y*m + z;
        j = i + m;
        // Value of upper elements
        val = sqrt((double)(y+1));
        ierr = MatSetValues(C,1,&i,1,&j,&val,INSERT_VALUES);CHKERRQ(ierr);
        // Lower element
        val = sign * val;
        ierr = MatSetValues(C,1,&j,1,&i,&val,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  return 0;
}


int TwoOscilHam::apply(double t){
  int ierr;
  Vector control_Re(2);
  Vector control_Im(2);

  /* Compute time-dependent control functions */
  oscil_vec[0]->getControl(t, &control_Re(0), &control_Im(0));
  oscil_vec[1]->getControl(t, &control_Re(1), &control_Im(1));

  /* Sum up real part of hamiltonian operator Re = Im1*A1 + Im2*A2 */ 
  ierr = MatZeroEntries(Re);CHKERRQ(ierr);
  // ierr = MatAXPY(Re,control_Im(0),A1,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(Re,control_Im(1),A2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  /* Sum up imaginary part of system matrix B = f1*B1 + f2*B2 + H_const  */
  ierr = MatZeroEntries(Im);CHKERRQ(ierr);
  ierr = MatAXPY(Im,control_Re(0),B1,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
  // ierr = MatAXPY(B,control_Re(1),B2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  Hamiltonian::apply(t);

  return 0;
}
