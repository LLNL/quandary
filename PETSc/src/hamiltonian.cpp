#include "hamiltonian.hpp"


Hamiltonian::Hamiltonian(){
  dim = 0;
  nlevels = 0;
  noscillators = 0;
  oscil_vec = NULL;
  Re = NULL;
  Im = NULL;
}


Hamiltonian::Hamiltonian(int nlevels_, int noscillators_, Oscillator** oscil_vec_){
  int ierr;

  /* Set dimensions */
  dim = (int) pow(nlevels_, noscillators_*2); // n^osc : pure states, (n^osc)^2 : Density matrix
  nlevels = nlevels_;
  noscillators = noscillators_;
  /* Set oscillator vector */
  oscil_vec = oscil_vec_;

  /* Allocate Re */
  MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,0,NULL,&Re);
  MatSetFromOptions(Re);
  MatSetUp(Re);
  MatSetOption(Re, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
  for (int irow = 0; irow < dim; irow++)
  {
    MatSetValue(Re, irow, irow, 0.0, INSERT_VALUES);
  }
  MatAssemblyBegin(Re,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Re,MAT_FINAL_ASSEMBLY);


  /* Allocate Im */
  MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,0,NULL,&Im);
  MatSetFromOptions(Im);
  MatSetUp(Im);
  MatAssemblyBegin(Im,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Im,MAT_FINAL_ASSEMBLY);

  /* Allocate H, dimension: 2*dim x 2*dim for the real-valued system */
  MatCreate(PETSC_COMM_SELF,&M);
  MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE,2*dim,2*dim);
  MatSetOptionsPrefix(M, "system");
  MatSetFromOptions(M);
  MatSetUp(M);
  MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);


}


Hamiltonian::~Hamiltonian(){
  if (dim > 0){
    MatDestroy(&Re);
    MatDestroy(&Im);
    MatDestroy(&M);
  }
}


int Hamiltonian::getDim(){ return dim; }

bool Hamiltonian::ExactSolution(double t, Vec x) { return false; }

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
   * M(0, 0) =  Re    M(0,1) = Im
   * M(0, 1) = -Im    M(1,1) = Re
   */
  for (int irow = 0; irow < dim; irow++) {
    PetscInt irow_shift = irow + dim;

    /* Get row in Re */
    ierr = MatGetRow(Re, irow, &ncol, &col_idx, &vals);CHKERRQ(ierr);
    for (int icol = 0; icol < ncol; icol++)
    {
      col_idx_shift[icol] = col_idx[icol] + dim;
    }
    // Set A in M: M(0,0) = Re  M(1,1) = Re
    ierr = MatSetValues(M,1,&irow,ncol,col_idx,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(M,1,&irow_shift,ncol,col_idx_shift,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(Re,irow,&ncol,&col_idx,&vals);CHKERRQ(ierr);

    /* Get row in Im */
    ierr = MatGetRow(Im, irow, &ncol, &col_idx, &vals);CHKERRQ(ierr);
    for (int icol = 0; icol < ncol; icol++)
    {
      col_idx_shift[icol] = col_idx[icol] + dim;
      negvals[icol] = -vals[icol];
    }
    // Set Im in M: M(1,0) = Im, M(0,1) = -Im
    ierr = MatSetValues(M,1,&irow,ncol,col_idx_shift,negvals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(M,1,&irow_shift,ncol,col_idx,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(Im,irow,&ncol,&col_idx,&vals);CHKERRQ(ierr);
  }

  /* Assemble M */
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  // MatView(M, PETSC_VIEWER_STDOUT_SELF);

  /* Cleanup */
  ierr = PetscFree(col_idx_shift);
  ierr = PetscFree(negvals);

  return 0;
}

Mat Hamiltonian::getM(){
  return M;
}

TwoOscilHam::TwoOscilHam(){
  A1 = NULL;
  A2 = NULL;
  B1 = NULL;
  B2 = NULL;
  Hd = NULL;
  xi = NULL;
}

TwoOscilHam::TwoOscilHam(int nlevels_, double* xi_, Oscillator** oscil_vec_)
                :  Hamiltonian(nlevels_, 2, oscil_vec_){
  xi = xi_;

  /* Create constant matrices */
  MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,1,NULL,&A1);
  MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,1,NULL,&A2);
  MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,1,NULL,&B1);
  MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,1,NULL,&B2);

  Mat tmp;
  /* --- Set up A1 = C^{-}(n^2, n) - C^-(1,n^3)^T --- */
  MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,1,NULL,&tmp);
  BuildingBlock(tmp, -1, 1, (int) pow(nlevels, 3));
  MatTranspose(tmp, MAT_INPLACE_MATRIX, &tmp);
  BuildingBlock(A1, -1, (int)pow(nlevels, 2), nlevels);
  MatAXPY(A1, -1., tmp, DIFFERENT_NONZERO_PATTERN);
  MatDestroy(&tmp);

  /* --- Set up A2 = C^{-}(n^3,1) - C^{-}(n,n^2)^T --- */
  MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,1,NULL,&tmp);
  BuildingBlock(tmp, -1,nlevels, (int) pow(nlevels, 2));
  MatTranspose(tmp, MAT_INPLACE_MATRIX, &tmp);
  BuildingBlock(A2, -1, (int)pow(nlevels, 3), 1);
  MatAXPY(A2, -1., tmp, DIFFERENT_NONZERO_PATTERN);
  MatDestroy(&tmp);
  
  /* --- Set up B1 = C^+(1,n^3)^T - C^+(n^2, n) --- */
  MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,1,NULL,&tmp);
  BuildingBlock(tmp, 1, (int) pow(nlevels, 2), nlevels);
  BuildingBlock(B1, 1, 1, (int)pow(nlevels, 3));
  MatTranspose(B1, MAT_INPLACE_MATRIX, &B1);
  MatAXPY(B1, -1., tmp, DIFFERENT_NONZERO_PATTERN);
  MatDestroy(&tmp);

  /* --- Set up B2 = C^+(n,n^2)^T - C^+(n^3, 1) --- */
  MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,1,NULL,&tmp);
  BuildingBlock(tmp, 1, nlevels, (int) pow(nlevels, 2));
  BuildingBlock(B2, 1, nlevels, (int)pow(nlevels, 2));
  MatTranspose(B2, MAT_INPLACE_MATRIX, &B2);
  MatAXPY(B2, -1., tmp, DIFFERENT_NONZERO_PATTERN);
  MatDestroy(&tmp);

  /* --- Set up Hd --- */
  // tmp = Hs, Hd = -In\kron Hs + Hs^T\kron In
  Vec diag;
  VecCreate(PETSC_COMM_WORLD, &diag);
  VecSetSizes(diag, PETSC_DECIDE, nlevels*nlevels);
  VecSetFromOptions(diag);
  for (int i=1; i<nlevels; i++){  // first block is always empty
    for (int j=0; j<nlevels; j++){
      int rowid = i * nlevels + j;
      double val = 0.0;
      val += - xi[0] / 2. * i*(i-1);
      val += - xi[1] / 2. * j*(j-1);
      val += - xi[2] * i*j;
      VecSetValue(diag, rowid, val, INSERT_VALUES);
    }
  }
  VecAssemblyBegin(diag);
  VecAssemblyEnd(diag);
  // VecView(diag, PETSC_VIEWER_STDOUT_WORLD);

  /* Create diagonal Hs */
  MatCreateSeqAIJ(PETSC_COMM_SELF, nlevels*nlevels, nlevels*nlevels, nlevels*nlevels, NULL, &tmp);
  MatSetUp(tmp);
  MatDiagonalSet(tmp, diag, INSERT_VALUES);
  MatAssemblyBegin(tmp, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(tmp, MAT_FINAL_ASSEMBLY);
  // MatView(tmp, PETSC_VIEWER_STDOUT_WORLD);

  MatCreateSeqAIJ(PETSC_COMM_SELF,dim,dim,dim,NULL,&Hd); // only diagonal is nonzero
  MatSetUp(Hd);
  kronI(tmp, nlevels*nlevels, &Hd, INSERT_VALUES);  // Hd = tmp \kron I
  MatScale(tmp, -1.0); 
  Ikron(tmp, nlevels*nlevels, &Hd, ADD_VALUES);  // Hd += -I \kron tmp
  MatDestroy(&tmp);

  MatAssemblyBegin(Hd, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(Hd, MAT_FINAL_ASSEMBLY);
  // MatView(Hd, PETSC_VIEWER_STDOUT_WORLD);

  /* Assemble the matrices */
  MatAssemblyBegin(A1, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A1, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(A2, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A2, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(B1, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(B1, MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin(B2, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(B2, MAT_FINAL_ASSEMBLY);

}



TwoOscilHam::~TwoOscilHam() {

  MatDestroy(&A1);
  MatDestroy(&A2);
  MatDestroy(&B1);
  MatDestroy(&B2);
  MatDestroy(&Hd);
  MatDestroy(&M);

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

  /* Sum up real part of hamiltonian operator Re = controlIm1*A1 + controlIm2*A2 */ 
  ierr = MatZeroEntries(Re);CHKERRQ(ierr);
  ierr = MatAXPY(Re,control_Im(0),A1,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(Re,control_Im(1),A2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  /* Sum up imaginary part of system matrix B = f1*B1 + f2*B2 + H_const  */
  ierr = MatZeroEntries(Im);CHKERRQ(ierr);
  ierr = MatAXPY(Im,control_Re(0),B1,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
  ierr = MatAXPY(Im,control_Re(1),B2,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(Im, 1.0, Hd, DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);

  /* Set M from Re and Im */
  Hamiltonian::apply(t);

  return 0;
}


int TwoOscilHam::initialCondition(Vec x) {
  VecZeroEntries(x); // set zero for now. TODO: Set initial condition

  return 0;
}

AnalyticHam::AnalyticHam(double* xi_, Oscillator** oscil_vec_) : TwoOscilHam(2, xi_, oscil_vec_) {}


PetscScalar F1_analytic(PetscReal t, PetscReal freq)
{
  PetscScalar f = (1./4.) * (1. - PetscCosScalar(freq * t));
  return f;
}


PetscScalar G2_analytic(PetscReal t,PetscReal freq)
{
  PetscScalar g = (1./4.) * (1. - PetscSinScalar(freq * t));
  return g;
}



bool AnalyticHam::ExactSolution(PetscReal t,Vec s)
{
  double f1, f2, g1, g2;
  oscil_vec[0]->getParams(&f1, &g1);
  oscil_vec[1]->getParams(&f2, &g2);

  if (fabs(f1) < 1e-12 || fabs(g2) < 1e-12){
    printf("\n ERROR: Can't use 0.0 for FunctionOscillatorFrequency!\n");
    exit(1);
  }

  PetscScalar    *s_localptr;

  /* Get a pointer to vector data. */
  VecGetArray(s,&s_localptr);;

  /* Write the solution into the array locations.
   *  Alternatively, we could use VecSetValues() or VecSetValuesLocal(). */
  PetscScalar phi = (1./4.) * (t - (1./ f1)*PetscSinScalar(f1*t));
  PetscScalar theta = (1./4.) * (t + (1./ g2)*PetscCosScalar(g2*t) - 1.);
  PetscScalar cosphi = PetscCosScalar(phi);
  PetscScalar costheta = PetscCosScalar(theta);
  PetscScalar sinphi = PetscSinScalar(phi);
  PetscScalar sintheta = PetscSinScalar(theta);
  

  // /* Real part */
  s_localptr[0] = cosphi*costheta*cosphi*costheta;
  s_localptr[1] = -1.*cosphi*sintheta*cosphi*costheta;
  s_localptr[2] = 0.;
  s_localptr[3] = 0.;
  s_localptr[4] = -1.*cosphi*costheta*cosphi*sintheta;
  s_localptr[5] = cosphi*sintheta*cosphi*sintheta;
  s_localptr[6] = 0.;
  s_localptr[7] = 0.;
  s_localptr[8] = 0.;
  s_localptr[9] = 0.;
  s_localptr[10] = sinphi*costheta*sinphi*costheta;
  s_localptr[11] = -1.*sinphi*sintheta*sinphi*costheta;
  s_localptr[12] = 0.;
  s_localptr[13] = 0.;
  s_localptr[14] = -1.*sinphi*costheta*sinphi*sintheta;
  s_localptr[15] = sinphi*sintheta*sinphi*sintheta;
  /* Imaginary part */
  s_localptr[16] = 0.;
  s_localptr[17] = 0.;
  s_localptr[18] = - sinphi*costheta*cosphi*costheta;
  s_localptr[19] = sinphi*sintheta*cosphi*costheta;
  s_localptr[20] = 0.;
  s_localptr[21] = 0.;
  s_localptr[22] = sinphi*costheta*cosphi*sintheta;
  s_localptr[23] = - sinphi*sintheta*cosphi*sintheta;
  s_localptr[24] = cosphi*costheta*sinphi*costheta;
  s_localptr[25] = - cosphi*sintheta*sinphi*costheta;
  s_localptr[26] = 0.;
  s_localptr[27] = 0.;
  s_localptr[28] = - cosphi*costheta*sinphi*sintheta;
  s_localptr[29] = cosphi*sintheta*sinphi*sintheta;
  s_localptr[30] = 0.;
  s_localptr[31] = 0.;

  // /* Restore solution vector */
  VecRestoreArray(s,&s_localptr);

  return true;
}


PetscErrorCode AnalyticHam::initialCondition(Vec x)
{
  ExactSolution(0,x);
  return 0;
}


