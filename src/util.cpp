#include "util.hpp"

int getIndexReal(const int i) {
  return 2*i;
}

int getIndexImag(const int i) {
  return 2*i + 1;
}

int getVecID(const int row, const int col, const int dim){
  return row + col * dim;  
} 


int mapEssToFull(const int i, const std::vector<int> &nlevels, const std::vector<int> &nessential){
  if (nlevels.size() != 2) {
    printf("\n ERROR: guard levels currently only for two oscillators.\n");
    printf("\n TODO: To make this work, generalize mapEssToFull(i).\n");
    assert(nlevels.size() == 2); // TODO: Generalize this formula for more than two oscillators 
  }

  return ( (int) i/nessential[1] ) * nlevels[1] + i % nessential[1];
}


int isEssential(const int i, const std::vector<int> &nlevels, const std::vector<int> &nessential) {
  if (nlevels.size() != 2) {
  printf("\n ERROR: guard levels currently only for two oscillators.\n");
    printf("\n TODO: To make this work, generalize isEssential(i).\n");
    assert(nlevels.size() == 2); // TODO: Generalize this formula for more than two oscillators 
  }

  int isEss;

  /* i corresponds to an essential in rho = rho^A \kron \rho^B iff (i/n^B <= n^A_e) && (i mod n^b <= n^B_e) */
  int a = i / nlevels[1];  
  int b = i % nlevels[1];
  if ( a < nessential[0] && b < nessential[1]) isEss = 1;
  else isEss = 0;

  return isEss; 
}


PetscErrorCode Ikron(const Mat A,const  int dimI, const double alpha, Mat *Out, InsertMode insert_mode){

    int ierr;
    int ncols;
    const PetscInt* cols; 
    const PetscScalar* Avals;
    PetscInt* shiftcols;
    PetscScalar* vals;
    int dimA;
    int dimOut;
    int nonzeroOut;
    int rowID;

    MatGetSize(A, &dimA, NULL);

    ierr = PetscMalloc1(dimA, &shiftcols); CHKERRQ(ierr);
    ierr = PetscMalloc1(dimA, &vals); CHKERRQ(ierr);

    /* Loop over dimension of I */
    for (int i = 0; i < dimI; i++){

        /* Set the diagonal block (i*dimA)::(i+1)*dimA */
        for (int j=0; j<dimA; j++){
            MatGetRow(A, j, &ncols, &cols, &Avals);
            rowID = i*dimA + j;
            for (int k=0; k<ncols; k++){
                shiftcols[k] = cols[k] + i*dimA;
                vals[k] = Avals[k] * alpha;
            }
            MatSetValues(*Out, 1, &rowID, ncols, shiftcols, vals, insert_mode);
            MatRestoreRow(A, j, &ncols, &cols, &Avals);
        }

    }
    // MatAssemblyBegin(*Out, MAT_FINAL_ASSEMBLY);
    // MatAssemblyEnd(*Out, MAT_FINAL_ASSEMBLY);

    PetscFree(shiftcols);
    PetscFree(vals);
    return 0;
}

PetscErrorCode kronI(const Mat A, const int dimI, const double alpha, Mat *Out, InsertMode insert_mode){
    
    int ierr;
    int dimA;
    const PetscInt* cols; 
    const PetscScalar* Avals;
    int rowid;
    int colid;
    double insertval;
    int dimOut;
    int nonzeroOut;
    int ncols;
    MatInfo Ainfo;
    MatGetSize(A, &dimA, NULL);

    ierr = PetscMalloc1(dimA, &cols); CHKERRQ(ierr);
    ierr = PetscMalloc1(dimA, &Avals);

    /* Loop over rows in A */
    for (int i = 0; i < dimA; i++){
        MatGetRow(A, i, &ncols, &cols, &Avals);

        /* Loop over non negative columns in row i */
        for (int j = 0; j < ncols; j++){
            //printf("A: row = %d, col = %d, val = %f\n", i, cols[j], Avals[j]);
            
            // dimI rows. global row indices: i, i+dimI
            for (int k=0; k<dimI; k++) {
               rowid = i*dimI + k;
               colid = cols[j]*dimI + k;
               insertval = Avals[j] * alpha;
               MatSetValues(*Out, 1, &rowid, 1, &colid, &insertval, insert_mode);
              //  printf("Setting %d,%d %f\n", rowid, colid, insertval);
            }
        }
        MatRestoreRow(A, i, &ncols, &cols, &Avals);
    }

    // MatAssemblyBegin(*Out, MAT_FINAL_ASSEMBLY);
    // MatAssemblyEnd(*Out, MAT_FINAL_ASSEMBLY);

    PetscFree(cols);
    PetscFree(Avals);

    return 0;
}



PetscErrorCode AkronB(const Mat A, const Mat B, const double alpha, Mat *Out, InsertMode insert_mode){
    int Adim1, Adim2, Bdim1, Bdim2;
    MatGetSize(A, &Adim1, &Adim2);
    MatGetSize(B, &Bdim1, &Bdim2);

    int ncolsA, ncolsB;
    const int *colsA, *colsB;
    const double *valsA, *valsB;
    // Iterate over rows of A 
    for (int irowA = 0; irowA < Adim1; irowA++){
        // Iterate over non-zero columns in this row of A
        MatGetRow(A, irowA, &ncolsA, &colsA, &valsA);
        for (int j=0; j<ncolsA; j++) {
            int icolA = colsA[j];
            double valA = valsA[j];
            /* put a B-block at position (irowA*Bdim1, icolA*Bdim2): */
            // Iterate over rows of B 
            for (int irowB = 0; irowB < Bdim1; irowB++){
                // Iterate over non-zero columns in this B-row
                MatGetRow(B, irowB, &ncolsB, &colsB, &valsB);
                for (int k=0; k< ncolsB; k++) {
                    int icolB = colsB[k];
                    double valB = valsB[k];
                    /* Insert values in Out */
                    int rowOut = irowA*Bdim1 + irowB;
                    int colOut = icolA*Bdim2 + icolB;
                    double valOut = valA * valB * alpha; 
                    MatSetValue(*Out, rowOut, colOut, valOut, insert_mode);
                }
                MatRestoreRow(B, irowB, &ncolsB, &colsB, &valsB);
            }
        }   
        MatRestoreRow(A, irowA, &ncolsA, &colsA, &valsA);
    }  

  return 0;
}


PetscErrorCode MatIsAntiSymmetric(Mat A, PetscReal tol, PetscBool *flag) {
  
  int ierr; 

  /* Create B = -A */
  Mat B;
  ierr = MatConvert(A, MATSAME, MAT_INITIAL_MATRIX, &B); CHKERRQ(ierr);
  ierr = MatScale(B, -1.0); CHKERRQ(ierr);

  /* Test if B^T = A */
  ierr = MatIsTranspose(B, A, tol, flag); CHKERRQ(ierr);

  /* Cleanup */
  ierr = MatDestroy(&B); CHKERRQ(ierr);

  return ierr;
}



PetscErrorCode StateIsHermitian(Vec x, PetscReal tol, PetscBool *flag) {
  int ierr;
  int i, j;

  /* TODO: MAKE THIS WORK IN PARALLEL */
  
  /* Get u and v from x */
  int dim;
  ierr = VecGetSize(x, &dim); CHKERRQ(ierr);
  dim = dim/2;
  Vec u, v;
  IS isu, isv;

  int dimis = dim;
  ierr = ISCreateStride(PETSC_COMM_WORLD, dimis, 0, 2, &isu); CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD, dimis, 1, 2, &isv); CHKERRQ(ierr);
  ierr = VecGetSubVector(x, isu, &u); CHKERRQ(ierr);
  ierr = VecGetSubVector(x, isv, &v); CHKERRQ(ierr);

  /* Init flags*/
  *flag = PETSC_TRUE;

  /* Check for symmetric u and antisymmetric v */
  const double *u_array;
  const double *v_array;
  double u_diff, v_diff;
  ierr = VecGetArrayRead(u, &u_array); CHKERRQ(ierr);
  ierr = VecGetArrayRead(v, &v_array); CHKERRQ(ierr);
  int N = sqrt(dim);
  for (i=0; i<N; i++) {
    for (j=i; j<N; j++) {
      u_diff = u_array[i*N+j] - u_array[j*N+i];
      v_diff = v_array[i*N+j] + v_array[j*N+i];
      if (fabs(u_diff) > tol || fabs(v_diff) > tol ) {
        *flag = PETSC_FALSE;
        break;
      }
    }
  }

  ierr = VecRestoreArrayRead(u, &u_array);
  ierr = VecRestoreArrayRead(v, &v_array);
  ierr = VecRestoreSubVector(x, isu, &u);
  ierr = VecRestoreSubVector(x, isv, &v);
  ISDestroy(&isu);
  ISDestroy(&isv);

  return ierr;
}



PetscErrorCode StateHasTrace1(Vec x, PetscReal tol, PetscBool *flag) {

  int ierr;
  int i;

  /* Get u and v from x */
  int dim;
  ierr = VecGetSize(x, &dim); CHKERRQ(ierr);
  int dimis = dim/2;
  Vec u, v;
  IS isu, isv;
  ierr = ISCreateStride(PETSC_COMM_WORLD, dimis, 0, 2, &isu); CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD, dimis, 1, 2, &isv); CHKERRQ(ierr);
  ierr = VecGetSubVector(x, isu, &u); CHKERRQ(ierr);
  ierr = VecGetSubVector(x, isv, &v); CHKERRQ(ierr);

  /* Init flags*/
  *flag = PETSC_FALSE;
  PetscBool u_hastrace1 = PETSC_FALSE;
  PetscBool v_hasdiag0  = PETSC_FALSE;

  /* Check if diagonal of u sums to 1, and diagonal elements of v are 0 */ 
  const double *u_array;
  const double *v_array;
  double u_sum = 0.0;
  double v_sum = 0.0;
  ierr = VecGetArrayRead(u, &u_array); CHKERRQ(ierr);
  ierr = VecGetArrayRead(v, &v_array); CHKERRQ(ierr);
  int N = sqrt(dimis);
  for (i=0; i<N; i++) {
    u_sum += u_array[i*N+i];
    v_sum += fabs(v_array[i*N+i]);
  }
  if ( fabs(u_sum - 1.0) < tol ) u_hastrace1 = PETSC_TRUE;
  if ( fabs(v_sum      ) < tol ) v_hasdiag0  = PETSC_TRUE;

  /* Restore vecs */
  ierr = VecRestoreArrayRead(u, &u_array);
  ierr = VecRestoreArrayRead(v, &v_array);
  ierr = VecRestoreSubVector(x, isu, &u);
  ierr = VecRestoreSubVector(x, isv, &v);

  /* Answer*/
  if (u_hastrace1 && v_hasdiag0) {
    *flag = PETSC_TRUE;
  }
  
  /* Destroy vector strides */
  ISDestroy(&isu);
  ISDestroy(&isv);


  return ierr;
}



PetscErrorCode SanityTests(Vec x, double time){

  /* Sanity check. Be careful: This is costly! */
  printf("Trace check %f ...\n", time);
  PetscBool check;
  double tol = 1e-10;
  StateIsHermitian(x, tol, &check);
  if (!check) {
    printf("WARNING at t=%f: rho is not hermitian!\n", time);
    printf("\n rho :\n");
    VecView(x, PETSC_VIEWER_STDOUT_WORLD);
    exit(1);
  }
  else printf("IsHermitian check passed.\n");
  StateHasTrace1(x, tol, &check);
  if (!check) {
    printf("WARNING at t=%f: Tr(rho) is NOT one!\n", time);
    printf("\n rho :\n");
    VecView(x, PETSC_VIEWER_STDOUT_WORLD);
    exit(1);
  }
  else printf("Trace1 check passed.\n");

  return 0;
}


void read_vector(const char *filename, double *var, int dim) {

  FILE *file;
  double tmp;

  /* Open file */
  file = fopen(filename, "r");
  if (file != NULL) {
    /* Read data */
    printf("Reading file %s\n", filename);
    for (int ix = 0; ix < dim; ix++) {
      fscanf(file, "%lf", &tmp);
      var[ix] = tmp;
    }
  } else {
    printf("ERROR: Can't open initialization file %s\n", filename);
    exit(1);
  }

  fclose(file);
}


/* Compute eigenvalues */
int getEigvals(const Mat A, const int neigvals, std::vector<double>& eigvals, std::vector<Vec>& eigvecs){

int nconv = 0;
#ifdef WITH_SLEPC

  /* Create Slepc's eigensolver */
  EPS eigensolver;       
  EPSCreate(PETSC_COMM_WORLD, &eigensolver);
  EPSSetOperators(eigensolver, A, NULL);
  EPSSetProblemType(eigensolver, EPS_NHEP);
  EPSSetFromOptions(eigensolver);

  /* Number of requested eigenvalues */
  EPSSetDimensions(eigensolver,neigvals,PETSC_DEFAULT,PETSC_DEFAULT);

  // Solve eigenvalue problem
  int ierr = EPSSolve(eigensolver); CHKERRQ(ierr);

  /* Get information about convergence */
  int its, nev, maxit;
  EPSType type;
  double tol;
  EPSGetIterationNumber(eigensolver,&its);
  EPSGetType(eigensolver,&type);
  EPSGetDimensions(eigensolver,&nev,NULL,NULL);
  EPSGetTolerances(eigensolver,&tol,&maxit);
  EPSGetConverged(eigensolver, &nconv );

  PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n",type);
  PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenpairs: %D\n",nev);
  PetscPrintf(PETSC_COMM_WORLD," Number of iterations taken: %D / %D\n",its, maxit);
  PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g\n",(double)tol);
  PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %D\n\n",nconv);

  /* Allocate eigenvectors */
  Vec eigvec;
  MatCreateVecs(A, &eigvec, NULL);

  // Get the result
  double kr, ki, error;
  // printf("Eigenvalues: \n");
  for (int j=0; j<nconv; j++) {
      EPSGetEigenpair( eigensolver, j, &kr, &ki, eigvec, NULL);
      EPSComputeError( eigensolver, j, EPS_ERROR_RELATIVE, &error );
      // printf("%f + i%f (err %f)\n", kr, ki, error);

      /* Store the eigenpair */
      eigvals.push_back(kr);
      eigvecs.push_back(eigvec);
      if (ki != 0.0) printf("Warning: eigenvalue imaginary! : %f", ki);
  }
  // printf("\n");
  // EPSView(eigensolver, PETSC_VIEWER_STDOUT_WORLD);

  /* Clean up*/
  EPSDestroy(&eigensolver);
#endif
  return nconv;
}
