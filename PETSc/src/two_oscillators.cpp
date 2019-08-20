#include "two_oscillators.hpp"

PetscErrorCode ExactSolution(PetscReal t,Vec s, PetscReal freq)
{
  PetscScalar    *s_localptr;
  PetscErrorCode ierr;

  /* Get a pointer to vector data. */
  ierr = VecGetArray(s,&s_localptr);CHKERRQ(ierr);

  /* Write the solution into the array locations.
   *  Alternatively, we could use VecSetValues() or VecSetValuesLocal(). */
  PetscScalar phi = (1./4.) * (t - (1./ freq)*PetscSinScalar(freq*t));
  PetscScalar theta = (1./4.) * (t + (1./freq)*PetscCosScalar(freq*t) - 1.);
  PetscScalar cosphi = PetscCosScalar(phi);
  PetscScalar costheta = PetscCosScalar(theta);
  PetscScalar sinphi = PetscSinScalar(phi);
  PetscScalar sintheta = PetscSinScalar(theta);
  

  /* Real part */
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

  /* Restore solution vector */
  ierr = VecRestoreArray(s,&s_localptr);CHKERRQ(ierr);
  return 0;
}


PetscErrorCode InitialConditions(Vec x,TS_App *petsc_app)
{
  ExactSolution(0,x,petsc_app->w);
  return 0;
}


PetscScalar F(PetscReal t, PetscReal freq)
{
  PetscScalar f = (1./4.) * (1. - PetscCosScalar(freq * t));
  return f;
}


PetscScalar G(PetscReal t,PetscReal freq)
{
  PetscScalar g = (1./4.) * (1. - PetscSinScalar(freq * t));
  return g;
}


PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec u,Mat M,Mat P,void *ctx)
{
  TS_App *petsc_app = (TS_App*)ctx;
  PetscInt nvec = petsc_app->nvec;
  PetscScalar f, g;
  PetscErrorCode ierr;
  PetscInt ncol;
  const PetscInt *col_idx;
  const PetscScalar *vals;
  PetscScalar *negvals;
  PetscInt *col_idx_shift;

  /* Allocate tmp vectors */
  ierr = PetscMalloc1(nvec, &col_idx_shift);CHKERRQ(ierr);
  ierr = PetscMalloc1(nvec, &negvals);CHKERRQ(ierr);

  /* Compute time-dependent control functions */
  f = F(t, petsc_app->w);
  g = G(t, petsc_app->w);


  /* Set up real part of system matrix (A) */
  ierr = MatZeroEntries(petsc_app->A);CHKERRQ(ierr);
  ierr = MatAXPY(petsc_app->A,g,petsc_app->IKbMbd,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(petsc_app->A,-1.*g,petsc_app->bMbdTKI,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  /* Set up imaginary part of system matrix (B) */
  ierr = MatZeroEntries(petsc_app->B);CHKERRQ(ierr);
  ierr = MatAXPY(petsc_app->B,f,petsc_app->aPadTKI,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(petsc_app->B,-1.*f,petsc_app->IKaPad,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  // MatView(petsc_app->A, PETSC_VIEWER_STDOUT_SELF);
  // MatView(petsc_app->B, PETSC_VIEWER_STDOUT_SELF);

  /* Set up Jacobian M 
   * M(0, 0) =  A    M(0,1) = B
   * M(0, 1) = -B    M(1,1) = A
   */
  for (int irow = 0; irow < nvec; irow++) {
    PetscInt irow_shift = irow + nvec;

    /* Get row in A */
    ierr = MatGetRow(petsc_app->A, irow, &ncol, &col_idx, &vals);CHKERRQ(ierr);
    for (int icol = 0; icol < ncol; icol++)
    {
      col_idx_shift[icol] = col_idx[icol] + nvec;
    }
    // Set A in M: M(0,0) = A  M(1,1) = A
    ierr = MatSetValues(M,1,&irow,ncol,col_idx,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(M,1,&irow_shift,ncol,col_idx_shift,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(petsc_app->A,irow,&ncol,&col_idx,&vals);CHKERRQ(ierr);

    /* Get row in B */
    ierr = MatGetRow(petsc_app->B, irow, &ncol, &col_idx, &vals);CHKERRQ(ierr);
    for (int icol = 0; icol < ncol; icol++)
    {
      col_idx_shift[icol] = col_idx[icol] + nvec;
      negvals[icol] = -vals[icol];
    }
    // Set B in M: M(1,0) = B, M(0,1) = -B
    ierr = MatSetValues(M,1,&irow,ncol,col_idx_shift,negvals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(M,1,&irow_shift,ncol,col_idx,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(petsc_app->B,irow,&ncol,&col_idx,&vals);CHKERRQ(ierr);
  }

  /* Assemble M */
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  // MatView(M, PETSC_VIEWER_STDOUT_SELF);

  /* Cleanup */
  ierr = PetscFree(col_idx_shift);
  ierr = PetscFree(negvals);


  /* TODO: Do we really need to store A and B explicitely? They are only used here, so maybe we can assemble M from g,f,IKbMDb, bMbdTKI, aPadTKI, IKaPad directly... */


  return 0;
}




PetscErrorCode SetUpMatrices(TS_App *petsc_app)
{
  PetscInt       nvec = petsc_app->nvec;
  PetscInt       i, j;
  PetscScalar    v[1];
  PetscErrorCode ierr;

  /* Set up IKbMbd */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nvec,nvec,1,NULL,&petsc_app->IKbMbd);CHKERRQ(ierr);
  ierr = MatSetFromOptions(petsc_app->IKbMbd);CHKERRQ(ierr);
  ierr = MatSetUp(petsc_app->IKbMbd);CHKERRQ(ierr);

  i = 1;
  j = 0;
  v[0] = -1;
  ierr = MatSetValues(petsc_app->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 0;
  j = 1;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 3;
  j = 2;
  v[0] = -1;
  ierr = MatSetValues(petsc_app->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 2;
  j = 3;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 5;
  j = 4;
  v[0] = -1;
  ierr = MatSetValues(petsc_app->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 4;
  j = 5;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 7;
  j = 6;
  v[0] = -1;
  ierr = MatSetValues(petsc_app->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 6;
  j = 7;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 9;
  j = 8;
  v[0] = -1;
  ierr = MatSetValues(petsc_app->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 8;
  j = 9;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 11;
  j = 10;
  v[0] = -1;
  ierr = MatSetValues(petsc_app->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 10;
  j = 11;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 13;
  j = 12;
  v[0] = -1;
  ierr = MatSetValues(petsc_app->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 12;
  j = 13;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 15;
  j = 14;
  v[0] = -1;
  ierr = MatSetValues(petsc_app->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 14;
  j = 15;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(petsc_app->IKbMbd,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(petsc_app->IKbMbd,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

///////////////////////////////////////////////////////////////////////////////

  /* Set up bMbdTKI */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nvec,nvec,1,NULL,&petsc_app->bMbdTKI);CHKERRQ(ierr);
  ierr = MatSetFromOptions(petsc_app->bMbdTKI);CHKERRQ(ierr);
  ierr = MatSetUp(petsc_app->bMbdTKI);CHKERRQ(ierr);

  i = 4;
  j = 0;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 5;
  j = 1;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 6;
  j = 2;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 7;
  j = 3;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 0;
  j = 4;
  v[0] = -1;
  ierr = MatSetValues(petsc_app->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 1;
  j = 5;
  v[0] = -1;
  ierr = MatSetValues(petsc_app->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 2;
  j = 6;
  v[0] = -1;
  ierr = MatSetValues(petsc_app->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 3;
  j = 7;
  v[0] = -1;
  ierr = MatSetValues(petsc_app->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 12;
  j = 8;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 13;
  j = 9;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 14;
  j = 10;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 15;
  j = 11;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 8;
  j = 12;
  v[0] = -1;
  ierr = MatSetValues(petsc_app->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 9;
  j = 13;
  v[0] = -1;
  ierr = MatSetValues(petsc_app->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 10;
  j = 14;
  v[0] = -1;
  ierr = MatSetValues(petsc_app->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 11;
  j = 15;
  v[0] = -1;
  ierr = MatSetValues(petsc_app->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(petsc_app->bMbdTKI,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(petsc_app->bMbdTKI,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

///////////////////////////////////////////////////////////////////////////////

  /* Set up aPadTKI */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nvec,nvec,1,NULL,&petsc_app->aPadTKI);CHKERRQ(ierr);
  ierr = MatSetFromOptions(petsc_app->aPadTKI);CHKERRQ(ierr);
  ierr = MatSetUp(petsc_app->aPadTKI);CHKERRQ(ierr);

  i = 8;
  j = 0;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 9;
  j = 1;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 10;
  j = 2;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 11;
  j = 3;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 12;
  j = 4;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 13;
  j = 5;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 14;
  j = 6;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 15;
  j = 7;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 0;
  j = 8;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 1;
  j = 9;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 2;
  j = 10;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 3;
  j = 11;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 4;
  j = 12;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 5;
  j = 13;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 6;
  j = 14;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 7;
  j = 15;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(petsc_app->aPadTKI,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(petsc_app->aPadTKI,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

////////////////////////////////////////////////////////////////////////////////

  /* Set up IKaPad */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nvec,nvec,1,NULL,&petsc_app->IKaPad);CHKERRQ(ierr);
  ierr = MatSetFromOptions(petsc_app->IKaPad);CHKERRQ(ierr);
  ierr = MatSetUp(petsc_app->IKaPad);CHKERRQ(ierr);

  i = 2;
  j = 0;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 3;
  j = 1;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 0;
  j = 2;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 1;
  j = 3;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 6;
  j = 4;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 7;
  j = 5;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 4;
  j = 6;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 5;
  j = 7;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 10;
  j = 8;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 11;
  j = 9;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 8;
  j = 10;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 9;
  j = 11;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 14;
  j = 12;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 15;
  j = 13;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 12;
  j = 14;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 13;
  j = 15;
  v[0] = 1;
  ierr = MatSetValues(petsc_app->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(petsc_app->IKaPad,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(petsc_app->IKaPad,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

//////////////////////////////////////////////////////////////////////////////

  /* Allocate A */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nvec,nvec,0,NULL,&petsc_app->A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(petsc_app->A);CHKERRQ(ierr);
  ierr = MatSetUp(petsc_app->A);CHKERRQ(ierr);
  ierr = MatSetOption(petsc_app->A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);CHKERRQ(ierr);
  for (int irow = 0; irow < nvec; irow++)
  {
    ierr = MatSetValue(petsc_app->A, irow, irow, 0.0, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(petsc_app->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(petsc_app->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  // MatView(petsc_app->A, PETSC_VIEWER_STDOUT_SELF);

  /* Allocate B */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nvec,nvec,0,NULL,&petsc_app->B);CHKERRQ(ierr);
  ierr = MatSetFromOptions(petsc_app->B);CHKERRQ(ierr);
  ierr = MatSetUp(petsc_app->B);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(petsc_app->B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(petsc_app->B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  return 0;
}


