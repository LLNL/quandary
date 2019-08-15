#include <petscts.h>
/*
   Petsc's application context containing data needed to perform a time step.
*/
typedef struct {
  PetscInt    nvec;    /* Dimension of vectorized system */
  Mat         IKbMbd, bMbdTKI, aPadTKI, IKaPad, A, B;
  PetscReal   w;       /* Oscillator frequencies */
} TS_App;


/*
 *   Compute the exact solution at a given time.
 *   Input:
 *      t - current time
 *      s - vector in which exact solution will be computed
 *      TS_App - application context
 *   Output:
 *      s - vector with the newly computed exact solution
 */
PetscErrorCode ExactSolution(PetscReal t,Vec s,TS_App*petsc_app)
{
  PetscScalar    *s_localptr;
  PetscErrorCode ierr;

  /* Get a pointer to vector data. */
  ierr = VecGetArray(s,&s_localptr);CHKERRQ(ierr);

  /* Write the solution into the array locations.
   *  Alternatively, we could use VecSetValues() or VecSetValuesLocal(). */
  PetscScalar phi = (1./4.) * (t - (1./petsc_app->w)*PetscSinScalar(petsc_app->w*t));
  PetscScalar theta = (1./4.) * (t + (1./petsc_app->w)*PetscCosScalar(petsc_app->w*t) - 1.);
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

/*
 *  Set the initial condition at time t_0
 *  Input:
 *     u - uninitialized solution vector (global)
 *     TS_App - application context
 *  Output Parameter:
 *     u - vector with solution at initial time (global)
 */
PetscErrorCode InitialConditions(Vec x,TS_App *petsc_app)
{
  ExactSolution(0,x,petsc_app);
  return 0;
}
/*
 * Oscillator 1 (real part)
 */
PetscScalar F(PetscReal t,TS_App *petsc_app)
{
  PetscScalar f = (1./4.) * (1. - PetscCosScalar(petsc_app->w*t));
  return f;
}

/*
 * Oscillator 2 (imaginary part)
 */
PetscScalar G(PetscReal t,TS_App *petsc_app)
{
  PetscScalar g = (1./4.) * (1. - PetscSinScalar(petsc_app->w*t));
  return g;
}

/*
 * Evaluate the right-hand side system Matrix (real, vectorized Hamiltonian system matrix)
 * In: ts - time stepper
 *      t - current time
 *      u - solution vector x(t) 
 *      M - right hand side system Matrix
 *      P - ??
 *    ctx - Application context 
 */
PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec u,Mat M,Mat P,void *ctx)
{
  TS_App   *petsc_app = (TS_App*)ctx;
  PetscInt        nvec = petsc_app->nvec;
  PetscScalar f, g;
  PetscScalar a[(nvec * nvec)],  b[(nvec * nvec)]; 
  PetscInt idx[nvec], idxn[nvec];  
  PetscErrorCode ierr;

/* Setup indices */
  for(int i = 0; i < nvec; i++)
  {
    idx[i] = i;
    idxn[i] = i + nvec;
  }

  /* Compute time-dependent control functions */
  f = F(t, petsc_app);
  g = G(t, petsc_app);


  /* Set up real part of system matrix (A) */
  ierr = MatZeroEntries(petsc_app->A);CHKERRQ(ierr);
  ierr = MatAXPY(petsc_app->A,g,petsc_app->IKbMbd,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(petsc_app->A,-1.*g,petsc_app->bMbdTKI,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  /* Set up imaginary part of system matrix (B) */
  ierr = MatZeroEntries(petsc_app->B);CHKERRQ(ierr);
  ierr = MatAXPY(petsc_app->B,f,petsc_app->aPadTKI,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(petsc_app->B,-1.*f,petsc_app->IKaPad,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  //MatView(petsc_app->A, PETSC_VIEWER_STDOUT_SELF);
  //MatView(petsc_app->B, PETSC_VIEWER_STDOUT_SELF);

  /* Get values of A */
  MatGetValues(petsc_app->A, nvec, idx, nvec, idx, a);

  /* M(0, 0) = A */
  MatSetValues(M, nvec, idx, nvec, idx, a , INSERT_VALUES);
  /* Set M(1, 1) = A */
  MatSetValues(M, nvec, idxn, nvec, idxn, a , INSERT_VALUES);

  /* Get values of B */
  MatGetValues(petsc_app->B, nvec, idx, nvec, idx, b);

  /* Set M(1, 0) = B */
  MatSetValues(M, nvec, idxn, nvec, idx, b, INSERT_VALUES);
  /* Set M(0, 1) = -B */
  for(int i = 0; i < nvec * nvec; i++)
  {
    b[i] = -1.0 * b[i];
  }
  MatSetValues(M, nvec, idx, nvec, idxn, b, INSERT_VALUES);

  /* Assemble the system matrix */
  ierr = MatAssemblyBegin(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(M,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* TODO: Store M in sparse matrix format! */
  /* Do we really need to store A and B explicitely? They are only used here, so maybe we can assemble M from g,f,IKbMDb, bMbdTKI, aPadTKI, IKaPad directly... */

  // MatView(M, PETSC_VIEWER_STDOUT_SELF);

  return 0;
}




/*
 * Initialize fixed matrices for assembling system Hamiltonian
 */
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
  ierr = MatAssemblyBegin(petsc_app->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(petsc_app->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Allocate B */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nvec,nvec,0,NULL,&petsc_app->B);CHKERRQ(ierr);
  ierr = MatSetFromOptions(petsc_app->B);CHKERRQ(ierr);
  ierr = MatSetUp(petsc_app->B);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(petsc_app->B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(petsc_app->B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  return 0;
}


