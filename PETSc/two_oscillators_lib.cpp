#include <petscts.h>

static char help[] ="Solves the Liouville-von-Neumann equations, two oscillators.\n\
Input parameters:\n\
  -nlvl <int>      : Set the number of levels (default: 2) \n\
  -nosci <int> : Set the number of oscillators (default: 2) \n\
  -ntime <int>        : Set the number of time steps \n\
  -dt <double>        : Set the time step size \n\
  -w  <double>        : Set the oscillator frequency\n\n";

/*
   Application context contains data needed to perform a time step.
*/
typedef struct {
  Vec         s;       /* global exact solution vector */
  PetscInt    nvec;    /* Dimension of vectorized system */
  Mat         IKbMbd, bMbdTKI, aPadTKI, IKaPad, A, B;
  PetscReal   w;       /* Oscillator frequencies */
} AppCtx;


// /*  Declare external routines */
// extern PetscErrorCode SetUpMatrices(AppCtx*);
// extern PetscErrorCode InitialConditions(Vec,AppCtx*);
// extern PetscErrorCode ExactSolution(PetscReal,Vec,AppCtx*);
// extern PetscScalar F(PetscReal,AppCtx*);
// extern PetscScalar G(PetscReal,AppCtx*);
// extern PetscErrorCode RHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);




/*
 *   Compute the exact solution at a given time.
 *   Input:
 *      t - current time
 *      s - vector in which exact solution will be computed
 *      appctx - application context
 *   Output:
 *      s - vector with the newly computed exact solution
 */
PetscErrorCode ExactSolution(PetscReal t,Vec s,AppCtx *appctx)
{
  PetscScalar    *s_localptr;
  PetscErrorCode ierr;

  /* Get a pointer to vector data. */
  ierr = VecGetArray(s,&s_localptr);CHKERRQ(ierr);

  /* Write the solution into the array locations.
   *  Alternatively, we could use VecSetValues() or VecSetValuesLocal(). */
  PetscScalar phi = (1./4.) * (t - (1./appctx->w)*PetscSinScalar(appctx->w*t));
  PetscScalar theta = (1./4.) * (t + (1./appctx->w)*PetscCosScalar(appctx->w*t) - 1.);
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
 *     appctx - application context
 *  Output Parameter:
 *     u - vector with solution at initial time (global)
 */
PetscErrorCode InitialConditions(Vec x,AppCtx *appctx)
{
  ExactSolution(0,x,appctx);
  return 0;
}
/*
 * Oscillator 1 (real part)
 */
PetscScalar F(PetscReal t,AppCtx *appctx)
{
  PetscScalar f = (1./4.) * (1. - PetscCosScalar(appctx->w*t));
  return f;
}

/*
 * Oscillator 2 (imaginary part)
 */
PetscScalar G(PetscReal t,AppCtx *appctx)
{
  PetscScalar g = (1./4.) * (1. - PetscSinScalar(appctx->w*t));
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
  AppCtx         *appctx = (AppCtx*)ctx;
  PetscInt        nvec = appctx->nvec;
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
  f = F(t, appctx);
  g = G(t, appctx);


  /* Set up real part of system matrix (A) */
  ierr = MatZeroEntries(appctx->A);CHKERRQ(ierr);
  ierr = MatAXPY(appctx->A,g,appctx->IKbMbd,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(appctx->A,-1.*g,appctx->bMbdTKI,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  /* Set up imaginary part of system matrix (B) */
  ierr = MatZeroEntries(appctx->B);CHKERRQ(ierr);
  ierr = MatAXPY(appctx->B,f,appctx->aPadTKI,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatAXPY(appctx->B,-1.*f,appctx->IKaPad,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  //MatView(appctx->A, PETSC_VIEWER_STDOUT_SELF);
  //MatView(appctx->B, PETSC_VIEWER_STDOUT_SELF);

  /* Get values of A */
  MatGetValues(appctx->A, nvec, idx, nvec, idx, a);

  /* M(0, 0) = A */
  MatSetValues(M, nvec, idx, nvec, idx, a , INSERT_VALUES);
  /* Set M(1, 1) = A */
  MatSetValues(M, nvec, idxn, nvec, idxn, a , INSERT_VALUES);

  /* Get values of B */
  MatGetValues(appctx->B, nvec, idx, nvec, idx, b);

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
PetscErrorCode SetUpMatrices(AppCtx *appctx)
{
  PetscInt       nvec = appctx->nvec;
  PetscInt       i, j;
  PetscScalar    v[1];
  PetscErrorCode ierr;

  /* Set up IKbMbd */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nvec,nvec,1,NULL,&appctx->IKbMbd);CHKERRQ(ierr);
  ierr = MatSetFromOptions(appctx->IKbMbd);CHKERRQ(ierr);
  ierr = MatSetUp(appctx->IKbMbd);CHKERRQ(ierr);

  i = 1;
  j = 0;
  v[0] = -1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 0;
  j = 1;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 3;
  j = 2;
  v[0] = -1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 2;
  j = 3;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 5;
  j = 4;
  v[0] = -1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 4;
  j = 5;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 7;
  j = 6;
  v[0] = -1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 6;
  j = 7;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 9;
  j = 8;
  v[0] = -1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 8;
  j = 9;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 11;
  j = 10;
  v[0] = -1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 10;
  j = 11;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 13;
  j = 12;
  v[0] = -1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 12;
  j = 13;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 15;
  j = 14;
  v[0] = -1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 14;
  j = 15;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKbMbd,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(appctx->IKbMbd,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(appctx->IKbMbd,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

///////////////////////////////////////////////////////////////////////////////

  /* Set up bMbdTKI */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nvec,nvec,1,NULL,&appctx->bMbdTKI);CHKERRQ(ierr);
  ierr = MatSetFromOptions(appctx->bMbdTKI);CHKERRQ(ierr);
  ierr = MatSetUp(appctx->bMbdTKI);CHKERRQ(ierr);

  i = 4;
  j = 0;
  v[0] = 1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 5;
  j = 1;
  v[0] = 1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 6;
  j = 2;
  v[0] = 1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 7;
  j = 3;
  v[0] = 1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 0;
  j = 4;
  v[0] = -1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 1;
  j = 5;
  v[0] = -1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 2;
  j = 6;
  v[0] = -1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 3;
  j = 7;
  v[0] = -1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 12;
  j = 8;
  v[0] = 1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 13;
  j = 9;
  v[0] = 1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 14;
  j = 10;
  v[0] = 1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 15;
  j = 11;
  v[0] = 1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 8;
  j = 12;
  v[0] = -1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 9;
  j = 13;
  v[0] = -1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 10;
  j = 14;
  v[0] = -1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 11;
  j = 15;
  v[0] = -1;
  ierr = MatSetValues(appctx->bMbdTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(appctx->bMbdTKI,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(appctx->bMbdTKI,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

///////////////////////////////////////////////////////////////////////////////

  /* Set up aPadTKI */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nvec,nvec,1,NULL,&appctx->aPadTKI);CHKERRQ(ierr);
  ierr = MatSetFromOptions(appctx->aPadTKI);CHKERRQ(ierr);
  ierr = MatSetUp(appctx->aPadTKI);CHKERRQ(ierr);

  i = 8;
  j = 0;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 9;
  j = 1;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 10;
  j = 2;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 11;
  j = 3;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 12;
  j = 4;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 13;
  j = 5;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 14;
  j = 6;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 15;
  j = 7;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 0;
  j = 8;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 1;
  j = 9;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 2;
  j = 10;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 3;
  j = 11;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 4;
  j = 12;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 5;
  j = 13;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 6;
  j = 14;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 7;
  j = 15;
  v[0] = 1;
  ierr = MatSetValues(appctx->aPadTKI,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(appctx->aPadTKI,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(appctx->aPadTKI,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

////////////////////////////////////////////////////////////////////////////////

  /* Set up IKaPad */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nvec,nvec,1,NULL,&appctx->IKaPad);CHKERRQ(ierr);
  ierr = MatSetFromOptions(appctx->IKaPad);CHKERRQ(ierr);
  ierr = MatSetUp(appctx->IKaPad);CHKERRQ(ierr);

  i = 2;
  j = 0;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 3;
  j = 1;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 0;
  j = 2;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 1;
  j = 3;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 6;
  j = 4;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 7;
  j = 5;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 4;
  j = 6;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 5;
  j = 7;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 10;
  j = 8;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 11;
  j = 9;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 8;
  j = 10;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 9;
  j = 11;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 14;
  j = 12;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 15;
  j = 13;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 12;
  j = 14;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);
  i = 13;
  j = 15;
  v[0] = 1;
  ierr = MatSetValues(appctx->IKaPad,1,&i,1,&j,v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(appctx->IKaPad,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(appctx->IKaPad,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

//////////////////////////////////////////////////////////////////////////////

  /* Allocate A */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nvec,nvec,0,NULL,&appctx->A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(appctx->A);CHKERRQ(ierr);
  ierr = MatSetUp(appctx->A);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(appctx->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(appctx->A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Allocate B */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,nvec,nvec,0,NULL,&appctx->B);CHKERRQ(ierr);
  ierr = MatSetFromOptions(appctx->B);CHKERRQ(ierr);
  ierr = MatSetUp(appctx->B);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(appctx->B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(appctx->B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  return 0;
}


