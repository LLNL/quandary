#include "braid_wrapper.hpp"


int my_Step(braid_App    app,
        braid_Vector     ustop,
        braid_Vector     fstop,
        braid_Vector     u,
        braid_StepStatus status)
{
    double tstart, tstop;
    int tindex;
    int done;

    /* Grab current time from XBraid and pass it to Petsc time-stepper */
    braid_StepStatusGetTstartTstop(status, &tstart, &tstop);
    braid_StepStatusGetTIndex(status, &tindex);
    braid_StatusGetDone((braid_Status) status, &done);
    // printf("\nBraid %d %f->%f \n", tindex, tstart, tstop);

#ifdef SANITY_CHECK
     printf("Performing check Hermitian, Trace... \n");
    /* Sanity check. Be careful: This is costly! */
    PetscBool check;
    double tol = 1e-14;
    StateIsHermitian(u->x, tol, &check);
    if (!check) {
      printf("%f: WARNING! State is not hermitian!\n", tstart);
      exit(1);
    }
    StateHasTrace1(u->x, tol, &check);
    if (!check) {
      printf("%f: WARNING! Trace(State) is not one!\n", tstart);
      exit(1);
    }
  #endif

    /* Set the time */
    TSSetTime(app->ts, tstart);
    TSSetTimeStep(app->ts, tstop - tstart);

    /* Pass the curent state to the Petsc time-stepper */
    // TSSetSolution(app->ts, u->x);

    // app->ts->steps = 0;
    TSSetStepNumber(app->ts, 0);
    TSSetMaxSteps(app->ts, 1);
    TSSolve(app->ts, u->x);

    // int ml = 0;
    // braid_StatusGetNLevels((braid_Status) status, &ml);
    

    /* Take a step forward */
    // bool tj_save = false;
    // if (done || ml <= 1) tj_save = true;
    // TSStepMod(app->ts, tj_save);

    /* Calling the access routine here, because I removed it from the end of the braid_Drive() routine. This might give wrong tindex values... TODO: Check! */
    // if (done) my_Access(app, u, (braid_AccessStatus) status);

    return 0;
}



int my_Init(braid_App     app,
        double        t,
        braid_Vector *u_ptr)
{

    int nreal = 2 * app->hamiltonian->getDim();

    /* Allocate a new braid vector */
    my_Vector* u = (my_Vector*) malloc(sizeof(my_Vector));

    /* Allocate the Petsc Vector */
    VecCreateSeq(app->comm_petsc,nreal,&(u->x));
    VecZeroEntries(u->x);

    /* Set initial condition at t=0.0 */
    // if (t == 0.0)
    {
        app->hamiltonian->initialCondition(u->x);
    }

    /* Set the return pointer */
    *u_ptr = u;

    return 0;
}



/* Create a copy of a braid vector */
int my_Clone(braid_App     app,
         braid_Vector  u,
         braid_Vector *v_ptr)
{

    /* Allocate a new vector */
    my_Vector* ucopy = (my_Vector*) malloc(sizeof(my_Vector));

    /* First duplicate storage, then copy values */
    VecDuplicate(u->x, &(ucopy->x));
    VecCopy(u->x, ucopy->x);

    /* Set the return pointer */
    *v_ptr = ucopy;
    return 0;
}



int my_Free(braid_App    app,
        braid_Vector u)
{

    /* Destroy Petsc's vector */
    VecDestroy(&(u->x));

    /* Destroy XBraid vector */
    free(u);
    return 0;
}


/* Sum AXPBY: y = alpha * x + beta * y */
int my_Sum(braid_App    app,
       double       alpha,
       braid_Vector x,
       double       beta,
       braid_Vector y)
{
    const PetscScalar *x_ptr;
    PetscScalar *y_ptr;

    VecGetArrayRead(x->x, &x_ptr);
    VecGetArray(y->x, &y_ptr);
    for (int i = 0; i< 2 * app->hamiltonian->getDim(); i++)
    {
        y_ptr[i] = alpha * x_ptr[i] + beta * y_ptr[i];
    }
    VecRestoreArray(y->x, &y_ptr);

    // VecAXPBY(y->x, alpha, beta, x->x);

    return 0;
}



int my_Access(braid_App       app,
          braid_Vector        u,
          braid_AccessStatus  astatus)
{
    int istep;
    double t;
    double err_norm, exact_norm;
    Vec err;
    Vec exact;

    /* Get time information */
    braid_AccessStatusGetTIndex(astatus, &istep);
    braid_AccessStatusGetT(astatus, &t);

    // printf("\nAccess %d %f\n", istep, t);
    // VecView(u->x, PETSC_VIEWER_STDOUT_WORLD);

    if (t == 0.0) return 0;

    /* Get access to Petsc's vector */
    const PetscScalar *x_ptr;
    VecGetArrayRead(u->x, &x_ptr);

    /* Write solution to files */
    fprintf(app->ufile,  "%.2f  ", t);
    fprintf(app->vfile,  "%.2f  ", t);
    for (int i = 0; i < 2*app->hamiltonian->getDim(); i++)
    {

      if (i < app->hamiltonian->getDim()) // real part
      {
        fprintf(app->ufile, "%1.14e  ", x_ptr[i]);  
      }  
      else  // imaginary part
      {
        fprintf(app->vfile, "%1.14e  ", x_ptr[i]); 
      }
      
    }
    fprintf(app->ufile, "\n");
    fprintf(app->vfile, "\n");


    /* If set, compare to the exact solution */
    VecDuplicate(u->x,&exact);
    if (app->hamiltonian->ExactSolution(t, exact) ) {  

      /* Compute relative error norm */
      VecDuplicate(u->x,&err);
      VecWAXPY(err,-1.0,u->x, exact);
      VecNorm(err, NORM_2,&err_norm);
      VecNorm(exact, NORM_2,&exact_norm);
      err_norm = err_norm / exact_norm;

      /* Print error */
      const PetscScalar *exact_ptr;
      VecGetArrayRead(exact, &exact_ptr);
      if (istep == app->ntime){
          printf("Last step: ");
          printf("%5d  %1.5f  x[1] = %1.14e  exact[1] = %1.14e  err = %1.14e \n",istep,(double)t, x_ptr[1], exact_ptr[1], err_norm);
      } 

      VecDestroy(&err);
    }


    VecDestroy(&exact);

    return 0;
}


int my_SpatialNorm(braid_App     app,
               braid_Vector  u,
               double       *norm_ptr)
{
    double norm;
    VecNorm(u->x, NORM_2, &norm);

    *norm_ptr = norm;
    return 0;
}

int my_BufSize(braid_App           app,
               int                 *size_ptr,
               braid_BufferStatus  bstatus)
{

    *size_ptr = 2 * app->hamiltonian->getDim() * sizeof(double);
    return 0;
}


int my_BufPack(braid_App       app,
           braid_Vector        u,
           void                *buffer,
           braid_BufferStatus  bstatus)
{
    const PetscScalar *x_ptr;
    double* dbuffer = (double*) buffer;
    int N = 2*app->hamiltonian->getDim();



    /* Copy the values into the buffer */
    VecGetArrayRead(u->x, &x_ptr);
    for (int i=0; i < N; i++)
    {
        dbuffer[i] = x_ptr[i];
    }
    VecRestoreArrayRead(u->x, &x_ptr);

    int size =  N * sizeof(double);
    braid_BufferStatusSetSize(bstatus, size);

    return 0;
}


int my_BufUnpack(braid_App        app,
             void                *buffer,
             braid_Vector        *u_ptr,
             braid_BufferStatus   status)
{
    double* dbuffer = (double*) buffer;
    int N = 2*app->hamiltonian->getDim();


    /* Create a new vector */
    braid_Vector u;
    my_Init(app, 0.0, &u);

    /* Get write access to the Petsc Vector */
    PetscScalar *x_ptr;
    VecGetArray(u->x, &x_ptr);

    /* Copy buffer into the vector */
    for (int i=0; i < N; i++)
    {
        x_ptr[i] = dbuffer[i];
    }

    /* Restore Petsc's vector */
    VecRestoreArray(u->x, &x_ptr);

    /* Pass vector to XBraid */
    *u_ptr = u;
    
    return 0;
}


int my_Step_adj(braid_App app, braid_Vector ustop, braid_Vector fstop, braid_Vector u, braid_StepStatus status){

    double tstart, tstop;
   int ierr;
    int tindex;
    int level, done;
    bool update_gradient;
    int mpirank;
    MPI_Comm_rank(app->comm_braid, &mpirank);


    /* Update gradient only on the finest grid */
    braid_StepStatusGetLevel(status, &level);
    braid_StatusGetDone((braid_Status) status, &done);
    if (done){
      update_gradient = true;
    }
    else {
      update_gradient = false;
    }

    /* Grab current time from XBraid and pass it to Petsc time-stepper */
    braid_StepStatusGetTstartTstop(status, &tstart, &tstop);
    braid_StepStatusGetTIndex(status, &tindex);

    double dt = tstop - tstart;
    // printf("\n %d: Braid %d %f->%f, dt=%f \n", mpirank, tindex, tstart, tstop, dt);


    /* Get primal state */
    int finegrid = 0;
    int tstop_id = GetTimeStepIndex(tstop, app->total_time / app->ntime);
    int primaltimestep = app->ntime - tstop_id;
    braid_BaseVector ubaseprimal;
    my_Vector *uprimal;
    Vec x;
    // printf("Accessing primal %d\n", primaltimestep);
    _braid_UGetVectorRef(app->primalcore, finegrid, primaltimestep, &ubaseprimal);
    if (ubaseprimal == NULL) printf("ubaseprimal is null!\n");
    uprimal = (my_Vector*) ubaseprimal->userVector;
    VecDuplicate(uprimal->x, &x);
    VecCopy(uprimal->x, x);
    // VecView(x, PETSC_VIEWER_STDOUT_WORLD);


    /* Solve forward while saving trajectory */
    TSDestroy(&app->ts);
    TSCreate(PETSC_COMM_SELF,&app->ts);CHKERRQ(ierr);
    // TSReset(app->ts);
    TSInit(app->ts, app->hamiltonian, app->ntime  , dt, app->total_time, x, &(u->x), &(app->mu), app->monitor);

    ierr = TSSetSaveTrajectory(app->ts);CHKERRQ(ierr);
    ierr = TSTrajectorySetSolutionOnly(app->ts->trajectory, (PetscBool) true);
    ierr = TSTrajectorySetType(app->ts->trajectory, app->ts, TSTRAJECTORYMEMORY);

    TSSetTime(app->ts, app->total_time - tstop);
    TSSetTimeStep(app->ts, dt);

    TSSetStepNumber(app->ts, 0);
    TSSetMaxSteps(app->ts, 1);

    TSSetSolution(app->ts, x);

    TSSolve(app->ts, x);

    /* Set adjoint vars */ 
    if (!update_gradient) VecZeroEntries(app->mu);
    TSSetCostGradients(app->ts, 1, &u->x, &app->mu); CHKERRQ(ierr);

    /* Solve adjoint */
    TSSetTimeStep(app->ts, -dt);
    TSAdjointSolve(app->ts);

    VecDestroy(&x);
    // app->ts->ptime_prev = tstop;
    // TSSetTimeStep(app->ts, -dt) ; 
    // TSSetTime(app->ts, app->total_time - tstart);
    // app->ts->ptime_prev = app->total_time - tstop;    
    // TSSetTimeStep(app->ts, 5000.0) ; // -(ptime - ptime_prev) ?


    /* Pass adjoint and derivative to Petsc */
    // TSSetAdjointSolution(app->ts, u_bar->x, app->mu);   // this one works too!?

    // VecCopy(u->x, app->ts->vecs_sensi[0]);
    // if (update_gradient) {
    //   VecCopy(app->mu, app->ts->vecs_sensip[0]);
    // } else {
    //   VecZeroEntries(app->ts->vecs_sensip[0]);
    // }

    // /* Take an adjoint step */
    // bool tj_save = true;
    // app->ts->steps = app->ntime - tindex;
    // TSAdjointStepMod(app->ts, tj_save);

    // /* Grab derivatives from Petsc and pass to XBraid */
    // VecCopy(app->ts->vecs_sensi[0], u->x);
    // if (update_gradient) VecCopy(app->ts->vecs_sensip[0], app->mu);


  return 0;
}



int my_Init_adj(braid_App app, double t, braid_Vector *u_ptr){

  braid_Vector u;
  my_Init(app, 0.0, &u);
  VecZeroEntries(u->x);
  VecZeroEntries(app->mu);

  /* Set the differentiated objective function */
  if (t==0){
    double t = 0.0;
    // TSTrajectoryGet(app->ts->trajectory,app->ts, 0, &t);
    app->hamiltonian->evalObjective_diff(t, app->ts->vec_sol, &u->x, &app->mu);
  }

  *u_ptr = u;
  return 0;
}

int my_Access_adj(braid_App       app,
          braid_Vector        u,
          braid_AccessStatus  astatus)
{

  return 0;
}


int braid_printConvHistory(braid_Core core, const char* filename) {

  FILE* braidlog;
  int niter, ntime, nlevels;
  int cfactor;
  double dt;

  /* Get some general information from the core */
  ntime = _braid_CoreElt(core, ntime); 
  braid_GetNLevels(core, &nlevels);
  _braid_GetCFactor(core, 0, &cfactor);

  /* Get braid's residual norms for all iterations */
  braid_GetNumIter(core, &niter);
  double *norms     = new double[niter];
  braid_GetRNorms(core, &niter, norms);

  /* Write to file */
  braidlog = fopen(filename, "w");
  fprintf(braidlog,"# ntime %d\n", (int) ntime);
  fprintf(braidlog,"# cfactor %d\n", (int) cfactor);
  fprintf(braidlog,"# nlevels %d\n", (int) nlevels);
  for (int i=0; i<niter; i++)
  {
    fprintf(braidlog, "%d  %1.14e\n", i, norms[i]);
  }
  fprintf(braidlog, "\n\n\n");

  delete [] norms;
 
  return 0;
}


int GetTimeStepIndex(double t, double dt){
  int ts = round(t / dt);
  
  return ts;
}
