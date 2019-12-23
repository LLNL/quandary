#include "braid_wrapper.hpp"


myBraidVector::myBraidVector() {
  x = NULL;
}

myBraidVector::myBraidVector(MPI_Comm comm, int dim) {

    /* Allocate the Petsc Vector */
    VecCreateSeq(comm, dim, &x);
    VecZeroEntries(x);
}


myBraidVector::~myBraidVector() {
  VecDestroy(&x);
}



myBraidApp::myBraidApp(MPI_Comm comm_braid_, MPI_Comm comm_petsc_, double total_time_, int ntime_, TS ts_, Hamiltonian* ham_, MapParam* config) 
          : BraidApp(comm_braid_, 0.0, total_time_, ntime_) {

  ntime = ntime_;
  total_time = total_time_;
  timestepper = ts_;
  hamiltonian = ham_;
  comm_petsc = comm_petsc_;
  ufile = NULL;
  vfile = NULL;


  /* Init Braid core */
  core = new BraidCore(comm_braid_, this);

  /* Get and set Braid options */
  int printlevel = config->GetIntParam("printlevel", 2);
  core->SetPrintLevel(printlevel);
  int iolevel = config->GetIntParam("iolevel", 1);
  core->SetAccessLevel( iolevel);
  int maxlevels = config->GetIntParam("maxlevels", 20);
  core->SetMaxLevels(maxlevels);
  int cfactor = config->GetIntParam("cfactor", 5);
  core->SetCFactor(-1, cfactor);
  int maxiter = config->GetIntParam("maxiter", 50);
  core->SetMaxIter( maxiter);
  bool skip = (PetscBool) config->GetBoolParam("skip", false);
  core->SetSkip( skip);
  bool fmg = (PetscBool) config->GetBoolParam("fmg", false);
  if (fmg) core->SetFMG();

  core->SetNRelax(-1, 1);
  core->SetAbsTol(1e-6);
  core->SetSeqSoln(0);

  // _braid_SetVerbosity(core->GetCore(), 1);
}

myBraidApp::~myBraidApp() {
  /* Delete the core, if drive() has been called */
  if (core->GetWarmRestart()) delete core;
}

int myBraidApp::getTimeStepIndex(double t, double dt){
  int ts = round(t / dt);
  return ts;
}


BraidCore* myBraidApp::getCore() { return core; }

int myBraidApp::printConvHistory(const char* filename){ 
  FILE* braidlog;
  int niter, ntime, nlevels;
  int cfactor;
  double dt;

  /* Get some general information from the core */
  core->GetNLevels(&nlevels);
  core->GetCFactor(&cfactor);

  /* Get braid's residual norms for all iterations */
  core->GetNumIter(&niter);
  double *norms     = new double[niter];
  core->GetRNorms(&niter, norms);

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


braid_Int myBraidApp::Step(braid_Vector u_, braid_Vector ustop_, braid_Vector fstop_, BraidStepStatus &pstatus){
    double tstart, tstop;
    int tindex;
    int done;

    /* Cast input u to class definition */
    myBraidVector *u = (myBraidVector *)u_;


    /* Grab current time from XBraid and pass it to Petsc time-stepper */
    pstatus.GetTstartTstop(&tstart, &tstop);
    pstatus.GetTIndex(&tindex);
    pstatus.GetDone(&done); 
  
    // printf("\nBraid %d %f->%f \n", tindex, tstart, tstop);

#ifdef SANITY_CHECK
    //  printf("Performing check Hermitian, Trace... \n");
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
    TSSetTime(timestepper, tstart);
    TSSetTimeStep(timestepper, tstop - tstart);

    /* Pass the curent state to the Petsc time-stepper */
    // TSSetSolution(app->ts, u->x);

    // app->ts->steps = 0;
    TSSetStepNumber(timestepper, 0);
    TSSetMaxSteps(timestepper, 1);
    TSSolve(timestepper, u->x);

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


braid_Int myBraidApp::Residual(braid_Vector u_, braid_Vector r_, BraidStepStatus &pstatus){ return 0; }

braid_Int myBraidApp::Clone(braid_Vector u_, braid_Vector *v_ptr){ 
    /* Cast input braid vector to class vector definition */
    myBraidVector *u = (myBraidVector *)u_;

    /* Allocate a new vector */
    myBraidVector* ucopy = new myBraidVector();

    /* First duplicate storage, then copy values */
    VecDuplicate(u->x, &(ucopy->x));
    VecCopy(u->x, ucopy->x);

    /* Set the return pointer */
    *v_ptr = (braid_Vector) ucopy;

  return 0; 
}

braid_Int myBraidApp::Init(braid_Real t, braid_Vector *u_ptr){ 

  /* Get dimensions */
  int nreal = 2 * hamiltonian->getDim();

  /* Create the vector */
  myBraidVector *u = new myBraidVector(comm_petsc, nreal);

  /* Set the initial condition */
  hamiltonian->initialCondition(u->x);

  /* Return vector to braid */
  *u_ptr = (braid_Vector) u;
  
  return 0; 
}


braid_Int myBraidApp::Free(braid_Vector u_){ 
  myBraidVector *u = (myBraidVector *)u_;
  delete u;
  return 0; 
}


braid_Int myBraidApp::Sum(braid_Real alpha, braid_Vector x_, braid_Real beta, braid_Vector y_){ 
  myBraidVector *x = (myBraidVector *)x_;
  myBraidVector *y = (myBraidVector *)y_;

    int dim;
    const PetscScalar *x_ptr;
    PetscScalar *y_ptr;

    VecGetSize(x->x, &dim);
    VecGetArrayRead(x->x, &x_ptr);
    VecGetArray(y->x, &y_ptr);
    for (int i = 0; i< 2 * hamiltonian->getDim(); i++)
    {
        y_ptr[i] = alpha * x_ptr[i] + beta * y_ptr[i];
    }
    VecRestoreArray(y->x, &y_ptr);

  return 0; 
}


braid_Int myBraidApp::SpatialNorm(braid_Vector u_, braid_Real *norm_ptr){ 
  myBraidVector *u = (myBraidVector *)u_;

  double norm;
  VecNorm(u->x, NORM_2, &norm);

  *norm_ptr = norm;
  return 0; 
}


braid_Int myBraidApp::Access(braid_Vector u_, BraidAccessStatus &astatus){ 
  myBraidVector *u = (myBraidVector *)u_;
  int istep;
  double t;
  double err_norm, exact_norm;
  Vec err;
  Vec exact;

  /* Get time information */
  astatus.GetTIndex(&istep);
  astatus.GetT(&t);

  // printf("\nAccess %d %f\n", istep, t);
  // VecView(u->x, PETSC_VIEWER_STDOUT_WORLD);

  if (t == 0.0) return 0;

  if (ufile != NULL && vfile != NULL) {

    /* Get access to Petsc's vector */
    const PetscScalar *x_ptr;
    VecGetArrayRead(u->x, &x_ptr);

    /* Write solution to files */
    fprintf(ufile,  "%.8f  ", t);
    fprintf(vfile,  "%.8f  ", t);
    for (int i = 0; i < 2*hamiltonian->getDim(); i++)
    {
      if (i < hamiltonian->getDim()) // real part
      { 
        fprintf(ufile, "%1.14e  ", x_ptr[i]);  
      }  
      else  // imaginary part
      {
        fprintf(vfile, "%1.14e  ", x_ptr[i]); 
      }

    }
    fprintf(ufile, "\n");
    fprintf(vfile, "\n");
  }

    /* TODO */ 
    // /* If set, compare to the exact solution */
    // VecDuplicate(u->x,&exact);
    // if (app->hamiltonian->ExactSolution(t, exact) ) {  

    //   /* Compute relative error norm */
    //   VecDuplicate(u->x,&err);
    //   VecWAXPY(err,-1.0,u->x, exact);
    //   VecNorm(err, NORM_2,&err_norm);
    //   VecNorm(exact, NORM_2,&exact_norm);
    //   err_norm = err_norm / exact_norm;

    //   /* Print error */
    //   const PetscScalar *exact_ptr;
    //   VecGetArrayRead(exact, &exact_ptr);
    //   if (istep == app->ntime){
    //       printf("Last step: ");
    //       printf("%5d  %1.5f  x[1] = %1.14e  exact[1] = %1.14e  err = %1.14e \n",istep,(double)t, x_ptr[1], exact_ptr[1], err_norm);
    //   } 

    //   VecDestroy(&err);
    // }


    // VecDestroy(&exact);



  return 0; 
}


braid_Int myBraidApp::BufSize(braid_Int *size_ptr, BraidBufferStatus &bstatus){ 

  *size_ptr = 2 * hamiltonian->getDim() * sizeof(double);
  return 0; 
}


braid_Int myBraidApp::BufPack(braid_Vector u_, void *buffer, BraidBufferStatus &bstatus){ 
  
  /* Cast input */
  myBraidVector *u = (myBraidVector *)u_;
  double* dbuffer = (double*) buffer;

  const PetscScalar *x_ptr;
  int dim;
  VecGetSize(u->x, &dim);

  /* Copy the values into the buffer */
  VecGetArrayRead(u->x, &x_ptr);
  for (int i=0; i < dim; i++)
  {
      dbuffer[i] = x_ptr[i];
  }
  VecRestoreArrayRead(u->x, &x_ptr);

  int size =  dim * sizeof(double);
  bstatus.SetSize(size);

  return 0; 
}


braid_Int myBraidApp::BufUnpack(void *buffer, braid_Vector *u_ptr, BraidBufferStatus &bstatus){ 

  /* Cast buffer to double */
  double* dbuffer = (double*) buffer;

  /* Allocate a new vector */
  int dim = 2 * hamiltonian->getDim();
  myBraidVector *u = new myBraidVector(comm_petsc, dim);

  /* Copy buffer into the vector */
  PetscScalar *x_ptr;
  VecGetArray(u->x, &x_ptr);
  for (int i=0; i < dim; i++)
  {
      x_ptr[i] = dbuffer[i];
  }

  /* Restore Petsc's vector */
  VecRestoreArray(u->x, &x_ptr);

  /* Pass vector to XBraid */
  *u_ptr = (braid_Vector) u;

  return 0; 
}

braid_Int myBraidApp::SetInitialCondition(){ 
/* Apply initial condition if warm_restart (otherwise it is set in my_Init().
 * Can not be set here if !(warm_restart) because the braid_grid is created only when braid_drive() is called. 
 */

  braid_BaseVector ubase;
  myBraidVector *u;
      
  if (core->GetWarmRestart()) {
    /* Get vector at t == 0 */
    _braid_UGetVectorRef(core->GetCore(), 0, 0, &ubase);
    if (ubase != NULL)  // only true on one first processor !
    {
      u = (myBraidVector *)ubase->userVector;
      hamiltonian->initialCondition(u->x);
    }
  }

  return 0; 
}


double myBraidApp::run() { 
  
  int nreq = -1;
  double norm;

  SetInitialCondition();
  core->Drive();
  core->GetRNorms(&nreq, &norm);

  return norm; 
}


/* ================================================================*/
/* Adjoint Braid App */
/* ================================================================*/
myAdjointBraidApp::myAdjointBraidApp(MPI_Comm comm_braid_, MPI_Comm comm_petsc_, double total_time_, int ntime_, TS ts_, Hamiltonian* ham_, Vec redgrad_, MapParam* config, BraidCore *Primalcoreptr_)
        : myBraidApp(comm_braid_, comm_petsc_, total_time_, ntime_, ts_, ham_, config) {

  /* Store the primal core */
  primalcore = Primalcoreptr_;

  /* Store reduced gradient */
  redgrad = redgrad_;

  /* Ensure that primal core stores all points */
  primalcore->SetStorage(0);

  /* Revert processor ranks for solving adjoint */
  core->SetRevertedRanks(1);
  // _braid_SetVerbosity(core->GetCore(), 1);
}

myAdjointBraidApp::~myAdjointBraidApp() {}

int myAdjointBraidApp::getPrimalIndex(int ts) { 
  /* TODO: Check this! Might need -1 */
  return ntime - ts; 
}

braid_Int myAdjointBraidApp::Step(braid_Vector u_, braid_Vector ustop_, braid_Vector fstop_, BraidStepStatus &pstatus) {

  myBraidVector *u = (myBraidVector *)u_;

  double tstart, tstop;
  int ierr;
  int tindex;
  int level, done;
  bool update_gradient;

  /* Update gradient only on the finest grid */
  pstatus.GetLevel(&level);
  pstatus.GetDone(&done);
  if (done){
    update_gradient = true;
  }
  else {
    update_gradient = false;
  }

    /* Grab current time from XBraid and pass it to Petsc time-stepper */
    pstatus.GetTstartTstop(&tstart, &tstop);
    pstatus.GetTIndex(&tindex);

    double dt = tstop - tstart;
    // printf("\n %d: Braid %d %f->%f, dt=%f \n", mpirank, tindex, tstart, tstop, dt);

    /* Get primal state */
    int finegrid = 0;
    int tstop_id = getTimeStepIndex(tstop, total_time / ntime);
    int primaltimestep = ntime - tstop_id;
    braid_BaseVector ubaseprimal;
    myBraidVector *uprimal;
    Vec x;
    // printf("Accessing primal %d\n", primaltimestep);
    _braid_UGetVectorRef(primalcore->GetCore(), finegrid, primaltimestep, &ubaseprimal);
    if (ubaseprimal == NULL) printf("ubaseprimal is null!\n");
    uprimal = (myBraidVector*) ubaseprimal->userVector;
    VecDuplicate(uprimal->x, &x);
    VecCopy(uprimal->x, x);
    // VecView(x, PETSC_VIEWER_STDOUT_WORLD);


    /* Solve forward while saving trajectory */
    TSDestroy(&timestepper);
    TSCreate(PETSC_COMM_SELF,&timestepper);CHKERRQ(ierr);
    // TSReset(app->ts);
    TSInit(timestepper, hamiltonian, ntime  , dt, total_time, x, &(u->x), &redgrad, false);

    ierr = TSSetSaveTrajectory(timestepper);CHKERRQ(ierr);
    ierr = TSTrajectorySetSolutionOnly(timestepper->trajectory, (PetscBool) true);
    ierr = TSTrajectorySetType(timestepper->trajectory, timestepper, TSTRAJECTORYMEMORY);

    TSSetTime(timestepper, total_time - tstop);
    TSSetTimeStep(timestepper, dt);

    TSSetStepNumber(timestepper, 0);
    TSSetMaxSteps(timestepper, 1);

    TSSetSolution(timestepper, x);

    TSSolve(timestepper, x);

    /* Set adjoint vars */ 
    if (!update_gradient) VecZeroEntries(redgrad);
    TSSetCostGradients(timestepper, 1, &u->x, &redgrad); CHKERRQ(ierr);

    /* Solve adjoint */
    TSSetTimeStep(timestepper, -dt);
    TSAdjointSolve(timestepper);

    VecDestroy(&x);
    // app->ts->ptime_prev = tstop;
    // TSSetTimeStep(app->ts, -dt) ; 
    // TSSetTime(app->ts, app->total_time - tstart);
    // app->ts->ptime_prev = app->total_time - tstop;    
    // TSSetTimeStep(app->ts, 5000.0) ; // -(ptime - ptime_prev) ?


    /* Pass adjoint and derivative to Petsc */
    // TSSetAdjointSolution(app->ts, u_bar->x, app->redgrad);   // this one works too!?

    // VecCopy(u->x, app->ts->vecs_sensi[0]);
    // if (update_gradient) {
    //   VecCopy(app->redgrad, app->ts->vecs_sensip[0]);
    // } else {
    //   VecZeroEntries(app->ts->vecs_sensip[0]);
    // }

    // /* Take an adjoint step */
    // bool tj_save = true;
    // app->ts->steps = app->ntime - tindex;
    // TSAdjointStepMod(app->ts, tj_save);

    // /* Grab derivatives from Petsc and pass to XBraid */
    // VecCopy(app->ts->vecs_sensi[0], u->x);
    // if (update_gradient) VecCopy(app->ts->vecs_sensip[0], app->redgrad);



  return 0;  
}


braid_Int myAdjointBraidApp::Init(braid_Real t, braid_Vector *u_ptr) {

  /* Allocate the adjoint vector and set to zero */
  myBraidVector *u = new myBraidVector(comm_petsc, 2*hamiltonian->getDim());

  /* Reset the reduced gradient */
  VecZeroEntries(redgrad); 

  /* Set the differentiated objective function */
  if (t==0){
    double t = 0.0;
    // TSTrajectoryGet(app->ts->trajectory,app->ts, 0, &t);
    hamiltonian->evalObjective_diff(t, timestepper->vec_sol, &u->x, &redgrad);
  }

  /* Return new vector to braid */
  *u_ptr = (braid_Vector) u;

  return 0;
}


int myAdjointBraidApp::SetInitialCondition() {
/* If warm_restart: set adjoint initial condition here. Otherwise it's set in my_Init_Adj.
 * It can not be done here if drive() has not been called before, because the braid grid is allocated only at the beginning of drive() 
*/
  braid_BaseVector ubaseadjoint;
  myBraidVector *uadjoint;

  if (core->GetWarmRestart()) {
    /* Get adjoint state */
    _braid_UGetVectorRef(core->GetCore(), 0, 0, &ubaseadjoint);
    if (ubaseadjoint != NULL) {   // this on true at the last processor only
      uadjoint = (myBraidVector *) ubaseadjoint->userVector;
    }

    /* Evaluate differentiated objective function */
    hamiltonian->evalObjective_diff(0.0, NULL, &uadjoint->x, &redgrad);
  }


  return 0;
}

