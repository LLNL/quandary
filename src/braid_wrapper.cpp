#include "braid_wrapper.hpp"


myBraidVector::myBraidVector() {
  x = NULL;
}

myBraidVector::myBraidVector(int dim) {

    /* Allocate the Petsc Vector */
    VecCreateSeq(PETSC_COMM_WORLD, dim, &x);
    VecZeroEntries(x);
}


myBraidVector::~myBraidVector() {
  VecDestroy(&x);
}



myBraidApp::myBraidApp(MPI_Comm comm_braid_, double total_time_, int ntime_, TS ts_petsc_, TimeStepper* mytimestepper_, Hamiltonian* ham_, Gate* targate_, MapParam* config) 
          : BraidApp(comm_braid_, 0.0, total_time_, ntime_) {

  ntime = ntime_;
  total_time = total_time_;
  ts_petsc = ts_petsc_;
  mytimestepper = mytimestepper_;
  hamiltonian = ham_;
  targetgate = targate_;
  comm_braid = comm_braid_;
  MPI_Comm_rank(comm_braid, &braidrank);
  ufile = NULL;
  vfile = NULL;

  usepetscts = config->GetBoolParam("usepetscts", false);

  /* Init Braid core */
  core = new BraidCore(comm_braid_, this);

  /* Get and set Braid options */
  int printlevel = config->GetIntParam("braid_printlevel", 2);
  core->SetPrintLevel(printlevel);
  int maxlevels = config->GetIntParam("braid_maxlevels", 20);
  core->SetMaxLevels(maxlevels);
  int cfactor = config->GetIntParam("braid_cfactor", 5);
  core->SetCFactor(-1, cfactor);
  int maxiter = config->GetIntParam("braid_maxiter", 50);
  core->SetMaxIter( maxiter);
  bool skip = (PetscBool) config->GetBoolParam("braid_skip", false);
  core->SetSkip( skip);
  bool fmg = (PetscBool) config->GetBoolParam("braid_fmg", false);
  if (fmg) core->SetFMG();

  core->SetNRelax(-1, 1);
  core->SetAbsTol(1e-6);
  core->SetSeqSoln(0);


  /* Output */
  accesslevel = config->GetIntParam("braid_accesslevel", 1);
  core->SetAccessLevel( accesslevel );
  datadir = config->GetStrParam("datadir", "./data_out");
  // _braid_SetVerbosity(core->GetCore(), 1);

  int worldrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &worldrank);
  if (worldrank == 0) {
    mkdir(datadir.c_str(), 0777);
    cout << "# Data directory: " << datadir << endl; 
  }
}

myBraidApp::~myBraidApp() {
  /* Delete the core, if drive() has been called */
  if (core->GetWarmRestart()) delete core;
}

int myBraidApp::getTimeStepIndex(double t, double dt){
  int ts = round(t / dt);
  return ts;
}


Vec myBraidApp::getStateVec(double time) {
  if (time != total_time) {
   printf("ERROR: getState not implemented yet for (t != final_time)\n\n");
   exit(1);
  }

  Vec x = NULL;
  braid_BaseVector ubase;
  myBraidVector *u;
  const double* state_ptr= NULL;
  _braid_UGetLast(core->GetCore(), &ubase);
  if (ubase != NULL) { // only true on last processor 
    u = (myBraidVector *)ubase->userVector;
    x = u->x;
  }
  return x;
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

  if (usepetscts) {
    /* -------------------------------------------------------------*/
    /* --- PETSC timestepper --- */
    /* -------------------------------------------------------------*/
    TSSetTime(ts_petsc, tstart);
    TSSetTimeStep(ts_petsc, tstop - tstart);
    TSSetStepNumber(ts_petsc, 0);
    TSSetMaxSteps(ts_petsc, 1);
    TSSolve(ts_petsc, u->x);
  } else {
    /* -------------------------------------------------------------*/
    /* --- my timestepper --- */
    /* -------------------------------------------------------------*/

    /* Evolve solution forward from tstart to tstop */
    mytimestepper->evolveFWD(tstart, tstop, u->x);
  }

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
  myBraidVector *u = new myBraidVector(nreal);

  /* Set the initial condition */
  hamiltonian->initialCondition(0,u->x);

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
  int done = 0;
  double t;

  /* Get time information */
  astatus.GetTIndex(&istep);
  astatus.GetT(&t);
  astatus.GetDone(&done);

  /* Don't print first time step. Something is fishy herre.*/
  if (t == 0.0) return 0;

  if (done && ufile != NULL && vfile != NULL) {

    
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
    // double err_norm, exact_norm;
    // Vec err;
    // Vec exact;
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
  myBraidVector *u = new myBraidVector(dim);

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

int myBraidApp::PreProcess(int iinit, double f_Re, double f_Im){
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
      hamiltonian->initialCondition(iinit, u->x);
    }
  }

  /* Open output files */
  if (accesslevel > 0) {
    char filename[255];
    sprintf(filename, "%s/out_u.iinit%04d.rank%04d.dat", datadir.c_str(),iinit, braidrank);
    ufile = fopen(filename, "w");
    sprintf(filename, "%s/out_v.iinit%04d.rank%04d.dat", datadir.c_str(), iinit, braidrank);
    vfile = fopen(filename, "w");
  }


  return 0; 
}



int myBraidApp::PostProcess(int iinit, double* f_Re, double* f_Im) {

  braid_BaseVector ubase;
  myBraidVector *u;
  int maxlevels = _braid_CoreElt(core->GetCore(), max_levels);

  /* If multilevel solve: Sweep over all points to access */
  if (maxlevels > 1) {
    _braid_CoreElt(core->GetCore(), done) = 1;
    _braid_FCRelax(core->GetCore(), 0);
  }

  /* Eval objective function for initial condition iinit */
  double obj_Re_local = 0.0;
  double obj_Im_local = 0.0;
  Vec finalstate = getStateVec(total_time); // this returns NULL for all but the last processors! 
  if (finalstate != NULL) {
    /* Compare to target gate */
    targetgate->apply(iinit, finalstate, obj_Re_local, obj_Im_local);
  }

  /* Set return value */
  *f_Re = obj_Re_local;
  *f_Im = obj_Im_local;

  /* Close output files */
  if (ufile != NULL) fclose(ufile);
  if (vfile != NULL) fclose(vfile);

  return 0;
}


double myBraidApp::Drive() { 
  
  int nreq = -1;
  double norm;

  core->Drive();
  core->GetRNorms(&nreq, &norm);

  // braid_printConvHistory(braid_core, "braid.out.log");

  return norm; 
}


/* ================================================================*/
/* Adjoint Braid App */
/* ================================================================*/
myAdjointBraidApp::myAdjointBraidApp(MPI_Comm comm_braid_, double total_time_, int ntime_, TS ts_, TimeStepper* mytimestepper_, Hamiltonian* ham_, Gate* targate_, Vec redgrad_, MapParam* config, BraidCore *Primalcoreptr_)
        : myBraidApp(comm_braid_, total_time_, ntime_, ts_, mytimestepper_, ham_, targate_, config) {

  /* Store the primal core */
  primalcore = Primalcoreptr_;

  /* Store reduced gradient */
  redgrad = redgrad_;

  /* Ensure that primal core stores all points */
  primalcore->SetStorage(0);

  /* Revert processor ranks for solving adjoint */
  core->SetRevertedRanks(1);
  // _braid_SetVerbosity(core->GetCore(), 1);

  int ndesign;
  VecGetSize(redgrad, &ndesign);
  mygrad = new double[ndesign];

}

myAdjointBraidApp::~myAdjointBraidApp() {
  delete [] mygrad;
}


const double* myAdjointBraidApp::getReducedGradientPtr(){

    const PetscScalar *grad_ptr;
    VecGetArrayRead(redgrad, &grad_ptr);

    return grad_ptr;
}

int myAdjointBraidApp::getPrimalIndex(int ts) { 
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

  if (usepetscts) {
    /* ------------------------------------------------------------------*/
    /* ---- PETSC time stepper ---- */
    /* ------------------------------------------------------------------*/

    /* Get primal state */
    int finegrid = 0;
    int tstop_id = getTimeStepIndex(tstop, total_time / ntime);
    int primaltimestep = ntime - tstop_id;
    braid_BaseVector ubaseprimal;
    myBraidVector *uprimal;
    Vec x;
    _braid_UGetVectorRef(primalcore->GetCore(), finegrid, primaltimestep, &ubaseprimal);
    if (ubaseprimal == NULL) printf("ubaseprimal is null!\n");
    uprimal = (myBraidVector*) ubaseprimal->userVector;
    VecDuplicate(uprimal->x, &x);
    VecCopy(uprimal->x, x);


    /* Solve forward while saving trajectory */
    TSDestroy(&ts_petsc);
    ierr = TSCreate(PETSC_COMM_WORLD,&ts_petsc);CHKERRQ(ierr);
    TSInit(ts_petsc, hamiltonian, ntime  , dt, total_time, x, &(u->x), &redgrad, false);

    ierr = TSSetSaveTrajectory(ts_petsc);CHKERRQ(ierr);
    ierr = TSTrajectorySetSolutionOnly(ts_petsc->trajectory, (PetscBool) true);
    ierr = TSTrajectorySetType(ts_petsc->trajectory, ts_petsc, TSTRAJECTORYMEMORY);

    TSSetTime(ts_petsc, total_time - tstop);
    TSSetTimeStep(ts_petsc, dt);

    TSSetStepNumber(ts_petsc, 0);
    TSSetMaxSteps(ts_petsc, 1);

    TSSetSolution(ts_petsc, x);

    TSSolve(ts_petsc, x);

    /* Set adjoint vars */ 
    if (!update_gradient) VecZeroEntries(redgrad);
    TSSetCostGradients(ts_petsc, 1, &u->x, &redgrad); CHKERRQ(ierr);

    /* Solve adjoint */
    TSSetTimeStep(ts_petsc, -dt);
    TSAdjointSolve(ts_petsc);

    VecDestroy(&x);

  } else {
    /* --------------------------------------------------------------------------*/
    /* --- New timestepper --- */
    /* --------------------------------------------------------------------------*/

    /* Get original time */
    double tstart_orig = total_time - tstart;
    double tstop_orig  = total_time - tstop;

    /* Get uprimal at tstop_orig */
    myBraidVector *uprimal_tstop;
    braid_BaseVector ubaseprimal_tstop;
    int tstop_orig_id  = getTimeStepIndex(tstop_orig, total_time/ntime);
    _braid_UGetVectorRef(primalcore->GetCore(), 0, tstop_orig_id, &ubaseprimal_tstop);
    if (ubaseprimal_tstop == NULL) printf("ubaseprimal_tstop is null!\n");
    uprimal_tstop  = (myBraidVector*) ubaseprimal_tstop->userVector;

    /* Reset gradient, if neccessary */
    if (!done) VecZeroEntries(redgrad);

    /* Evolve u backwards in time and update gradient */
    mytimestepper->evolveBWD(tstop_orig, tstart_orig, uprimal_tstop->x, u->x, redgrad, done);

  }


  return 0;  
}


braid_Int myAdjointBraidApp::Init(braid_Real t, braid_Vector *u_ptr) {

  /* Allocate the adjoint vector and set to zero */
  myBraidVector *u = new myBraidVector(2*hamiltonian->getDim());

  /* Reset the reduced gradient */
  VecZeroEntries(redgrad); 

  /* Set the differentiated objective function */
  if (t==0){

    /* Set derivative of objective function value */
    /* TODO: THIS IS BULLSHIT! MAKE SURE THAT PreProcess is called after this!! */
    double obj_bar = - 1./(hamiltonian->getDim() * hamiltonian->getDim());
    targetgate->apply_diff(0, u->x, 1.0, 1.0);
  }

  /* Return new vector to braid */
  *u_ptr = (braid_Vector) u;

  return 0;
}


int myAdjointBraidApp::PreProcess(int iinit, double f_Re_bar, double f_Im_bar){
/* If warm_restart: set adjoint initial condition here. Otherwise it's set in my_Init_Adj.
 * It can not be done here if drive() has not been called before, because the braid grid is allocated only at the beginning of drive() 
*/
  braid_BaseVector ubaseadjoint;
  myBraidVector *uadjoint;

  /* Set initial condition for adjoint: Derivative of objective function */
  if (core->GetWarmRestart()) {
    /* Get adjoint state at t=0.0 */
    _braid_UGetVectorRef(core->GetCore(), 0, 0, &ubaseadjoint);
    if (ubaseadjoint != NULL) {   // this is true only at the last processor
      uadjoint = (myBraidVector *) ubaseadjoint->userVector;

      /* Set derivative of objective function value */
      VecZeroEntries(uadjoint->x);
      targetgate->apply_diff(iinit, uadjoint->x, f_Re_bar, f_Im_bar);
    }
  }

  /* Reset the reduced gradient */
  VecZeroEntries(redgrad); 

  /* Open output files */
  if (accesslevel > 0) {
    char filename[255];
    sprintf(filename, "%s/out_uadj.iinit%04d.rank%04d.dat", datadir.c_str(),iinit, braidrank);
    ufile = fopen(filename, "w");
    sprintf(filename, "%s/out_vadj.iinit%04d.rank%04d.dat", datadir.c_str(),iinit, braidrank);
    vfile = fopen(filename, "w");
  }

  return 0;
}

int myAdjointBraidApp::PostProcess(int i, double* f_Re, double* f_Im) {

  int maxlevels;
  maxlevels = _braid_CoreElt(core->GetCore(), max_levels);

  /* Sweep over all points to collect the gradient */
  VecZeroEntries(redgrad);
  _braid_CoreElt(core->GetCore(), done) = 1;
  _braid_FCRelax(core->GetCore(), 0);

  /* Close output files */
  if (ufile != NULL) fclose(ufile);
  if (vfile != NULL) fclose(vfile);

  return 0;
}


// void evalObjective(braid_Core core, braid_App app, double *obj_ptr) {
//   braid_BaseVector ulast_base;
//   my_Vector *ulast;
//   double obj;

//   _braid_UGetLast(core, &ulast_base);
//   if (ulast_base == NULL) { 
//       printf("Error: can't get ulast from braid!\n"); 
//       exit(1); 
//   }
//   ulast = (my_Vector*) ulast_base->userVector;
//   app->hamiltonian->evalObjective(app->total_time, ulast->x, &obj);

//   *obj_ptr = obj;
// }