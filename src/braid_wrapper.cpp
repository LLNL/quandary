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
  core->SetStorage(0);  // store all 

}

myBraidApp::~myBraidApp() {
  /* Delete the core, if drive() has been called */
  if (core->GetWarmRestart()) delete core;
}

BraidCore* myBraidApp::getCore() { return core; }

braid_Int myBraidApp::printConvHistory(BraidCore core, const char* filename){ return 0;}
braid_Int myBraidApp::getTimeStepIndex(double t, double dt){return 0;}

braid_Int myBraidApp::Step(braid_Vector u_, braid_Vector ustop_, braid_Vector fstop_, BraidStepStatus &pstatus){
    double tstart, tstop;
    int tindex;
    int done;

    /* Cast input u to class definition */
    myBraidVector *u = (myBraidVector *)u_;


    /* Grab current time from XBraid and pass it to Petsc time-stepper */
    pstatus.GetTstartTstop(&tstart, &tstop);
    pstatus.GetTIndex(&tindex);
    // pstatus.GetDone(&done); // TODO: Implement C++ interface for braid_StatusGetDone().
  
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
braid_Int myBraidApp::Clone(braid_Vector u_, braid_Vector *v_ptr){ return 0; }
braid_Int myBraidApp::Init(braid_Real t, braid_Vector *u_ptr){ return 0; }
braid_Int myBraidApp::Free(braid_Vector u_){ return 0; }
braid_Int myBraidApp::Sum(braid_Real alpha, braid_Vector x_, braid_Real beta, braid_Vector y_){ return 0; }
braid_Int myBraidApp::SpatialNorm(braid_Vector u_, braid_Real *norm_ptr){ return 0; }
braid_Int myBraidApp::Access(braid_Vector u_, BraidAccessStatus &astatus){ return 0; }
braid_Int myBraidApp::BufSize(braid_Int *size_ptr, BraidBufferStatus &bstatus){ return 0; }
braid_Int myBraidApp::BufPack(braid_Vector u_, void *buffer, BraidBufferStatus &bstatus){ return 0; }
braid_Int myBraidApp::BufUnpack(void *buffer, braid_Vector *u_ptr, BraidBufferStatus &bstatus){ return 0; }
braid_Int myBraidApp::SetInitialCondition(){ return 0; }
double run() { return 0.0; }



/* ================================================================*/
/* Adjoint Braid App */
/* ================================================================*/
myAdjointBraidApp::myAdjointBraidApp(MPI_Comm comm_braid_, MPI_Comm comm_petsc_, double total_time_, int ntime_, TS ts_, Hamiltonian* ham_, MapParam* config, BraidCore *Primalcoreptr_)
        : myBraidApp(comm_braid_, comm_petsc_, total_time_, ntime_, ts_, ham_, config) {

  primalcore = Primalcoreptr_;

  primalcore->SetStorage(0);
  core->SetRevertedRanks(1);
}

myAdjointBraidApp::~myAdjointBraidApp() {}

int myAdjointBraidApp::getPrimalIndex(int ts) { return 0;}
braid_Int myAdjointBraidApp::Step(braid_Vector u_, braid_Vector ustop_, braid_Vector fstop_, BraidStepStatus &pstatus) {return 0;}
int myAdjointBraidApp::SetInitialCondition() {return 0;}

