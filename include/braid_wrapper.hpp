#include "config.hpp"
#include "timestepper.hpp"
#include "hamiltonian.hpp"
#include "braid.hpp"
#include "util.hpp"

#pragma once

class myBraidVector {
  public: 
    Vec x;
    
    myBraidVector();
    myBraidVector(MPI_Comm comm, int dim);
    ~myBraidVector();
};

class myBraidApp : public BraidApp {
  protected: 
    int          ntime;             /* number of time steps */
    double       total_time;        /* total time  */
    TS           timestepper;       /* Petsc Time-stepper struct */
    Hamiltonian *hamiltonian;       /* Hamiltonian system */
    MPI_Comm comm_petsc;            /* Petsc's communicator */

    BraidCore *core;                /* Braid core for running PinT simulation */

  public:
    FILE *ufile;
    FILE *vfile;

  public:

  myBraidApp(MPI_Comm comm_braid_, MPI_Comm comm_petsc_, double total_time_, int ntime_, TS ts_, Hamiltonian* ham_, MapParam* config);
  ~myBraidApp();

    /* Dumps xbraid's convergence history to a file */
    int printConvHistory(const char* filename);

    int getTimeStepIndex(double t, double dt);

    /* Return the core */
    BraidCore *getCore();

    /* Apply one time step */
    virtual braid_Int Step(braid_Vector u_, braid_Vector ustop_,
                          braid_Vector fstop_, BraidStepStatus &pstatus);

    /* Compute residual: Does nothing. */
    braid_Int Residual(braid_Vector u_, braid_Vector r_,
                      BraidStepStatus &pstatus);

    /* Allocate a new vector in *v_ptr, which is a deep copy of u_. */
    braid_Int Clone(braid_Vector u_, braid_Vector *v_ptr);

    /* Allocate a new vector in *u_ptr and initialize it with an initial guess appropriate for time t. */
    virtual braid_Int Init(braid_Real t, braid_Vector *u_ptr);

    /* De-allocate the vector @a u_. */
    braid_Int Free(braid_Vector u_);

    /* Perform the operation: y_ = alpha * x_ + beta * @a y_. */
    braid_Int Sum(braid_Real alpha, braid_Vector x_, braid_Real beta,
                  braid_Vector y_);

    /* Compute in *norm_ptr an appropriate spatial norm of u_. */
    braid_Int SpatialNorm(braid_Vector u_, braid_Real *norm_ptr);

    /* @see braid_PtFcnAccess. */
    braid_Int Access(braid_Vector u_, BraidAccessStatus &astatus);

    /* @see braid_PtFcnBufSize. */
    virtual braid_Int BufSize(braid_Int *size_ptr, BraidBufferStatus &bstatus);

    /* @see braid_PtFcnBufPack. */
    virtual braid_Int BufPack(braid_Vector u_, void *buffer,
                              BraidBufferStatus &bstatus);

    /* @see braid_PtFcnBufUnpack. */
    virtual braid_Int BufUnpack(void *buffer, braid_Vector *u_ptr,
                                BraidBufferStatus &bstatus);

    /* Sets the initial condition if warm_restart (otherwise it is set in my_Init().
    * Can not be set here if !(warm_restart) because the braid_grid is created only when braid_drive() is called. 
    */
    virtual int SetInitialCondition();

    /* Postprocess. This is called inside run(), after braid_Drive.  */
    virtual int PostProcess();

    /* Sets the initial condition, then calls braid_drive. 
    * Return residual norm of last iteration.
    */
    double run();
};

/**
 * Adjoint braid App for solving adjoint eqations with xbraid.
 */
class myAdjointBraidApp : public myBraidApp {
 protected:
  BraidCore *primalcore;    /* pointer to primal core for accessing primal states */
  Vec   redgrad;            /* reduced gradient */

  public:

    myAdjointBraidApp(MPI_Comm comm_braid_, MPI_Comm comm_petsc_, double total_time_, int ntime_, TS ts_, Hamiltonian* ham_, Vec redgrad_, MapParam* config, BraidCore *Primalcoreptr_);
    ~myAdjointBraidApp();

    /* Get the storage index of primal (reversed) time point index of a certain time t, on the grid created with spacing dt  */
    int getPrimalIndex(int ts);

    /* Apply one adjoint time step */
    braid_Int Step(braid_Vector u_, braid_Vector ustop_, braid_Vector fstop_, BraidStepStatus &pstatus);

    /* Set adjoint initial condition */
    braid_Int Init(braid_Real t, braid_Vector *u_ptr);

    /* Set the adjoint initial condition (derivative of primal objective function) */
    int SetInitialCondition();

    int PostProcess();

};
