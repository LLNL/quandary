#include "config.hpp"
#include "timestepper.hpp"
#include "hamiltonian.hpp"
#include "braid.hpp"
#include "util.hpp"
#include "gate.hpp"

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
    TS           ts_petsc;       /* Petsc Time-stepper struct */
    MPI_Comm comm_petsc;            /* Petsc's communicator */
    MPI_Comm comm_braid;            /* Braid's communicator */
    double       total_time;        /* total time  */
    Gate         *targetgate;
    TimeStepper  *mytimestepper;

    BraidCore *core;                /* Braid core for running PinT simulation */

    int usepetscts;

  public:
    Hamiltonian *hamiltonian;       /* Hamiltonian system */
    FILE *ufile;
    FILE *vfile;

  public:

  myBraidApp(MPI_Comm comm_braid_, MPI_Comm comm_petsc_, double total_time_, int ntime_, TS ts_petsc_, TimeStepper* mytimestepper_, Hamiltonian* ham_, Gate* targate_, MapParam* config);
  ~myBraidApp();

    /* Dumps xbraid's convergence history to a file */
    int printConvHistory(const char* filename);

    int getTimeStepIndex(double t, double dt);

    /* Return read-only state at a certain time point. CURRENTLY ONLY VALID FOR time == total_time */
    const double* getStateRead(double time);

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

    /* Sets the initial condition with index i if warm_restart (otherwise it is set in my_Init() */
    virtual int PreProcess(int i);

    /* Performs one last FRelax, evaluates the objective function value for init i */
    virtual int PostProcess(int i, double* f);

    /* Call braid_drive and postprocess. Return braid norm */
    double Drive();

};

/**
 * Adjoint braid App for solving adjoint eqations with xbraid.
 */
class myAdjointBraidApp : public myBraidApp {
  protected:
    BraidCore *primalcore;    /* pointer to primal core for accessing primal states */
    Vec   redgrad;            /* reduced gradient */

  private:
    double* mygrad; /* auxiliary vector used to MPI_Allreduce the gradient */

  public:

    myAdjointBraidApp(MPI_Comm comm_braid_, MPI_Comm comm_petsc_, double total_time_, int ntime_, TS ts_petsc_,TimeStepper* mytimestepper_, Hamiltonian* ham_, Gate* targate_, Vec redgrad_, MapParam* config, BraidCore *Primalcoreptr_);
    ~myAdjointBraidApp();

    /* Get pointer to reduced gradient. READ ONLY!! */
    const double* getReducedGradientPtr();

    /* Get the storage index of primal (reversed) time point index of a certain time t, on the grid created with spacing dt  */
    int getPrimalIndex(int ts);

    /* Apply one adjoint time step */
    braid_Int Step(braid_Vector u_, braid_Vector ustop_, braid_Vector fstop_, BraidStepStatus &pstatus);

    /* Set adjoint initial condition */
    braid_Int Init(braid_Real t, braid_Vector *u_ptr);

    /* Sets the adjoint initial condition if warmrestart (derivative of primal objective function) */
    int PreProcess(int i);

    /* Performs one last FRelax and MPI_Allreduce the gradient. */
    int PostProcess(int i, double* f);

};
