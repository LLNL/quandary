#include "config.hpp"
#include "timestepper.hpp"
#include "hamiltonian.hpp"
#include "braid.hpp"

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
    int     ntime;    // number of time steps
    double  total_time;
    TS           timestepper;       // Petsc Time-stepper struct
    Hamiltonian *hamiltonian; 
    // Vec   mu;                       // reduced gradient 
    // FILE *ufile;
    // FILE *vfile;
    // MPI_Comm comm_braid;
    // MPI_Comm comm_petsc;

    int monitor;
    BraidCore *core;  // Braid core for running PinT simulation */

  public:

    myBraidApp(MPI_Comm comm_braid_, MPI_Comm comm_petsc_, double total_time_, int ntime_, TS ts_, Hamiltonian* ham_, MapParam* config);
    ~myBraidApp();

    /* Dumps xbraid's convergence history to a file */
    braid_Int printConvHistory(BraidCore core, const char* filename);


    /* Return the time point index of a certain time t, on the grid created with spacing dt  */
    braid_Int getTimeStepIndex(double t, double dt);

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

  /* Compute in @a *norm_ptr an appropriate spatial norm of @a u_. */
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

  /* Set the initial condition */
  virtual braid_Int SetInitialCondition();

  /* Run Braid drive, return norm */
  double run();


};

/**
 * Adjoint braid App for solving adjoint eqations with xbraid.
 */
class myAdjointBraidApp : public myBraidApp {
 protected:
  BraidCore *primalcore; /* pointer to primal core for accessing primal states */

  public:

    myAdjointBraidApp(MPI_Comm comm_braid_, MPI_Comm comm_petsc_, double total_time_, int ntime_, TS ts_, Hamiltonian* ham_, MapParam* config, BraidCore *Primalcoreptr_);
    ~myAdjointBraidApp();

  /* Get the storage index of primal (reversed) time point index of a certain time t, on the grid created with spacing dt  */
  int getPrimalIndex(int ts);

  /* Apply one adjoint time step */
  braid_Int Step(braid_Vector u_, braid_Vector ustop_, braid_Vector fstop_,
                 BraidStepStatus &pstatus);

  /* Set the adjoint initial condition (derivative of primal objective function)
   */
  int SetInitialCondition();

};
