#include "config.hpp"
#include "timestepper.hpp"
#include "mastereq.hpp"
#include "braid.hpp"
#include "util.hpp"
#include "gate.hpp"
#include <iostream> 
#include <sys/stat.h> 

#pragma once

// define as extern here, they are needed for penalty integral term, implemented in optimproble.cpp
extern double objectiveT(MasterEq* mastereq, ObjectiveType objective_type, const std::vector<int>& obj_oscilIDs, const Vec state, const Vec rho_t0, Gate* targetgate);
extern void objectiveT_diff(MasterEq* mastereq, ObjectiveType objective_type, const std::vector<int>& obj_oscilIDs, Vec state, Vec state_bar, const Vec rho_t0, const double obj_bar, Gate* targetgate);

class myBraidVector {
  public: 
    Vec x;
    
    myBraidVector();
    myBraidVector(int dim);
    ~myBraidVector();
};

class myBraidApp : public BraidApp {
  protected: 
    TS           ts_petsc;        /* Petsc Time-stepper struct */
    TimeStepper  *mytimestepper;  /* My new time-stepper */
    BraidCore *core;                /* Braid core for running PinT simulation */

    /* output stuff */
    FILE *ufile;
    FILE *vfile;
    std::vector<FILE *>expectedfile;
    std::vector<FILE *>populationfile;
    std::vector<std::vector<std::string> > outputstr; // List of outputs for each oscillator

    /* MPI stuff */
    bool usepetscts;
    int mpirank_petsc;
    int mpirank_braid;
    int mpirank_world;

    VecScatter scat;    /* Petsc's scatter context to communicate a state across petsc's cores */
    Vec xseq;           /* A sequential vector for IO. */


  public:
    MPI_Comm comm_braid;            /* Braid's communicator */
    int          ntime;             /* number of time steps */
    int          accesslevel;
    std::string  datadir;           /* Name of output data directory */
    double       total_time;        /* total time  */
    MasterEq *mastereq;             /* Master equation */

    /* Stuff for evaluating penalty term objective function */
    ObjectiveType objective_type;
    std::vector<int> obj_oscilIDs;
    double Jbar;
    double penalty_exp;
    double penalty_coeff;
    double penalty_integral;

  public:

  myBraidApp(MPI_Comm comm_braid_, double total_time_, int ntime_, TS ts_petsc_, TimeStepper* mytimestepper_, MasterEq* ham_, MapParam* config);
  ~myBraidApp();

    /* Dumps xbraid's convergence history to a file */
    int printConvHistory(const char* filename);

    int getTimeStepIndex(const double t,const  double dt);

    /* Return  state vector at a certain time point. CURRENTLY ONLY VALID FOR time == total_time. Be careful: might return NULL! */
    Vec getStateVec(const double time);

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

    /* Pass initial condition to braid, open output files*/
    virtual void PreProcess(int iinit, const Vec rho_t0, double Jbar);

    /* Performs one last FRelax. Returns state at last time step or NULL if not stored on this processor */
    virtual Vec PostProcess();

    /* Call braid_drive and postprocess. Return braid norm */
    double Drive();

    /* Initialize braids time grids */
    void InitGrids();

    /* Pass the initial condition rho_t0 to braid at t=0 */
    void setInitCond(const Vec rho_t0);
};

/**
 * Adjoint braid App for solving adjoint eqations with xbraid.
 */
class myAdjointBraidApp : public myBraidApp {
  protected:
    BraidCore *primalcore;    /* pointer to primal core for accessing primal states */
  
  public:
    Vec        redgrad;       /* reduced gradient */

  private:
    double* mygrad; /* auxiliary vector used to MPI_Allreduce the gradient */

  public:

    myAdjointBraidApp(MPI_Comm comm_braid_, double total_time_, int ntime_, TS ts_petsc_,TimeStepper* mytimestepper_, MasterEq* ham_, MapParam* config, BraidCore *Primalcoreptr_);
    ~myAdjointBraidApp();

    /* Get the storage index of primal (reversed) time point index of a certain time t, on the grid created with spacing dt  */
    int getPrimalIndex(int ts);

    /* Apply one adjoint time step */
    braid_Int Step(braid_Vector u_, braid_Vector ustop_, braid_Vector fstop_, BraidStepStatus &pstatus);

    /* Set adjoint initial condition */
    braid_Int Init(braid_Real t, braid_Vector *u_ptr);

    /* Pass initial condition to braid, reset gradient, open output files */
    virtual void PreProcess(int iinit, const Vec rho_t0, double Jbar);

    /* Performs one last FRelax and MPI_Allreduce the gradient. */
    Vec PostProcess();

};
