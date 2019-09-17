#include "timestepper.hpp"
#include "hamiltonian.hpp"
#include "braid.h"

#pragma once


/* Define the solution at one time step */
typedef struct _braid_Vector_struct {
    Vec x;      // solution
} my_Vector;


/* Define the braid application structure */
typedef struct _braid_App_struct
{
    TS      ts;       // Petsc Time-stepper struct
    int     ntime;    // number of time steps
    Hamiltonian *hamiltonian; 
    Vec   mu;         // reduced gradient 
    FILE *ufile;
    FILE *vfile;
    MPI_Comm comm_braid;
    MPI_Comm comm_petsc;
} XB_App;


/* 
 * Perform one time-step forward from tstart to tstop
 * In: app - XBraid's application struct
 *     ustop - state at tstop
 *     fstop - ?
 *     u     - state at tstart
 *     status - struct to query current time tstart / tstop 
 * Out: u - updated state  
 */
int my_Step(braid_App app, braid_Vector ustop, braid_Vector fstop, braid_Vector u, braid_StepStatus status);
int my_Step_adj(braid_App app, braid_Vector ustop, braid_Vector fstop, braid_Vector u, braid_StepStatus status);


/* 
 * Allocate and initialize a braid vector at time t
 * In: app - XBraid's application struct
 *     t   - current time
 * Out: u_ptr - Pointer to the newly allocated state vector
 */
int my_Init(braid_App app, double t, braid_Vector *u_ptr);
int my_Init_adj(braid_App app, double t, braid_Vector *u_ptr);


/* 
 * Create a copy of a braid vector v <- u
 * In: app - XBraid's application struct
 *     u   - state vector
 * Out: v_ptr - a pointer to the copy of u 
 */
int my_Clone(braid_App app, braid_Vector  u, braid_Vector *v_ptr);


/* 
 * Deallocate a braid vector 
 */
int my_Free(braid_App app, braid_Vector u);


/* 
 * Sum two vectors (AXPBY): y = alpha * x + beta * y 
 */
int my_Sum(braid_App app, double alpha, braid_Vector x, double beta, braid_Vector y);


/*
 * Access a state vector
 * In: app - XBraid's application struct
 *     u   - state at current time
 *     astatus - struct to query current time 
 */
int my_Access(braid_App app, braid_Vector u, braid_AccessStatus astatus);



/* 
 * Compute the norm of a state vector
 * In: u - current state
 * Out: norm_ptr - pointer to the computed norm value 
 */
int my_SpatialNorm(braid_App app, braid_Vector u, double *norm_ptr);


/* 
 * Return the size of a state vector
 */
int my_BufSize(braid_App app, int *size_ptr, braid_BufferStatus bstatus);
int my_BufSize_adj(braid_App app, int *size_ptr, braid_BufferStatus bstatus);



/* 
 * Pack a braid vector into a buffer 
 */
int my_BufPack(braid_App app, braid_Vector u, void *buffer, braid_BufferStatus bstatus);
int my_BufPack_adj(braid_App app, braid_Vector u, void *buffer, braid_BufferStatus bstatus);


/*
 * Unpack a braid vector from a buffer
 */
int my_BufUnpack(braid_App app, void *buffer, braid_Vector *u_ptr, braid_BufferStatus status);
int my_BufUnpack_adj(braid_App app, void *buffer, braid_Vector *u_ptr, braid_BufferStatus status);


/*
 * Evaluate the objective function at time t 
 */
int my_ObjectiveT(braid_App app, braid_Vector u, braid_ObjectiveStatus ostatus, double *objectiveT_ptr);

/*
 * Derivative of the objectiveT function 
 */
int my_ObjectiveT_diff(braid_App app, braid_Vector u, braid_Vector u_bar, braid_Real F_bar, braid_ObjectiveStatus ostatus);

/*
 * Derivative of my_Step
 */
int my_Step_diff(braid_App app, braid_Vector ustop, braid_Vector u, braid_Vector ustop_bar, braid_Vector u_bar, braid_StepStatus status);

/*
 * Set the gradient to zero
 */
int my_ResetGradient(braid_App app);