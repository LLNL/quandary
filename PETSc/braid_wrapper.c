#include "two_oscillators_lib.cpp"
#include "braid.h"


/* Define the solution at one time step */
typedef struct _braid_Vector_struct {
    Vec x;
   
} my_Vector;


/* Define the braid application structure */
typedef struct _braid_App_struct
{
   int nreal;
} my_App;




int my_Step(braid_App        app,
        braid_Vector     ustop,
        braid_Vector     fstop,
        braid_Vector     u,
        braid_StepStatus status)
{
    double tstart, tstop;

    /* Grab current time from XBraid */
    braid_StepStatusGetTstartTstop(status, &tstart, &tstop);

    return 0;
}



/* Allocate and initialize a braid vector */
int my_Init(braid_App     app,
        double        t,
        braid_Vector *u_ptr)
{
    /* Allocate the struct */
    my_Vector* u = (my_Vector*) malloc(sizeof(my_Vector));

    /* Allocate the Petsc Vector */
    VecCreateSeq(PETSC_COMM_SELF,app->nreal,&(u->x));

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

    return 0;
}



/* Free a braid vector */
int my_Free(braid_App    app,
        braid_Vector u)
{
    VecDestroy(&u->x);
    return 0;
}


/* Sum AXPBY: y = alpha * x + beta * y */
int my_Sum(braid_App    app,
       double       alpha,
       braid_Vector x,
       double       beta,
       braid_Vector y)
{

    VecAXPBY(y->x, alpha, beta, x->x);

    return 0;
}



int my_Access(braid_App           app,
          braid_Vector        u,
          braid_AccessStatus  astatus)
{
    return 0;
}


int my_SpatialNorm(braid_App     app,
               braid_Vector  u,
               double       *norm_ptr)
{
    double norm;
    VecNorm(u->x, NORM_2, &norm);
   return 0;
}



int my_BufPack(braid_App           app,
           braid_Vector        u,
           void                *buffer,
           braid_BufferStatus  status)
{
    return 0;
}


int my_BufUnpack(braid_App           app,
             void                *buffer,
             braid_Vector        *u_ptr,
             braid_BufferStatus  status)
{
    return 0;
}