#include "braid_wrapper.hpp"


int my_Step(braid_App    app,
        braid_Vector     ustop,
        braid_Vector     fstop,
        braid_Vector     u,
        braid_StepStatus status)
{
    double tstart, tstop;

    /* Grab current time from XBraid and pass it to Petsc time-stepper */
    braid_StepStatusGetTstartTstop(status, &tstart, &tstop);
    TSSetTime(app->ts, tstart);
    TSSetTimeStep(app->ts, tstop - tstart);

    /* Pass the curent state to the Petsc time-stepper */
    TSSetSolution(app->ts, u->x);

    /* Take a step */
    TSStep(app->ts);

    return 0;
}



int my_Init(braid_App     app,
        double        t,
        braid_Vector *u_ptr)
{

    int nreal = 2 * app->petsc_app->nvec;

    /* Allocate a new braid vector */
    my_Vector* u = (my_Vector*) malloc(sizeof(my_Vector));

    /* Allocate the Petsc Vector */
    VecCreateSeq(PETSC_COMM_SELF,nreal,&(u->x));

    /* Set initial condition at t=0.0 */
    // if (t == 0.0)
    {
        InitialConditions(u->x, app->petsc_app);
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

    for (int i = 0; i< 2 * app->petsc_app->nvec; i++)
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
    Vec error;
    Vec exact;

    /* Get time information */
    braid_AccessStatusGetTIndex(astatus, &istep);
    braid_AccessStatusGetT(astatus, &t);

    if (t == 0.0) return 0;

    /* Get exact solution */
    VecDuplicate(u->x,&exact);
    ExactSolution(t,exact,app->petsc_app->w);

    /* Compute relative error norm */
    VecDuplicate(u->x,&error);
    VecWAXPY(error,-1.0,u->x, exact);
    VecNorm(error, NORM_2,&err_norm);
    VecNorm(exact, NORM_2,&exact_norm);
    err_norm = err_norm / exact_norm;

    const PetscScalar *x_ptr, *exact_ptr;
    VecGetArrayRead(u->x, &x_ptr);
    VecGetArrayRead(exact, &exact_ptr);

    /* Screen Output of error norm */
    if (istep == app->ntime){
        printf("Last step: ");
        printf("%5d  %1.5f  x[1] = %1.14e  exact[1] = %1.14e  err = %1.14e \n",istep,(double)t, x_ptr[1], exact_ptr[1], err_norm);
    } 

    /* Write numeric and analytic solution to files */
    fprintf(app->ufile,  "%.2f  ", t);
    fprintf(app->vfile,  "%.2f  ", t);
    fprintf(app->sufile, "%.2f  ", t);
    fprintf(app->svfile, "%.2f  ", t);
    for (int i = 0; i < 2*app->petsc_app->nvec; i++)
    {

      if (i < app->petsc_app->nvec) // real part
      {
        fprintf(app->ufile, "%1.14e  ", x_ptr[i]);  
        fprintf(app->sufile, "%1.14e  ", exact_ptr[i]);
      }  
      else  // imaginary part
      {
        fprintf(app->vfile, "%1.14e  ", x_ptr[i]); 
        fprintf(app->svfile, "%1.14e  ", exact_ptr[i]);
      }
      
    }
    fprintf(app->ufile, "\n");
    fprintf(app->vfile, "\n");
    fprintf(app->sufile, "\n");
    fprintf(app->svfile, "\n");


    VecDestroy(&error);
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

    *size_ptr = 2 * app->petsc_app->nvec * sizeof(double);
    return 0;
}


int my_BufPack(braid_App       app,
           braid_Vector        u,
           void                *buffer,
           braid_BufferStatus  bstatus)
{
    const PetscScalar *x_ptr;
    double* dbuffer = (double*) buffer;


    /* Get read access to the Petsc Vector */
    VecGetArrayRead(u->x, &x_ptr);

    /* Copy the values into the buffer */
    dbuffer[0] = 0.0;
    for (int i=0; i < 2*app->petsc_app->nvec; i++)
    {
        dbuffer[i] = x_ptr[i];
    }

    int size = 2 * app->petsc_app->nvec * sizeof(double);
    braid_BufferStatusSetSize(bstatus, size);

    return 0;
}


int my_BufUnpack(braid_App        app,
             void                *buffer,
             braid_Vector        *u_ptr,
             braid_BufferStatus   status)
{
    double* dbuffer = (double*) buffer;


    /* Create a new vector */
    braid_Vector u;
    my_Init(app, 0.0, &u);

    /* Get write access to the Petsc Vector */
    PetscScalar *x_ptr;
    VecGetArray(u->x, &x_ptr);

    /* Copy buffer into the vector */
    for (int i=0; i < 2* app->petsc_app->nvec; i++)
    {
        x_ptr[i] = dbuffer[i];
    }

    /* Restore Petsc's vector */
    VecRestoreArray(u->x, &x_ptr);

    /* Pass vector to XBraid */
    *u_ptr = u;
    
    return 0;
}

