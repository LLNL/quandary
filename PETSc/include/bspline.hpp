#include <stdio.h>
#include <math.h>
#pragma once

/* 
 * Implements quadratic Bspline a la Anders Peterson
 * Basis function have local support with width = 3*dtknot, 
 * where dtknot = T/(nsplines -2) is the time knot vector spacing.
 */
class Bspline{
    int nsplines;     // number of splines basis functions

    double dtknot;    // spacing of time knot vector    
    double width;     // support of each spline basis function (d*dtknot)
    double *tcenter;  // vector of basis function center positions

    public:
        /* Constructor */
        Bspline(int NSplines, double T);

        /* Destructor */
        ~Bspline();

        /* 
         * Evaluate the spline at time t using the coefficients coeff.
         * (only the first nspline elements in coeff will be used!)
         */
        double evaluate(double t, double* coeff);

};