#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#pragma once

/* 
 * Implements quadratic Bspline a la Anders Peterson, including carrier waves
 * Basis function have local support with width = 3*dtknot, 
 * where dtknot = T/(nsplines -2) is the time knot vector spacing.
 */
class Bspline{

    protected:
        int    nbasis;     // number of basis functions
        double dtknot;     // spacing of time knot vector    
        double *tcenter;   // vector of basis function center positions
        double width;      // support of each basis function (m*dtknot)
        std::vector<double> carrier_freq; // Frequencies of the carrier waves

        /* Evaluate b_k(tau_k(t)) */
        double basisfunction(int id, double t);

    public:
        /* Constructor */
        Bspline(int NBasis, double T, std::vector<double> carrier_freq_);

        /* Destructor */
        ~Bspline();

        /* Evaluate the spline at time t using the coefficients coeff. */
        double evaluate(double t, double* coeff);

        /* Evaluates the derivative at time t, multiplied with fbar. */
        void derivative(double t, double* coeff_diff, double fbar);
};