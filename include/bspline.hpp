#include "defs.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <cassert>
#pragma once

/* 
 * Discretization of the Controls. 
 * We use quadratic Bsplines a la Anders Peterson combined with carrier waves
 * Bspline basis functions have local support with width = 3*dtknot, 
 * where dtknot = T/(nsplines -2) is the time knot vector spacing.
 */
class ControlBasis{
    protected:
        int    nbasis;                    // number of basis functions
        double dtknot;                    // spacing of time knot vector    
        double *tcenter;                  // vector of basis function center positions
        double width;                     // support of each basis function (m*dtknot)

        /* Evaluate the bspline basis functions B_l(tau_l(t)) */
        double basisfunction(int id, double t);

    public:
        ControlBasis(int NBasis, double T);
        ~ControlBasis();

        /* Return the number of basis functions */
        int getNSplines() { return nbasis; };

        /* Evaluate the spline at time t using the coefficients coeff. */
        double evaluate(const double t, const std::vector<double>& coeff, const double ground_freq, std::vector<double>& carrier_freq, const ControlType controltype);

        /* Evaluates the derivative at time t, multiplied with fbar. */
        void derivative(const double t, double* coeff_diff, const double fbar, std::vector<double>& carrier_freq, const ControlType controltype);

        /* For debugging: evaluate spline number s at time t */
        double evalSpline_Re(int s, double t, const std::vector<double>& coeff, std::vector<double>& carrier_freq);
};
