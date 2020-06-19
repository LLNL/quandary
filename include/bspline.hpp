#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#pragma once

/* 
 * Discretization of the Controls. 
 * We use quadratic Bsplines a la Anders Peterson combined with carrier waves
 * Bspline basis functions have local support with width = 3*dtknot, 
 * where dtknot = T/(nsplines -2) is the time knot vector spacing.
 */
class ControlBasis{
    public:
        enum ControlType {RE, IM, LAB};   // Type of control: Rotating frame Real p(t), rotating frame imaginary q(t), or Lab frame f(t)

    protected:
        int    nbasis;                    // number of basis functions
        double dtknot;                    // spacing of time knot vector    
        double *tcenter;                  // vector of basis function center positions
        double width;                     // support of each basis function (m*dtknot)
        std::vector<double> carrier_freq; // Frequencies of the carrier waves

        /* Evaluate the bspline basis functions B_l(tau_l(t)) */
        double basisfunction(int id, double t);

    public:
        ControlBasis(int NBasis, double T, std::vector<double> carrier_freq_);
        ~ControlBasis();

        /* Evaluate the spline at time t using the coefficients coeff. */
        double evaluate(const double t, const std::vector<double>& coeff, const double ground_freq, const ControlType controltype);

        /* Evaluates the derivative at time t, multiplied with fbar. */
        void derivative(const double t, double* coeff_diff, const double fbar, const ControlType controltype);
};