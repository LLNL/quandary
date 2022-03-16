#include "defs.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include "BSplineCurve.h"
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
        std::vector<double> carrier_freq; // Frequencies of the carrier waves

        /* Evaluate the bspline basis functions B_l(tau_l(t)) */
        double basisfunction(int id, double t);

    public:
        ControlBasis(int NBasis, double T, std::vector<double> carrier_freq_);
        ~ControlBasis();

        /* Return the number of basis functions */
        int getNSplines() { return nbasis; };
        int getNCarrierwaves() { return carrier_freq.size(); };

        /* Evaluate the spline at time t using the coefficients coeff. */
        double evaluate(const double t, const std::vector<double>& coeff, const double ground_freq, const ControlType controltype);

        /* Evaluates the derivative at time t, multiplied with fbar. */
        void derivative(const double t, double* coeff_diff, const double fbar, const ControlType controltype);
};


/* 
 * Class to represent transfer functions that act on the controls: evaluate u(p(t)) and v(q(t)).
 * Default: u = v = identity. 
 * Otherwise: u and v are splines, read from the python interface. */
class TransferFunction{
    public:
        TransferFunction();
        ~TransferFunction();

        // Default: these are the identity functions
        double eval_re(double p);
        double eval_im(double q);
        // Derivative
        double der_re(double p);
        double der_im(double q);
};

class SplineTransferFunction : public TransferFunction {
    protected:
        fitpackpp::BSplineCurve* transfer_func_re;
        fitpackpp::BSplineCurve* transfer_func_im;
    public:
        SplineTransferFunction(int order_re, std::vector<double>knots_re, std::vector<double>coeffs_re, int order_im, std::vector<double>knots_im, std::vector<double>coeffs_im);
        ~SplineTransferFunction();

        // Evaluate the splines
        double eval_re(double p);
        double eval_im(double q);

        // Derivative
        double der_re(double p);
        double der_im(double q);
};