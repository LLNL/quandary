#include "defs.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#ifdef WITH_FITPACK
#include "BSplineCurve.h"
#endif
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
 * Abstract class to represent transfer functions that act on the controls: evaluate u(p(t)), or v(q(t)).
 * Default: u = v = IdentityTransferFunctions. 
 * Otherwise: u=v are splineTransferFunction, read from the python interface. */
class TransferFunction{
    public:
        TransferFunction();
        virtual ~TransferFunction();

        virtual double eval(double p) =0;
        virtual double der(double p) =0;
};

class IdentityTransferFunction : public TransferFunction {
    public:
        IdentityTransferFunction();
        ~IdentityTransferFunction();
        // This is the identity function
        double eval(double p);
        // Derivative = 1.0
        double der(double p);
};

class SplineTransferFunction : public TransferFunction {
    protected:
#ifdef WITH_FITPACK
        fitpackpp::BSplineCurve* transfer_func;
#endif
    public:
        SplineTransferFunction(int order, std::vector<double>knots, std::vector<double>coeffs);
        ~SplineTransferFunction();
        // Evaluate the spline
        double eval(double p);
        // Derivative
        double der(double p);
};


class CosineTransferFunction : public TransferFunction {
    protected:
        double freq;
        double amp;
    public:
        CosineTransferFunction(double amp, double freq);
        ~CosineTransferFunction();
        // This is amp*cos(freq*t)
        double eval(double p);
        double der(double p);
};

class SineTransferFunction : public TransferFunction {
    protected:
        double freq;
        double amp;
    public:
        SineTransferFunction(double amp, double freq);
        ~SineTransferFunction();
        // This is amp*sin(freq*t)
        double eval(double p);
        double der(double p);
};

