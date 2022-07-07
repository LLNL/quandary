#include "defs.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <cassert>
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
    protected: 
        std::vector<double> onofftimes;  // Stores when transfer functions are active: They return their value only in [t0,t1] U [t2,t3] U ... and they return 0.0 otherwise (i.e. in [t1,t2] and [t3,t4], ... )
    public:
        TransferFunction();
        TransferFunction(std::vector<double> onofftimes);
        virtual ~TransferFunction();

        virtual double eval(double p, double time) =0;
        virtual double der(double p, double time) =0;

        // Checks whether the transferFunction is ON at this time (determined by the onofflist). Returns p if it is, and returns 0.0 otherwise.
        double isOn(double p, double time);

        // Pass a list of times points when the transfer is active.
        void storeOnOffTimes(std::vector<double>onofftimes_);
};

/* 
 * Transfer function that is constant: u(x) = const, u'(x) = 0.0
 */
class ConstantTransferFunction : public TransferFunction {
    double constant;
    public:
        ConstantTransferFunction();
        ConstantTransferFunction(double constant_);
        ConstantTransferFunction(double constant_, std::vector<double> onofftimes);
        ~ConstantTransferFunction();

        double eval(double x, double time) {return isOn(constant, time); }; 
        double der(double x, double time) {return 0.0; }; 
};

/*
 * Transfer function that is the identity u(x) = x, u'(x) = 1.0
 */
class IdentityTransferFunction : public TransferFunction {
    public:
        IdentityTransferFunction();
        IdentityTransferFunction(std::vector<double> onofftimes);
        ~IdentityTransferFunction();

        double eval(double x, double time) {return isOn(x, time); };
        double der(double x, double time) {return isOn(1.0, time); };
};

class SplineTransferFunction : public TransferFunction {
    protected:
        double knot_min;  // Lower bound for spline evaluation 
        double knot_max;  // Upper bound for spline evaluation
#ifdef WITH_FITPACK
        fitpackpp::BSplineCurve* spline_func;
#endif
    public:
        SplineTransferFunction(int order, std::vector<double>knots, std::vector<double>coeffs);
        SplineTransferFunction(int order, std::vector<double>knots, std::vector<double>coeffs, std::vector<double> onofftimes);
        ~SplineTransferFunction();
        // Evaluate the spline
        double eval(double p, double time);
        // Derivative
        double der(double p, double time);

        // Check if evaluation point is inside bounds [tmin, tmax]. Prints a warning otherwise. 
        void checkBounds(double p);
};



class CosineTransferFunction : public TransferFunction {
    protected:
        double freq;
        double amp;
    public:
        CosineTransferFunction(double amp, double freq);
        CosineTransferFunction(double amp, double freq, std::vector<double>onofftimes);
        ~CosineTransferFunction();
        // This is amp*cos(freq*t)
        double eval(double x, double time){ return isOn(amp*cos(freq*x), time); };
        double der(double x, double time) { return isOn(amp*freq*sin(freq*x), time); };
};

class SineTransferFunction : public TransferFunction {
    protected:
        double freq;
        double amp;
    public:
        SineTransferFunction(double amp, double freq);
        SineTransferFunction(double amp, double freq, std::vector<double> onofftimes);
        ~SineTransferFunction();
        // This is amp*sin(freq*t)
        double eval(double x, double time) { return isOn(amp*sin(freq*x), time); };
        double der(double x, double time) { return isOn(-1.0*amp*freq*cos(freq*x), time); };
};

