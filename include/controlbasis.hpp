#include "defs.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <cassert>
#include "util.hpp"
#pragma once


/* Abstract base class */
class ControlBasis {
    protected:
        int nparams;            // number of parameters that define the controls 
        double tstart;         // Interval [tstart,tstop] where this control basis is applied in
        double tstop;           
        int skip;              // Constant to skip to the starting location for this basis inside the (global) control vector. 
        ControlType controltype;
        bool enforceZeroBoundary;   // Flag to determine whether controls should start and end at zero.

    public: 
        ControlBasis();
        ControlBasis(int nparams_, double tstart, double tstop, bool enforceZeroBoundary);
        virtual ~ControlBasis();

        int getNparams() {return nparams; };
        double getTstart() {return tstart; };
        double getTstop() {return tstop; };
        ControlType getType() {return controltype;};
        void setSkip(int skip_) {skip = skip_;};

        virtual int getNSplines() {return 0;};

        /* Default: do nothing. For some control parameterizations, this can be used to enforce that the controls start and end at zero. E.g. the Splines will overwrite the parameters x of the first and last two splines by zero, so that the splines start and end at zero. */
        virtual void enforceBoundary(double* x, int carrier_id) {};

        /* Evaluate the Basis(alpha, t) at time t using the coefficients coeff. */
        virtual void evaluate(const double t, const std::vector<double>& coeff, int carrier_freq_id, double* Blt1, double*Blt2) = 0;

        /* Evaluates the derivative at time t, multiplied with fbar. */
        virtual void derivative(const double t, const std::vector<double>& coeff, double* coeff_diff, const double valbar1, const double valbar2, int carrier_freq_id)= 0;
};

/* 
 * Discretization of the Controls using quadratic Bsplines ala Anders Petersson
 * Bspline basis functions have local support with width = 3*dtknot, 
 * where dtknot = T/(nsplines -2) is the time knot vector spacing.
 */
class BSpline2nd : public ControlBasis {
    protected:
        int nsplines;                     // Number of splines
        double dtknot;                    // spacing of time knot vector    
        double *tcenter;                  // vector of basis function center positions
        double width;                     // support of each basis function (m*dtknot)

        /* Evaluate the bspline basis functions B_l(tau_l(t)) */
        double basisfunction(int id, double t);

    public:
        BSpline2nd(int nsplines, double tstart, double tstop, bool enforceZeroBoundary);
        ~BSpline2nd();

        int getNSplines() {return nsplines;};

        /* Sets the first and last two spline coefficients in x to zero, so that the controls start and end at zero */
        void enforceBoundary(double* x, int carrier_id);

        /* Evaluate the spline at time t using the coefficients coeff. */
        void evaluate(const double t, const std::vector<double>& coeff, int carrier_freq_id, double* Blt1_ptr, double* Blt2_ptr);

        /* Evaluates the derivative at time t, multiplied with fbar. */
        void derivative(const double t, const std::vector<double>& coeff, double* coeff_diff, const double valbar1, const double valbar2, int carrier_freq_id);
};

/* 
 * Amplitude is parameterized by Bsplines, phase is time-independent.
 * Discretization of the Controls using quadratic Bsplines ala Anders Petersson
 * Bspline basis functions have local support with width = 3*dtknot, 
 * where dtknot = T/(nsplines -2) is the time knot vector spacing.
 */
class BSpline2ndAmplitude : public ControlBasis {
    protected:
        int nsplines;                     // Number of splines
        double dtknot;                    // spacing of time knot vector    
        double *tcenter;                  // vector of basis function center positions
        double width;                     // support of each basis function (m*dtknot)
        double scaling;                   // scaling for the phase

        /* Evaluate the bspline basis functions B_l(tau_l(t)) */
        double basisfunction(int id, double t);

    public:
        BSpline2ndAmplitude(int nsplines, double scaling, double tstart, double tstop, bool enforceZeroBoundary);
        ~BSpline2ndAmplitude();

        int getNSplines() {return nsplines;};

        /* Sets the first and last two spline coefficients in x to zero, so that the controls start and end at zero */
        void enforceBoundary(double* x, int carrier_id);

        /* Evaluate the spline at time t using the coefficients coeff. */
        void evaluate(const double t, const std::vector<double>& coeff, int carrier_freq_id, double* Blt1_ptr, double* Blt2_ptr);

        /* Evaluates the derivative at time t, multiplied with fbar. */
        void derivative(const double t, const std::vector<double>& coeff, double* coeff_diff, const double valbar1, const double valbar2, int carrier_freq_id);
};

/* 
 * Parameterization of the controls using step function with constant amplitude and variable width 
 */
class Step : public ControlBasis {
    protected:
        double step_amp1;
        double step_amp2;
        double tramp;

    public: 
        Step(double step_amp1_, double step_amp2_, double t0, double t1, double tramp, bool enforceZeroBoundary);
        ~Step();

       /* Evaluate the spline at time t using the coefficients coeff. */
        void evaluate(const double t, const std::vector<double>& coeff, int carrier_freq_id, double* Blt1, double*Blt2);

        /* Evaluates the derivative at time t, multiplied with fbar. */
        void derivative(const double t, const std::vector<double>& coeff, double* coeff_diff, const double valbar1, const double valbar2, int carrier_freq_id);
};

/* 
 * Discretization of the Controls using piece-wise constant Bsplines
 * Bspline basis functions have local support with width = dtknot, 
 * where dtknot = T/nsplines is the time knot vector spacing.
 */
class BSpline0 : public ControlBasis {
    protected:
        int nsplines;                     // Number of splines
        double dtknot;                    // spacing of time knot vector    
        double *tcenter;                  // vector of basis function center positions
        double width;                     // support of each basis function (m*dtknot)

        /* Evaluate the bspline basis functions B_l(tau_l(t)) NOT USED */
        // double bspl0(int id, double t);

    public:
        BSpline0(int nsplines, double tstart, double tstop, bool enforceZeroBoundary);
        ~BSpline0();

        int getNSplines() {return nsplines;};

        /* Evaluate the spline at time t using the coefficients coeff. */
        void evaluate(const double t, const std::vector<double>& coeff, int carrier_freq_id, double* Blt1_ptr, double* Blt2_ptr);

        /* Evaluates the derivative at time t, multiplied with fbar. */
        void derivative(const double t, const std::vector<double>& coeff, double* coeff_diff, const double valbar1, const double valbar2, int carrier_freq_id);
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

