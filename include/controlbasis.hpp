#include "defs.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
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

    public: 
        ControlBasis();
        ControlBasis(int nparams_, double tstart, double tstop);
        virtual ~ControlBasis();

        int getNparams() {return nparams; };
        double getTstart() {return tstart; };
        double getTstop() {return tstop; };
        ControlType getType() {return controltype;};
        void setSkip(int skip_) {skip = skip_;};

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
        BSpline2nd(int nsplines, double tstart, double tstop);
        ~BSpline2nd();


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
        Step(double step_amp1_, double step_amp2_, double t0, double t1, double tramp);
        ~Step();

       /* Evaluate the spline at time t using the coefficients coeff. */
        void evaluate(const double t, const std::vector<double>& coeff, int carrier_freq_id, double* Blt1, double*Blt2);

        /* Evaluates the derivative at time t, multiplied with fbar. */
        void derivative(const double t, const std::vector<double>& coeff, double* coeff_diff, const double valbar1, const double valbar2, int carrier_freq_id);
};
