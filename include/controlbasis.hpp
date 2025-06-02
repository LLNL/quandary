#include "defs.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <cassert>
#include "util.hpp"
#pragma once


/**
 * @brief Abstract base class for control parameterizations.
 *
 * The `ControlBasis` class defines the interface for various control parameterizations
 * used in quantum optimal control. Derived classes implement specific control schemes.
 */
class ControlBasis {
    protected:
        int nparams; ///< Number of parameters that define the controls.
        double tstart; ///< Start time of the interval where the control basis is applied.
        double tstop; ///< Stop time of the interval where the control basis is applied.
        int skip; ///< Offset to the starting location for this basis inside the global control vector.
        ControlType controltype; ///< Type of control parameterization.
        bool enforceZeroBoundary; ///< Flag to enforce zero boundary conditions for controls.

    public: 
        ControlBasis();

        /**
         * @brief Constructor with parameters.
         *
         * @param nparams_ Number of parameters defining the controls.
         * @param tstart Start time of the interval.
         * @param tstop Stop time of the interval.
         * @param enforceZeroBoundary Flag to enforce zero boundary conditions. If true, ensures that the controls start and end at zero.
         */
        ControlBasis(int nparams_, double tstart, double tstop, bool enforceZeroBoundary);

        virtual ~ControlBasis();

        /**
         * @brief Retrieves the number of parameters defining the controls.
         *
         * @return int Number of parameters.
         */
        int getNparams() {return nparams; };

        /**
         * @brief Retrieves the start time of the interval.
         *
         * @return double Start time.
         */
        double getTstart() {return tstart; };

        /**
         * @brief Retrieves the stop time of the interval.
         *
         * @return double Stop time.
         */
        double getTstop() {return tstop; };

        /**
         * @brief Retrieves the type of control parameterization.
         *
         * @return ControlType Type of control.
         */
        ControlType getType() {return controltype;};

        /**
         * @brief Sets the offset for the starting location in the global control vector.
         *
         * @param skip_ Offset value.
         */
        void setSkip(int skip_) {skip = skip_;};

        /**
         * @brief Retrieves the offset for the starting location in the global control vector.
         *
         * @return int Offset value.
         */
        int getSkip(){return skip;};

        /**
         * @brief Retrieves the number of splines (default implementation returns 0).
         *
         * @return int Number of splines.
         */
        virtual int getNSplines() {return 0;};

        /**
         * @brief Computes the variation of control parameters (default implementation returns 0.0).
         *
         * @param params Vector of control parameters.
         * @param carrierfreqID ID of the carrier frequency.
         * @return double Variation value.
         */
        virtual double computeVariation(std::vector<double>& /*params*/, int /*carrierfreqID*/){return 0.0;};

        /**
         * @brief Computes the gradient of the variation (default implementation does nothing).
         *
         * @param grad Pointer to the gradient array.
         * @param params Vector of control parameters.
         * @param var_bar Variation multiplier.
         * @param carrierfreqID ID of the carrier frequency.
         */
        virtual void computeVariation_diff(double* /*grad*/, std::vector<double>& /*params*/, double /*var_bar*/, int /*carrierfreqID*/){};

        /**
         * @brief Enforces boundary conditions for controls (default implementation does nothing).
         *
         * For some control parameterizations, this can be used to enforce that the controls start and end at zero.
         * For example, splines will overwrite the parameters `x` of the first and last two splines by zero, ensuring
         * that the splines start and end at zero.
         *
         * @param x Pointer to the control parameters.
         * @param carrier_id ID of the carrier wave.
         */
        virtual void enforceBoundary(double* /*x*/, int /*carrier_id*/) {};

        /**
         * @brief Evaluates the control basis at a given time using coefficients.
         *
         * @param t Time at which to evaluate.
         * @param coeff Vector of coefficients.
         * @param carrier_freq_id ID of the carrier frequency.
         * @param Blt1 Pointer to store the first evaluated control value.
         * @param Blt2 Pointer to store the second evaluated control value.
         */
        virtual void evaluate(const double t, const std::vector<double>& coeff, int carrier_freq_id, double* Blt1, double*Blt2) = 0;

        /**
         * @brief Evaluates the derivative of the control basis at a given time.
         *
         * @param t Time at which to evaluate.
         * @param coeff Vector of coefficients.
         * @param coeff_diff Pointer to the derivative coefficients.
         * @param valbar1 Multiplier for the first derivative term.
         * @param valbar2 Multiplier for the second derivative term.
         * @param carrier_freq_id ID of the carrier frequency.
         */
        virtual void derivative(const double t, const std::vector<double>& coeff, double* coeff_diff, const double valbar1, const double valbar2, int carrier_freq_id)= 0;
};

/**
 * @brief Discretization of the Controls using quadratic Bsplines ala Anders Petersson.
 *
 * Bspline basis functions have local support with width = 3*dtknot,
 * where dtknot = T/(nsplines -2) is the time knot vector spacing.
 */
class BSpline2nd : public ControlBasis {
    protected:
        int nsplines; ///< Number of splines.
        double dtknot; ///< Spacing of time knot vector.
        double *tcenter; ///< Vector of basis function center positions.
        double width; ///< Support of each basis function (m*dtknot).

        /**
         * @brief Evaluate the bspline basis functions B_l(tau_l(t)).
         *
         * @param id Index of the basis function.
         * @param t Time at which to evaluate.
         * @return double Value of the basis function.
         */
        double basisfunction(int id, double t);

    public:
        /**
         * @brief Constructor for quadratic Bsplines.
         *
         * @param nsplines Number of splines.
         * @param tstart Start time of the interval.
         * @param tstop Stop time of the interval.
         * @param enforceZeroBoundary Flag to enforce zero boundary conditions.
         */
        BSpline2nd(int nsplines, double tstart, double tstop, bool enforceZeroBoundary);

        ~BSpline2nd();

        int getNSplines() {return nsplines;};

        /**
         * @brief Sets the first and last two spline coefficients in x to zero, so that the controls start and end at zero.
         *
         * @param x Pointer to the control parameters.
         * @param carrier_id ID of the carrier wave.
         */
        void enforceBoundary(double* x, int carrier_id);

        void evaluate(const double t, const std::vector<double>& coeff, int carrier_freq_id, double* Blt1_ptr, double* Blt2_ptr);

        void derivative(const double t, const std::vector<double>& coeff, double* coeff_diff, const double valbar1, const double valbar2, int carrier_freq_id);
};

/**
 * @brief Amplitude is parameterized by Bsplines, phase is time-independent.
 *
 * Discretization of the controls using quadratic Bsplines ala Anders Petersson.
 * Bspline basis functions have local support with width = 3 * dtknot, 
 * where dtknot = T / (nsplines - 2) is the time knot vector spacing.
 */
class BSpline2ndAmplitude : public ControlBasis {
    protected:
        int nsplines; ///< Number of splines.
        double dtknot; ///< Spacing of the time knot vector.
        double *tcenter; ///< Vector of basis function center positions.
        double width; ///< Support of each basis function (m * dtknot).
        double scaling; ///< Scaling for the phase.

        /**
         * @brief Evaluate the bspline basis functions B_l(tau_l(t)).
         *
         * @param id Index of the basis function.
         * @param t Time at which to evaluate.
         * @return double Value of the basis function.
         */
        double basisfunction(int id, double t);

    public:
        /**
         * @brief Constructor for quadratic Bsplines with amplitude parameterization.
         *
         * @param nsplines Number of splines.
         * @param scaling Scaling factor for the phase.
         * @param tstart Start time of the interval.
         * @param tstop Stop time of the interval.
         * @param enforceZeroBoundary Flag to enforce zero boundary conditions.
         */
        BSpline2ndAmplitude(int nsplines, double scaling, double tstart, double tstop, bool enforceZeroBoundary);

        ~BSpline2ndAmplitude();

        /**
         * @brief Retrieves the number of splines.
         *
         * @return int Number of splines.
         */
        int getNSplines() {return nsplines;};

        /**
         * @brief Sets the first and last two spline coefficients in x to zero, so that the controls start and end at zero.
         *
         * @param x Pointer to the control parameters.
         * @param carrier_id ID of the carrier wave.
         */
        void enforceBoundary(double* x, int carrier_id);

        void evaluate(const double t, const std::vector<double>& coeff, int carrier_freq_id, double* Blt1_ptr, double* Blt2_ptr);

        void derivative(const double t, const std::vector<double>& coeff, double* coeff_diff, const double valbar1, const double valbar2, int carrier_freq_id);
};

/**
 * @brief Parameterization of the controls using step functions with constant amplitude and variable width.
 *
 * This class represents a step function with configurable amplitudes and ramp time.
 */
class Step : public ControlBasis {
    protected:
        double step_amp1; ///< Amplitude of the first step.
        double step_amp2; ///< Amplitude of the second step.
        double tramp; ///< Ramp time.

    public: 
        /**
         * @brief Constructor for the step function.
         *
         * @param step_amp1_ Amplitude of the first step.
         * @param step_amp2_ Amplitude of the second step.
         * @param t0 Start time of the interval.
         * @param t1 Stop time of the interval.
         * @param tramp Ramp time.
         * @param enforceZeroBoundary Flag to enforce zero boundary conditions.
         */
        Step(double step_amp1_, double step_amp2_, double t0, double t1, double tramp, bool enforceZeroBoundary);

        ~Step();

       /* Evaluate the spline at time t using the coefficients coeff. */
        void evaluate(const double t, const std::vector<double>& coeff, int carrier_freq_id, double* Blt1, double*Blt2);

        void derivative(const double t, const std::vector<double>& coeff, double* coeff_diff, const double valbar1, const double valbar2, int carrier_freq_id);
};

/**
 * @brief Discretization of the Controls using piece-wise constant Bsplines.
 *
 * Bspline basis functions have local support with width = dtknot, 
 * where dtknot = T/nsplines is the time knot vector spacing.
 */
class BSpline0 : public ControlBasis {
    protected:
        /* Evaluate the bspline basis functions B_l(tau_l(t)) NOT USED */
        int nsplines; ///< Number of splines.
        double dtknot; ///< Spacing of time knot vector.
        double width; ///< Support of each basis function (m*dtknot).

        // double bspl0(int id, double t);

    public:
        BSpline0(int nsplines, double tstart, double tstop, bool enforceZeroBoundary);
        ~BSpline0();

        int getNSplines() {return nsplines;};

        /* Set the first and last parameter to zero, for this carrier wave */
        void enforceBoundary(double* x, int carrier_id);

        /* Variation of the control parameters: 1/nsplines * sum_splines (alpha_i - alpha_{i-1})^2 */
        double computeVariation(std::vector<double>& params, int carrierfreqID);
        virtual void computeVariation_diff(double* grad, std::vector<double>&params, double var_bar, int carrierfreqID);

        /* Evaluate the spline at time t using the coefficients coeff. */
        void evaluate(const double t, const std::vector<double>& coeff, int carrier_freq_id, double* Blt1_ptr, double* Blt2_ptr);

        /* Evaluates the derivative at time t, multiplied with fbar. */
        void derivative(const double t, const std::vector<double>& coeff, double* coeff_diff, const double valbar1, const double valbar2, int carrier_freq_id);
};

/* 
 * @brief Abstract class to represent transfer functions that act on the controls.
 *
 * This class evaluates `u(p(t))` or `v(q(t))`. By default, `u` and `v` are IdentityTransferFunctions.
 * Alternatively, `u` and `v` can be splineTransferFunctions, read from the Python interface.
 */
class TransferFunction{
    protected: 
        std::vector<double> onofftimes;  // Stores when transfer functions are active: They return their value only in [t0,t1] U [t2,t3] U ... and they return 0.0 otherwise (i.e., in [t1,t2] and [t3,t4], ...).

    public:
        TransferFunction();

        /**
         * @brief Constructor with active intervals.
         *
         * @param onofftimes Vector of time points defining active intervals.
         */
        TransferFunction(std::vector<double> onofftimes);

        virtual ~TransferFunction();

        /**
         * @brief Evaluates the transfer function at a given time.
         *
         * @param p Input parameter.
         * @param time Current time.
         * @return double Value of the transfer function at the given time.
         */
        virtual double eval(double p, double time) =0;

        /**
         * @brief Evaluates the derivative of the transfer function at a given time.
         *
         * @param p Input parameter.
         * @param time Current time.
         * @return double Derivative of the transfer function at the given time.
         */
        virtual double der(double p, double time) =0;

        /**
         * @brief Checks whether the transfer function is active at a given time.
         *
         * This method determines whether the transfer function is active based on the `onofftimes` list.
         * If active, it returns the input parameter `p`; otherwise, it returns 0.0.
         *
         * @param p Input parameter.
         * @param time Current time.
         * @return double Returns `p` if active, otherwise 0.0.
         */
        double isOn(double p, double time);

        /**
         * @brief Stores a list of time points when the transfer function is active.
         *
         * This method updates the list of time intervals during which the transfer function is active.
         *
         * @param onofftimes_ Vector of time points defining active intervals.
         */
        void storeOnOffTimes(std::vector<double> onofftimes_);
};

/**
 * @brief Transfer function that is constant: u(x) = const, u'(x) = 0.0.
 *
 * This class represents a transfer function with a constant value and zero derivative.
 */
class ConstantTransferFunction : public TransferFunction {
    double constant; ///< Constant value of the transfer function.
    public:
        ConstantTransferFunction();

        /**
         * @brief Constructor with a constant value.
         *
         * @param constant_ Constant value of the transfer function.
         */
        ConstantTransferFunction(double constant_);

        /**
         * @brief Constructor with a constant value and active intervals.
         *
         * @param constant_ Constant value of the transfer function.
         * @param onofftimes Vector of time points defining active intervals.
         */
        ConstantTransferFunction(double constant_, std::vector<double> onofftimes);
        ~ConstantTransferFunction();

        double eval(double /*x*/, double time) {return isOn(constant, time); }; 
        double der(double /*x*/, double /*time*/) {return 0.0; }; 
};

/**
 * @brief Transfer function that is the identity: u(x) = x, u'(x) = 1.0.
 *
 * This class represents a transfer function that returns the input value and has a constant derivative of 1.0.
 */
class IdentityTransferFunction : public TransferFunction {
    public:
        IdentityTransferFunction();

        /**
         * @brief Constructor with active intervals.
         *
         * @param onofftimes Vector of time points defining active intervals.
         */
        IdentityTransferFunction(std::vector<double> onofftimes);
        ~IdentityTransferFunction();

        double eval(double x, double time) {return isOn(x, time); };
        double der(double /*x*/, double time) {return isOn(1.0, time); };
};


/**
 * @brief Transfer function that represents a cosine wave: u(x) = amp * cos(freq * x).
 *
 * This class represents a transfer function based on a cosine wave with configurable amplitude and frequency.
 */
class CosineTransferFunction : public TransferFunction {
    protected:
        double freq; ///< Frequency of the cosine wave.
        double amp; ///< Amplitude of the cosine wave.

    public:
        /**
         * @brief Constructor with amplitude and frequency.
         *
         * @param amp Amplitude of the cosine wave.
         * @param freq Frequency of the cosine wave.
         */
        CosineTransferFunction(double amp, double freq);

        /**
         * @brief Constructor with amplitude, frequency, and active intervals.
         *
         * @param amp Amplitude of the cosine wave.
         * @param freq Frequency of the cosine wave.
         * @param onofftimes Vector of time points defining active intervals.
         */
        CosineTransferFunction(double amp, double freq, std::vector<double>onofftimes);

        ~CosineTransferFunction();
        // This is amp*cos(freq*t)
        double eval(double x, double time){ return isOn(amp*cos(freq*x), time); };
        double der(double x, double time) { return isOn(amp*freq*sin(freq*x), time); };
};

/**
 * @brief Transfer function that represents a sine wave: u(x) = amp * sin(freq * x).
 *
 * This class represents a transfer function based on a sine wave with configurable amplitude and frequency.
 */
class SineTransferFunction : public TransferFunction {
    protected:
        double freq; ///< Frequency of the sine wave.
        double amp; ///< Amplitude of the sine wave.

    public:
        /**
         * @brief Constructor with amplitude and frequency.
         *
         * @param amp Amplitude of the sine wave.
         * @param freq Frequency of the sine wave.
         */
        SineTransferFunction(double amp, double freq);

        /**
         * @brief Constructor with amplitude, frequency, and active intervals.
         *
         * @param amp Amplitude of the sine wave.
         * @param freq Frequency of the sine wave.
         * @param onofftimes Vector of time points defining active intervals.
         */
        SineTransferFunction(double amp, double freq, std::vector<double> onofftimes);

        ~SineTransferFunction();
        // This is amp*sin(freq*t)
        double eval(double x, double time) { return isOn(amp*sin(freq*x), time); };
        double der(double x, double time) { return isOn(-1.0*amp*freq*cos(freq*x), time); };
};

