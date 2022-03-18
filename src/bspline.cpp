#include "bspline.hpp"

ControlBasis::ControlBasis(int NBasis, double T, std::vector<double> carrier_freq_){
    nbasis = NBasis;
    carrier_freq = carrier_freq_;
    for (int i=0; i<carrier_freq.size(); i++) {
        carrier_freq[i] *= 2.*M_PI;
    }

    dtknot = T / (double)(nbasis - 2);
	width = 3.0*dtknot;

    /* Compute center points of the splines */
    tcenter = new double[nbasis];
    for (int i = 0; i < nbasis; i++){
        tcenter[i] = dtknot * ( (i+1) - 1.5 );
    }

}

ControlBasis::~ControlBasis(){

    delete [] tcenter;
}


double ControlBasis::evaluate(const double t, const std::vector<double>& coeff, const double ground_freq, const ControlType controltype){

    double sum = 0.0;
    /* Sum over basis function */
    for (int l=0; l<nbasis; l++) {
        double Blt = basisfunction(l,t);
        /* Sum over carrier wave frequencies */
        for (int f=0; f < carrier_freq.size(); f++) {
            double alpha1 = coeff[l*carrier_freq.size()*2 + f*2];
            double alpha2 = coeff[l*carrier_freq.size()*2 + f*2 + 1];
            double cos_omt = cos(carrier_freq[f]*t);
            double sin_omt = sin(carrier_freq[f]*t);
            switch (controltype) {
                case ControlType::RE:
                    sum += alpha1 * cos_omt * Blt; 
                    sum -= alpha2 * sin_omt * Blt;
                    // sum += alpha1 * Blt * sin(carrier_freq[f]*t); // Match rigetti case
                    break;
                case ControlType::IM:
                    sum += alpha1 * sin_omt * Blt;
                    sum += alpha2 * cos_omt * Blt;
                    break;
                case ControlType::LAB:
                    sum += 2. * alpha1 * Blt * cos((ground_freq + carrier_freq[f])*t);
                    sum -= 2. * alpha2 * Blt * sin((ground_freq + carrier_freq[f])*t);
                    break;
            }   
        }
    }

    return sum;
}

void ControlBasis::derivative(const double t, double* coeff_diff, const double valbar, const ControlType controltype) {

    /* Iterate over basis function */
    for (int l=0; l<nbasis; l++) {
        double basis = basisfunction(l, t); 
        /* Iterate over carrier frequencies */
        for (int f=0; f < carrier_freq.size(); f++) {
            double freq = carrier_freq[f] * t;
            int coeff_id = l * carrier_freq.size() * 2 + f * 2;
            if (controltype == ControlType::RE) {
                coeff_diff[coeff_id]     +=   basis * cos(freq) * valbar; // alpha1
                coeff_diff[coeff_id + 1] += - basis * sin(freq) * valbar; // alpha2
                // coeff_diff[coeff_id]     +=   basis * sin(freq) * valbar;
            } else {
                coeff_diff[coeff_id]     += basis * sin(freq) * valbar;
                coeff_diff[coeff_id + 1] += basis * cos(freq) * valbar;
            }
        }
    }
}

double ControlBasis::basisfunction(int id, double t){

    /* compute scaled time tau = (t-tcenter[k])  */
    double tau = (t - tcenter[id]) / width;

    /* Return 0 if tau not in local support */
    if ( tau < -1./2. || tau >= 1./2. ) return 0.0;

    /* Evaluate basis function */
    double val = 0.0;
    if       (-1./2. <= tau && tau < -1./6.) val = 9./8. + 9./2. * tau + 9./2. * pow(tau,2);
    else if  (-1./6. <= tau && tau <  1./6.) val = 3./4. - 9. * pow(tau,2);
    else if  ( 1./6. <= tau && tau <  1./2.) val = 9./8. - 9./2. * tau + 9./2. * pow(tau,2);

    return val;
}


TransferFunction::TransferFunction(){}
TransferFunction::~TransferFunction() {}


IdentityTransferFunction::IdentityTransferFunction() : TransferFunction() {}
IdentityTransferFunction::~IdentityTransferFunction() {}

double IdentityTransferFunction::eval(double p) {
    return p;
}

double IdentityTransferFunction::der(double p){
    return 1.0;
}

SplineTransferFunction::SplineTransferFunction(int order, std::vector<double>knots, std::vector<double>coeffs) : TransferFunction() {

#ifdef WITH_FITPACK
    transfer_func = new fitpackpp::BSplineCurve(knots, coeffs, order);
#else
    printf("# Warning: Can not process spline transfer function from python interface, because you didn't link to the FITPACK package. Check your Makefile for WITH_FITPACK=true, if you want to use this feature. Using identity transfer function now.\n");
#endif
}

SplineTransferFunction::~SplineTransferFunction() {
#ifdef WITH_FITPACK
    delete transfer_func;
#endif
}

double SplineTransferFunction::eval(double p) {
#ifdef WITH_FITPACK
    double out = transfer_func->eval(p); 
    return out;
#else 
    return p;
#endif
}


double SplineTransferFunction::der(double p){
#ifdef WITH_FITPACK
    return transfer_func->der(p);
#else 
    return 1.0;
#endif
}

CosineTransferFunction::CosineTransferFunction(double amp_, double freq_) : TransferFunction() {
    freq = freq_;
    amp = amp_;
}

CosineTransferFunction::~CosineTransferFunction(){}

double CosineTransferFunction::eval(double t){
    return amp*cos(freq*t);
}

double CosineTransferFunction::der(double t){
    return amp*freq*sin(freq*t);
}

SineTransferFunction::SineTransferFunction(double amp_, double freq_) : TransferFunction() {
    freq = freq_;
    amp = amp_;
}

SineTransferFunction::~SineTransferFunction(){}

double SineTransferFunction::eval(double t){
    return amp*sin(freq*t);
}

double SineTransferFunction::der(double t){
    return -1.*amp*freq*cos(freq*t);
}