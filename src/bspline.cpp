#include "bspline.hpp"

ControlBasis::ControlBasis(int NBasis, double T, std::vector<double> carrier_freq_){
    nbasis = NBasis;
    carrier_freq = carrier_freq_;

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

    double freq;

    double sum = 0.0;
    /* Sum over basis function */
    for (int l=0; l<nbasis; l++) {
        double ampl = 0.0;
        /* Sum over carrier wave frequencies */
        for (int f=0; f < carrier_freq.size(); f++) {
            int coeff_id = l * carrier_freq.size() * 2 + f * 2;     // alpha^{k(1)}_{l,f}
            switch (controltype) {
                case RE:
                    freq = -2.0 * M_PI * carrier_freq[f];
                    ampl += coeff[coeff_id] * cos(freq*t) - coeff[coeff_id + 1] * sin(freq*t);
                    break;
                case IM:
                    freq = -2.0 * M_PI * carrier_freq[f];
                    ampl += coeff[coeff_id] * sin(freq*t) + coeff[coeff_id + 1] * cos(freq*t);
                    break;
                case LAB:
                    freq = -2.0 * M_PI * (ground_freq + carrier_freq[f]);
                    ampl += coeff[coeff_id] * cos(freq*t) - coeff[coeff_id + 1] * sin(freq*t);
                    break;
            }   
        }
        sum += basisfunction(l,t) * ampl;
    }
    if (controltype == LAB) sum *= 2.0;

    return sum;
}

void ControlBasis::derivative(const double t, double* coeff_diff, const double valbar, const ControlType controltype) {

    /* Iterate over basis function */
    for (int l=0; l<nbasis; l++) {
        double basis = basisfunction(l, t); 
        /* Iterate over carrier frequencies */
        for (int f=0; f < carrier_freq.size(); f++) {
            double tmp = -2 * M_PI * carrier_freq[f] * t;
            int coeff_id = l * carrier_freq.size() * 2 + f * 2;
            if (controltype == RE) {
                coeff_diff[coeff_id]     +=   basis * cos(tmp) * valbar;
                coeff_diff[coeff_id + 1] += - basis * sin(tmp) * valbar;
            } else {
                coeff_diff[coeff_id]     += basis * sin(tmp) * valbar;
                coeff_diff[coeff_id + 1] += basis * cos(tmp) * valbar;
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