#include "bspline.hpp"

Bspline::Bspline(int NBasis, double T, std::vector<double> carrier_freq_){
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

Bspline::~Bspline(){

    delete [] tcenter;
}


double Bspline::evaluate(double t, double* coeff, int sign){

    double val = 0.0;
    double tau;

    /* Sum up basis function */
    double sum = 0.0;
    for (int k=0; k<nbasis; k++) {
        for (int l=0; l < carrier_freq.size(); l++) {
            double tmp = carrier_freq[l] * t;
            int coeff_id = k * carrier_freq.size() * 2 + l * 2;
            sum += basisfunction(k,t) * ( coeff[coeff_id] * cos(tmp) + sign * coeff[coeff_id + 1] * sin(tmp) );
            // sum += coeff[k] * basisfunction(k, t);
        }
    }

    return sum;
}

void Bspline::derivative(double t, double* coeff_diff, double valbar, int sign) {

    /* Iterate over basis function */
    for (int k=0; k<nbasis; k++) {
        for (int l=0; l < carrier_freq.size(); l++) {
            double tmp = carrier_freq[l] * t;
            int coeff_id = k * carrier_freq.size() * 2 + l * 2;
            coeff_diff[coeff_id]     +=  basisfunction(k, t) * cos(tmp) * valbar;
            coeff_diff[coeff_id + 1] +=  basisfunction(k, t) * sin(tmp) * valbar * sign;
        }
    }
 
}

double Bspline::basisfunction(int id, double t){

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