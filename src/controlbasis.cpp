#include "controlbasis.hpp"


ControlBasis::ControlBasis() {
    nparams= 0;
    skip = 0;
    controltype = ControlType::NONE;
}

ControlBasis::ControlBasis(int nparams_, double tstart_, double tstop_) : ControlBasis() {
    nparams = nparams_;
    tstart = tstart_;
    tstop = tstop_;
}
ControlBasis::~ControlBasis(){}


BSpline2nd::BSpline2nd(int nsplines_, double t0, double T) : ControlBasis(2*nsplines_, t0, T){
    nsplines = nsplines_;
    controltype = ControlType::BSPLINE;

    dtknot = (T-t0) / (double)(nsplines - 2);
	width = 3.0*dtknot;

    /* Compute center points of the splines */
    tcenter = new double[nsplines];
    for (int i = 0; i < nsplines; i++){
        tcenter[i] = t0 + dtknot * ( (i+1) - 1.5 );
    }

}

BSpline2nd::~BSpline2nd(){

    delete [] tcenter;
}


void BSpline2nd::evaluate(const double t, const std::vector<double>& coeff, int carrier_freq_size, int carrier_freq_id, double* Bl1_ptr, double* Bl2_ptr){

    double sum1 = 0.0;
    double sum2 = 0.0;
    /* Sum over basis function */
    for (int l=0; l<nsplines; l++) {
        if (l<=1 || l >= nsplines- 2) continue; // skip first and last two splines (set to zero) so that spline starts and ends at zero 
        double Blt = basisfunction(l,t);
        double alpha1 = coeff[skip + l*carrier_freq_size*2 + carrier_freq_id*2];
        double alpha2 = coeff[skip + l*carrier_freq_size*2 + carrier_freq_id*2 + 1];
        sum1 += alpha1 * Blt;
        sum2 += alpha2 * Blt;
    }
    *Bl1_ptr = sum1;
    *Bl2_ptr = sum2;

}

void BSpline2nd::derivative(const double t, const std::vector<double>& coeff, double* coeff_diff, const double valbar1, const double valbar2, int carrier_freq_size, int carrier_freq_id) {

    /* Iterate over basis function */
    for (int l=0; l<nsplines; l++) {
        if (l<=1 || l >= nsplines- 2) continue; // skip first and last two splines (set to zero) so that spline starts and ends at zero       
        double Blt = basisfunction(l, t); 
        coeff_diff[skip + l * carrier_freq_size *2 + carrier_freq_id*2] += Blt * valbar1;
        coeff_diff[skip + l * carrier_freq_size *2 + carrier_freq_id*2+1] += Blt * valbar2;
    }
}

double BSpline2nd::basisfunction(int id, double t){

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

Step::Step(double step_amp1_, double step_amp2_, double t0, double t1, double tramp_) : ControlBasis(1, t0, t1) {
    step_amp1 = step_amp1_;
    step_amp2 = step_amp2_;
    tramp = tramp_;
    controltype = ControlType::STEP;
}

Step::~Step(){}

void Step::evaluate(const double t, const std::vector<double>& coeff, int carrier_freq_size, int carrier_freq_id, double* Blt1, double*Blt2){
    // Access the control
    double alpha = coeff[skip + carrier_freq_id*2];

    // The control enters as tstop for the ramping function
    double tstepend = tstart + alpha*(tstop - tstart);
    double ramp = getRampFactor(t, tstart, tstepend, tramp);

    *Blt1 = ramp*step_amp1;
    *Blt2 = ramp*step_amp2;
}

void Step::derivative(const double t, const std::vector<double>& coeff, double* coeff_diff, const double valbar1, const double valbar2, int carrier_freq_size, int carrier_freq_id) {

    double alpha = coeff[skip + carrier_freq_id*2];    
    double tstepend = tstart + alpha*(tstop - tstart);

    double rbar = 0.0;
    rbar += step_amp1 * valbar1;
    rbar += step_amp2 * valbar2;

    double tstependbar = getRampFactor_diff(t, tstart, tstepend, tramp) * rbar;

    coeff_diff[skip + carrier_freq_id*2] += (tstop - tstart) * tstependbar; 
}