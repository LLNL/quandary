#include "controlbasis.hpp"


ControlBasis::ControlBasis() {
    nparams= 0;
    skip = 0;
    controltype = ControlType::NONE;
    enforceZeroBoundary = false;
}

ControlBasis::ControlBasis(int nparams_, double tstart_, double tstop_, bool enforceZeroBoundary_) : ControlBasis() {
    nparams = nparams_;
    tstart = tstart_;
    tstop = tstop_;
    enforceZeroBoundary = enforceZeroBoundary_;
}
ControlBasis::~ControlBasis(){}


BSpline2nd::BSpline2nd(int nsplines_, double t0, double T, bool enforceZeroBoundary_) : ControlBasis(2*nsplines_, t0, T, enforceZeroBoundary_){
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

void BSpline2nd::enforceBoundary(double* x, int carrier_id){
    // set first and last two splines to zero so that spline starts and ends at zero 
    for (int l=0; l<nsplines; l++) {
        if (l<=1 || l >= nsplines- 2) {
            x[skip + carrier_id*nsplines*2 + l] = 0.0;
            x[skip + carrier_id*nsplines*2 + l + nsplines] = 0.0;
        }
    }
}

void BSpline2nd::evaluate(const double t, const std::vector<double>& coeff, int carrier_freq_id, double* Bl1_ptr, double* Bl2_ptr){

    double sum1 = 0.0;
    double sum2 = 0.0;
    /* Sum over basis function */
    for (int l=0; l<nsplines; l++) {
        if (enforceZeroBoundary) {
            if (l<=1 || l >= nsplines- 2) continue; // skip first and last two splines (set to zero) so that spline starts and ends at zero 
        }
        double Blt = basisfunction(l,t);
        double alpha1 = coeff[skip + carrier_freq_id*nsplines*2 + l];
        double alpha2 = coeff[skip + carrier_freq_id*nsplines*2 + l + nsplines];
        sum1 += alpha1 * Blt;
        sum2 += alpha2 * Blt;
    }
    *Bl1_ptr = sum1;
    *Bl2_ptr = sum2;

}

void BSpline2nd::derivative(const double t, const std::vector<double>& coeff, double* coeff_diff, const double valbar1, const double valbar2, int carrier_freq_id) {

    /* Iterate over basis function */
    for (int l=0; l<nsplines; l++) {
        if (enforceZeroBoundary){
            if (l<=1 || l >= nsplines- 2) continue; // skip first and last two splines (set to zero) so that spline starts and ends at zero       
        }
        double Blt = basisfunction(l, t); 
        coeff_diff[skip + carrier_freq_id*nsplines*2 + l]            += Blt * valbar1;
        coeff_diff[skip + carrier_freq_id*nsplines*2 + l + nsplines] += Blt * valbar2;
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


BSpline2ndAmplitude::BSpline2ndAmplitude(int nsplines_, double scaling_, double t0, double T, bool enforceZeroBoundary_) : ControlBasis(nsplines_ + 1, t0, T, enforceZeroBoundary_){
    nsplines = nsplines_;
    scaling = scaling_;
    controltype = ControlType::BSPLINEAMP;


    dtknot = (T-t0) / (double)(nsplines - 2);
	width = 3.0*dtknot;

    /* Compute center points of the splines */
    tcenter = new double[nsplines];
    for (int i = 0; i < nsplines; i++){
        tcenter[i] = t0 + dtknot * ( (i+1) - 1.5 );
    }
}

BSpline2ndAmplitude::~BSpline2ndAmplitude(){
    delete [] tcenter;
}

void BSpline2ndAmplitude::enforceBoundary(double* x, int carrier_id){
    // set first and last two splines to zero so that spline starts and ends at zero 
    for (int l=0; l<nsplines; l++) {
        if (l<=1 || l >= nsplines- 2) {
            x[skip + carrier_id*(nsplines+1) + l] = 0.0;
        }
    }
}

void BSpline2ndAmplitude::evaluate(const double t, const std::vector<double>& coeff, int carrier_freq_id, double* Bl1_ptr, double* Bl2_ptr){

    /* Sum over basis function for amplitudes */
    double ampsum = 0.0;
    for (int l=0; l<nsplines; l++) {
        if (enforceZeroBoundary){
            if (l<=1 || l >= nsplines- 2) continue; // skip first and last two splines (set to zero) so that spline starts and ends at zero       
        }
        double Blt = basisfunction(l,t);
        double alpha1 = coeff[skip + carrier_freq_id*(nsplines+1) + l];
        ampsum += alpha1 * Blt;
    }
    *Bl1_ptr = ampsum;
    // last one is for the phase
    *Bl2_ptr = scaling*coeff[skip + carrier_freq_id*(nsplines+1) + nsplines];  // last one is the phase
}

void BSpline2ndAmplitude::derivative(const double t, const std::vector<double>& coeff, double* coeff_diff, const double valbar1, const double valbar2, int carrier_freq_id) {
    // valbar1 holds the current carrierfrequency. 
    double cos_omt = cos(valbar1*t + scaling * coeff[skip + carrier_freq_id*(nsplines+1) + nsplines]);
    double sin_omt = sin(valbar1*t + scaling * coeff[skip + carrier_freq_id*(nsplines+1) + nsplines]);

    /* Iterate over basis function */
    double ampsum = 0.0;
    for (int l=0; l<nsplines; l++) {
        if (enforceZeroBoundary){
            if (l<=1 || l >= nsplines- 2) continue; // skip first and last two splines (set to zero) so that spline starts and ends at zero       
        }
        double Blt = basisfunction(l, t); 
        double alpha1 = coeff[skip + carrier_freq_id*(nsplines+1) + l];
        ampsum += alpha1 * Blt;
        // Update derivative for the amplitude splines
        // valbar2 holds the flag whether this is p or q
        if (valbar2 > 0.0) coeff_diff[skip + carrier_freq_id*(nsplines+1) + l] += Blt * cos_omt;
        else               coeff_diff[skip + carrier_freq_id*(nsplines+1) + l] += Blt * sin_omt;
    }
    // Update derivate for phase
    if (valbar2 > 0.0) coeff_diff[skip + carrier_freq_id*(nsplines+1) + nsplines] += -ampsum * scaling * sin_omt;
    else               coeff_diff[skip + carrier_freq_id*(nsplines+1) + nsplines] +=  ampsum * scaling * cos_omt;
}

double BSpline2ndAmplitude::basisfunction(int id, double t){

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

Step::Step(double step_amp1_, double step_amp2_, double t0, double t1, double tramp_, bool enforceZeroBoundary_) : ControlBasis(1, t0, t1, enforceZeroBoundary_) {
    step_amp1 = step_amp1_;
    step_amp2 = step_amp2_;
    tramp = tramp_;
    controltype = ControlType::STEP;
}

Step::~Step(){}

void Step::evaluate(const double t, const std::vector<double>& coeff, int carrier_freq_id, double* Blt1, double*Blt2){
    // Access the control
    double alpha = coeff[skip + carrier_freq_id*2];

    // The control enters as tstop for the ramping function
    double tstepend = tstart + alpha*(tstop - tstart);
    double ramp = 1.0;
    if (tramp > 1e-13) ramp = getRampFactor(t, tstart, tstepend, tramp);

    *Blt1 = ramp*step_amp1;
    *Blt2 = ramp*step_amp2;
}

void Step::derivative(const double t, const std::vector<double>& coeff, double* coeff_diff, const double valbar1, const double valbar2, int carrier_freq_id) {

    double alpha = coeff[skip + carrier_freq_id*2];    
    double tstepend = tstart + alpha*(tstop - tstart);

    double dramp = getRampFactor_diff(t, tstart, tstepend, tramp);
    coeff_diff[skip + carrier_freq_id*2] += step_amp1*valbar1 * dramp * (tstop - tstart); 
    coeff_diff[skip + carrier_freq_id*2] += step_amp2*valbar2 * dramp * (tstop - tstart); 
}

//
// Zeroth order B-splines, i.e., piecewise constant
//
BSpline0::BSpline0(int nsplines_, double t0, double T, bool enforceZeroBoundary_) : ControlBasis(2*nsplines_, t0, T, enforceZeroBoundary_){
    nsplines = nsplines_;
    controltype = ControlType::BSPLINE0;

    dtknot = (T-t0) / (double)nsplines;
	width = dtknot;

    /* Compute center points of the splines */
    tcenter = new double[nsplines];
    for (int i = 0; i < nsplines; i++){
        tcenter[i] = t0 + dtknot * ( i + 0.5 );
    }

}

BSpline0::~BSpline0(){

    delete [] tcenter;
}


void BSpline0::evaluate(const double t, const std::vector<double>& coeff, int carrier_freq_id, double* Bl1_ptr, double* Bl2_ptr){
    /* NO need to sum over basis functions! */
    // double sum1 = 0.0;
    // double sum2 = 0.0;

    // first calculate lc index from t
    int lc = floor((t-tstart)/dtknot);

    // Ctrl function defined to be zero outside [tstart, tend]
    if (lc < 0 || lc >= nsplines){
        *Bl1_ptr = 0.0;
        *Bl2_ptr = 0.0;
    }
    else{
        *Bl1_ptr = coeff[skip + carrier_freq_id*nsplines*2 + lc];
        *Bl2_ptr = coeff[skip + carrier_freq_id*nsplines*2 + lc + nsplines];
    }

    // for (int l=lstart; l<lend; l++) { 
    //     double Blt = bspl0(l,t);
    //     double alpha1 = coeff[skip + carrier_freq_id*nsplines*2 + l];
    //     double alpha2 = coeff[skip + carrier_freq_id*nsplines*2 + l + nsplines];
    //     sum1 += alpha1 * Blt;
    //     sum2 += alpha2 * Blt;
    // }
    // *Bl1_ptr = sum1;
    // *Bl2_ptr = sum2;
}

void BSpline0::derivative(const double t, const std::vector<double>& coeff, double* coeff_diff, const double valbar1, const double valbar2, int carrier_freq_id) {

    // first calculate lc index from t
    int lc = floor((t-tstart)/dtknot);

    if (lc >= 0 && lc < nsplines){
        coeff_diff[skip + carrier_freq_id*nsplines*2 + lc] += valbar1;
        coeff_diff[skip + carrier_freq_id*nsplines*2 + lc + nsplines] += valbar2;
    }

    // for (int l=lstart; l<lend; l++) {
    //     double Blt = basisfunction(l, t); 
    //     coeff_diff[skip + carrier_freq_id*nsplines*2 + l]            += Blt * valbar1;
    //     coeff_diff[skip + carrier_freq_id*nsplines*2 + l + nsplines] += Blt * valbar2;
    // }
}

// This fcn is not needed anymore
// double BSpline0::bspl0(int id, double t){

//     /* compute scaled time tau = (t-tcenter[k])  */
//     double tau = (t - tcenter[id]) / width;

//     /* Return 0 if tau not in local support */
//     if ( tau < -1./2. || tau >= 1./2. ) 
//         return 0.0;
//     else 
//         return 1.0;

// }

TransferFunction::TransferFunction(){}
TransferFunction::TransferFunction(std::vector<double> onofftimes_){
   storeOnOffTimes(onofftimes_); 
}
TransferFunction::~TransferFunction() {}


void TransferFunction::storeOnOffTimes(std::vector<double>onofftimes_){
    // Make sure the list contains an even number of times (on,off, on,off,... needs to end with an 'off')
    assert(onofftimes_.size()%2 == 0);

    // Copy the list of time points that determine when the transfer functions are active
    onofftimes.clear();
    for (int i=0; i<onofftimes_.size(); i++) {
        onofftimes.push_back( onofftimes_[i] );
    }   
}


double TransferFunction::isOn(double p, double time){
    // Default: always on. 
    bool ison = true; 
    // If list onofftimes is given, check if time \in [t_{2i}, t_{2i+1}] (i.e. transfer is ON)
    if (onofftimes.size()>0) ison = false;
    for (int i=0; i<(int)(onofftimes.size()/2); i++) {
        if ( onofftimes[2*i] <= time && time <= onofftimes[2*i+1] ) {
            ison = true;
            break;
        }
    }

    if (ison) return p;
    else return 0.0;
}


IdentityTransferFunction::IdentityTransferFunction() : TransferFunction() {}
IdentityTransferFunction::IdentityTransferFunction(std::vector<double> onofftimes) : TransferFunction(onofftimes) {}
IdentityTransferFunction::~IdentityTransferFunction() {}

ConstantTransferFunction::ConstantTransferFunction() : TransferFunction() {
    constant = 1.0;
}
ConstantTransferFunction::ConstantTransferFunction(double constant_) : TransferFunction() {
    constant = constant_;
}


ConstantTransferFunction::ConstantTransferFunction(double constant_, std::vector<double> onofftimes) : TransferFunction(onofftimes) {
    constant = constant_;
}

ConstantTransferFunction::~ConstantTransferFunction() {}


CosineTransferFunction::CosineTransferFunction(double amp_, double freq_) : TransferFunction() {
    freq = freq_;
    amp = amp_;
}
CosineTransferFunction::CosineTransferFunction(double amp_, double freq_, std::vector<double> onofftimes) : TransferFunction(onofftimes) {
    freq = freq_;
    amp = amp_;
}
CosineTransferFunction::~CosineTransferFunction(){}

SineTransferFunction::SineTransferFunction(double amp_, double freq_) : TransferFunction() {
    freq = freq_;
    amp = amp_;
}
SineTransferFunction::SineTransferFunction(double amp_, double freq_, std::vector<double> onofftimes) : TransferFunction(onofftimes) {
    freq = freq_;
    amp = amp_;
}
SineTransferFunction::~SineTransferFunction(){}
