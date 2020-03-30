#include "bspline.hpp"

Bspline::Bspline(int NBasis, double T){
    nbasis = NBasis;

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



double Bspline::evaluate(double t, double* coeff){

    double val = 0.0;
    double tau;

    /* Sum up basis function */
    double sum = 0.0;
    for (int k=1; k<=nbasis; k++) {
        double tmp = coeff[k-1] * basisfunction(k-1, t);
        sum += tmp;
    }
    return sum;
}

void Bspline::derivative(double t, double* coeff, double valbar, double* coeff_diff) {

    double tau  = 0.0;
    
    /* Find k such that t \in [t_k, t_k+1) */
    int k = floor(t / dtknot) + 1;
    if (k <= 0 || k >= nbasis) {  // sanity check
        printf("\n ERROR: Can't find interval for spline evaluation!\n\n");
        exit(0);
    }

    /* 3rd segment of basis function k */
    tau = (t - tcenter[k-1]) / width;
    coeff_diff[k-1] += (9./8. - 4.5*tau + 4.5 * pow(tau,2)) * valbar;

    /* 2nd segment of basis function k+1 */
    tau = (t - tcenter[k]) / width;
    coeff_diff[k] += (0.75 - 9. * pow(tau,2)) * valbar;

    /* 1st segment of basis function k+2 */
    if (k < nbasis - 1)
    {
        tau = (t - tcenter[k+1]) / width;
        coeff_diff[k+1] += (9./8. + 4.5*tau + 4.5 * pow(tau,2));
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