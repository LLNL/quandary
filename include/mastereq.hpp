#include "defs.hpp"
#include "oscillator.hpp"
#include "util.hpp"
#include <petscts.h>
#include <vector>
#include <assert.h>
#include <iostream> 
#include "gate.hpp"
#pragma once


/* Define a matshell context containing pointers to data needed for applying the RHS matrix to a vector */
typedef struct {
  std::vector<int> nlevels;
  IS *isu, *isv;
  Oscillator** oscil_vec;
  std::vector<double> crosskerr;
  std::vector<double> Jkl;
  std::vector<double> eta;
  LindbladType lindbladtype;
  bool addT1, addT2;
  std::vector<double> control_Re, control_Im;
  Mat** Ac_vec;
  Mat** Bc_vec;
  Mat *Ad, *Bd;
  Mat** Ad_vec;
  Mat** Bd_vec;
  Vec *aux;
  double time;
} MatShellCtx;


/* Define the Matrix-Vector products for the RHS MatShell */
int myMatMult_matfree_2Osc(Mat RHS, Vec x, Vec y);              // Matrix free solver for 2 oscillators 
int myMatMultTranspose_matfree_2Osc(Mat RHS, Vec x, Vec y);
int myMatMult_matfree_3Osc(Mat RHS, Vec x, Vec y);              // Matrix free solver for 3 oscillators 
int myMatMultTranspose_matfree_3Osc(Mat RHS, Vec x, Vec y);
int myMatMult_matfree_4Osc(Mat RHS, Vec x, Vec y);              // Matrix free solver for 4 oscillators 
int myMatMultTranspose_matfree_4Osc(Mat RHS, Vec x, Vec y);
int myMatMult_matfree_5Osc(Mat RHS, Vec x, Vec y);              // Matrix free solver for 5 oscillators 
int myMatMultTranspose_matfree_5Osc(Mat RHS, Vec x, Vec y);
int myMatMult_sparsemat(Mat RHS, Vec x, Vec y);                 // Sparse matrix solver
int myMatMultTranspose_sparsemat(Mat RHS, Vec x, Vec y);


/* 
 * Implements the Lindblad master equation
 */
class MasterEq{

  protected:
    int dim;                   // Dimension of full vectorized system: either N^2 if Lindblad, or N if Schrodinger
    int dim_rho;               // Dimension of Hilbertspace = N
    int dim_ess;               // Dimension of system of essential levels = N_e
    int noscillators;          // Number of oscillators
    Oscillator** oscil_vec;    // Vector storing pointers to the oscillators

    Mat RHS;                // Realvalued, vectorized systemmatrix (2N^2 x 2N^2)
    MatShellCtx RHSctx;     // MatShell context that contains data needed to apply the RHS

    Mat* Ac_vec;  // Vector of constant mats for time-varying control term (real)
    Mat* Bc_vec;  // Vector of constant mats for time-varying control term (imag)
    Mat  Ad, Bd;  // Real and imaginary part of constant system matrix
    Mat* Ad_vec;  // Vector of constant mats for Jaynes-Cummings coupling term in drift Hamiltonian (real)
    Mat* Bd_vec;  // Vector of constant mats for Jaynes-Cummings coupling term in drift Hamiltonian (imag)

    std::vector<double> crosskerr;    // Cross ker coefficients (rad/time) $\xi_{kl} for zz-coupling ak^d ak al^d al
    std::vector<double> Jkl;          // Jaynes-Cummings coupling coefficient (rad/time), multiplies ak^d al + ak al^d
    std::vector<double> eta;          // Delta in rotational frame frequencies (rad/time). Used for Jaynes-Cummings coupling terms in rotating frame
    bool addT1, addT2;                // flags for including Lindblad collapse operators T1-decay and/or T2-dephasing

    /* Auxiliary stuff */
    int mpirank_petsc;   // Rank of Petsc's communicator
    int mpirank_world;   // Rank of global communicator
    int nparams_max;     // Maximum number of design parameters per oscilator 
    IS isu, isv;         // Vector strides for accessing u=Re(x), v=Im(x) 

    double *dRedp;
    double *dImdp;
    Vec aux;              // auxiliary vector 
    PetscInt* cols;           // holding columns when evaluating dRHSdp
    PetscScalar* vals;   // holding values when evaluating dRHSdp
 
  public:
    std::vector<int> nlevels;  // Number of levels per oscillator
    std::vector<int> nessential; // Number of essential levels per oscillator
    bool usematfree;  // Flag for using matrix free solver
    LindbladType lindbladtype;        // Flag that determines which lindblad terms are added. if NONE, than Schroedingers eq. is solved

  public:
    MasterEq();
    MasterEq(std::vector<int> nlevels, std::vector<int> nessential, Oscillator** oscil_vec_, const std::vector<double> crosskerr_, const std::vector<double> Jkl_, const std::vector<double> eta_, LindbladType lindbladtype_, bool usematfree_);
    ~MasterEq();

    /* initialize matrices needed for applying sparse-mat solver */
    void initSparseMatSolver();

    /* Return the i-th oscillator */
    Oscillator* getOscillator(const int i);

    /* Return number of oscillators */
    int getNOscillators();

    /* Return dimension of vectorized system N^2 (for Lindblad solver) or N (for Schroedinger solver) */
    int getDim();

    /* Return dimension of essential level system: N_e */
    int getDimEss();
    
    /* Return dimension of system matrix rho: N */
    int getDimRho();

    /* 
     * Uses Re and Im to build the vectorized Hamiltonian operator M = vec(-i(Hq-qH)+Lindblad). 
     * This should always be called before applying the RHS matrix.
     */
    int assemble_RHS(const double t);

    /* Access the right-hand-side matrix */
    Mat getRHS();

    /* 
     * Compute gradient of RHS wrt control parameters:
     * grad += alpha * RHS(x)^T * x_bar  
     */
    void computedRHSdp(const double t,const Vec x,const Vec x_bar, const double alpha, Vec grad);

    // /* Compute reduced density operator for a sub-system defined by IDs in the oscilIDs vector */
    // void createReducedDensity(const Vec rho, Vec *reduced, const std::vector<int>& oscilIDs);
    // /* Derivative of reduced density computation */
    // void createReducedDensity_diff(Vec rhobar, const Vec reducedbar, const std::vector<int>& oscilIDs);

    /* Set the oscillators control function parameters from global design vector x */
    void setControlAmplitudes(const Vec x);

    /* Set initial conditions 
     * In:   iinit -- index in processors range [rank * ninit_local .. (rank+1) * ninit_local - 1]
     *       ninit -- number of initial conditions 
     *       initcond_type -- type of initial condition (pure, fromfile, diagona, basis)
     *       oscilIDs -- ID of oscillators defining the subsystem for the initial conditions  
     * Out: initID -- Idenifyier for this initial condition: Element number in matrix vectorization. 
     *       rho0 -- Vector for setting initial condition 
     */
    int getRhoT0(const int iinit, const int ninit, const InitialConditionType initcond_type, const std::vector<int>& oscilIDs, Vec rho0);

};


// Mat-free solver inlines for 2 oscillator
inline double H_detune(const double detuning0, const double detuning1, const int a, const int b) {
  return detuning0*a + detuning1*b;
};
inline double H_selfkerr(const double xi0, const double xi1, const int a, const int b) {
  return - xi0 / 2.0 * a * (a-1) - xi1 / 2.0 * b * (b-1);
};
inline double H_crosskerr(const double xi01, const int a, const int b) {
  return - xi01 * a * b;
};
inline double L2(const double dephase0, const double dephase1, const int i0, const int i1, const int i0p, const int i1p){
  return dephase0 * ( i0*i0p - 1./2. * (i0*i0 + i0p*i0p) ) + dephase1 * ( i1*i1p - 1./2. * (i1*i1 + i1p*i1p) );
};
inline double L1diag(const double decay0, const double decay1, const int i0, const int i1, const int i0p, const int i1p){
  return - decay0 / 2.0 * ( i0 + i0p ) - decay1 / 2.0 * ( i1 + i1p );
};
inline int TensorGetIndex(const int nlevels0, const int nlevels1,const  int i0, const int i1, const int i0p, const int i1p){
  return i0*nlevels1 + i1 + (nlevels0 * nlevels1) * ( i0p * nlevels1 + i1p);
};


// Matfree solver inlines for 3 oscillator
inline double H_detune(const double detuning0, const double detuning1, const double detuning2, const int i0, const int i1, const int i2) {
  return detuning0*i0 + detuning1*i1 + detuning2*i2;
};
inline double H_selfkerr(const double xi0, const double xi1, const double xi2, const int i0, const int i1, const int i2) {
  return - xi0 / 2.0 * i0 * (i0-1) - xi1 / 2.0 * i1 * (i1-1) - xi2 / 2.0 * i2 * (i2-1);
};
inline double H_crosskerr(const double xi01, const double xi02, const double xi12, const int i0, const int i1, const int i2) {
  return - xi01 * i0 * i1 - xi02 * i0 * i2 - xi12 * i1 * i2;
};
inline double L2(const double dephase0, const double dephase1, const double dephase2, const int i0, const int i1, const int i2, const int i0p, const int i1p, const int i2p){
  return dephase0 * ( i0*i0p - 1./2. * (i0*i0 + i0p*i0p) ) 
       + dephase1 * ( i1*i1p - 1./2. * (i1*i1 + i1p*i1p) )
       + dephase2 * ( i2*i2p - 1./2. * (i2*i2 + i2p*i2p) );
};
inline double L1diag(const double decay0, const double decay1, const double decay2, const int i0, const int i1, const int i2, const int i0p, const int i1p, const int i2p){
  return - decay0 / 2.0 * ( i0 + i0p ) 
         - decay1 / 2.0 * ( i1 + i1p )
         - decay2 / 2.0 * ( i2 + i2p );
};
inline int TensorGetIndex(const int nlevels0, const int nlevels1, const int nlevels2, const  int i0, const int i1, const int i2, const int i0p, const int i1p, const int i2p){
  return i0*nlevels1*nlevels2 + i1*nlevels2 + i2 + (nlevels0 * nlevels1 * nlevels2) * ( i0p * nlevels1*nlevels2 + i1p*nlevels2 + i2p);
};


// Matfree solver inlines for 4 oscillators
inline double H_detune(const double detuning0, const double detuning1, const double detuning2, const double detuning3, const int i0, const int i1, const int i2, const int i3) {
  return detuning0*i0 + detuning1*i1 + detuning2*i2 + detuning3*i3;
};
inline double H_selfkerr(const double xi0, const double xi1, const double xi2, const double xi3, const int i0, const int i1, const int i2, const int i3) {
  return - xi0 / 2.0 * i0 * (i0-1) - xi1 / 2.0 * i1 * (i1-1) - xi2 / 2.0 * i2 * (i2-1) - xi3/2.0 * i3 * (i3-1);
};
inline double H_crosskerr(const double xi01, const double xi02, const double xi03, const double xi12, const double xi13, const double xi23, const int i0, const int i1, const int i2, const int i3) {
  return - xi01 * i0 * i1 - xi02 * i0 * i2  - xi03*i0*i3 - xi12 * i1 * i2 - xi13*i1*i3 - xi23*i2*i3;
};
inline double L2(const double dephase0, const double dephase1, const double dephase2, const double dephase3, const int i0, const int i1, const int i2, const int i3, const int i0p, const int i1p, const int i2p, const int i3p){
  return dephase0 * ( i0*i0p - 1./2. * (i0*i0 + i0p*i0p) ) 
       + dephase1 * ( i1*i1p - 1./2. * (i1*i1 + i1p*i1p) )
       + dephase2 * ( i2*i2p - 1./2. * (i2*i2 + i2p*i2p) )
       + dephase3 * ( i3*i3p - 1./2. * (i3*i3 + i3p*i3p) );
};
inline double L1diag(const double decay0, const double decay1, const double decay2, const double decay3, const int i0, const int i1, const int i2, const int i3, const int i0p, const int i1p, const int i2p, const int i3p){
  return - decay0 / 2.0 * ( i0 + i0p ) 
         - decay1 / 2.0 * ( i1 + i1p )
         - decay2 / 2.0 * ( i2 + i2p )
         - decay3 / 2.0 * ( i3 + i3p );
};
inline int TensorGetIndex(const int nlevels0, const int nlevels1, const int nlevels2, const int nlevels3, const  int i0, const int i1, const int i2, const int i3, const int i0p, const int i1p, const int i2p, const int i3p){
  return i0*nlevels1*nlevels2*nlevels3 + i1*nlevels2*nlevels3 + i2*nlevels3 + i3 + (nlevels0 * nlevels1 * nlevels2 * nlevels3) * ( i0p * nlevels1*nlevels2*nlevels3 + i1p*nlevels2*nlevels3 + i2p*nlevels3 + i3p);
}

// Matfree solver inlines for 5 oscillators
inline double H_detune(const double detuning0, const double detuning1, const double detuning2, const double detuning3, const double detuning4, const int i0, const int i1, const int i2, const int i3, const int i4) {
  return detuning0*i0 + detuning1*i1 + detuning2*i2 + detuning3*i3 + detuning4*i4;
};
inline double H_selfkerr(const double xi0, const double xi1, const double xi2, const double xi3, const double xi4, const int i0, const int i1, const int i2, const int i3, const int i4) {
  return - xi0 / 2.0 * i0 * (i0-1) - xi1 / 2.0 * i1 * (i1-1) - xi2 / 2.0 * i2 * (i2-1) - xi3/2.0 * i3 * (i3-1) - xi4/2.0 * i4 * (i4-1);
};
inline double H_crosskerr(const double xi01, const double xi02, const double xi03, const double xi04, const double xi12, const double xi13, const double xi14, const double xi23, const double xi24, const double xi34, const int i0, const int i1, const int i2, const int i3, const int i4) {
  return - xi01 * i0 * i1 - xi02 * i0 * i2  - xi03*i0*i3 - xi04*i0*i4 - xi12 * i1 * i2 - xi13*i1*i3 - xi14*i1*i4 - xi23*i2*i3 - xi24*i2*i4 - xi34*i3*i4;
};
inline double L2(const double dephase0, const double dephase1, const double dephase2, const double dephase3, const double dephase4, const int i0, const int i1, const int i2, const int i3, const int i4, const int i0p, const int i1p, const int i2p, const int i3p, const int i4p){
  return dephase0 * ( i0*i0p - 1./2. * (i0*i0 + i0p*i0p) ) 
       + dephase1 * ( i1*i1p - 1./2. * (i1*i1 + i1p*i1p) )
       + dephase2 * ( i2*i2p - 1./2. * (i2*i2 + i2p*i2p) )
       + dephase3 * ( i3*i3p - 1./2. * (i3*i3 + i3p*i3p) )
       + dephase4 * ( i4*i4p - 1./2. * (i4*i4 + i4p*i4p) );
};
inline double L1diag(const double decay0, const double decay1, const double decay2, const double decay3, const double decay4, const int i0, const int i1, const int i2, const int i3, const int i4, const int i0p, const int i1p, const int i2p, const int i3p, const int i4p){
  return - decay0 / 2.0 * ( i0 + i0p ) 
         - decay1 / 2.0 * ( i1 + i1p )
         - decay2 / 2.0 * ( i2 + i2p )
         - decay3 / 2.0 * ( i3 + i3p )
         - decay4 / 2.0 * ( i4 + i4p );
};
inline int TensorGetIndex(const int nlevels0, const int nlevels1, const int nlevels2, const int nlevels3, const int nlevels4, const  int i0, const int i1, const int i2, const int i3, const int i4, const int i0p, const int i1p, const int i2p, const int i3p, const int i4p){
  return i0*nlevels1*nlevels2*nlevels3*nlevels4 + i1*nlevels2*nlevels3*nlevels4 + i2*nlevels3*nlevels4 + i3*nlevels4 + i4 + (nlevels0 * nlevels1 * nlevels2 * nlevels3*nlevels4) * ( i0p * nlevels1*nlevels2*nlevels3*nlevels4 + i1p*nlevels2*nlevels3*nlevels4 + i2p*nlevels3*nlevels4+ i3p*nlevels4 + i4p);
}


// Mat-free solver inline for gradient updates for oscillator i
inline void dRHSdp_getcoeffs(const int it, const int n, const int np, const int i, const int ip, const int stridei, const int strideip, const double* xptr, double* res_p_re, double* res_p_im, double* res_q_re, double* res_q_im) {

  *res_p_re = 0.0;
  *res_p_im = 0.0;
  *res_q_re = 0.0;
  *res_q_im = 0.0;

  /* ik+1..,ik'.. term */
  if (i < n-1) {
    int itx = it + stridei;
    double xre = xptr[2 * itx];
    double xim = xptr[2 * itx + 1];
    double sq = sqrt(i + 1);
    *res_p_re +=   sq * xim;
    *res_p_im += - sq * xre;
    *res_q_re +=   sq * xre;
    *res_q_im +=   sq * xim;
  }
  /* \rho(ik..,ik'+1..) */
  if (ip < np-1) {
    int itx = it + strideip;
    double xre = xptr[2 * itx];
    double xim = xptr[2 * itx + 1];
    double sq = sqrt(ip + 1);
    *res_p_re += - sq * xim;
    *res_p_im += + sq * xre;
    *res_q_re +=   sq * xre;
    *res_q_im +=   sq * xim;
  }
  /* \rho(ik-1..,ik'..) */
  if (i > 0) {
    int itx = it - stridei;
    double xre = xptr[2 * itx];
    double xim = xptr[2 * itx + 1];
    double sq = sqrt(i);
    *res_p_re += + sq * xim;
    *res_p_im += - sq * xre;
    *res_q_re += - sq * xre;
    *res_q_im += - sq * xim;
  }
  /* \rho(ik..,ik'-1..) */
  if (ip > 0) {
    int itx = it - strideip;
    double xre = xptr[2 * itx];
    double xim = xptr[2 * itx + 1];
    double sq = sqrt(ip);
    *res_p_re += - sq * xim;
    *res_p_im += + sq * xre;
    *res_q_re += - sq * xre;
    *res_q_im += - sq * xim;
  }
}

// Mat-free solver inline for Jkl coupling between oscillator i and oscillator j
inline void Jkl_coupling(const int it, const int ni, const int nj, const int nip, const int njp, const int i, const int ip, const int j, const int jp, const int stridei, const int strideip, const int stridej, const int stridejp, const double* xptr, const double Jij, const double cosij, const double sinij, double* yre, double* yim) {
  if (fabs(Jij)>1e-10) {
    //  1) J_kl (-icos + sin) * ρ_{E−k+l i, i′}
    if (i > 0 && j < nj-1) {
      int itx = it - stridei + stridej;
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(i * (j + 1));
      // sin u + cos v + i ( -cos u + sin v)
      *yre += Jij * sq * (   cosij * xim + sinij * xre);
      *yim += Jij * sq * ( - cosij * xre + sinij * xim);
    }
    // 2) J_kl (−icos − sin)sqrt(il*(ik +1)) ρ_{E+k−li,i′}
    if (i < ni-1 && j > 0) {
      int itx = it + stridei - stridej;  // E+k-l i, i'
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(j * (i + 1)); // sqrt( il*(ik+1))
      // -sin u + cos v + i (-cos u - sin v)
      *yre += Jij * sq * (   cosij * xim - sinij * xre);
      *yim += Jij * sq * ( - cosij * xre - sinij * xim);
    }
    // 3) J_kl ( icos + sin)sqrt(ik'*(il' +1)) ρ_{i,E-k+li'}
    if (ip > 0 && jp < njp-1) {
      int itx = it - strideip + stridejp;  // i, E-k+l i'
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(ip * (jp + 1)); // sqrt( ik'*(il'+1))
      //  sin u - cos v + i ( cos u + sin v)
      *yre += Jij * sq * ( - cosij * xim + sinij * xre);
      *yim += Jij * sq * (   cosij * xre + sinij * xim);
    }
    // 4) J_kl ( icos - sin)sqrt(il'*(ik' +1)) ρ_{i,E+k-li'}
    if (ip < nip-1 && jp > 0) {
      int itx = it + strideip - stridejp;  // i, E+k-l i'
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(jp * (ip + 1)); // sqrt( il'*(ik'+1))
      // - sin u - cos v + i ( cos u - sin v)
      *yre += Jij * sq * ( - cosij * xim - sinij * xre);
      *yim += Jij * sq * (   cosij * xre - sinij * xim);
    }
  }
}

// transpose of Jkl coupling
inline void Jkl_coupling_T(const int it, const int ni, const int nj, const int nip, const int njp, const int i, const int ip, const int j, const int jp, const int stridei, const int strideip, const int stridej, const int stridejp, const double* xptr, const double Jij, const double cosij, const double sinij, double* yre, double* yim) {
  if (fabs(Jij)>1e-10) {
    //  1) [...] * \bar y_{E+k-l i, i′}
    if (i < ni-1 && j > 0) {
      int itx = it + stridei - stridej;
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(j * (i + 1));
      *yre += Jij * sq * ( - cosij * xim + sinij * xre);
      *yim += Jij * sq * ( + cosij * xre + sinij * xim);
    }
    // 2) J_kl (−icos − sin)sqrt(ik*(il +1)) \bar y_{E-k+li,i′}
    if (i > 0 && j < nj-1) {
      int itx = it - stridei + stridej;  // E-k+l i, i'
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(i * (j + 1)); // sqrt( ik*(il+1))
      *yre += Jij * sq * ( - cosij * xim - sinij * xre);
      *yim += Jij * sq * ( + cosij * xre - sinij * xim);
    }
    // 3) J_kl ( icos + sin)sqrt(il'*(ik' +1)) \bar y_{i,E+k-li'}
    if (ip < nip-1 && jp > 0) {
      int itx = it + strideip - stridejp;  // i, E+k-l i'
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(jp * (ip + 1)); // sqrt( il'*(ik'+1))
      *yre += Jij * sq * (   cosij * xim + sinij * xre);
      *yim += Jij * sq * ( - cosij * xre + sinij * xim);
    }
    // 4) J_kl ( icos - sin)sqrt(ik'*(il' +1)) \bar y_{i,E-k+li'}
    if (ip > 0 && jp < njp-1) {
      int itx = it - strideip + stridejp;  // i, E-k+l i'
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(ip * (jp + 1)); // sqrt( ik'*(il'+1))
      *yre += Jij * sq * (   cosij * xim - sinij * xre);
      *yim += Jij * sq * ( - cosij * xre - sinij * xim);
    }
  }

}

// Mat-free solver inline for off-diagonal L1decay term
inline void L1decay(const int it, const int n, const int i, const int ip, const int stridei, const int strideip, const double* xptr, const double decayi, double* yre, double* yim){
  if  (fabs(decayi) > 1e-12) {
    if (i < n-1 && ip < n-1) {
      double l1off = decayi * sqrt((i+1)*(ip+1));
      int itx = it + stridei + strideip;
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      *yre += l1off * xre;
      *yim += l1off * xim;
    }
  }
}


// Transpose of offdiagonal L1decay
inline void L1decay_T(const int it, const int n, const int i, const int ip, const int stridei, const int strideip, const double* xptr, const double decayi, double* yre, double* yim){
  if (fabs(decayi) > 1e-12) {
      if (i > 0 && ip > 0) {
        double l1off = decayi * sqrt(i*ip);
        int itx = it - stridei - strideip;
        double xre = xptr[2 * itx];
        double xim = xptr[2 * itx + 1];
        *yre += l1off * xre;
        *yim += l1off * xim;
      }
    }
}

// Matfree solver inline for Control terms
inline void control(const int it, const int n, const int i, const int np, const int ip, const int stridei, const int strideip, const double* xptr, const double pt, const double qt, double* yre, double* yim){
  /* \rho(ik+1..,ik'..) term */
  if (i < n-1) {
      int itx = it + stridei;
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(i + 1);
      *yre += sq * (   pt * xim + qt * xre);
      *yim += sq * ( - pt * xre + qt * xim);
    }
    /* \rho(ik..,ik'+1..) */
    if (ip < np-1) {
      int itx = it + strideip;
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(ip + 1);
      *yre += sq * ( -pt * xim + qt * xre);
      *yim += sq * (  pt * xre + qt * xim);
    }
    /* \rho(ik-1..,ik'..) */
    if (i > 0) {
      int itx = it - stridei;
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(i);
      *yre += sq * (  pt * xim - qt * xre);
      *yim += sq * (- pt * xre - qt * xim);
    }
    /* \rho(ik..,ik'-1..) */
    if (ip > 0) {
      int itx = it - strideip;
      double xre = xptr[2 * itx];
      double xim = xptr[2 * itx + 1];
      double sq = sqrt(ip);
      *yre += sq * (- pt * xim - qt * xre);
      *yim += sq * (  pt * xre - qt * xim);
    }
}


// Transpose of control terms
inline void control_T(const int it, const int n, const int i, const int np, const int ip, const int stridei, const int strideip, const double* xptr, const double pt, const double qt, double* yre, double* yim){
  /* \rho(ik+1..,ik'..) term */
  if (i > 0) {
    int itx = it - stridei;
    double xre = xptr[2 * itx];
    double xim = xptr[2 * itx + 1];
    double sq = sqrt(i);
    *yre += sq * ( - pt * xim + qt * xre);
    *yim += sq * (   pt * xre + qt * xim);
  }
  /* \rho(ik..,ik'+1..) */
  if (ip > 0) {
    int itx = it - strideip;
    double xre = xptr[2 * itx];
    double xim = xptr[2 * itx + 1];
    double sq = sqrt(ip);
    *yre += sq * (  pt * xim + qt * xre);
    *yim += sq * ( -pt * xre + qt * xim);
  }
  /* \rho(ik-1..,ik'..) */
  if (i < n-1) {
    int itx = it + stridei;
    double xre = xptr[2 * itx];
    double xim = xptr[2 * itx + 1];
    double sq = sqrt(i+1);
    *yre += sq * (- pt * xim - qt * xre);
    *yim += sq * (  pt * xre - qt * xim);
  }
  /* \rho(ik..,ik'-1..) */
  if (ip < np-1) {
    int itx = it + strideip;
    double xre = xptr[2 * itx];
    double xim = xptr[2 * itx + 1];
    double sq = sqrt(ip+1);
    *yre += sq * (+ pt * xim - qt * xre);
    *yim += sq * (- pt * xre - qt * xim);
  }
}