#include "braid_wrapper.hpp"
#include "hiopInterface.hpp"
#include "math.h"
#include <assert.h>

#pragma once


class OptimProblem : public hiop::hiopInterfaceDenseConstraints {

    protected:
        myBraidApp* primalbraidapp;         /* Primal BraidApp to carry out PinT forward sim.*/
        myAdjointBraidApp* adjointbraidapp; /* Adjoint BraidApp to carry out PinT backward sim. */
        Gate  *targetgate;                  
        double objective;                 /* holds the current objective value */
        double fidelity;                /* holds the current fidelity value */
        double regul;                       /* Parameter for L2 regularization */
        std::vector<double> bounds;    /* Bounds for the control function amplitudes for each oscillator */
        std::string datadir;           /* Directory for data output */
        std::string optiminit_type;           /* Type of design initialization */
        int printlevel;                
        int ninit;                            /* Number of initial conditions to be considered (N^2, N, or 1) */
        int ninit_local;                      /* Local number of initial conditions on this processor */

        MPI_Comm comm_hiop, comm_init;
        int mpirank_braid, mpisize_braid;
        int mpirank_space, mpisize_space;
        int mpirank_optim, mpisize_optim;
        int mpirank_world, mpisize_world;
        int mpirank_init, mpisize_init;

        FILE* optimfile;
    
    private: 
        Vec initcond_re, initcond_im;           /* Storage for real and imag part of initial condition */
        Vec initcond_re_bar, initcond_im_bar;   /* Storage for real and imag part of adjoint initial condition */

    public:
        OptimProblem();
        OptimProblem(MapParam config, myBraidApp* primalbraidapp_, myAdjointBraidApp* adjointbraidapp_, Gate* targate_, MPI_Comm comm_hiop_, MPI_Comm comm_init_, int ninit_);
        virtual ~OptimProblem();


        /* Pass x to the oscillator parameters */
        void setDesign(int n, const double* x);
        /* Pass the oscillator parameters to x */
        void getDesign(int n, double* x);

        /* Set the initial condition of index iinit */
        int assembleInitialCondition(int iinit);

        /* Required interface routines. These are purely virtual in HiOp. */
        bool get_prob_sizes(long long& n, long long& m);
        bool get_vars_info(const long long& n, double *xlow, double* xupp, NonlinearityType* type);
        bool get_cons_info(const long long& m, double* clow, double* cupp, NonlinearityType* type);
        bool eval_f(const long long& n, const double* x_in, bool new_x, double& obj_value);
        bool eval_grad_f(const long long& n, const double* x_in, bool new_x, double* gradf);
        bool eval_cons(const long long& n, const long long& m, const long long& num_cons, const long long* idx_cons, const double* x_in, bool new_x, double* cons);
        bool eval_Jac_cons(const long long& n, const long long& m, const long long& num_cons, const long long* idx_cons, const double* x_in, bool new_x, double** Jac);

        /* Optional interface routines. These have a default implementation. */
        bool get_starting_point(const long long &global_n, double* x0);
        bool get_MPI_comm(MPI_Comm& comm_out);
        void solution_callback(hiop::hiopSolveStatus status, int n, const double* x, const double* z_L, const double* z_U, int m, const double* g, const double* lambda, double obj_value);
        bool iterate_callback(int iter, double obj_value, int n, const double* x, const double* z_L, const double* z_U, int m, const double* g, const double* lambda, double inf_pr, double inf_du, double mu, double alpha_du, double alpha_pr, int ls_trials) ;
 

	private:
	/* Methods to block default compiler methods.
	 * The compiler automatically generates the following three methods.
	 * Since the default compiler implementation is generally not what
	 * you want (for all but the most simple classes), we usually
	 * put the declarations of these methods in the private section
	 * and never implement them. This prevents the compiler from
	 * implementing an incorrect "default" behavior without us
	 * knowing. (See Scott Meyers book, "Effective C++") 
     */
	//  HS071_NLP();
	OptimProblem(const OptimProblem&);
	OptimProblem& operator=(const OptimProblem&);
};