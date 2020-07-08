#include <sys/stat.h> 
#include <petscmat.h>
#include <iostream> 
#include "config.hpp"
#include "mastereq.hpp"
#pragma once


class Output{

  int mpirank_world;  /* Rank of processor of MPI_COMM_WORLD */
  int mpirank_petsc;  /* Rank of processor for parallelizing Petsc */
  int mpirank_init;   /* Rank of processor for parallelizing initial conditions */
  int mpirank_braid;  /* Rank of processor for parallelizing XBraid, or zero if compiling without XBraid */
  
  FILE* optimfile;      /* Output file to log optimization progress */
  int optim_outputfreq; /* Write state output to file every <outfreq> iterations */
  std::vector<std::vector<std::string> > outputstr; // List of outputs for each oscillator

  bool writefullstate;  /* Flag to determin if full state vector should be written to file */
  FILE *ufile;          /* File for writing real part of solution vector */
  FILE *vfile;          /* File for writing imaginary part of solution vector */
  std::vector<FILE *>expectedfile;  /* Files for writing expected energy levels over time */

  // VecScatter scat;    /* Petsc's scatter context to communicate a state across petsc's cores */
  // Vec xseq;           /* A sequential vector for IO. */

  public:
    std::string datadir;
    int optim_iter;       /* Current optimization iteration */

  public:
    Output();
    Output(MapParam& config, int mpirank_petsc, int mpirank_init);
    Output(MapParam& config, int mpirank_petsc, int mpirank_init, int mpirank_braid);
    ~Output();

    /* Write to optimization history file in every optim iteration */
    void writeOptimFile(double objective, double gnorm, double stepsize, double cost, double tikh_regul,  double penalty);

    /* Write current controls and parameters every <optim_outputfreq> iterations */
    void writeControls(Vec params, MasterEq* mastereq, int ntime, double dt);

    /* Open, write and close files for fullstate and expected energy levels over time */
    void openDataFiles(std::string prefix, int initid, int rank);
    void writeDataFiles(double time, const Vec state, MasterEq* mastereq);
    void closeDataFiles();

};