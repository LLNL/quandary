#include <sys/stat.h> 
#include <petscmat.h>
#include <iostream> 
#include "config.hpp"
#include "mastereq.hpp"
#pragma once


class Output{

  int mpirank_world;  /* Rank of processor of MPI_COMM_WORLD */
  int mpirank_petsc;  /* Rank of processor for parallelizing Petsc */
  int mpisize_petsc;  /* Size of communicator for parallelizing Petsc */
  int mpirank_init;   /* Rank of processor for parallelizing initial conditions */

  bool quietmode; /* Reduced screen output */
  
  FILE* optimfile;      /* Output file to log optimization progress */
  int output_frequency;   /* Output frequency in time domain: write output at every <num> time step. */
  std::vector<std::vector<std::string> > outputstr; // List of outputs for each oscillator

  bool writeFullState;  /* Flag to determine if full state vector should be written to file */
  FILE *ufile;          /* File for writing real part of solution vector */
  FILE *vfile;          /* File for writing imaginary part of solution vector */

  std::vector<bool> writeExpectedEnergy; ///< Flag to determine if the evolution of the expected energy per oscillator should be written to files.
  bool writeExpectedEnergy_comp; ///< Flag to determine if the evolution of the expected energy of the full composite system should be written to file
  std::vector<FILE *>expectedfile;    /* Files for writing the evolution of the expected energy per oscillator */
  FILE *expectedfile_comp;    /* File for writing the evolution of the expected energy level of the full composite system */

  std::vector<bool> writePopulation; ///< Flag to determine if the evolution of the energy level occupations per oscillator should be written to files.
  bool writePopulation_comp; ///< Flag to determine if the evolution of the energy level occupations of the full composite system should be written to file
  std::vector<FILE *>populationfile;  /* Files for writing population over time */
  FILE *populationfile_comp;    /* File for writing the evolution of the population of the composite system */

  // VecScatter scat;    /* Petsc's scatter context to communicate a state across petsc's cores */
  // Vec xseq;           /* A sequential vector for IO. */

  public:
    std::string datadir;
    int optim_monitor_freq; /* Write output files every <num> optimization iterations */

  public:
    Output();
    Output(MapParam& config, MPI_Comm comm_petsc, MPI_Comm comm_init, int noscillators, bool quietmode=false);
    ~Output();

    /* Write to optimization history file in every optim iteration */
    void writeOptimFile(int optim_iter, double objective, double gnorm, double stepsize, double Favg, double cost, double tikh_regul,  double penalty, double penalty_dpdm, double penalty_energy, double penalty_variation, double obj_robust);

    /* Write current controls and parameters every <optim_monitor_freq> iterations */
    void writeControls(Vec params, MasterEq* mastereq, int ntime, double dt);

    /* Write gradient for adjoint mode */
    void writeGradient(Vec grad);

    /* Open, write and close files for fullstate and expected energy levels over time */
    void openTrajectoryDataFiles(std::string prefix, int initid);
    void writeTrajectoryDataFiles(int timestep, double time, const Vec state, MasterEq* mastereq);
    void closeTrajectoryDataFiles();

};
