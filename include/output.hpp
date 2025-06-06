#include <sys/stat.h> 
#include <petscmat.h>
#include <iostream> 
#include "config.hpp"
#include "mastereq.hpp"
#pragma once

/**
 * @brief Output management for quantum control simulations and optimization.
 *
 * This class handles all file I/O operations for Quandary, including optimization
 * progress logging, control pulse output, state evolution data, and population dynamics.
 * It manages MPI-aware output to avoid duplicate writes in parallel runs.
 */
class Output{

  int mpirank_world; ///< Rank of processor in MPI_COMM_WORLD
  int mpirank_petsc; ///< Rank of processor for PETSc parallelization
  int mpisize_petsc; ///< Size of communicator for PETSc parallelization
  int mpirank_init; ///< Rank of processor for initial condition parallelization

  bool quietmode; ///< Flag for reduced screen output
  
  FILE* optimfile; ///< Output file for logging optimization progress
  int output_frequency; ///< Time domain output frequency (write every N time steps)
  std::vector<std::vector<std::string> > outputstr; ///< List of output specifications for each oscillator

  bool writefullstate; ///< Flag to determine if full state vector should be written to file
  FILE *ufile; ///< File for writing real part of solution vector
  FILE *vfile; ///< File for writing imaginary part of solution vector
  std::vector<FILE *>expectedfile; ///< Files for expected energy evolution per oscillator
  FILE *expectedfile_comp; ///< File for expected energy evolution of composite system
  FILE *populationfile_comp; ///< File for population evolution of composite system
  std::vector<FILE *>populationfile; ///< Files for population evolution per oscillator

  // VecScatter scat; ///< PETSc's scatter context for state communication across cores
  // Vec xseq; ///< Sequential vector for I/O operations

  public:
    std::string datadir; ///< Directory path for output data files
    int optim_monitor_freq; ///< Write output files every N optimization iterations

  public:
    Output();

    /**
     * @brief Constructor with configuration and MPI setup.
     *
     * @param config Configuration parameters from input file
     * @param comm_petsc MPI communicator for PETSc parallelization
     * @param comm_init MPI communicator for initial condition parallelization
     * @param noscillators Number of oscillators in the system
     * @param quietmode Flag for reduced output (default: false)
     */
    Output(MapParam& config, MPI_Comm comm_petsc, MPI_Comm comm_init, int noscillators, bool quietmode=false);

    ~Output();

    /**
     * @brief Writes optimization progress to history file.
     *
     * Called at every optimization iteration to log convergence data.
     *
     * @param optim_iter Current optimization iteration
     * @param objective Total objective function value
     * @param gnorm Gradient norm
     * @param stepsize Optimization step size
     * @param Favg Average fidelity
     * @param cost Final-time cost term
     * @param tikh_regul Tikhonov regularization term
     * @param penalty Penalty term
     * @param penalty_dpdm Second-order derivative penalty
     * @param penalty_energy Energy penalty term
     * @param penalty_variation Control variation penalty
     */
    void writeOptimFile(int optim_iter, double objective, double gnorm, double stepsize, double Favg, double cost, double tikh_regul,  double penalty, double penalty_dpdm, double penalty_energy, double penalty_variation);

    /**
     * @brief Writes current control pulses and parameters.
     *
     * Called every optim_monitor_freq optimization iterations.
     *
     * @param params Current parameter vector
     * @param mastereq Pointer to master equation solver
     * @param ntime Number of time steps
     * @param dt Time step size
     */
    void writeControls(Vec params, MasterEq* mastereq, int ntime, double dt);

    /**
     * @brief Writes gradient vector for debugging adjoint calculations.
     *
     * @param grad Gradient vector to output
     */
    void writeGradient(Vec grad);

    /**
     * @brief Opens data files for time evolution output.
     *
     * Prepares files for writing full state, expected energy, and population data.
     *
     * @param prefix Filename prefix for output files
     * @param initid Initial condition identifier
     */
    void openDataFiles(std::string prefix, int initid);

    /**
     * @brief Writes time evolution data to files.
     *
     * Outputs state vector, expected energies, and populations at current time step.
     *
     * @param timestep Current time step number
     * @param time Current time value
     * @param state Current state vector
     * @param mastereq Pointer to master equation solver
     */
    void writeDataFiles(int timestep, double time, const Vec state, MasterEq* mastereq);

    /**
     * @brief Closes all open data files.
     *
     * Properly closes and flushes all output files after simulation completion.
     */
    void closeDataFiles();

};
