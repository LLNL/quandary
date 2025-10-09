#include "defs.hpp"

#include <cstddef>
#include <vector>
#include <string>
#include <petsc.h>
#include <sstream>
#include <optional>
#include <variant>

#pragma once

/**
 * @brief Structure for storing pi-pulse parameters for one segment.
 *
 * Stores timing and amplitude information for pi-pulse sequences.
 */
struct PiPulseSegment {
  double tstart; ///< Start time for pulse segment
  double tstop; ///< Stop time for pulse segment
  double amp; ///< Amplitude for pulse segment
};

struct SplineParams {
  size_t nspline; ///< Number of basis functions in this segment
  double tstart; ///< Start time of the control segment
  double tstop; ///< Stop time of the control segment
};

struct SplineAmpParams : SplineParams {
  double scaling;
};

struct StepParams {
  double step_amp1; ///< Real part of amplitude of the step pulse.
  double step_amp2; ///< Imaginary part of amplitude of the step pulse.
  double tramp; ///< Ramp time.
  double tstart; ///< Start time of the control segment
  double tstop; ///< Stop time of the control segment // TODO default value ntime * dt
};

using ControlParams = std::variant<SplineParams, SplineAmpParams, StepParams>;

/**
 * @brief Structure for defining control segments.
 *
 * Defines a controllable segment for an oscillator and the type of parameterization,
 * with corresponding starting and finish times.
 */
struct ControlSegment {
  ControlType type; ///< Type of control segment
  ControlParams params; ///< Parameters for control pulse for segment
};

/**
 * @brief Structure for defining a control segment's initialization
 */
struct ControlSegmentInitialization {
  ControlInitializationType type; ///< Type of control initialization
  double amplitude; ///< Initial control pulse amplitude
  double phase; ///< Initial control pulse phase
};

// TODO make a struct for all per oscillator settings and a vector of these
/**
 * @brief Final validated configuration class.
 *
 * Contains only validated, typed configuration parameters. All fields are required
 * and have been validated by ConfigBuilder. This class is immutable after construction.
 */
class Config {
  public:
    // MPI and logging (still needed for runtime operations)
    MPI_Comm comm; ///< MPI communicator for parallel operations.
    int mpi_rank; ///< MPI rank of the current process.

  private:

    // General options
    std::vector<size_t> nlevels;  ///< Number of levels per subsystem
    std::vector<size_t> nessential;  ///< Number of essential levels per subsystem (Default: same as nlevels)
    int ntime = 1000;  ///< Number of time steps used for time-integration
    double dt = 0.1;  ///< Time step size (ns). Determines final time: T=ntime*dt
    std::vector<double> transfreq;  ///< Fundamental transition frequencies for each oscillator (GHz)
    std::vector<double> selfkerr;  ///< Self-kerr frequencies for each oscillator (GHz)
    std::vector<double> crosskerr;  ///< Cross-kerr coupling frequencies for each oscillator coupling (GHz)
    std::vector<double> Jkl;  ///< Dipole-dipole coupling frequencies for each oscillator coupling (GHz)
    std::vector<double> rotfreq;  ///< Rotational wave approximation frequencies for each subsystem (GHz)
    LindbladType collapse_type = LindbladType::NONE;  ///< Switch between Schroedinger and Lindblad solver
    std::vector<double> decay_time;  ///< Time of decay collapse operation (T1) per oscillator (for Lindblad solver)
    std::vector<double> dephase_time;  ///< Time of dephase collapse operation (T2) per oscillator (for Lindblad solver)
    InitialConditionType initial_condition_type = InitialConditionType::BASIS;  ///< Type of initial condition
    int n_initial_conditions;  ///< Number of initial conditions
    std::vector<size_t> initial_condition_IDs;  ///< IDs of initial conditions for pure-state initialization
    std::string initial_condition_file;  ///< File to read initial conditions from (if applicable)
    std::vector<std::vector<PiPulseSegment>> apply_pipulse;  ///< Apply a pi-pulse to oscillator with specified parameters

    // Optimization options
    std::vector<std::vector<ControlSegment>> control_segments;  ///< Define the control segments for each oscillator
    bool control_enforceBC = false;  ///< Decide whether control pulses should start and end at zero
    std::vector<std::vector<ControlSegmentInitialization>> control_initializations;  ///< Set the initial control pulse parameters for each oscillator
    std::optional<std::string> control_initialization_file;  ///< File to read the control initializations from (optional)
    std::vector<std::vector<double>> control_bounds;  ///< Maximum amplitude bound for the control pulses for each oscillator segment (GHz)
    std::vector<std::vector<double>> carrier_frequencies;  ///< Carrier wave frequencies for each oscillator (GHz)
    // TODO struct or something?
    TargetType optim_target_type = TargetType::PURE;  ///< Optimization target
    std::string optim_target_file;  ///< File to read the target state from (if applicable)
    GateType optim_target_gate_type = GateType::NONE;  ///< Target gate for gate optimization (if applicable)
    std::string optim_target_gate_file;  ///< File to read the target gate matrix from (if applicable)
    std::vector<size_t> optim_target_purestate_levels;  ///< Levels of each oscillator for pure states (if applicable)
    std::vector<double> gate_rot_freq;  ///< Frequency of rotation of the target gate, for each oscillator (GHz)
    ObjectiveType optim_objective = ObjectiveType::JFROBENIUS;  ///< Objective function measure // TODO not used?
    std::vector<double> optim_weights;  ///< Weights for summing up the objective function
    double optim_atol = 1e-8;  ///< Optimization stopping tolerance based on gradient norm (absolute)
    double optim_rtol = 1e-4;  ///< Optimization stopping tolerance based on gradient norm (relative)
    double optim_ftol = 1e-8;  ///< Optimization stopping criterion based on the final time cost (absolute)
    double optim_inftol = 1e-5;  ///< Optimization stopping criterion based on the infidelity (absolute)
    int optim_maxiter = 200;  ///< Maximum number of optimization iterations
    double optim_regul = 1e-4;  ///< Coefficient of Tikhonov regularization for the design variables
    double optim_penalty = 0.0;  ///< Coefficient for adding first integral penalty term
    double optim_penalty_param = 0.5;  ///< Integral penalty parameter inside the weight (gaussian variance a)
    double optim_penalty_dpdm = 0.0;  ///< Coefficient for penalizing the integral of the second derivative of state populations
    double optim_penalty_energy = 0.0;  ///< Coefficient for penalizing the control pulse energy integral
    double optim_penalty_variation = 0.01;  ///< Coefficient for penalizing variations in control amplitudes
    bool optim_regul_tik0 = false;  ///< Switch to use Tikhonov regularization with ||x - x_0||^2 instead of ||x||^2
    // bool optim_regul_interpolate = false;  ///< TODO deprecated version

    // Output and runtypes
    std::string datadir = "./data_out";  ///< Directory for output files
    std::vector<std::vector<OutputType>> output;  ///< Specify the desired output for each oscillator
    int output_frequency = 1;  ///< Output frequency in the time domain: write output every <num> time-step
    int optim_monitor_frequency = 10;  ///< Frequency of writing output during optimization iterations
    RunType runtype = RunType::SIMULATION;  ///< Runtype options: simulation, gradient, or optimization
    bool usematfree = false;  ///< Use matrix free solver, instead of sparse matrix implementation
    LinearSolverType linearsolver_type = LinearSolverType::GMRES;  ///< Solver type for solving the linear system at each time step
    int linearsolver_maxiter = 10;  ///< Set maximum number of iterations for the linear solver
    TimeStepperType timestepper_type = TimeStepperType::IMR;  ///< The time-stepping algorithm
    int rand_seed;  ///< Fixed seed for the random number generator for reproducability
    std::string hamiltonian_file_Hsys;  ///< File to read the system Hamiltonian from
    std::string hamiltonian_file_Hc;  ///< File to read the control Hamiltonian from

  public:
    // Constructor takes all validated parameters (to be called by ConfigBuilder)
    Config(
        MPI_Comm comm_,
        // System parameters
        const std::vector<size_t>& nlevels_,
        const std::vector<size_t>& nessential_,
        int ntime_,
        double dt_,
        const std::vector<double>& transfreq_,
        const std::vector<double>& selfkerr_,
        const std::vector<double>& crosskerr_,
        const std::vector<double>& Jkl_,
        const std::vector<double>& rotfreq_,
        LindbladType collapse_type_,
        const std::vector<double>& decay_time_,
        const std::vector<double>& dephase_time_,
        InitialConditionType initial_condition_type_,
        int n_initial_conditions_,
        const std::vector<size_t>& initial_condition_IDs_,
        const std::string& initial_condition_file_,
        const std::vector<std::vector<PiPulseSegment>>& apply_pipulse_,
        // Control parameters
        const std::vector<std::vector<ControlSegment>>& control_segments_,
        bool control_enforceBC_,
        const std::vector<std::vector<ControlSegmentInitialization>>& control_initializations_,
        const std::optional<std::string>& control_initialization_file_,
        const std::vector<std::vector<double>>& control_bounds_,
        const std::vector<std::vector<double>>& carrier_frequencies_,
        // Optimization parameters
        TargetType optim_target_type_,
        const std::string& optim_target_file_,
        GateType optim_target_gate_type_,
        const std::string& optim_target_gate_file_,
        const std::vector<size_t>& optim_target_purestate_levels_,
        const std::vector<double>& gate_rot_freq_,
        ObjectiveType optim_objective_,
        const std::vector<double>& optim_weights_,
        double optim_atol_,
        double optim_rtol_,
        double optim_ftol_,
        double optim_inftol_,
        int optim_maxiter_,
        double optim_regul_,
        double optim_penalty_,
        double optim_penalty_param_,
        double optim_penalty_dpdm_,
        double optim_penalty_energy_,
        double optim_penalty_variation_,
        bool optim_regul_tik0_,
        // Output parameters
        const std::string& datadir_,
        const std::vector<std::vector<OutputType>>& output_,
        int output_frequency_,
        int optim_monitor_frequency_,
        RunType runtype_,
        bool usematfree_,
        LinearSolverType linearsolver_type_,
        int linearsolver_maxiter_,
        TimeStepperType timestepper_type_,
        int rand_seed_,
        const std::string& hamiltonian_file_Hsys_,
        const std::string& hamiltonian_file_Hc_
    );

    ~Config();

    static Config fromCfg(std::string filename, std::stringstream* log, bool quietmode = false);

    void printConfig() const;

    // getters
    const std::vector<size_t>& getNLevels() const { return nlevels; }
    const std::vector<size_t>& getNEssential() const { return nessential; }
    int getNTime() const { return ntime; }
    double getDt() const { return dt; }
    const std::vector<double>& getTransFreq() const { return transfreq; }
    const std::vector<double>& getSelfKerr() const { return selfkerr; }
    const std::vector<double>& getCrossKerr() const { return crosskerr; }
    const std::vector<double>& getJkl() const { return Jkl; }
    const std::vector<double>& getRotFreq() const { return rotfreq; }
    LindbladType getCollapseType() const { return collapse_type; }
    const std::vector<double>& getDecayTime() const { return decay_time; }
    const std::vector<double>& getDephaseTime() const { return dephase_time; }
    InitialConditionType getInitialConditionType() const { return initial_condition_type; }
    int getNInitialConditions() const { return n_initial_conditions; }
    const std::vector<size_t>& getInitialConditionIDs() const { return initial_condition_IDs; }
    const std::string& getInitialConditionFile() const { return initial_condition_file; }
    const std::vector<std::vector<PiPulseSegment>>& getApplyPiPulse() const { return apply_pipulse; }
    const std::vector<PiPulseSegment>& getApplyPiPulse(size_t i) const { return apply_pipulse[i]; }

    const std::vector<std::vector<ControlSegment>>& getControlSegments() const { return control_segments; }
    const std::vector<ControlSegment>& getControlSegment(size_t i) const { return control_segments[i]; }
    bool getControlEnforceBC() const { return control_enforceBC; }
    const std::vector<std::vector<ControlSegmentInitialization>>& getControlInitialization() const { return control_initializations; }
    const std::vector<ControlSegmentInitialization>& getControlInitialization(size_t i) const { return control_initializations[i]; }
    const std::optional<std::string>& getControlInitializationFile() const { return control_initialization_file; }
    const std::vector<std::vector<double>>& getControlBounds() const { return control_bounds; }
    const std::vector<double>& getControlBounds(size_t i_osc) const { return control_bounds[i_osc]; }
    double getControlBounds(size_t i_osc, size_t i_seg) const { return control_bounds[i_osc][i_seg]; }
    const std::vector<std::vector<double>>& getCarrierFrequencies() const { return carrier_frequencies; }
    const std::vector<double>& getCarrierFrequency(size_t i) const { return carrier_frequencies[i]; }
    TargetType getOptimTargetType() const { return optim_target_type; }
    const std::string& getOptimTargetFile() const { return optim_target_file; }
    GateType getOptimTargetGateType() const { return optim_target_gate_type; }
    const std::string& getOptimTargetGateFile() const { return optim_target_gate_file; }
    const std::vector<size_t>& getOptimTargetPurestateLevels() const { return optim_target_purestate_levels; }
    const std::vector<double>& getGateRotFreq() const { return gate_rot_freq; }
    ObjectiveType getOptimObjective() const { return optim_objective; }
    const std::vector<double>& getOptimWeights() const { return optim_weights; }
    double getOptimAtol() const { return optim_atol; }
    double getOptimRtol() const { return optim_rtol; }
    double getOptimFtol() const { return optim_ftol; }
    double getOptimInftol() const { return optim_inftol; }
    int getOptimMaxiter() const { return optim_maxiter; }
    double getOptimRegul() const { return optim_regul; }
    double getOptimPenalty() const { return optim_penalty; }
    double getOptimPenaltyParam() const { return optim_penalty_param; }
    double getOptimPenaltyDpdm() const { return optim_penalty_dpdm; }
    double getOptimPenaltyEnergy() const { return optim_penalty_energy; }
    double getOptimPenaltyVariation() const { return optim_penalty_variation; }
    bool getOptimRegulTik0() const { return optim_regul_tik0; }
    bool getOptimRegulInterpolate() const { return false; } // Deprecated - always return false

    const std::string& getDataDir() const { return datadir; }
    const std::vector<std::vector<OutputType>>& getOutput() const { return output; }
    const std::vector<OutputType>& getOutput(size_t i) const { return output[i]; }
    int getOutputFrequency() const { return output_frequency; }
    int getOptimMonitorFrequency() const { return optim_monitor_frequency; }
    RunType getRuntype() const { return runtype; }
    bool getUseMatFree() const { return usematfree; }
    LinearSolverType getLinearSolverType() const { return linearsolver_type; }
    int getLinearSolverMaxiter() const { return linearsolver_maxiter; }
    TimeStepperType getTimestepperType() const { return timestepper_type; }
    int getRandSeed() const { return rand_seed; }
    const std::string& getHamiltonianFileHsys() const { return hamiltonian_file_Hsys; }
    const std::string& getHamiltonianFileHc() const { return hamiltonian_file_Hc; }

  };
