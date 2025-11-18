#include "defs.hpp"
#include "config_types.hpp"

#include <toml++/toml.hpp>

#include <cstddef>
#include <vector>
#include <string>
#include <petsc.h>
#include <sstream>
#include <optional>
#include <variant>

#pragma once

// Individual structs for each initial condition type
struct FromFileInitialCondition {
  std::string filename; ///< File to read initial condition from

  std::string toString() const;
};

struct PureInitialCondition {
  std::vector<size_t> levels; ///< Quantum level for each oscillator

  std::string toString() const;
};

struct OscillatorIDsInitialCondition {
  std::vector<size_t> osc_IDs; ///< Oscillator IDs

  std::string toString(std::string name) const;
};
struct EnsembleInitialCondition : public OscillatorIDsInitialCondition {
  std::string toString() const {
    return OscillatorIDsInitialCondition::toString("ensemble");
  }
};

struct DiagonalInitialCondition : public OscillatorIDsInitialCondition {
  std::string toString() const {
    return OscillatorIDsInitialCondition::toString("diagonal");
  }
};

struct BasisInitialCondition : public OscillatorIDsInitialCondition {
  std::string toString() const {
    return OscillatorIDsInitialCondition::toString("basis");
  }
};

struct ThreeStatesInitialCondition {
  std::string toString() const {
    return "type = \"3states\"";
  }
};

struct NPlusOneInitialCondition {
  std::string toString() const {
    return "type = \"nplus1\"";
  }
};

struct PerformanceInitialCondition {
  std::string toString() const {
    return "type = \"performance\"";
  }
};

using InitialCondition = std::variant<
  FromFileInitialCondition,
  PureInitialCondition,
  EnsembleInitialCondition,
  DiagonalInitialCondition,
  BasisInitialCondition,
  ThreeStatesInitialCondition,
  NPlusOneInitialCondition,
  PerformanceInitialCondition
>;

/**
 * @brief Grouped optimization tolerance settings.
 *
 * Groups all optimization stopping criteria and iteration limits.
 */
struct OptimTolerance {
  double atol = 1e-8;      ///< Absolute gradient tolerance
  double rtol = 1e-4;      ///< Relative gradient tolerance
  double ftol = 1e-8;      ///< Final time cost tolerance
  double inftol = 1e-5;    ///< Infidelity tolerance
  size_t maxiter = 200;   ///< Maximum iterations
};

/**
 * @brief Grouped optimization penalty settings.
 *
 * Groups all penalty terms used for control pulse regularization.
 */
struct OptimPenalty {
  double penalty = 0.0;           ///< First integral penalty coefficient
  double penalty_param = 0.5;     ///< Gaussian variance parameter
  double penalty_dpdm = 0.0;      ///< Second derivative penalty coefficient
  double penalty_energy = 0.0;    ///< Energy penalty coefficient
  double penalty_variation = 0.01; ///< Amplitude variation penalty coefficient
};

struct GateOptimTarget {
  GateType gate_type = GateType::NONE;          ///< Gate type (for gate targets)
  std::string gate_file;                        ///< Gate file (for gate from file)

  std::string toString() const {
    if (gate_type == GateType::FILE) {
      return "{target_type = \"gate\", gate_type = \"file\", gate_file = \"" + gate_file + "\"}";
    } else {
      auto it = std::find_if(GATE_TYPE_MAP.begin(), GATE_TYPE_MAP.end(),
                             [this](const auto& pair) { return pair.second == gate_type; });
      std::string gate_name = (it != GATE_TYPE_MAP.end() ? it->first : "unknown");
      return "{target_type = \"gate\", gate_type = \"" + gate_name + "\"}";
    }
  }
};

struct PureOptimTarget {
  std::vector<size_t> purestate_levels;         ///< Pure state levels (for pure targets)

  std::string toString() const {
    std::string out = "{target_type = \"pure\", levels = [";
    for (size_t i = 0; i < purestate_levels.size(); ++i) {
      out += std::to_string(purestate_levels[i]);
      if (i < purestate_levels.size() - 1) out += ", ";
    }
    out += "]}";
    return out;
  }
};

struct FileOptimTarget {
  std::string file;                             ///< Target file (for file targets)

  std::string toString() const {
    return "{target_type = \"file\", filename = \"" + file + "\"}";
  }
};

using OptimTargetSettings = std::variant<GateOptimTarget, PureOptimTarget, FileOptimTarget>;

inline std::string toString(const OptimTargetSettings& target) {
  return std::visit([](const auto& t) { return t.toString(); }, target);
}

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
  double tstop; ///< Stop time of the control segment
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

struct ControlSegmentInitialization {
  ControlSegmentInitType type;
  double amplitude; ///< Initial control pulse amplitude
  double phase; ///< Initial control pulse phase

  std::string toString() const;
};

/**
 * @brief Final validated configuration class.
 *
 * Contains only validated, typed configuration parameters. All fields are required
 * and have been validated by CfgParser. This class is immutable after construction.
 */
class Config {
  private:
    // MPI and logging (still needed for runtime operations)
    int mpi_rank; ///< MPI rank of the current process.

    std::stringstream& log; ///< Reference to log stream for output messages.
    bool quietmode; ///< Flag to control verbose output.

    // General options
    std::vector<size_t> nlevels;  ///< Number of levels per subsystem
    std::vector<size_t> nessential;  ///< Number of essential levels per subsystem (Default: same as nlevels)
    size_t ntime = 1000;  ///< Number of time steps used for time-integration
    double dt = 0.1;  ///< Time step size (ns). Determines final time: T=ntime*dt
    std::vector<double> transfreq;  ///< Fundamental transition frequencies for each oscillator (GHz)
    std::vector<double> selfkerr;  ///< Self-kerr frequencies for each oscillator (GHz)
    std::vector<double> crosskerr;  ///< Cross-kerr coupling frequencies for each oscillator coupling (GHz)
    std::vector<double> Jkl;  ///< Dipole-dipole coupling frequencies for each oscillator coupling (GHz)
    std::vector<double> rotfreq;  ///< Rotational wave approximation frequencies for each subsystem (GHz)
    LindbladType collapse_type = LindbladType::NONE;  ///< Switch between Schroedinger and Lindblad solver
    std::vector<double> decay_time;  ///< Time of decay collapse operation (T1) per oscillator (for Lindblad solver)
    std::vector<double> dephase_time;  ///< Time of dephase collapse operation (T2) per oscillator (for Lindblad solver)
    size_t n_initial_conditions;  ///< Number of initial conditions
    InitialCondition initial_condition;  ///< Initial condition configuration
    std::vector<std::vector<PiPulseSegment>> apply_pipulse;  ///< Apply a pi-pulse to oscillator with specified parameters
    std::optional<std::string> hamiltonian_file_Hsys;  ///< File to read the system Hamiltonian from
    std::optional<std::string> hamiltonian_file_Hc;  ///< File to read the control Hamiltonian from

    // Optimization options
    bool control_enforceBC = true;  ///< Decide whether control pulses should start and end at zero
    std::optional<std::string> control_initialization_file;  ///< Global control initialization file for all oscillators
    std::vector<std::vector<ControlSegment>> control_segments;  ///< Control segments for each oscillator
    std::vector<std::vector<ControlSegmentInitialization>> control_initializations;  ///< Control initializations for each oscillator
    std::vector<std::vector<double>> control_bounds;  ///< Control bounds for each oscillator
    std::vector<std::vector<double>> carrier_frequencies;  ///< Carrier frequencies for each oscillator
    OptimTargetSettings optim_target;  ///< Grouped optimization target configuration
    std::vector<double> gate_rot_freq = std::vector<double>{0.0};  ///< Frequency of rotation of the target gate, for each oscillator (GHz)
    ObjectiveType optim_objective = ObjectiveType::JFROBENIUS;  ///< Objective function measure
    std::vector<double> optim_weights;  ///< Weights for summing up the objective function
    OptimTolerance tolerance;  ///< Grouped optimization stopping criteria and iteration limits
    double optim_regul = 1e-4;  ///< Coefficient of Tikhonov regularization for the design variables
    OptimPenalty penalty;  ///< Grouped optimization penalty coefficients
    bool optim_regul_tik0 = false;  ///< Switch to use Tikhonov regularization with ||x - x_0||^2 instead of ||x||^2

    // Output and runtypes
    std::string datadir = "./data_out";  ///< Directory for output files
    std::vector<std::vector<OutputType>> output_to_write;  ///< Specify the desired output for each oscillator
    size_t output_frequency = 1;  ///< Output frequency in the time domain: write output every <num> time-step
    size_t optim_monitor_frequency = 10;  ///< Frequency of writing output during optimization iterations
    RunType runtype = RunType::SIMULATION;  ///< Runtype options: simulation, gradient, or optimization
    bool usematfree = false;  ///< Use matrix free solver, instead of sparse matrix implementation
    LinearSolverType linearsolver_type = LinearSolverType::GMRES;  ///< Solver type for solving the linear system at each time step
    size_t linearsolver_maxiter = 10;  ///< Set maximum number of iterations for the linear solver
    TimeStepperType timestepper_type = TimeStepperType::IMR;  ///< The time-stepping algorithm
    int rand_seed;  ///< Fixed seed for the random number generator for reproducibility

  public:
    Config(
      int mpi_rank_,
      std::stringstream& log_,
      bool quietmode_,
      const toml::table& table
    );

    Config(
      int mpi_rank_,
      std::stringstream& log_,
      bool quietmode_,
      const ConfigSettings& settings
    );

    ~Config();

    static Config fromFile(int mpi_rank, const std::string& filename, std::stringstream* log, bool quietmode = false);
    static Config fromToml(int mpi_rank, const std::string& toml_filename, std::stringstream* log, bool quietmode = false);
    static Config fromTomlString(int mpi_rank, const std::string& toml_content, std::stringstream* log, bool quietmode = false);
    static Config fromCfg(int mpi_rank, const std::string& cfg_filename, std::stringstream* log, bool quietmode = false);
    static Config fromCfgString(int mpi_rank, const std::string& cfg_content, std::stringstream* log, bool quietmode = false);

    void printConfig() const;

    // getters
    const std::vector<size_t>& getNLevels() const { return nlevels; }
    const std::vector<size_t>& getNEssential() const { return nessential; }
    size_t getNTime() const { return ntime; }
    double getDt() const { return dt; }

    const std::vector<double>& getTransFreq() const { return transfreq; }
    const std::vector<double>& getSelfKerr() const { return selfkerr; }
    const std::vector<double>& getCrossKerr() const { return crosskerr; }
    const std::vector<double>& getJkl() const { return Jkl; }
    const std::vector<double>& getRotFreq() const { return rotfreq; }
    LindbladType getCollapseType() const { return collapse_type; }
    const std::vector<double>& getDecayTime() const { return decay_time; }
    const std::vector<double>& getDephaseTime() const { return dephase_time; }
    size_t getNInitialConditions() const { return n_initial_conditions; }
    const InitialCondition& getInitialCondition() const { return initial_condition; }
    const std::vector<std::vector<PiPulseSegment>>& getApplyPiPulses() const { return apply_pipulse; }
    const std::vector<PiPulseSegment>& getApplyPiPulse(size_t i_osc) const { return apply_pipulse[i_osc]; }
    const std::optional<std::string>& getHamiltonianFileHsys() const { return hamiltonian_file_Hsys; }
    const std::optional<std::string>& getHamiltonianFileHc() const { return hamiltonian_file_Hc; }

    const std::vector<ControlSegment>& getControlSegments(size_t i_osc) const { return control_segments[i_osc]; }
    bool getControlEnforceBC() const { return control_enforceBC; }
    const std::vector<ControlSegmentInitialization>& getControlInitializations(size_t i_osc) const { return control_initializations[i_osc]; }
    const std::optional<std::string> getControlInitializationFile() const { return control_initialization_file; }
    const std::vector<double>& getControlBounds(size_t i_osc) const { return control_bounds[i_osc]; }
    double getControlBound(size_t i_osc, size_t i_seg) const { return control_bounds[i_osc][i_seg]; }
    const std::vector<double>& getCarrierFrequencies(size_t i_osc) const { return carrier_frequencies[i_osc]; }
    double getCarrierFrequency(size_t i_osc, size_t i_seg) const { return carrier_frequencies[i_osc][i_seg]; }
    const OptimTargetSettings& getOptimTarget() const { return optim_target; }
    const std::vector<double>& getGateRotFreq() const { return gate_rot_freq; }
    ObjectiveType getOptimObjective() const { return optim_objective; }
    const std::vector<double>& getOptimWeights() const { return optim_weights; }
    const OptimTolerance& getOptimTolerance() const { return tolerance; }
    double getOptimRegul() const { return optim_regul; }
    const OptimPenalty& getOptimPenalty() const { return penalty; }
    bool getOptimRegulTik0() const { return optim_regul_tik0; }

    const std::string& getDataDir() const { return datadir; }
    const std::vector<std::vector<OutputType>>& getOutput() const { return output_to_write; }
    const std::vector<OutputType>& getOutput(size_t i) const { return output_to_write[i]; }
    size_t getOutputFrequency() const { return output_frequency; }
    size_t getOptimMonitorFrequency() const { return optim_monitor_frequency; }
    RunType getRuntype() const { return runtype; }
    bool getUseMatFree() const { return usematfree; }
    LinearSolverType getLinearSolverType() const { return linearsolver_type; }
    size_t getLinearSolverMaxiter() const { return linearsolver_maxiter; }
    TimeStepperType getTimestepperType() const { return timestepper_type; }
    int getRandSeed() const { return rand_seed; }

private:
    void finalize();
    void validate() const;

    size_t computeNumInitialConditions() const;
    void setRandSeed(std::optional<int> rand_seed_);

    // Conversion helper methods
    InitialCondition parseInitialCondition(const InitialConditionConfig& config) const;
    InitialCondition parseInitialCondition(const std::optional<InitialConditionConfig>& config) const;
    OptimTargetSettings parseOptimTarget(const std::optional<OptimTargetConfig>& opt_config,const std::vector<size_t>& nlevels) const;
    std::vector<std::vector<ControlSegment>> parseControlSegments(const std::optional<std::map<int, std::vector<ControlSegmentConfig>>>& segments_opt) const;
    std::vector<ControlSegment> parseOscControlSegments(const std::vector<ControlSegmentConfig>& segments) const;
    ControlSegment parseControlSegment(const ControlSegmentConfig& seg_config) const;
    ControlSegment parseControlSegment(const toml::table& table) const;
    std::vector<std::vector<ControlSegmentInitialization>>  parseControlInitializations(const std::optional<std::map<int, std::vector<ControlInitializationConfig>>>& init_configs) const;
    ControlSegmentInitialization parseControlInitialization(const toml::table& table) const;
    std::vector<double> parseOptimWeights(const std::optional<std::vector<double>>& optim_weights_) const;

    template<typename T>
    std::vector<std::vector<T>> parseIndexedWithDefaults(
        const std::optional<std::map<int, std::vector<T>>>& indexed,
        size_t num_entries,
        const std::vector<T>& default_values = {}) const;
};
