#include "defs.hpp"
#include "config_types.hpp"

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

  std::string toString() const {
    return "file, " + filename;
  }
};

struct PureInitialCondition {
  std::vector<size_t> level_indices; ///< Quantum level for each oscillator

  std::string toString() const {
    std::string out = "pure";
    for (size_t idx : level_indices) {
      out += ", " + std::to_string(idx);
    }
    return out;
  }
};

struct OscillatorIDsInitialCondition {
  std::vector<size_t> osc_IDs; ///< Oscillator IDs

    std::string toString(std::string name) const {
    std::string out = name;
    for (size_t idx : osc_IDs) {
      out += ", " + std::to_string(idx);
    }
    return out;
  }
};
struct EnsembleInitialCondition : public OscillatorIDsInitialCondition {
  std::string toString() const {
    return OscillatorIDsInitialCondition::toString("ensemble");
  }
};

struct DiagonalInitialCondition : public OscillatorIDsInitialCondition {
  std::string toString() const {
    return OscillatorIDsInitialCondition::toString("ensemble");
  }
};

struct BasisInitialCondition : public OscillatorIDsInitialCondition {
  std::string toString() const {
    return OscillatorIDsInitialCondition::toString("basis");
  }
};

struct ThreeStatesInitialCondition {
  std::string toString() const {
    return "3states";
  }
};

struct NPlusOneInitialCondition {
  std::string toString() const {
    return "nplus1";
  }
};

struct PerformanceInitialCondition {
  std::string toString() const {
    return "performance";
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
  double atol;      ///< Absolute gradient tolerance
  double rtol;      ///< Relative gradient tolerance
  double ftol;      ///< Final time cost tolerance
  double inftol;    ///< Infidelity tolerance
  size_t maxiter;   ///< Maximum iterations
};

/**
 * @brief Grouped optimization penalty settings.
 *
 * Groups all penalty terms used for control pulse regularization.
 */
struct OptimPenalty {
  double penalty;           ///< First integral penalty coefficient
  double penalty_param;     ///< Gaussian variance parameter
  double penalty_dpdm;      ///< Second derivative penalty coefficient
  double penalty_energy;    ///< Energy penalty coefficient
  double penalty_variation; ///< Amplitude variation penalty coefficient
};

struct GateOptimTarget {
  GateType gate_type = GateType::NONE;          ///< Gate type (for gate targets)
  std::string gate_file;                        ///< Gate file (for gate from file)

  std::string toString() const {
    if (gate_type == GateType::FILE) {
      return "gate, file, " + gate_file;
    } else {
      auto it = std::find_if(GATE_TYPE_MAP.begin(), GATE_TYPE_MAP.end(),
                             [this](const auto& pair) { return pair.second == gate_type; });
      return "gate, " + (it != GATE_TYPE_MAP.end() ? it->first : "unknown");
    }
  }
};

struct PureOptimTarget {
  std::vector<size_t> purestate_levels;         ///< Pure state levels (for pure targets)

  std::string toString() const {
    std::string out = "pure";
    for (size_t level : purestate_levels) {
      out += ", " + std::to_string(level);
    }
    return out;
  }
};

struct FileOptimTarget {
  std::string file;                             ///< Target file (for file targets)

  std::string toString() const {
    return "file, " + file;
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

struct ControlSegmentInitializationFile {
  std::string filename; ///< Filename to read control segment initialization from

  std::string toString() const {
    return "file, " + filename;
  }
};

struct ControlSegmentInitializationConstant {
  double amplitude; ///< Initial control pulse amplitude
  double phase; ///< Initial control pulse phase

  std::string toString() const {
    return "constant, " + std::to_string(amplitude) + ", " + std::to_string(phase);
  }
};

struct ControlSegmentInitializationRandom {
  double amplitude; ///< Initial control pulse amplitude
  double phase; ///< Initial control pulse phase

  std::string toString() const {
    return "random, " + std::to_string(amplitude) + ", " + std::to_string(phase);
  }
};

using ControlSegmentInitialization = std::variant<
  ControlSegmentInitializationFile,
  ControlSegmentInitializationConstant,
  ControlSegmentInitializationRandom
>;

// TODO remove carrier_frequencies? the others are per segment?
/**
 * @brief Per-oscillator optimization configuration settings.
 *
 * Groups all optimization-related settings for a single oscillator.
 */
struct OscillatorOptimization {
  std::vector<ControlSegment> control_segments;                    ///< Control segments for this oscillator
  std::vector<ControlSegmentInitialization> control_initializations; ///< Control initializations for this oscillator for each segment
  std::vector<double> control_bounds;                              ///< Control bounds for this oscillator for each segment
  std::vector<double> carrier_frequencies;                         ///< Carrier frequencies for this oscillator
};
/**
 * @brief Final validated configuration class.
 *
 * Contains only validated, typed configuration parameters. All fields are required
 * and have been validated by ConfigBuilder. This class is immutable after construction.
 */
class Config {
  private:
    // MPI and logging (still needed for runtime operations)
    MPI_Comm comm; ///< MPI communicator for parallel operations.
    int mpi_rank; ///< MPI rank of the current process.

    std::stringstream* log; ///< Pointer to log stream for output messages.
    bool quietmode; ///< Flag to control verbose output.

    // General options
    std::vector<size_t> nlevels;  ///< Number of levels per subsystem
    std::vector<size_t> nessential;  ///< Number of essential levels per subsystem (Default: same as nlevels)
    size_t ntime;  ///< Number of time steps used for time-integration
    double dt;  ///< Time step size (ns). Determines final time: T=ntime*dt
    std::vector<double> transfreq;  ///< Fundamental transition frequencies for each oscillator (GHz)
    std::vector<double> selfkerr;  ///< Self-kerr frequencies for each oscillator (GHz)
    std::vector<double> crosskerr;  ///< Cross-kerr coupling frequencies for each oscillator coupling (GHz)
    std::vector<double> Jkl;  ///< Dipole-dipole coupling frequencies for each oscillator coupling (GHz)
    std::vector<double> rotfreq;  ///< Rotational wave approximation frequencies for each subsystem (GHz)
    LindbladType collapse_type;  ///< Switch between Schroedinger and Lindblad solver
    std::vector<double> decay_time;  ///< Time of decay collapse operation (T1) per oscillator (for Lindblad solver)
    std::vector<double> dephase_time;  ///< Time of dephase collapse operation (T2) per oscillator (for Lindblad solver)
    size_t n_initial_conditions;  ///< Number of initial conditions
    InitialCondition initial_condition;  ///< Initial condition configuration
    std::vector<std::vector<PiPulseSegment>> apply_pipulse;  ///< Apply a pi-pulse to oscillator with specified parameters
    std::optional<std::string> hamiltonian_file_Hsys;  ///< File to read the system Hamiltonian from
    std::optional<std::string> hamiltonian_file_Hc;  ///< File to read the control Hamiltonian from

    // Optimization options
    bool control_enforceBC;  ///< Decide whether control pulses should start and end at zero
    std::vector<OscillatorOptimization> oscillator_optimization;  ///< Optimization configuration for each oscillator
    OptimTargetSettings target;  ///< Grouped optimization target configuration
    std::vector<double> gate_rot_freq;  ///< Frequency of rotation of the target gate, for each oscillator (GHz)
    ObjectiveType optim_objective;  ///< Objective function measure
    std::vector<double> optim_weights;  ///< Weights for summing up the objective function
    OptimTolerance tolerance;  ///< Grouped optimization stopping criteria and iteration limits
    double optim_regul;  ///< Coefficient of Tikhonov regularization for the design variables
    OptimPenalty penalty;  ///< Grouped optimization penalty coefficients
    bool optim_regul_tik0;  ///< Switch to use Tikhonov regularization with ||x - x_0||^2 instead of ||x||^2

    // Output and runtypes
    std::string datadir;  ///< Directory for output files
    std::vector<std::vector<OutputType>> output;  ///< Specify the desired output for each oscillator
    size_t output_frequency;  ///< Output frequency in the time domain: write output every <num> time-step
    size_t optim_monitor_frequency;  ///< Frequency of writing output during optimization iterations
    RunType runtype;  ///< Runtype options: simulation, gradient, or optimization
    bool usematfree;  ///< Use matrix free solver, instead of sparse matrix implementation
    LinearSolverType linearsolver_type;  ///< Solver type for solving the linear system at each time step
    size_t linearsolver_maxiter;  ///< Set maximum number of iterations for the linear solver
    TimeStepperType timestepper_type;  ///< The time-stepping algorithm
    int rand_seed;  ///< Fixed seed for the random number generator for reproducibility

  public:
    Config(
      MPI_Comm comm_,
      std::stringstream& log_,
      bool quietmode_,
      // General parameters
      const std::optional<std::vector<size_t>>& nlevels_,
      const std::optional<std::vector<size_t>>& nessential_,
      const std::optional<size_t>& ntime_,
      const std::optional<double>& dt_,
      const std::optional<std::vector<double>>& transfreq_,
      const std::optional<std::vector<double>>& selfkerr_,
      const std::optional<std::vector<double>>& crosskerr_,
      const std::optional<std::vector<double>>& Jkl_,
      const std::optional<std::vector<double>>& rotfreq_,
      const std::optional<LindbladType>& collapse_type_,
      const std::optional<std::vector<double>>& decay_time_,
      const std::optional<std::vector<double>>& dephase_time_,
      const std::optional<InitialConditionConfig>& initialcondition_,
      const std::optional<std::vector<PiPulseConfig>>& apply_pipulse_,
      const std::optional<std::string>& hamiltonian_file_Hsys_,
      const std::optional<std::string>& hamiltonian_file_Hc_,
      // Control and optimization parameters
      const std::optional<std::map<int, std::vector<ControlSegmentConfig>>>& indexed_control_segments_,
      const std::optional<bool>& control_enforceBC_,
      const std::optional<std::map<int, std::vector<ControlInitializationConfig>>>& indexed_control_init_,
      const std::optional<std::map<int, std::vector<double>>>& indexed_control_bounds_,
      const std::optional<std::map<int, std::vector<double>>>& indexed_carrier_frequencies_,
      const std::optional<OptimTargetConfig>& optim_target_,
      const std::optional<std::vector<double>>& gate_rot_freq_,
      const std::optional<ObjectiveType>& optim_objective_,
      const std::optional<std::vector<double>>& optim_weights_,
      const std::optional<double>& optim_atol_,
      const std::optional<double>& optim_rtol_,
      const std::optional<double>& optim_ftol_,
      const std::optional<double>& optim_inftol_,
      const std::optional<size_t>& optim_maxiter_,
      const std::optional<double>& optim_regul_,
      const std::optional<double>& optim_penalty_,
      const std::optional<double>& optim_penalty_param_,
      const std::optional<double>& optim_penalty_dpdm_,
      const std::optional<double>& optim_penalty_energy_,
      const std::optional<double>& optim_penalty_variation_,
      const std::optional<bool>& optim_regul_tik0_,
      // Output parameters
      const std::optional<std::string>& datadir_,
      const std::optional<std::map<int, std::vector<OutputType>>>& indexed_output_,
      const std::optional<size_t>& output_frequency_,
      const std::optional<size_t>& optim_monitor_frequency_,
      const std::optional<RunType>& runtype_,
      const std::optional<bool>& usematfree_,
      const std::optional<LinearSolverType>& linearsolver_type_,
      const std::optional<size_t>& linearsolver_maxiter_,
      const std::optional<TimeStepperType>& timestepper_type_,
      const std::optional<int>& rand_seed_
    );

    ~Config();

    static Config fromCfg(std::string filename, std::stringstream* log, bool quietmode = false);

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

    const std::vector<OscillatorOptimization>& getOscillators() const { return oscillator_optimization; } // TODO rename
    const OscillatorOptimization& getOscillator(size_t i) const { return oscillator_optimization[i]; }
    const std::vector<ControlSegment>& getControlSegments(size_t i_osc) const { return oscillator_optimization[i_osc].control_segments; }
    bool getControlEnforceBC() const { return control_enforceBC; }
    const std::vector<ControlSegmentInitialization>& getControlInitializations(size_t i_osc) const { return oscillator_optimization[i_osc].control_initializations; }

    const std::optional<std::string> getControlInitializationFile() const {
      if (!std::holds_alternative<ControlSegmentInitializationFile>(oscillator_optimization[0].control_initializations[0])) {
        return std::nullopt;
      }
      return std::get<ControlSegmentInitializationFile>(oscillator_optimization[0].control_initializations[0]).filename;
    }

    const std::vector<double>& getControlBounds(size_t i_osc) const { return oscillator_optimization[i_osc].control_bounds; }
    double getControlBound(size_t i_osc, size_t i_seg) const { return oscillator_optimization[i_osc].control_bounds[i_seg]; }
    const std::vector<double>& getCarrierFrequencies(size_t i_osc) const { return oscillator_optimization[i_osc].carrier_frequencies; }
    double getCarrierFrequency(size_t i_osc, size_t i_seg) const { return oscillator_optimization[i_osc].carrier_frequencies[i_seg]; }
    const OptimTargetSettings& getOptimTarget() const { return target; }
    TargetType getOptimTargetType() const {
      if (std::holds_alternative<GateOptimTarget>(target)) return TargetType::GATE;
      if (std::holds_alternative<PureOptimTarget>(target)) return TargetType::PURE;
      return TargetType::FROMFILE;
    }
    const std::string& getOptimTargetFile() const {
      return std::get<FileOptimTarget>(target).file;
    }
    GateType getOptimTargetGateType() const {
      return std::get<GateOptimTarget>(target).gate_type;
    }
    const std::string& getOptimTargetGateFile() const {
      return std::get<GateOptimTarget>(target).gate_file;
    }
    const std::vector<size_t>& getOptimTargetPurestateLevels() const {
      return std::get<PureOptimTarget>(target).purestate_levels;
    }
    const std::vector<double>& getGateRotFreq() const { return gate_rot_freq; }
    ObjectiveType getOptimObjective() const { return optim_objective; }
    const std::vector<double>& getOptimWeights() const { return optim_weights; }
    const OptimTolerance& getOptimTolerance() const { return tolerance; }
    double getOptimRegul() const { return optim_regul; }
    const OptimPenalty& getOptimPenalty() const { return penalty; }
    bool getOptimRegulTik0() const { return optim_regul_tik0; }
    bool getOptimRegulInterpolate() const { return false; } // Deprecated - always return false

    const std::string& getDataDir() const { return datadir; }
    const std::vector<std::vector<OutputType>>& getOutput() const { return output; }
    const std::vector<OutputType>& getOutput(size_t i) const { return output[i]; }
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

    // Conversion helper methods
    InitialCondition convertInitialCondition(const InitialConditionConfig& config);
    void convertInitialCondition(const std::optional<InitialConditionConfig>& config);
    void convertOptimTarget(const std::optional<OptimTargetConfig>& config);
    void convertControlSegments(const std::optional<std::map<int, std::vector<ControlSegmentConfig>>>& indexed);
    void convertControlInitializations(const std::optional<std::map<int, std::vector<ControlInitializationConfig>>>& indexed);
    void convertPiPulses(const std::optional<std::vector<PiPulseConfig>>& pulses);
    void convertIndexedOutput(const std::optional<std::map<int, std::vector<OutputType>>>& indexed);
    void convertIndexedControlBounds(const std::optional<std::map<int, std::vector<double>>>& indexed);
    void convertIndexedCarrierFreqs(const std::optional<std::map<int, std::vector<double>>>& indexed);

    void setNumInitialConditions();
    void setOptimWeights(const std::optional<std::vector<double>>& optim_weights_);

    // Helper for indexed map conversion
    template<typename T>
    std::vector<std::vector<T>> convertIndexedToVectorVector(const std::optional<std::map<int, std::vector<T>>>& indexed_map);
};
