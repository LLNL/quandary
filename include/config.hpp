#include "defs.hpp"
#include "util.hpp"

#include <cstddef>
#include <vector>
#include <string>
#include <petsc.h>
#include <sstream>
#include <set>
#include <optional>
# include <variant>

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
 * @brief Configuration parameter management class with typed member variables.
 *
 * The `Config` class provides a type-safe way to manage Quandary configuration
 * parameters. It supports both programmatic configuration and reading from
 * configuration files for backward compatibility.
 */
class Config {
  public:
    std::stringstream* log; ///< Pointer to log stream for output messages.
    bool quietmode; ///< Flag to control verbose output.

  private:
    std::unordered_map<std::string, std::function<void(const std::string&)>> setters; ///< Setters from config string

    // MPI and logging
    MPI_Comm comm; ///< MPI communicator for parallel operations.
    int mpi_rank; ///< MPI rank of the current process.

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
    // TODO builder pattern instead of setters and public empty constructor??
    // Constructors
    Config();
    Config(MPI_Comm comm_, std::stringstream& logstream, bool quietmode=false);
    ~Config();

    static Config createFromFile(const std::string& filename, MPI_Comm comm, std::stringstream& logstream, bool quietmode = false);
    void loadFromFile(const std::string& filename);
    void applyConfigLine(const std::string& line);
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
    bool getOptimRegulInterpolate() const { return optim_regul_interpolate; }

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

  private:
    void validate();

    // parse and set from string config
    void setNEssential(const std::string& value);
    void setInitialConditions(const std::string& init_cond_str, LindbladType collapse_type_);
    void setApplyPiPulse(const std::string& value);
    void setOptimTarget(const std::string& value);
    void setControlInitialization(const std::string& value, ControlType control_type);
    void setRandSeed(int value);

    std::vector<std::string> split(const std::string& str, char delimiter = ',');

    template<typename T>
    T convertFromString(const std::string& str) {
      return str;
    }

    template<>
    bool convertFromString<bool>(const std::string& str) {
      const std::set<std::string> trueValues = {"true", "yes", "1"};
      std::string lowerStr = str;
      std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
      return trueValues.find(lowerStr) != trueValues.end();
    }

    template<>
    int convertFromString<int>(const std::string& str) {
      return std::stoi(str);
    }

    template<>
    size_t convertFromString<size_t>(const std::string& str) {
      return std::stoi(str);
    }

    template<>
    double convertFromString<double>(const std::string& str) {
      return std::stod(str);
    }

    template<>
    std::vector<double> convertFromString<std::vector<double>>(const std::string& str) {
      std::vector<double> vec;
      auto parts = split(str);
      vec.reserve(parts.size());
      for (const auto& part : parts) {
        vec.push_back(convertFromString<double>(part));
      }
      return vec;
    }

    template<>
    std::vector<int> convertFromString<std::vector<int>>(const std::string& str) {
      std::vector<int> vec;
      auto parts = split(str);
      vec.reserve(parts.size());
      for (const auto& part : parts) {
        vec.push_back(convertFromString<int>(part));
      }
      return vec;
    }

    template<>
    RunType convertFromString<RunType>(const std::string& str) {
      auto it = RUN_TYPE_MAP.find(toLower(str));
      if (it == RUN_TYPE_MAP.end()) {
        logErrorToRank0(mpi_rank, "\n\n ERROR: Unknown run type: " + str + ".\n");
        exit(1);
      }
      return it->second;
    }

    template<>
    LindbladType convertFromString<LindbladType>(const std::string& str) {
      auto it = LINDBLAD_TYPE_MAP.find(toLower(str));
      if (it == LINDBLAD_TYPE_MAP.end()) {
        logErrorToRank0(mpi_rank, "\n\n ERROR: Unknown collapse type: " + str + ".\n");
        exit(1);
      }
      return it->second;
    }

    template<>
    LinearSolverType convertFromString<LinearSolverType>(const std::string& str) {
      auto it = LINEAR_SOLVER_TYPE_MAP.find(toLower(str));
      if (it == LINEAR_SOLVER_TYPE_MAP.end()) {
        logErrorToRank0(mpi_rank, "\n\n ERROR: Unknown linear solver type: " + str + ".\n");
        exit(1);
      }
      return it->second;
    }

    template<>
    TimeStepperType convertFromString<TimeStepperType>(const std::string& str) {
      auto it = TIME_STEPPER_TYPE_MAP.find(toLower(str));
      if (it == TIME_STEPPER_TYPE_MAP.end()) {
        logErrorToRank0(mpi_rank, "\n\n ERROR: Unknown time stepper type: " + str + ".\n");
        exit(1);
      }
      return it->second;
    }

    template<>
    TargetType convertFromString<TargetType>(const std::string& str) {
      auto it = TARGET_TYPE_MAP.find(toLower(str));
      if (it == TARGET_TYPE_MAP.end()) {
        logErrorToRank0(mpi_rank, "\n\n ERROR: Unknown target type: " + str + ".\n");
        exit(1);
      }
      return it->second;
    }

    template<>
    InitialConditionType convertFromString<InitialConditionType>(const std::string& str) {
      auto it = INITCOND_TYPE_MAP.find(toLower(str));
      if (it == INITCOND_TYPE_MAP.end()) {
        logErrorToRank0(mpi_rank, "\n\n ERROR: Unknown initial condition type: " + str + ".\n");
        exit(1);
      }
      return it->second;
    }

    template<>
    GateType convertFromString<GateType>(const std::string& str) {
      auto it = GATE_TYPE_MAP.find(toLower(str));
      if (it == GATE_TYPE_MAP.end()) {
        logErrorToRank0(mpi_rank, "\n\n ERROR: Unknown gate type: " + str + ".\n");
        exit(1);
      }
      return it->second;
    }

    template<>
    OutputType convertFromString<OutputType>(const std::string& str) {
      auto it = OUTPUT_TYPE_MAP.find(toLower(str));
      if (it == OUTPUT_TYPE_MAP.end()) {
        logErrorToRank0(mpi_rank, "\n\n ERROR: Unknown output type: " + str + ".\n");
        exit(1);
      }
      return it->second;
    }

    template<>
    ObjectiveType convertFromString<ObjectiveType>(const std::string& str) {
      auto it = OBJECTIVE_TYPE_MAP.find(toLower(str));
      if (it == OBJECTIVE_TYPE_MAP.end()) {
        logErrorToRank0(mpi_rank, "\n\n ERROR: Unknown objective type: " + str + ".\n");
        exit(1);
      }
      return it->second;
    }

    template<>
    std::vector<std::string> convertFromString<std::vector<std::string>>(const std::string& str) {
      std::vector<std::string> vec;
      auto parts = split(str);
      vec.reserve(parts.size());
      for (const auto& part : parts) {
        vec.push_back(convertFromString<std::string>(part));
      }
      return vec;
    }

    template<typename T>
    void registerScalar(const std::string& key, T& member) {
      setters[key] = [this, &member](const std::string& val) {
        member = convertFromString<T>(val);
      };
    }

    template<typename T>
    void registerVector(const std::string& key, std::vector<T>& member) {
      setters[key] = [this, &member](const std::string& val) {
        auto parts = split(val);
        member.clear();
        for (const auto& part : parts) {
          member.push_back(convertFromString<T>(part));
        }
      };
    }

    // Register a vector with a desired size and fill with the last value to desired size
    // If the vector is empty, it will be set to the default_value
    template<typename T>
    void registerAndFillVector(const std::string& key, std::vector<T>& member, size_t desired_size, std::vector<T> default_value) {
      setters[key] = [this, &member, desired_size, default_value](const std::string& val) {
        member = convertFromString<std::vector<T>>(val);
        if (!member.empty()) {
          member.resize(desired_size, member.back());
        } else {
          member = default_value;
        }
      };
    }

    // If empty default to vector with one element of default_value
    template<typename T>
    void registerVector(const std::string& key, std::vector<T>& member, T default_value) {
      setters[key] = [this, &member, default_value](const std::string& val) {
        if (val.empty()) {
          member.push_back(default_value);
        } else {
          auto parts = split(val);
          member.clear();
          for (const auto& part : parts) {
            member.push_back(convertFromString<T>(part));
          }
        }
      };
    }

    template<typename T>
    void registerVectorOfVectors(const std::string& key, std::vector<std::vector<T>>& member, size_t size, T default_value) {
      member.resize(size);
      for (size_t i = 0; i < size; i++) {
        std::string key_i = key + std::to_string(i);
        registerVector(key_i, member[i], default_value);
      }
    }
  };
