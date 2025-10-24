#include <cstddef>
#include <set>
#include <sstream>
#include <type_traits>
#include <vector>

#include <petsc.h>

#include "config.hpp"
#include "config_types.hpp"
#include "defs.hpp"
#include "util.hpp"

#pragma once


namespace {
  // Vector detection
  template<typename T>
  struct is_vector : std::false_type {};
  template<typename T>
  struct is_vector<std::vector<T>> : std::true_type {};
  template<typename T>
  inline constexpr bool is_vector_v = is_vector<T>::value;
} // namespace

/**
 * @brief Configuration builder that converts raw config to validated Config
 *
 * Handles the logic for applying defaults, parsing mixed-type strings,
 * and validating configuration parameters.
 */
class ConfigBuilder {
private:
  std::unordered_map<std::string, std::function<void(const std::string&)>> setters; ///< Setters from config string
  std::unordered_map<std::string, std::function<void(int, const std::string&)>> indexed_setters; ///< Setters for indexed config strings

  // MPI and logging
  MPI_Comm comm; ///< MPI communicator for parallel operations.
  int mpi_rank; ///< MPI rank of the current process.
  std::stringstream* log; ///< Pointer to log stream for output messages.
  bool quietmode; ///< Flag to control verbose output.

  // General options
  std::optional<std::vector<size_t>> nlevels;  ///< Number of levels per subsystem
  std::optional<std::vector<size_t>> nessential;  ///< Number of essential levels per subsystem
  std::optional<size_t> ntime;  ///< Number of time steps used for time-integration
  std::optional<double> dt;  ///< Time step size (ns). Determines final time: T=ntime*dt
  std::optional<std::vector<double>> transfreq;  ///< Fundamental transition frequencies for each oscillator (GHz)
  std::optional<std::vector<double>> selfkerr;  ///< Self-kerr frequencies for each oscillator (GHz)
  std::optional<std::vector<double>> crosskerr;  ///< Cross-kerr coupling frequencies for each oscillator coupling (GHz)
  std::optional<std::vector<double>> Jkl;  ///< Dipole-dipole coupling frequencies for each oscillator coupling (GHz)
  std::optional<std::vector<double>> rotfreq;  ///< Rotational wave approximation frequencies for each subsystem (GHz)
  std::optional<LindbladType> collapse_type;  ///< Switch between Schroedinger and Lindblad solver
  std::optional<std::vector<double>> decay_time;  ///< Time of decay collapse operation (T1) per oscillator (for Lindblad solver)
  std::optional<std::vector<double>> dephase_time;  ///< Time of dephase collapse operation (T2) per oscillator (for Lindblad solver)
  std::optional<InitialConditionConfig> initialcondition;  ///< Initial condition specification
  std::optional<std::vector<PiPulseConfig>> apply_pipulse;  ///< Apply a pi-pulse to oscillator with specified parameters
  std::optional<std::string> hamiltonian_file_Hsys;  ///< File to read the system Hamiltonian from
  std::optional<std::string> hamiltonian_file_Hc;  ///< File to read the control Hamiltonian from

  // Optimization options
  std::optional<std::vector<ControlSegmentConfig>> control_segments;  ///< Define the control segments for each oscillator (consolidated from control_segments0, control_segments1, etc.)
  std::optional<bool> control_enforceBC;  ///< Decide whether control pulses should start and end at zero
  std::optional<std::vector<ControlInitializationConfig>> control_initializations;  ///< Set the initial control pulse parameters for each oscillator (consolidated from control_initialization0, etc.)
  std::optional<std::vector<std::vector<double>>> control_bounds;  ///< Maximum amplitude bound for the control pulses for each oscillator segment (GHz) (consolidated from control_bounds0, etc.)
  std::optional<std::vector<std::vector<double>>> carrier_frequencies;  ///< Carrier wave frequencies for each oscillator (GHz) (consolidated from carrier_frequency0, etc.)
  std::optional<OptimTargetConfig> optim_target;  ///< Optimization target configuration
  std::optional<std::vector<double>> gate_rot_freq;  ///< Frequency of rotation of the target gate, for each oscillator (GHz)
  std::optional<ObjectiveType> optim_objective;  ///< Objective function measure
  std::optional<std::vector<double>> optim_weights;  ///< Weights for summing up the objective function
  std::optional<double> optim_atol;  ///< Optimization stopping tolerance based on gradient norm (absolute)
  std::optional<double> optim_rtol;  ///< Optimization stopping tolerance based on gradient norm (relative)
  std::optional<double> optim_ftol;  ///< Optimization stopping criterion based on the final time cost (absolute)
  std::optional<double> optim_inftol;  ///< Optimization stopping criterion based on the infidelity (absolute)
  std::optional<size_t> optim_maxiter;  ///< Maximum number of optimization iterations
  std::optional<double> optim_regul;  ///< Coefficient of Tikhonov regularization for the design variables
  std::optional<double> optim_penalty;  ///< Coefficient for adding first integral penalty term
  std::optional<double> optim_penalty_param;  ///< Integral penalty parameter inside the weight (gaussian variance a)
  std::optional<double> optim_penalty_dpdm;  ///< Coefficient for penalizing the integral of the second derivative of state populations
  std::optional<double> optim_penalty_energy;  ///< Coefficient for penalizing the control pulse energy integral
  std::optional<double> optim_penalty_variation;  ///< Coefficient for penalizing variations in control amplitudes
  std::optional<bool> optim_regul_tik0;  ///< Switch to use Tikhonov regularization with ||x - x_0||^2 instead of ||x||^2
  std::optional<bool> optim_regul_interpolate;  ///< Deprecated version of optim_regul_tik0

  // Output and runtypes
  std::optional<std::string> datadir;  ///< Directory for output files
  std::optional<std::vector<std::vector<OutputType>>> output;  ///< Specify the desired output for each oscillator
  std::optional<int> output_frequency;  ///< Output frequency in the time domain: write output every <num> time-step
  std::optional<int> optim_monitor_frequency;  ///< Frequency of writing output during optimization iterations
  std::optional<RunType> runtype;  ///< Runtype options: simulation, gradient, or optimization
  std::optional<bool> usematfree;  ///< Use matrix free solver, instead of sparse matrix implementation
  std::optional<LinearSolverType> linearsolver_type;  ///< Solver type for solving the linear system at each time step
  std::optional<size_t> linearsolver_maxiter;  ///< Set maximum number of iterations for the linear solver
  std::optional<TimeStepperType> timestepper;  ///< The time-stepping algorithm
  std::optional<int> rand_seed;  ///< Fixed seed for the random number generator for reproducibility

  // Indexed settings storage (per-oscillator)
  std::optional<std::map<int, std::vector<ControlSegmentConfig>>> indexed_control_segments;      ///< control_segments0, control_segments1, etc.
  std::optional<std::map<int, ControlInitializationConfig>> indexed_control_init;   ///< control_initialization0, control_initialization1, etc.
  std::optional<std::map<int, std::vector<double>>> indexed_control_bounds;         ///< control_bounds0, control_bounds1, etc.
  std::optional<std::map<int, std::vector<double>>> indexed_carrier_frequencies;    ///< carrier_frequency0, carrier_frequency1, etc.
  std::optional<std::map<int, std::vector<OutputType>>> indexed_output;             ///< output0, output1, etc.

public:
  ConfigBuilder(MPI_Comm comm, std::stringstream& logstream, bool quietmode = false);
  void loadFromFile(const std::string& filename);
  void loadFromString(const std::string& config_content);
  Config build();

private:
  std::vector<std::string> split(const std::string& str, char delimiter = ',');
  void applyConfigLine(const std::string& line);
  bool handleIndexedSetting(const std::string& key, const std::string& value);

  template<typename StreamType>
  void loadFromStream(StreamType& stream) {
    std::string line;
    while (getline(stream, line)) {
      applyConfigLine(line);
    }
  }

  std::vector<std::vector<double>> convertIndexedToVectorVector(
      const std::map<int, std::vector<double>>& indexed_map,
      size_t num_oscillators);
  std::vector<std::vector<OutputType>> convertIndexedToOutputVector(
      const std::map<int, std::vector<OutputType>>& indexed_map,
      size_t num_oscillators);

  template<typename T>
  void registerConfig(const std::string& key, std::optional<T>& member) {
    setters[key] = [this, &member](const std::string& value) {
      member = convertFromString<T>(value);
    };
  }

  template<typename T>
  void registerIndexedConfig(const std::string& base_key, std::optional<std::map<int, T>>& storage) {
    indexed_setters[base_key] = [this, &storage](int index, const std::string& value) {
      if (!storage.has_value()) {
        storage = std::map<int, T>{};
      }
      (*storage)[index] = convertFromString<T>(value);
    };
  }

  template<typename T>
  T convertFromString(const std::string& str) {
    if constexpr (std::is_same_v<T, std::string>) {
      return str;
    }
    else if constexpr (std::is_same_v<T, bool>) {
      const std::set<std::string> trueValues = {"true", "yes", "1"};
      std::string lowerStr = str;
      std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
      return trueValues.find(lowerStr) != trueValues.end();
    }
    else if constexpr (std::is_same_v<T, int>) {
      return std::stoi(str);
    }
    else if constexpr (std::is_same_v<T, size_t>) {
      return static_cast<size_t>(std::stoul(str));
    }
    else if constexpr (std::is_same_v<T, double>) {
      return std::stod(str);
    }
    else if constexpr (is_vector_v<T>) {
      return parseVector<T>(str);
    }
    else if constexpr (std::is_enum_v<T>) {
      return parseEnum<T>(str);
    }
    else {
      static_assert(false, "Unsupported type for convertFromString");
    }
  }

  template<typename VectorType>
  VectorType parseVector(const std::string& str) {
    using ElementType = typename VectorType::value_type;
    VectorType vec;
    auto parts = split(str);
    vec.reserve(parts.size());
    for (const auto& part : parts) {
      vec.push_back(convertFromString<ElementType>(part));
    }
    return vec;
  }

  // Enum types
  template<typename EnumType>
  EnumType parseEnum(const std::string& str);


  // Enum converter specializations
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
      logErrorToRank0(mpi_rank, "\n\n ERROR: Unknown Lindblad type: " + str + ".\n");
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
  ControlType convertFromString<ControlType>(const std::string& str) {
    auto it = CONTROL_TYPE_MAP.find(toLower(str));
    if (it == CONTROL_TYPE_MAP.end()) {
      logErrorToRank0(mpi_rank, "\n\n ERROR: Unknown control type: " + str + ".\n");
      exit(1);
    }
    return it->second;
  }

  template<>
  ControlInitializationType convertFromString<ControlInitializationType>(const std::string& str) {
    auto it = CONTROL_INITIALIZATION_TYPE_MAP.find(toLower(str));
    if (it == CONTROL_INITIALIZATION_TYPE_MAP.end()) {
      logErrorToRank0(mpi_rank, "\n\n ERROR: Unknown control initialization type: " + str + ".\n");
      exit(1);
    }
    return it->second;
  }

  // Struct converters
  template<>
  InitialConditionConfig convertFromString<InitialConditionConfig>(const std::string& str);

  template<>
  OptimTargetConfig convertFromString<OptimTargetConfig>(const std::string& str);

  template<>
  std::vector<PiPulseConfig> convertFromString<std::vector<PiPulseConfig>>(const std::string& str);

  template<>
  std::vector<ControlSegmentConfig> convertFromString<std::vector<ControlSegmentConfig>>(const std::string& str);

  template<>
  ControlInitializationConfig convertFromString<ControlInitializationConfig>(const std::string& str);
};