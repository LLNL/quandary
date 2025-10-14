#include "config.hpp"
#include <iostream>
#include <string>
#include <vector>

#include "configbuilder.hpp"
#include "util.hpp"

// Helper function to convert enum back to string using existing enum maps
template<typename EnumType>
std::string enumToString(EnumType value, const std::map<std::string, EnumType>& type_map) {
  for (const auto& [str, enum_val] : type_map) {
    if (enum_val == value) return str;
  }
  return "unknown";
}

Config::Config(
  MPI_Comm comm_,
  int mpi_rank_,
  std::stringstream* log_,
  bool quietmode_,
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
  const OptimTolerance& tolerance_,
  double optim_regul_,
  const OptimPenalty& penalty_,
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
) :
  comm(comm_),
  mpi_rank(mpi_rank_),
  log(log_),
  quietmode(quietmode_),
  nlevels(nlevels_),
  nessential(nessential_),
  ntime(ntime_),
  dt(dt_),
  transfreq(transfreq_),
  selfkerr(selfkerr_),
  crosskerr(crosskerr_),
  Jkl(Jkl_),
  rotfreq(rotfreq_),
  collapse_type(collapse_type_),
  decay_time(decay_time_),
  dephase_time(dephase_time_),
  initial_condition_type(initial_condition_type_),
  n_initial_conditions(n_initial_conditions_),
  initial_condition_IDs(initial_condition_IDs_),
  initial_condition_file(initial_condition_file_),
  apply_pipulse(apply_pipulse_),
  control_segments(control_segments_),
  control_enforceBC(control_enforceBC_),
  control_initializations(control_initializations_),
  control_initialization_file(control_initialization_file_),
  control_bounds(control_bounds_),
  carrier_frequencies(carrier_frequencies_),
  optim_target_type(optim_target_type_),
  optim_target_file(optim_target_file_),
  optim_target_gate_type(optim_target_gate_type_),
  optim_target_gate_file(optim_target_gate_file_),
  optim_target_purestate_levels(optim_target_purestate_levels_),
  gate_rot_freq(gate_rot_freq_),
  optim_objective(optim_objective_),
  optim_weights(optim_weights_),
  tolerance(tolerance_),
  optim_regul(optim_regul_),
  penalty(penalty_),
  optim_regul_tik0(optim_regul_tik0_),
  datadir(datadir_),
  output(output_),
  output_frequency(output_frequency_),
  optim_monitor_frequency(optim_monitor_frequency_),
  runtype(runtype_),
  usematfree(usematfree_),
  linearsolver_type(linearsolver_type_),
  linearsolver_maxiter(linearsolver_maxiter_),
  timestepper_type(timestepper_type_),
  rand_seed(rand_seed_),
  hamiltonian_file_Hsys(hamiltonian_file_Hsys_),
  hamiltonian_file_Hc(hamiltonian_file_Hc_)
{
  MPI_Comm_rank(comm, &mpi_rank);
  finalize();
}

Config::~Config(){}

void Config::finalize() {
  // Basic domain requirements
  if (nlevels.empty()) {
    exitWithError(mpi_rank, "ERROR: nlevels cannot be empty");
  }

  // Hamiltonian file + matrix-free compatibility check
  if ((!hamiltonian_file_Hsys.empty() || !hamiltonian_file_Hc.empty()) && usematfree) {
    if (!quietmode) {
      logOutputToRank0(mpi_rank, "# Warning: Matrix-free solver cannot be used when Hamiltonian is read from file. Switching to sparse-matrix version.\n");
    }
    usematfree = false;
  }

  // Sanity check for Schrodinger solver initial conditions
  if (collapse_type == LindbladType::NONE) {
    if (initial_condition_type == InitialConditionType::ENSEMBLE ||
        initial_condition_type == InitialConditionType::THREESTATES ||
        initial_condition_type == InitialConditionType::NPLUSONE) {
      exitWithError(mpi_rank, "\n\n ERROR for initial condition setting: \n When running Schroedingers solver (collapse_type == NONE), the initial condition needs to be either 'pure' or 'from file' or 'diagonal' or 'basis'. Note that 'diagonal' and 'basis' in the Schroedinger case are the same (all unit vectors).\n\n");
    }

    // DIAGONAL and BASIS initial conditions in the Schroedinger case are the same. Overwrite it to DIAGONAL
    if (initial_condition_type == InitialConditionType::BASIS) {
      initial_condition_type = InitialConditionType::DIAGONAL;
    }
  }

  // Validate initial conditions
  if (initial_condition_type == InitialConditionType::PURE) {
    if (initial_condition_IDs.size() != nlevels.size()) {
      exitWithError(mpi_rank, "ERROR during pure-state initialization: List of IDs must contain " +
        std::to_string(nlevels.size()) + " elements!\n");
    }

    for (size_t k = 0; k < initial_condition_IDs.size(); k++) {
      if (initial_condition_IDs[k] >= nlevels[k]) {
        exitWithError(mpi_rank, "Pure state initialization: ID " + std::to_string(initial_condition_IDs[k]) +
          " for oscillator " + std::to_string(k) +
          " exceeds maximum level " + std::to_string(nlevels[k] - 1));
      }
    }
  } else if (initial_condition_type == InitialConditionType::ENSEMBLE) {
    // Sanity check for the list in initcond_IDs!
    if (initial_condition_IDs.empty()) {
      exitWithError(mpi_rank, "ERROR: initial_condition_IDs cannot be empty for ensemble initialization");
    }
    if (initial_condition_IDs.back() >= nlevels.size()) {
      exitWithError(mpi_rank, "ERROR: Last element in initial_condition_IDs exceeds number of oscillators");
    }
    for (size_t i = 0; i < initial_condition_IDs.size() - 1; i++) { // list should be consecutive!
      if (initial_condition_IDs[i] + 1 != initial_condition_IDs[i + 1]) {
        exitWithError(mpi_rank, "ERROR: List of oscillators for ensemble initialization should be consecutive!\n");
      }
    }
  }

  // Validate essential levels don't exceed total levels
  if (nessential.size() != nlevels.size()) {
    exitWithError(mpi_rank, "nessential size must match nlevels size");
  }

  for (size_t i = 0; i < nlevels.size(); i++) {
    if (nessential[i] > nlevels[i]) {
      exitWithError(mpi_rank, "nessential[" + std::to_string(i) + "] = " + std::to_string(nessential[i]) +
        " cannot exceed nlevels[" + std::to_string(i) + "] = " + std::to_string(nlevels[i]));
    }
  }

  // Basic sanity checks
  if (ntime <= 0) {
    exitWithError(mpi_rank, "ntime must be positive, got " + std::to_string(ntime));
  }

  if (dt <= 0) {
    exitWithError(mpi_rank, "dt must be positive, got " + std::to_string(dt));
  }
}

Config Config::fromCfg(std::string filename, std::stringstream* log, bool quietmode) {
  ConfigBuilder builder(MPI_COMM_WORLD, *log, quietmode);
  builder.loadFromFile(filename);
  return builder.build();
}

namespace {
  template<typename T>
  std::string printVector(std::vector<T> vec) {
    std::string out = "";
    for (size_t i = 0; i < vec.size(); ++i) {
      out += std::to_string(vec[i]);
      if (i < vec.size() - 1) {
        out += ", ";
      }
    }
    return out;
  }
} //namespace

void Config::printConfig() const {
  std::string delim = ", ";
  std::cout << "# Configuration settings\n";
  std::cout << "# =============================================\n\n";

  std::cout << "nlevels = " << printVector(nlevels) << "\n";
  std::cout << "nessential = " << printVector(nessential) << "\n";
  std::cout << "ntime = " << ntime << "\n";
  std::cout << "dt = " << dt << "\n";
  std::cout << "transfreq = " << printVector(transfreq) << "\n";
  std::cout << "selfkerr = " << printVector(selfkerr) << "\n";
  std::cout << "crosskerr = " << printVector(crosskerr) << "\n";
  std::cout << "Jkl = " << printVector(Jkl) << "\n";
  std::cout << "rotfreq = " << printVector(rotfreq) << "\n";
  std::cout << "collapse_type = " << enumToString(collapse_type, LINDBLAD_TYPE_MAP) << "\n";
  std::cout << "decay_time = " << printVector(decay_time) << "\n";
  std::cout << "dephase_time = " << printVector(dephase_time) << "\n";
  std::cout << "initialcondition = " << enumToString(initial_condition_type, INITCOND_TYPE_MAP);
  if (!initial_condition_IDs.empty()) {
    for (size_t id : initial_condition_IDs) {
      std::cout << ", " << id;
    }
  }
  std::cout << "\n";
  for (size_t i = 0; i < apply_pipulse.size(); ++i) {
    for (const auto& segment : apply_pipulse[i]) {
      std::cout << "apply_pipulse = " << i
                << ", " << segment.tstart
                << ", " << segment.tstop
                << ", " << segment.amp << "\n";
    }
  }
  if (!hamiltonian_file_Hsys.empty()) {
    std::cout << "hamiltonian_file_Hsys = " << hamiltonian_file_Hsys << "\n";
  }
  if (!hamiltonian_file_Hc.empty()) {
    std::cout << "hamiltonian_file_Hc = " << hamiltonian_file_Hc << "\n";
  }

  // Optimization Parameters
  for (size_t i = 0; i < control_segments.size(); ++i) {
    if (!control_segments[i].empty()) {
      const auto& seg = control_segments[i][0];
      std::cout << "control_segments" << i << " = "
                << enumToString(seg.type, CONTROL_TYPE_MAP);
      // Add segment-specific parameters
      if (std::holds_alternative<SplineParams>(seg.params)) {
        auto params = std::get<SplineParams>(seg.params);
        std::cout << ", " << params.nspline;
      } else if (std::holds_alternative<SplineAmpParams>(seg.params)) {
        auto params = std::get<SplineAmpParams>(seg.params);
        std::cout << ", " << params.nspline;
        if (params.scaling != 1.0) std::cout << ", " << params.scaling;
      }
      std::cout << "\n";
    }
  }
  std::cout << "control_enforceBC = " << (control_enforceBC ? "true" : "false") << "\n";
  for (size_t i = 0; i < control_initializations.size(); ++i) {
    if (!control_initializations[i].empty()) {
      const auto& init = control_initializations[i][0];
      std::cout << "control_initialization" << i
                << " = " << enumToString(init.type, CONTROL_INITIALIZATION_TYPE_MAP)
                << ", " << init.amplitude;
      if (init.phase != 0.0) std::cout << ", " << init.phase;
      std::cout << "\n";
    }
  }
  for (size_t i = 0; i < control_bounds.size(); ++i) {
    std::cout << "control_bounds" << i << " = " << printVector(control_bounds[i]) << "\n";
  }
  for (size_t i = 0; i < carrier_frequencies.size(); ++i) {
    std::cout << "carrier_frequency" << i << " = " << printVector(carrier_frequencies[i]) << "\n";
  }

  std::cout << "runtype = " << enumToString(runtype, RUN_TYPE_MAP) << "\n";
  std::cout << "optim_target = " << enumToString(optim_target_type, TARGET_TYPE_MAP);
  if (optim_target_type == TargetType::GATE) {
    std::cout << ", " << enumToString(optim_target_gate_type, GATE_TYPE_MAP);
  }
  std::cout << "\n";

  std::cout << "optim_objective = " << enumToString(optim_objective, OBJECTIVE_TYPE_MAP) << "\n";
  if (!optim_weights.empty()) {
    std::cout << "optim_weights = " << printVector(optim_weights) << "\n";
  }
  std::cout << "optim_atol = " << tolerance.atol << "\n";
  std::cout << "optim_rtol = " << tolerance.rtol << "\n";
  std::cout << "optim_ftol = " << tolerance.ftol << "\n";
  std::cout << "optim_inftol = " << tolerance.inftol << "\n";
  std::cout << "optim_maxiter = " << tolerance.maxiter << "\n";
  std::cout << "optim_regul = " << optim_regul << "\n";
  std::cout << "optim_penalty = " << penalty.penalty << "\n";
  std::cout << "optim_penalty_param = " << penalty.penalty_param << "\n";
  std::cout << "optim_penalty_dpdm = " << penalty.penalty_dpdm << "\n";
  std::cout << "optim_penalty_energy = " << penalty.penalty_energy << "\n";
  std::cout << "optim_penalty_variation = " << penalty.penalty_variation << "\n";
  std::cout << "optim_regul_tik0 = " << (optim_regul_tik0 ? "true" : "false") << "\n";

  std::cout << "datadir = " << datadir << "\n";

  for (size_t i = 0; i < output.size(); ++i) {
    if (!output[i].empty()) {
      std::cout << "output" << i << " = ";
      for (size_t j = 0; j < output[i].size(); ++j) {
        std::cout << enumToString(output[i][j], OUTPUT_TYPE_MAP);
        if (j < output[i].size() - 1) std::cout << ", ";
      }
      std::cout << "\n";
    }
  }

  std::cout << "output_frequency = " << output_frequency << "\n";
  std::cout << "optim_monitor_frequency = " << optim_monitor_frequency << "\n";
  std::cout << "runtype = " << enumToString(runtype, RUN_TYPE_MAP) << "\n";
  std::cout << "usematfree = " << (usematfree ? "true" : "false") << "\n";
  std::cout << "linearsolver_type = " << enumToString(linearsolver_type, LINEAR_SOLVER_TYPE_MAP) << "\n";
  std::cout << "linearsolver_maxiter = " << linearsolver_maxiter << "\n";
  std::cout << "timestepper = " << enumToString(timestepper_type, TIME_STEPPER_TYPE_MAP) << "\n";
  std::cout << "rand_seed = " << rand_seed << "\n";

  std::cout << "# =============================================\n\n";
}
