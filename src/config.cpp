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
  std::stringstream& log_,
  bool quietmode_,
  // All parameters as optionals
  const std::optional<std::vector<size_t>>& nlevels_,
  const std::optional<std::vector<size_t>>& nessential_,
  const std::optional<int>& ntime_,
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
  // Control parameters (optional indexed data)
  const std::optional<std::map<int, ControlSegmentConfig>>& indexed_control_segments_,
  const std::optional<bool>& control_enforceBC_,
  const std::optional<std::map<int, ControlInitializationConfig>>& indexed_control_init_,
  const std::optional<std::map<int, std::vector<double>>>& indexed_control_bounds_,
  const std::optional<std::map<int, std::vector<double>>>& indexed_carrier_frequencies_,
  // Optimization parameters
  const std::optional<OptimTargetConfig>& optim_target_,
  const std::optional<std::vector<double>>& gate_rot_freq_,
  const std::optional<ObjectiveType>& optim_objective_,
  const std::optional<std::vector<double>>& optim_weights_,
  const std::optional<double>& optim_atol_,
  const std::optional<double>& optim_rtol_,
  const std::optional<double>& optim_ftol_,
  const std::optional<double>& optim_inftol_,
  const std::optional<int>& optim_maxiter_,
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
  const std::optional<int>& output_frequency_,
  const std::optional<int>& optim_monitor_frequency_,
  const std::optional<RunType>& runtype_,
  const std::optional<bool>& usematfree_,
  const std::optional<LinearSolverType>& linearsolver_type_,
  const std::optional<int>& linearsolver_maxiter_,
  const std::optional<TimeStepperType>& timestepper_type_,
  const std::optional<int>& rand_seed_
) :
  comm(comm_),
  log(&log_),
  quietmode(quietmode_)
{
  MPI_Comm_rank(comm, &mpi_rank);

  // First validate user-provided settings
  if (ntime_.has_value() && ntime_.value() <= 0) {
    exitWithError(mpi_rank, "ERROR: User-specified ntime must be positive, got " + std::to_string(ntime_.value()));
  }

  if (dt_.has_value() && dt_.value() <= 0) {
    exitWithError(mpi_rank, "ERROR: User-specified dt must be positive, got " + std::to_string(dt_.value()));
  }

  // Apply defaults for basic settings
  if (!nlevels_.has_value()) {
    exitWithError(mpi_rank, "ERROR: nlevels cannot be empty");
  }
  nlevels = nlevels_.value();

  nessential = nessential_.value_or(nlevels); // Default: same as nlevels
  ntime = ntime_.value_or(1000);
  dt = dt_.value_or(0.1);

  // Physics parameters
  size_t num_osc = nlevels.size();
  size_t num_pairs_osc = (num_osc - 1) * num_osc / 2;

  if (!transfreq_.has_value()) {
    exitWithError(mpi_rank, "ERROR: transfreq cannot be empty");
  }
  transfreq = transfreq_.value();
  copyLast(transfreq, num_osc);

  selfkerr = selfkerr_.value_or(std::vector<double>(num_osc, 0.0));
  copyLast(selfkerr, num_osc);

  crosskerr = crosskerr_.value_or(std::vector<double>(num_pairs_osc, 0.0));
  copyLast(crosskerr, num_pairs_osc);

  Jkl = Jkl_.value_or(std::vector<double>(num_pairs_osc, 0.0));
  copyLast(Jkl, num_pairs_osc);

  if (!rotfreq_.has_value()) {
    exitWithError(mpi_rank, "ERROR: rotfreq cannot be empty");
  }
  rotfreq = rotfreq_.value();
  copyLast(rotfreq, num_osc);

  collapse_type = collapse_type_.value_or(LindbladType::NONE);

  decay_time = decay_time_.value_or(std::vector<double>(num_osc, 0.0));
  copyLast(decay_time, num_osc);

  dephase_time = dephase_time_.value_or(std::vector<double>(num_osc, 0.0));
  copyLast(dephase_time, num_osc);

  // Extract and convert initial condition data
  if (initialcondition_.has_value()) {
    initial_condition_type = initialcondition_->type;
    if (!initialcondition_->params.empty()) {
      for (int param : initialcondition_->params) {
        initial_condition_IDs.push_back(static_cast<size_t>(param));
      }
    }
    initial_condition_file = initialcondition_->filename.value_or("");
    n_initial_conditions = initialcondition_->params.size();
  } else {
    initial_condition_type = InitialConditionType::BASIS;
    n_initial_conditions = 1;
    initial_condition_file = "";
  }

  // Convert from parsing structs to runtime format
  convertInitialCondition(initialcondition_);

  hamiltonian_file_Hsys = hamiltonian_file_Hsys_.value_or("");
  hamiltonian_file_Hc = hamiltonian_file_Hc_.value_or("");

  convertOptimTarget(optim_target_);
  convertPiPulses(apply_pipulse_);

  // Initialize oscillators vector if needed
  if (oscillator_optimization.size() != nlevels.size()) {
    oscillator_optimization.resize(nlevels.size());
  }
  convertControlSegments(indexed_control_segments_);
  convertControlInitializations(indexed_control_init_);
  convertIndexedControlBounds(indexed_control_bounds_);
  convertIndexedCarrierFreqs(indexed_carrier_frequencies_);
  convertIndexedOutput(indexed_output_);

  // Apply remaining optimization defaults
  gate_rot_freq = gate_rot_freq_.value_or(std::vector<double>{});
  optim_objective = optim_objective_.value_or(ObjectiveType::JFROBENIUS);
  optim_weights = optim_weights_.value_or(std::vector<double>{});
  control_initialization_file = std::nullopt; // Not used in current design

  // For now, set some basic defaults to prevent compilation errors
  control_enforceBC = control_enforceBC_.value_or(false);
  optim_regul = optim_regul_.value_or(1e-4);
  optim_regul_tik0 = optim_regul_tik0_.value_or(false);
  datadir = datadir_.value_or("./data_out");
  output_frequency = output_frequency_.value_or(1);
  optim_monitor_frequency = optim_monitor_frequency_.value_or(10);
  runtype = runtype_.value_or(RunType::SIMULATION);
  usematfree = usematfree_.value_or(false);
  linearsolver_type = linearsolver_type_.value_or(LinearSolverType::GMRES);
  linearsolver_maxiter = linearsolver_maxiter_.value_or(10);
  timestepper_type = timestepper_type_.value_or(TimeStepperType::IMR);
  rand_seed = rand_seed_.value_or(1234);

  // Build tolerance and penalty structs
  tolerance = OptimTolerance{
    optim_atol_.value_or(1e-8),
    optim_rtol_.value_or(1e-4),
    optim_ftol_.value_or(1e-8),
    optim_inftol_.value_or(1e-5),
    optim_maxiter_.value_or(200)
  };

  penalty = OptimPenalty{
    optim_penalty_.value_or(0.0),
    optim_penalty_param_.value_or(0.5),
    optim_penalty_dpdm_.value_or(0.0),
    optim_penalty_energy_.value_or(0.0),
    optim_penalty_variation_.value_or(0.01)
  };

  // Run final validation and normalization
  finalize();
}

Config::~Config(){}

void Config::finalize() {
  // Basic domain requirements
  if (nlevels.empty()) {
    exitWithError(mpi_rank, "ERROR: nlevels cannot be empty");
  }

  // Hamiltonian file + matrix-free compatibility check
  if ((hamiltonian_file_Hsys.has_value() || hamiltonian_file_Hc.has_value()) && usematfree) {
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
  if (hamiltonian_file_Hsys.has_value()) {
    std::cout << "hamiltonian_file_Hsys = " << hamiltonian_file_Hsys.value() << "\n";
  }
  if (hamiltonian_file_Hc.has_value()) {
    std::cout << "hamiltonian_file_Hc = " << hamiltonian_file_Hc.value() << "\n";
  }

  // Optimization Parameters
  for (size_t i = 0; i < oscillator_optimization.size(); ++i) {
    if (!oscillator_optimization[i].control_segments.empty()) {
      const auto& seg = oscillator_optimization[i].control_segments[0];
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
  for (size_t i = 0; i < oscillator_optimization.size(); ++i) {
    if (!oscillator_optimization[i].control_initializations.empty()) {
      const auto& init = oscillator_optimization[i].control_initializations[0];
      std::cout << "control_initialization" << i
                << " = " << enumToString(init.type, CONTROL_INITIALIZATION_TYPE_MAP)
                << ", " << init.amplitude;
      if (init.phase != 0.0) std::cout << ", " << init.phase;
      std::cout << "\n";
    }
    if (!oscillator_optimization[i].control_bounds.empty()) {
      std::cout << "control_bounds" << i << " = " << printVector(oscillator_optimization[i].control_bounds) << "\n";
    }
    if (!oscillator_optimization[i].carrier_frequencies.empty()) {
      std::cout << "carrier_frequency" << i << " = " << printVector(oscillator_optimization[i].carrier_frequencies) << "\n";
    }
  }

  std::cout << "runtype = " << enumToString(runtype, RUN_TYPE_MAP) << "\n";
  std::cout << "optim_target = " << enumToString(target.type, TARGET_TYPE_MAP);
  if (target.type == TargetType::GATE) {
    if (!target.gate_file.empty()) {
      std::cout << ", file, " << target.gate_file;
    } else {
      std::cout << ", " << enumToString(target.gate_type, GATE_TYPE_MAP);
    }
  } else if (target.type == TargetType::FROMFILE) {
    std::cout << ", " << target.file;
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

// Template helper implementation
template<typename T>
std::vector<std::vector<T>> Config::convertIndexedToVectorVector(const std::optional<std::map<int, std::vector<T>>>& indexed_map) {
  if (!indexed_map.has_value()) {
    return std::vector<std::vector<T>>(nlevels.size()); // Empty vectors for each oscillator
  }

  std::vector<std::vector<T>> result(nlevels.size());
  for (const auto& [osc_idx, values] : *indexed_map) {
    if (static_cast<size_t>(osc_idx) < result.size()) {
      result[osc_idx] = values;
    }
  }
  return result;
}

// Conversion helper implementations
void Config::convertInitialCondition(const std::optional<InitialConditionConfig>& config) {
  if (config.has_value()) {
    initial_condition_type = config->type;
    n_initial_conditions = config->params.size();

    // Convert int params to size_t IDs
    for (int param : config->params) {
      initial_condition_IDs.push_back(static_cast<size_t>(param));
    }

    initial_condition_file = config->filename.value_or("");
  } else {
    initial_condition_type = InitialConditionType::BASIS;
    n_initial_conditions = 1;
    initial_condition_file = "";
  }
}

void Config::convertOptimTarget(const std::optional<OptimTargetConfig>& config) {
  if (config.has_value()) {
    target.type = config->target_type;
    target.gate_type = config->gate_type.value_or(GateType::NONE);
    target.file = config->filename.value_or("");
    target.gate_file = config->gate_file.value_or("");

    // Convert int levels to size_t
    for (int level : config->levels) {
      target.purestate_levels.push_back(static_cast<size_t>(level));
    }
  } else {
    target.type = TargetType::PURE;
    target.gate_type = GateType::NONE;
    target.file = "";
    target.gate_file = "";
    target.purestate_levels.clear();
  }
}

void Config::convertPiPulses(const std::optional<std::vector<PiPulseConfig>>& pulses) {
  apply_pipulse.resize(nlevels.size());

  if (pulses.has_value()) {
    for (const auto& pulse_config : *pulses) {
      if (pulse_config.oscil_id >= 0 &&
          static_cast<size_t>(pulse_config.oscil_id) < nlevels.size()) {
        PiPulseSegment segment;
        segment.tstart = pulse_config.tstart;
        segment.tstop = pulse_config.tstop;
        segment.amp = pulse_config.amp;
        apply_pipulse[pulse_config.oscil_id].push_back(segment);
      }
    }
  }
}

void Config::convertControlSegments(const std::optional<std::map<int, ControlSegmentConfig>>& indexed) {
  if (indexed.has_value()) {
    for (const auto& [osc_idx, seg_config] : *indexed) {
      if (static_cast<size_t>(osc_idx) < oscillator_optimization.size()) {
        ControlSegment segment;
        segment.type = seg_config.control_type;

        // Create appropriate params variant based on type
        if (seg_config.control_type == ControlType::BSPLINE ||
            seg_config.control_type == ControlType::BSPLINE0) {
          SplineParams params;
          params.nspline = seg_config.num_basis_functions.value_or(10);
          params.tstart = seg_config.tstart.value_or(0.0);
          params.tstop = seg_config.tstop.value_or(ntime * dt);
          segment.params = params;
        } else if (seg_config.control_type == ControlType::BSPLINEAMP) {
          SplineAmpParams params;
          params.nspline = seg_config.num_basis_functions.value_or(10);
          params.tstart = seg_config.tstart.value_or(0.0);
          params.tstop = seg_config.tstop.value_or(ntime * dt);
          params.scaling = seg_config.scaling.value_or(1.0);
          segment.params = params;
        } else if (seg_config.control_type == ControlType::STEP) {
          StepParams params;
          params.step_amp1 = seg_config.amplitude_1.value_or(0.0);
          params.step_amp2 = seg_config.amplitude_2.value_or(0.0);
          params.tramp = 0.0;
          params.tstart = seg_config.tstart.value_or(0.0);
          params.tstop = seg_config.tstop.value_or(ntime * dt);
          segment.params = params;
        }

        oscillator_optimization[osc_idx].control_segments.push_back(segment);
      }
    }
  }
}

void Config::convertControlInitializations(const std::optional<std::map<int, ControlInitializationConfig>>& indexed) {
  if (indexed.has_value()) {
    for (const auto& [osc_idx, init_config] : *indexed) {
      if (static_cast<size_t>(osc_idx) < oscillator_optimization.size()) {
        ControlSegmentInitialization init;
        init.type = init_config.init_type;
        init.amplitude = init_config.amplitude.value_or(0.0);
        init.phase = init_config.phase.value_or(0.0);
        oscillator_optimization[osc_idx].control_initializations.push_back(init);
      }
    }
  }
}

void Config::convertIndexedOutput(const std::optional<std::map<int, std::vector<OutputType>>>& indexed) {
  output = convertIndexedToVectorVector(indexed);
}

void Config::convertIndexedControlBounds(const std::optional<std::map<int, std::vector<double>>>& indexed) {
  if (indexed.has_value()) {
    for (const auto& [osc_idx, bounds] : *indexed) {
      if (static_cast<size_t>(osc_idx) < oscillator_optimization.size()) {
        oscillator_optimization[osc_idx].control_bounds = bounds;
      }
    }
  }
}

void Config::convertIndexedCarrierFreqs(const std::optional<std::map<int, std::vector<double>>>& indexed) {
  if (indexed.has_value()) {
    for (const auto& [osc_idx, freqs] : *indexed) {
      if (static_cast<size_t>(osc_idx) < oscillator_optimization.size()) {
        oscillator_optimization[osc_idx].carrier_frequencies = freqs;
      }
    }
  }
}
