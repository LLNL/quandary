#include "config.hpp"
#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "config_types.hpp"
#include "configbuilder.hpp"
#include "defs.hpp"
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
  // General settings
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
  // Control parameters (optional indexed data)
  const std::optional<std::map<int, std::vector<ControlSegmentConfig>>>& indexed_control_segments_,
  const std::optional<bool>& control_enforceBC_,
  const std::optional<std::map<int, std::vector<ControlInitializationConfig>>>& indexed_control_init_,
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
) :
  comm(comm_),
  log(&log_),
  quietmode(quietmode_)
{
  MPI_Comm_rank(comm, &mpi_rank);

  if (!nlevels_.has_value()) {
    exitWithError(mpi_rank, "ERROR: nlevels cannot be empty");
  }
  nlevels = nlevels_.value();
  size_t num_osc = nlevels.size();
  size_t num_pairs_osc = (num_osc - 1) * num_osc / 2;

  nessential = nessential_.value_or(nlevels);
  copyLast(nessential, num_osc);

  if (ntime_.has_value() && ntime_.value() <= 0) {
    exitWithError(mpi_rank, "ERROR: User-specified ntime must be positive, got " + std::to_string(ntime_.value()));
  }
  ntime = ntime_.value_or(1000);

  if (dt_.has_value() && dt_.value() <= 0) {
    exitWithError(mpi_rank, "ERROR: User-specified dt must be positive, got " + std::to_string(dt_.value()));
  }
  dt = dt_.value_or(0.1);

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

  convertInitialCondition(initialcondition_);
  setNumInitialConditions();

  convertPiPulses(apply_pipulse_);

  hamiltonian_file_Hsys = hamiltonian_file_Hsys_.value_or("");
  hamiltonian_file_Hc = hamiltonian_file_Hc_.value_or("");

  // Control and optimization parameters
  oscillator_optimization.resize(num_osc);

  convertControlSegments(indexed_control_segments_);
  control_enforceBC = control_enforceBC_.value_or(true);
  convertControlInitializations(indexed_control_init_);
  convertIndexedControlBounds(indexed_control_bounds_);
  convertIndexedCarrierFreqs(indexed_carrier_frequencies_);
  convertOptimTarget(optim_target_);

  gate_rot_freq = gate_rot_freq_.value_or(std::vector<double>{0.0});
  copyLast(gate_rot_freq, num_osc);

  optim_objective = optim_objective_.value_or(ObjectiveType::JFROBENIUS);

  setOptimWeights(optim_weights_);

  control_initialization_file = std::nullopt; // Not used in current design

  tolerance = OptimTolerance{
    optim_atol_.value_or(1e-8),
    optim_rtol_.value_or(1e-4),
    optim_ftol_.value_or(1e-8),
    optim_inftol_.value_or(1e-5),
    optim_maxiter_.value_or(200)
  };

  optim_regul = optim_regul_.value_or(1e-4);

  penalty = OptimPenalty{
    optim_penalty_.value_or(0.0),
    optim_penalty_param_.value_or(0.5),
    optim_penalty_dpdm_.value_or(0.0),
    optim_penalty_energy_.value_or(0.0),
    optim_penalty_variation_.value_or(0.01)
  };

  optim_regul_tik0 = optim_regul_tik0_.value_or(false);

  // Output parameters
  convertIndexedOutput(indexed_output_);
  datadir = datadir_.value_or("./data_out");
  output_frequency = output_frequency_.value_or(1);
  optim_monitor_frequency = optim_monitor_frequency_.value_or(10);
  runtype = runtype_.value_or(RunType::SIMULATION);
  usematfree = usematfree_.value_or(false);
  linearsolver_type = linearsolver_type_.value_or(LinearSolverType::GMRES);
  linearsolver_maxiter = linearsolver_maxiter_.value_or(10);
  timestepper_type = timestepper_type_.value_or(TimeStepperType::IMR);
  rand_seed = rand_seed_.value_or(1234);

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

  std::string print(const InitialCondition& initial_condition) {
    return std::visit([](const auto& opt) {
        return opt.toString();
    }, initial_condition);
  }

  std::string print(const ControlSegmentInitialization& control_initialization) {
    return std::visit([](const auto& opt) {
        return opt.toString();
    }, control_initialization);
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
  std::cout << "initialcondition = " << print(initial_condition) << "\n";

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
  std::cout << "\n// Optimization Parameters\n";
  for (size_t i = 0; i < oscillator_optimization.size(); ++i) {
    if (!oscillator_optimization[i].control_segments.empty()) {
      const auto& seg = oscillator_optimization[i].control_segments[0];
      std::cout << "control_segments" << i << " = "
                << enumToString(seg.type, CONTROL_TYPE_MAP);
      // Add segment-specific parameters
      if (std::holds_alternative<SplineParams>(seg.params)) {
        auto params = std::get<SplineParams>(seg.params);
        std::cout << ", " << params.nspline << ", " << params.tstart << ", " << params.tstop;
      } else if (std::holds_alternative<SplineAmpParams>(seg.params)) {
        auto params = std::get<SplineAmpParams>(seg.params);
        std::cout << ", " << params.nspline << ", " << params.scaling << ", " << params.tstart << ", " << params.tstop;
      } else if (std::holds_alternative<StepParams>(seg.params)) {
        auto params = std::get<StepParams>(seg.params);
        std::cout << ", " << params.step_amp1 << ", " << params.step_amp2 << ", " << params.tramp << ", "
          << params.tstart << ", " << params.tstop;
      }
      std::cout << "\n";
    }
  }
  std::cout << "control_enforceBC = " << (control_enforceBC ? "true" : "false") << "\n";
  for (size_t i = 0; i < oscillator_optimization.size(); ++i) {
    if (!oscillator_optimization[i].control_initializations.empty()) {
      const auto& init = oscillator_optimization[i].control_initializations[0];
      std::cout << "control_initialization" << i << " = " << print(init) << "\n";
    }
  }
  for (size_t i = 0; i < oscillator_optimization.size(); ++i) {
    if (!oscillator_optimization[i].control_bounds.empty()) {
      std::cout << "control_bounds" << i << " = " << printVector(oscillator_optimization[i].control_bounds) << "\n";
    }
  }
  for (size_t i = 0; i < oscillator_optimization.size(); ++i) {
    if (!oscillator_optimization[i].carrier_frequencies.empty()) {
      std::cout << "carrier_frequency" << i << " = " << printVector(oscillator_optimization[i].carrier_frequencies) << "\n";
    }
  }

  std::cout << "runtype = " << enumToString(runtype, RUN_TYPE_MAP) << "\n";
  std::cout << "optim_target = " << toString(target) << "\n";

  std::cout << "optim_objective = " << enumToString(optim_objective, OBJECTIVE_TYPE_MAP) << "\n";
  std::cout << "optim_weights = " << printVector(optim_weights) << "\n";
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

  // Output and runtypes
  std::cout << "\n// Output and runtypes\n";
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

InitialCondition Config::convertInitialCondition(const InitialConditionConfig& config) {
  const auto& params = config.params;

    /* Sanity check for Schrodinger solver initial conditions */
  if (collapse_type == LindbladType::NONE){
    if (config.type == InitialConditionType::ENSEMBLE ||
        config.type == InitialConditionType::THREESTATES ||
        config.type == InitialConditionType::NPLUSONE ){
          exitWithError(mpi_rank, "\n\n ERROR for initial condition setting: \n When running Schroedingers solver"
            " (collapse_type == NONE), the initial condition needs to be either 'pure' or 'from file' or 'diagonal' or 'basis'."
            " Note that 'diagonal' and 'basis' in the Schroedinger case are the same (all unit vectors).\n\n");
    }
  }

  switch (config.type) {
    case InitialConditionType::FROMFILE:
      if (!config.filename.has_value()) {
        exitWithError(mpi_rank, "ERROR: initialcondition of type FROMFILE must have a filename");
      }
      return FromFileInitialCondition{config.filename.value()};
    case InitialConditionType::PURE:
      if (params.size() != nlevels.size()) {
        exitWithError(mpi_rank, "ERROR: initialcondition of type PURE must have exactly " +
          std::to_string(nlevels.size()) + " parameters, got " + std::to_string(params.size()));
      }
      for (size_t k=0; k < params.size(); k++) {
        if (params[k] >= nlevels[k]){
          exitWithError(mpi_rank, "ERROR in config setting. The requested pure state initialization "
            + std::to_string(params[k]) + " exceeds the number of allowed levels for that oscillator ("
            + std::to_string(nlevels[k]) + ").\n");
        }
      }
      return PureInitialCondition{params};

    case InitialConditionType::BASIS:
      if (collapse_type == LindbladType::NONE) {
        // DIAGONAL and BASIS initial conditions in the Schroedinger case are the same. Overwrite it to DIAGONAL
        return DiagonalInitialCondition{params};
      }
      return BasisInitialCondition{params};

    case InitialConditionType::ENSEMBLE:
      if (params.back() >= nlevels.size()) {
        exitWithError(mpi_rank, "ERROR: Last element in initialcondition params exceeds number of oscillators");
      }

      for (size_t i = 1; i < params.size()-1; i++){
        if (params[i]+1 != params[i+1]) {
          exitWithError(mpi_rank, "ERROR: List of oscillators for ensemble initialization should be consecutive!\n");
        }
      }
      return EnsembleInitialCondition{params};

    case InitialConditionType::DIAGONAL:
      return DiagonalInitialCondition{params};

    case InitialConditionType::THREESTATES:
      return ThreeStatesInitialCondition{};

    case InitialConditionType::NPLUSONE:
      return NPlusOneInitialCondition{};

    case InitialConditionType::PERFORMANCE:
      return PerformanceInitialCondition{};
  }
}

// Conversion helper implementations
void Config::convertInitialCondition(const std::optional<InitialConditionConfig>& config) {
  if (!config.has_value()) {
    initial_condition = BasisInitialCondition{}; // Default
    return;
  }

  initial_condition = convertInitialCondition(config.value());
}

void Config::convertOptimTarget(const std::optional<OptimTargetConfig>& config) {
  if (config.has_value()) {
    switch (config->target_type) {
      case TargetType::GATE: {
        GateOptimTarget gate_target;
        gate_target.gate_type = config->gate_type.value_or(GateType::NONE);
        gate_target.gate_file = config->gate_file.value_or("");
        target = gate_target;
        break;
      }
      case TargetType::PURE: {
        PureOptimTarget pure_target;
        for (auto level : config->levels) {
          pure_target.purestate_levels.push_back(static_cast<size_t>(level));
        }
        target = pure_target;
        break;
      }
      case TargetType::FROMFILE: {
        FileOptimTarget file_target;
        file_target.file = config->filename.value_or("");
        target = file_target;
        break;
      }
    }
  } else {
    target = PureOptimTarget{};
  }
}

void Config::convertPiPulses(const std::optional<std::vector<PiPulseConfig>>& pulses) {
  std::cout << "# Converting pi-pulse configurations...\n";
  std::cout << "# Number of oscillators: " << nlevels.size() << "\n";
  std::cout << "# Number of pi-pulse segments: "
            << (pulses.has_value() ? std::to_string(pulses->size()) : "0") << "\n";
  apply_pipulse.resize(nlevels.size());

  if (pulses.has_value()) {
    for (const auto& pulse_config : *pulses) {
      if (pulse_config.oscil_id < nlevels.size()) {
        // Set pipulse for this oscillator
        PiPulseSegment segment;
        segment.tstart = pulse_config.tstart;
        segment.tstop = pulse_config.tstop;
        segment.amp = pulse_config.amp;
        apply_pipulse[pulse_config.oscil_id].push_back(segment);

        logOutputToRank0(mpi_rank, "Applying PiPulse to oscillator " +
          std::to_string(pulse_config.oscil_id) + " in [" + std::to_string(segment.tstart) +
          ", " + std::to_string(segment.tstop) + "]: |p+iq|=" + std::to_string(segment.amp) + "\n");

        // Set zero control for all other oscillators during this pipulse
        for (size_t i = 0; i < nlevels.size(); i++){
          if (i != pulse_config.oscil_id) {
            PiPulseSegment zero_segment;
            zero_segment.tstart = pulse_config.tstart;
            zero_segment.tstop = pulse_config.tstop;
            zero_segment.amp = 0.0;
            apply_pipulse[i].push_back(zero_segment);
          }
        }
      }
    }
  }
}

void Config::convertControlSegments(const std::optional<std::map<int, std::vector<ControlSegmentConfig>>>& segments) {
  ControlSegment default_segment = {ControlType::BSPLINE, SplineParams{10, 0.0, ntime * dt}};

  for (size_t i = 0; i < oscillator_optimization.size(); i++) {
    if (!segments.has_value() || segments->find(static_cast<int>(i)) == segments->end()) {
      oscillator_optimization[i].control_segments = {default_segment};
      continue;
    }
    for (const auto& seg_config : segments->at(static_cast<int>(i))) {
      const auto& params = seg_config.parameters;

      // Create appropriate params variant based on type
      ControlSegment segment;
      segment.type = seg_config.control_type;

      if (seg_config.control_type == ControlType::BSPLINE ||
          seg_config.control_type == ControlType::BSPLINE0) {
        SplineParams spline_params;
        assert(params.size() >= 1); // nspline is required, should be validated in ConfigBuilder
        spline_params.nspline = static_cast<size_t>(params[0]);
        spline_params.tstart = params.size() > 1 ? params[1] : 0.0;
        spline_params.tstop = params.size() > 2 ? params[2] : ntime * dt;
        segment.params = spline_params;
      } else if (seg_config.control_type == ControlType::BSPLINEAMP) {
        SplineAmpParams spline_amp_params;
        assert(params.size() >= 2); // nspline and scaling are required, should be validated in ConfigBuilder
        spline_amp_params.nspline = static_cast<size_t>(params[0]);
        spline_amp_params.scaling = static_cast<double>(params[1]);
        spline_amp_params.tstart = params.size() > 2 ? params[2] : 0.0;
        spline_amp_params.tstop = params.size() > 3 ? params[3] : ntime * dt;
        segment.params = spline_amp_params;
      } else if (seg_config.control_type == ControlType::STEP) {
        StepParams step_params;
        assert(params.size() >= 3); // step_amp1, step_amp2, tramp are required, should be validated in ConfigBuilder
        step_params.step_amp1 = static_cast<double>(params[0]);
        step_params.step_amp2 = static_cast<double>(params[1]);
        step_params.tramp = static_cast<double>(params[2]);
        step_params.tstart = params.size() > 3 ? params[3] : 0.0;
        step_params.tstop = params.size() > 4 ? params[4] : ntime * dt;
        segment.params = step_params;
      }

      default_segment = segment;
      oscillator_optimization[i].control_segments.push_back(segment);
    }
  }
}

void Config::convertControlInitializations(const std::optional<std::map<int, std::vector<ControlInitializationConfig>>>& init_configs) {
  ControlSegmentInitialization default_init = ControlSegmentInitializationConstant{0.0, 0.0};

  for (size_t i = 0; i < oscillator_optimization.size(); i++) {
    if (!init_configs.has_value() || init_configs->find(static_cast<int>(i)) == init_configs->end()) {
      oscillator_optimization[i].control_initializations = {default_init};
      continue;
    }
    for (const auto& init_config : init_configs->at(static_cast<int>(i))) {
      ControlSegmentInitialization init;

      switch (init_config.init_type) {
        case ControlInitializationType::FILE:
          init = ControlSegmentInitializationFile{init_config.filename.value()};
          break;
        case ControlInitializationType::CONSTANT:
          assert(init_config.amplitude.has_value()); // should be validated in ConfigBuilder
          init = ControlSegmentInitializationConstant{init_config.amplitude.value(), init_config.phase.value_or(0.0)};
          break;
        case ControlInitializationType::RANDOM:
          assert(init_config.amplitude.has_value()); // should be validated in ConfigBuilder
          init = ControlSegmentInitializationRandom{init_config.amplitude.value(), init_config.phase.value_or(0.0)};
          break;
      }

      default_init = init;
      oscillator_optimization[i].control_initializations.push_back(init);
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
        size_t num_segments = oscillator_optimization[osc_idx].control_segments.size();
        oscillator_optimization[osc_idx].control_bounds = bounds;
        copyLast(oscillator_optimization[osc_idx].control_bounds, num_segments);
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

void Config::setNumInitialConditions() {
  if      (std::holds_alternative<FromFileInitialCondition>(initial_condition) ) n_initial_conditions = 1;
  else if (std::holds_alternative<PureInitialCondition>(initial_condition) ) n_initial_conditions = 1;
  else if (std::holds_alternative<PerformanceInitialCondition>(initial_condition) ) n_initial_conditions = 1;
  else if (std::holds_alternative<EnsembleInitialCondition>(initial_condition) ) n_initial_conditions = 1;
  else if (std::holds_alternative<ThreeStatesInitialCondition>(initial_condition) ) n_initial_conditions = 3;
  else if (std::holds_alternative<NPlusOneInitialCondition>(initial_condition) )  {
    // compute system dimension N
    n_initial_conditions = 1;
    for (size_t i=0; i<nlevels.size(); i++){
      n_initial_conditions *= nlevels[i];
    }
    n_initial_conditions +=1;
  }
  else if (std::holds_alternative<DiagonalInitialCondition>(initial_condition) ||
           std::holds_alternative<BasisInitialCondition>(initial_condition) ) {
    /* Compute ninit = dim(subsystem defined by list of oscil IDs) */
    n_initial_conditions = 1;
    for (size_t oscilID = 1; oscilID<nlevels.size(); oscilID++){
      if (oscilID < nessential.size()) n_initial_conditions *= nessential[oscilID];
    }
    if (std::holds_alternative<BasisInitialCondition>(initial_condition)  ) {
      // if Schroedinger solver: ninit = N, do nothing.
      // else Lindblad solver: ninit = N^2
      if (collapse_type == LindbladType::NONE) n_initial_conditions = (int) pow(n_initial_conditions,2.0);
    }
  }
  if (!quietmode) {
    logOutputToRank0(mpi_rank, "Number of initial conditions: " + std::to_string(n_initial_conditions) + "\n");
  }
}

void Config::setOptimWeights(const std::optional<std::vector<double>>& optim_weights_) {
  // Set optimization weights, default to uniform weights summing to one
  optim_weights = optim_weights_.value_or(std::vector<double>{1.0});
  copyLast(optim_weights, n_initial_conditions);
  // Scale the weights such that they sum up to one: beta_i <- beta_i / (\sum_i beta_i)
  double scaleweights = 0.0;
  for (size_t i = 0; i < n_initial_conditions; i++) scaleweights += optim_weights[i];
  for (size_t i = 0; i < n_initial_conditions; i++) optim_weights[i] = optim_weights[i] / scaleweights;
}
