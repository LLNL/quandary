#include "config.hpp"
#include <cassert>
#include <cstddef>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <toml++/impl/forward_declarations.hpp>
#include <toml++/impl/table.hpp>
#include <vector>

#include "config_types.hpp"
#include "cfgparser.hpp"
#include "defs.hpp"
#include "util.hpp"
#include "config_validators.hpp"

// Helper function to convert enum back to string using existing enum maps
template<typename EnumType>
std::string enumToString(EnumType value, const std::map<std::string, EnumType>& type_map) {
  for (const auto& [str, enum_val] : type_map) {
    if (enum_val == value) return "\"" + str + "\"";
  }
  return "\"unknown\"";
}

template<typename EnumType>
std::vector<EnumType> convertStringVectorToEnum(
  const std::vector<std::string>& strings,
  const std::map<std::string, EnumType>& type_map) {
  std::vector<EnumType> result;
  result.reserve(strings.size());
  for (const auto& str : strings) {
    auto enum_val = parseEnum(str, type_map);
    if (!enum_val) {
      throw std::runtime_error("Unknown enum value: " + str);
    }
    result.push_back(*enum_val);
  }
  return result;
}

namespace {

GateType parseGateType(const std::optional<std::string>& gate_type_str) {
  if (!gate_type_str.has_value()) {
    return GateType::NONE;
  }

  auto it = GATE_TYPE_MAP.find(gate_type_str.value());
  if (it != GATE_TYPE_MAP.end()) {
    return it->second;
  } else {
    return GateType::NONE;
  }
}

} // namespace

OptimTargetSettings Config::parseOptimTarget(
  const std::optional<OptimTargetConfig>& opt_config,
  const std::vector<size_t>& nlevels) const {
  if (!opt_config.has_value()) {
    return PureOptimTarget{};
  }

  const OptimTargetConfig& config = opt_config.value();

  // Convert target type string to enum
  auto type = parseEnum(config.target_type, TARGET_TYPE_MAP);
  if (!type.has_value()) {
    exitWithError(mpi_rank, "ERROR: Unknown optimization target type: " + config.target_type);
  }

  switch (*type) {
    case TargetType::GATE: {
      GateOptimTarget gate_target;
      gate_target.gate_type = parseGateType(config.gate_type);
      gate_target.gate_file = config.gate_file.value_or("");
      return gate_target;
    }

    case TargetType::PURE: {
      PureOptimTarget pure_target;

      if (!config.levels.has_value() || config.levels->empty()) {
        logOutputToRank0(mpi_rank, "# Warning: You want to prepare a pure state, but didn't specify which one."
          " Taking default: ground-state |0...0> \n");
        pure_target.purestate_levels = std::vector<size_t>(nlevels.size(), 0);
        return pure_target;
      }

      // Copy levels and validate
      for (auto level : config.levels.value()) {
        pure_target.purestate_levels.push_back(static_cast<size_t>(level));
      }
      pure_target.purestate_levels.resize(nlevels.size(), nlevels.back());

      for (size_t i = 0; i < nlevels.size(); i++) {
        if (pure_target.purestate_levels[i] >= nlevels[i]) {
          exitWithError(mpi_rank, "ERROR in config setting. The requested pure state target |" +
            std::to_string(pure_target.purestate_levels[i]) +
            "> exceeds the number of modeled levels for that oscillator (" +
            std::to_string(nlevels[i]) + ").\n");
        }
      }

      return pure_target;
    }

    case TargetType::FROMFILE: {
      FileOptimTarget file_target;
      file_target.file = config.filename.value_or("");
      return file_target;
    }
  }

  // Should never reach here, but satisfy compiler
  return PureOptimTarget{};
}

namespace {
// Helper to extract optional vectors directly from TOML
template<typename T>
std::optional<std::vector<T>> get_optional_vector(const toml::node_view<toml::node>& node) {
  auto* arr = node.as_array();
  if (!arr) return std::nullopt;

  std::vector<T> result;
  for (size_t i = 0; i < arr->size(); ++i) {
    auto val = arr->at(i).template value<T>();
    if (!val) return std::nullopt;  // Type mismatch in array element
    result.push_back(*val);
  }

  return result;
}

} // namespace

Config::Config(
  int mpi_rank_,
  std::stringstream& log_,
  bool quietmode_,
  const toml::table& table
) :
  mpi_rank(mpi_rank_),
  log(log_),
  quietmode(quietmode_) {

  try {
    // Parse system settings
    if(!table.contains("system")) {
      exitWithError(mpi_rank, "ERROR: [system] section required in TOML config");
    }
    auto system = *table["system"].as_table();

    nlevels = validators::vector_field<size_t>(system, "nlevels")
      .required()
      .min_length(1)
      .positive()
      .get();

    size_t num_osc = nlevels.size();
    size_t num_pairs_osc = (num_osc - 1) * num_osc / 2;

    nessential = validators::vector_field<size_t>(system, "nessential")
      .min_length(1)
      .positive()
      .get_or(nlevels);
    copyLast(nessential, num_osc);

    ntime = validators::field<size_t>(system, "ntime")
      .positive()
      .get_or(ntime);

    dt = validators::field<double>(system, "dt")
      .positive()
      .get_or(dt);

    transfreq = validators::vector_field<double>(system, "transfreq")
      .required()
      .min_length(1)
      .get();
    copyLast(transfreq, num_osc);

    selfkerr = validators::vector_field<double>(system, "selfkerr")
      .min_length(1)
      .get_or(std::vector<double>(num_osc, 0.0));
    copyLast(selfkerr, num_osc);

    crosskerr = validators::vector_field<double>(system, "crosskerr")
      .min_length(1)
      .get_or(std::vector<double>(num_pairs_osc, 0.0));
    copyLast(crosskerr, num_pairs_osc);

    Jkl = validators::vector_field<double>(system, "Jkl")
      .min_length(1)
      .get_or(std::vector<double>(num_pairs_osc, 0.0));
    copyLast(Jkl, num_pairs_osc);

    rotfreq = validators::vector_field<double>(system, "rotfreq")
      .required()
      .min_length(1)
      .get();
    copyLast(rotfreq, num_osc);

    std::string collapse_type_str = validators::field<std::string>(system, "collapse_type")
      .get_or("none");
    collapse_type = parseEnum(collapse_type_str, LINDBLAD_TYPE_MAP)
      .value_or(LindbladType::NONE);

    decay_time = validators::vector_field<double>(system, "decay_time")
      .get_or(std::vector<double>(num_osc, 0.0));
    copyLast(decay_time, num_osc);

    dephase_time = validators::vector_field<double>(system, "dephase_time")
      .get_or(std::vector<double>(num_osc, 0.0));
    copyLast(dephase_time, num_osc);

    std::optional<InitialConditionConfig> init_cond_config = std::nullopt;
    if (system.contains("initial_condition")) {
      auto init_cond_table = *system["initial_condition"].as_table();
      std::string type_str = validators::field<std::string>(init_cond_table, "type").required().get();
      std::optional<std::vector<size_t>> levels = get_optional_vector<size_t>(init_cond_table["levels"]);
      std::optional<std::vector<size_t>> osc_IDs = get_optional_vector<size_t>(init_cond_table["oscIDs"]);
      std::optional<std::string> filename = init_cond_table["filename"].value<std::string>();
      init_cond_config = {type_str, osc_IDs, levels, filename};
    }
    initial_condition = parseInitialCondition(init_cond_config);
    n_initial_conditions = computeNumInitialConditions();

    std::optional<std::vector<PiPulseConfig>> pipulses = std::nullopt;
    auto apply_pipulse_node = system["apply_pipulse"];
    if (apply_pipulse_node.is_array_of_tables()) {
      pipulses = std::vector<PiPulseConfig>();
      for (auto& elem : *apply_pipulse_node.as_array()) {
        auto table = *elem.as_table();
        size_t oscilID = validators::field<size_t>(table, "oscID").required().get();
        double tstart = validators::field<double>(table, "tstart").required().get();
        double tstop = validators::field<double>(table, "tstop").required().get();
        double amp = validators::field<double>(table, "amp").required().get();
        PiPulseConfig pi_pulse_config = {oscilID, tstart, tstop, amp};
        pipulses->push_back(pi_pulse_config);
      }
    }
    apply_pipulse = parsePiPulses(pipulses);

    hamiltonian_file_Hsys = system["hamiltonian_file_Hsys"].value<std::string>();
    hamiltonian_file_Hc = system["hamiltonian_file_Hc"].value<std::string>();

    // Parse optimization settings
    toml::table optimization = table.contains("optimization") ? *table["optimization"].as_table() : toml::table{};

    oscillator_optimization.resize(num_osc);

    std::map<size_t, std::vector<ControlSegment>> control_segments;
    auto control_seg_node = optimization["control_segments"];
    if (control_seg_node.is_array_of_tables()) {
      for (auto& elem : *control_seg_node.as_array()) {
        auto table = *elem.as_table();
        size_t oscilID = validators::field<size_t>(table, "oscID").required().get();
        ControlSegment control_seg = parseControlSegment(table);
        control_segments[oscilID].push_back(control_seg);
      }
    }

    // Parse control initialization - separate sections for file vs per-segment
    std::map<size_t, std::vector<ControlSegmentInitialization>> osc_inits;
    if (optimization.contains("control_initialization_file")) {
      // Case 1: Global file initialization
      auto file_table = *optimization["control_initialization_file"].as_table();
      control_initialization_file = validators::field<std::string>(file_table, "filename").required().get();
    } else if (optimization.contains("control_initialization")) {
      // Case 2: Per-oscillator initialization
      auto init_node = optimization["control_initialization"];
      if (init_node.is_array_of_tables()) {
        for (auto& elem : *init_node.as_array()) {
          auto table = *elem.as_table();
          size_t oscID = validators::field<size_t>(table, "oscID").required().get();
          std::string type = validators::field<std::string>(table, "type").required().get();

          ControlSegmentInitialization init;
          if (type == "constant") {
            double amplitude = validators::field<double>(table, "amplitude").required().get();
            double phase = validators::field<double>(table, "phase").get_or(0.0);
            init = {ControlSegmentInitType::CONSTANT, amplitude, phase};
          } else if (type == "random") {
            double amplitude = validators::field<double>(table, "amplitude").get_or(0.1);
            double phase = validators::field<double>(table, "phase").get_or(0.0);
            init = {ControlSegmentInitType::RANDOM, amplitude, phase};
          } else {
            throw std::invalid_argument("Unknown control initialization type: " + type);
          }

          osc_inits[oscID].push_back(init);
        }
      }
    }

    std::optional<std::map<int, std::vector<double>>> control_bounds_opt = std::nullopt;
    auto control_bounds_node = optimization["control_bounds"];
    if (control_bounds_node.is_array_of_tables()) {
      control_bounds_opt = std::map<int, std::vector<double>>();
      for (auto& elem : *control_bounds_node.as_array()) {
        auto table = *elem.as_table();
        size_t oscilID = validators::field<size_t>(table, "oscID").required().get();
        (*control_bounds_opt)[oscilID] = validators::vector_field<double>(table, "values").required().get();
      }
    }
    auto control_bounds = parseIndexedToVectorVector(control_bounds_opt);

    std::optional<std::map<int, std::vector<double>>> carrier_freq_opt = std::nullopt;
    auto carrier_freq_node = optimization["carrier_frequency"];
    if (carrier_freq_node.is_array_of_tables()) {
      carrier_freq_opt = std::map<int, std::vector<double>>();
      for (auto& elem : *carrier_freq_node.as_array()) {
        auto table = *elem.as_table();
        size_t oscilID = validators::field<size_t>(table, "oscID").required().get();
        (*carrier_freq_opt)[oscilID] = validators::vector_field<double>(table, "values").required().get();
      }
    }
    auto carrier_freq = parseIndexedToVectorVector(carrier_freq_opt);

    std::vector<ControlSegment> default_segments = {{ControlType::BSPLINE, SplineParams{10, 0.0, ntime * dt}}};
    std::vector<ControlSegmentInitialization> default_initialization = {ControlSegmentInitialization{ControlSegmentInitType::CONSTANT, 0.0, 0.0}};
    std::vector<double> default_bounds = {10000.0};
    std::vector<double> default_freq = {0.0};
    for (size_t i = 0; i < oscillator_optimization.size(); i++) {
      if (control_segments.find(i) != control_segments.end()) {
        oscillator_optimization[i].control_segments = control_segments[i];
        default_segments = control_segments[i];
      } else {
        oscillator_optimization[i].control_segments = default_segments;
      }
      if (osc_inits.find(i) != osc_inits.end()) {
        oscillator_optimization[i].control_initializations = osc_inits[i];
        default_initialization = osc_inits[i];
      } else {
        oscillator_optimization[i].control_initializations = default_initialization;
      }
      size_t num_segments = oscillator_optimization[i].control_segments.size();

      oscillator_optimization[i].control_bounds = !control_bounds[i].empty() ? control_bounds[i] : default_bounds;
      copyLast(oscillator_optimization[i].control_bounds, num_segments);

      oscillator_optimization[i].carrier_frequencies = !carrier_freq[i].empty() ? carrier_freq[i] : default_freq;
      copyLast(oscillator_optimization[i].carrier_frequencies, num_segments);
    }

    control_enforceBC = validators::field<bool>(optimization, "control_enforceBC")
      .get_or(control_enforceBC);

    // optim_target
    std::optional<OptimTargetConfig> optim_target_config;
    if (optimization.contains("optim_target")) {
      auto target_table = *optimization["optim_target"].as_table();
      std::string type_str = validators::field<std::string>(target_table, "target_type").required().get();
      std::optional<std::string> gate_type_str = target_table["gate_type"].value<std::string>();
      std::optional<std::string> gate_file = target_table["gate_file"].value<std::string>();
      std::optional<std::vector<size_t>> levels = get_optional_vector<size_t>(target_table["levels"]);
      std::optional<std::string> filename = target_table["filename"].value<std::string>();
      optim_target_config = {type_str, gate_type_str, filename, gate_file, levels};
    }
    optim_target = parseOptimTarget(optim_target_config, nlevels);

    std::string optim_objective_str = validators::field<std::string>(optimization, "optim_objective")
      .get_or("");
    optim_objective = parseEnum(optim_objective_str, OBJECTIVE_TYPE_MAP)
      .value_or(optim_objective);

    // TODO: optim_weights

    tolerance.atol = validators::field<double>(optimization, "optim_atol")
      .positive()
      .get_or(tolerance.atol);
    tolerance.rtol = validators::field<double>(optimization, "optim_rtol")
      .positive()
      .get_or(tolerance.rtol);
    tolerance.ftol = validators::field<double>(optimization, "optim_ftol")
      .positive()
      .get_or(tolerance.ftol);
    tolerance.inftol = validators::field<double>(optimization, "optim_inftol")
      .positive()
      .get_or(tolerance.inftol);
    tolerance.maxiter = validators::field<size_t>(optimization, "optim_maxiter")
      .positive()
      .get_or(tolerance.maxiter);
    optim_regul = validators::field<double>(optimization, "optim_regul")
      .greater_than_equal(0.0)
      .get_or(optim_regul);

    penalty.penalty = validators::field<double>(optimization, "optim_penalty")
      .greater_than_equal(0.0)
      .get_or(penalty.penalty);
    penalty.penalty_param = validators::field<double>(optimization, "optim_penalty_param")
      .greater_than(0.0)
      .get_or(penalty.penalty_param);
    penalty.penalty_dpdm = validators::field<double>(optimization, "optim_penalty_dpdm")
      .greater_than_equal(0.0)
      .get_or(penalty.penalty_dpdm);
    penalty.penalty_energy = validators::field<double>(optimization, "optim_penalty_energy")
      .greater_than_equal(0.0)
      .get_or(penalty.penalty_energy);
    penalty.penalty_variation = validators::field<double>(optimization, "optim_penalty_variation")
      .greater_than_equal(0.0)
      .get_or(penalty.penalty_variation);

    if (!optimization.contains("optim_regul_tik0") && optimization.contains("optim_regul_interpolate")) {
      // Handle deprecated optim_regul_interpolate logic
      optim_regul_tik0 = validators::field<bool>(optimization, "optim_regul_interpolate").get();
      logOutputToRank0(mpi_rank, "# Warning: 'optim_regul_interpolate' is deprecated. Please use 'optim_regul_tik0' instead.\n");
    }
    optim_regul_tik0 = validators::field<bool>(optimization, "optim_regul_tik0")
      .get_or(optim_regul_tik0);


    // Parse output settings
    toml::table output = table.contains("output") ? *table["output"].as_table() : toml::table{};

    datadir = validators::field<std::string>(output, "datadir")
      .get_or(datadir);

    std::optional<std::map<int, std::vector<OutputType>>> output_to_write_opt = std::nullopt;
    auto write_node = output["write"];
    if (write_node.is_array_of_tables()) {
      output_to_write_opt = std::map<int, std::vector<OutputType>>();
      for (auto& elem : *write_node.as_array()) {
        auto table = *elem.as_table();
        size_t oscilID = validators::field<size_t>(table, "oscID").required().get();
        std::vector<std::string> types_str = validators::vector_field<std::string>(table, "type").required().get();
        std::vector<OutputType> types = convertStringVectorToEnum(types_str, OUTPUT_TYPE_MAP);
        (*output_to_write_opt)[oscilID] = types;
      }
    }
    output_to_write = parseIndexedToVectorVector(output_to_write_opt);

    output_frequency = validators::field<size_t>(output, "output_frequency")
      .positive()
      .get_or(output_frequency);
    optim_monitor_frequency = validators::field<size_t>(output, "optim_monitor_frequency")
      .positive()
      .get_or(optim_monitor_frequency);

    std::string runtype_str = validators::field<std::string>(output, "runtype")
      .get_or("");
    runtype = parseEnum(runtype_str, RUN_TYPE_MAP)
      .value_or(runtype);

    usematfree = validators::field<bool>(output, "usematfree")
      .get_or(usematfree);

    std::string linearsolver_type_str = validators::field<std::string>(output, "linearsolver_type")
      .get_or("");
    linearsolver_type = parseEnum(linearsolver_type_str, LINEAR_SOLVER_TYPE_MAP)
      .value_or(linearsolver_type);

    linearsolver_maxiter = validators::field<size_t>(output, "linearsolver_maxiter")
      .positive()
      .get_or(linearsolver_maxiter);

    std::string timestepper_type_str = validators::field<std::string>(output, "timestepper")
      .get_or("");
    timestepper_type = parseEnum(timestepper_type_str, TIME_STEPPER_TYPE_MAP)
      .value_or(TimeStepperType::IMR);

    setRandSeed(output["rand_seed"].value<int>());

  } catch (const validators::ValidationError& e) {
    exitWithError(mpi_rank, "ERROR: " + std::string(e.what()));
  }

  // Finalize and validate
  finalize();
  validate();
}

Config::Config(
  int mpi_rank_,
  std::stringstream& log_,
  bool quietmode_,
  const ConfigSettings& settings
) :
  mpi_rank(mpi_rank_),
  log(log_),
  quietmode(quietmode_)
{

  if (!settings.nlevels.has_value()) {
    exitWithError(mpi_rank, "ERROR: nlevels cannot be empty");
  }
  nlevels = settings.nlevels.value();
  size_t num_osc = nlevels.size();
  size_t num_pairs_osc = (num_osc - 1) * num_osc / 2;

  nessential = settings.nessential.value_or(nlevels);
  copyLast(nessential, num_osc);

  if (settings.ntime.has_value()) ntime = settings.ntime.value();

  if (settings.dt.has_value()) dt = settings.dt.value();

  if (!settings.transfreq.has_value()) {
    exitWithError(mpi_rank, "ERROR: transfreq cannot be empty");
  }
  transfreq = settings.transfreq.value();
  copyLast(transfreq, num_osc);

  selfkerr = settings.selfkerr.value_or(std::vector<double>(num_osc, 0.0));
  copyLast(selfkerr, num_osc);

  crosskerr = settings.crosskerr.value_or(std::vector<double>(num_pairs_osc, 0.0));
  copyLast(crosskerr, num_pairs_osc);

  Jkl = settings.Jkl.value_or(std::vector<double>(num_pairs_osc, 0.0));
  copyLast(Jkl, num_pairs_osc);

  if (!settings.rotfreq.has_value()) {
    exitWithError(mpi_rank, "ERROR: rotfreq cannot be empty");
  }
  rotfreq = settings.rotfreq.value();
  copyLast(rotfreq, num_osc);

  if (settings.collapse_type.has_value()) collapse_type = settings.collapse_type.value();

  decay_time = settings.decay_time.value_or(std::vector<double>(num_osc, 0.0));
  copyLast(decay_time, num_osc);

  dephase_time = settings.dephase_time.value_or(std::vector<double>(num_osc, 0.0));
  copyLast(dephase_time, num_osc);

  initial_condition = parseInitialCondition(settings.initialcondition);
  n_initial_conditions = computeNumInitialConditions();

  apply_pipulse = parsePiPulses(settings.apply_pipulse);

  hamiltonian_file_Hsys = settings.hamiltonian_file_Hsys;
  hamiltonian_file_Hc = settings.hamiltonian_file_Hc;

  // Control and optimization parameters
  oscillator_optimization.resize(num_osc);

  auto control_segments = parseControlSegments(settings.indexed_control_segments);
  for (size_t i = 0; i < oscillator_optimization.size(); i++) {
    oscillator_optimization[i].control_segments = control_segments[i];
  }

  // Control initialization
  if (settings.indexed_control_init.has_value()) {
    auto init_map = settings.indexed_control_init.value();
    if (init_map.find(0) != init_map.end() && !init_map[0].empty() && init_map[0][0].filename.has_value()) {
      control_initialization_file = init_map[0][0].filename;
    } else {
      auto control_initializations = parseControlInitializations(settings.indexed_control_init);
      for (size_t i = 0; i < oscillator_optimization.size(); i++) {
        oscillator_optimization[i].control_initializations = control_initializations[i];
      }
    }
  }

  if (settings.control_enforceBC.has_value()) control_enforceBC = settings.control_enforceBC.value();
  convertIndexedControlBounds(settings.indexed_control_bounds);
  convertIndexedCarrierFreqs(settings.indexed_carrier_frequencies);
  optim_target = parseOptimTarget(settings.optim_target, nlevels);

  if (settings.gate_rot_freq.has_value()) gate_rot_freq = settings.gate_rot_freq.value();
  copyLast(gate_rot_freq, num_osc);

  if (settings.optim_objective.has_value()) optim_objective = settings.optim_objective.value();

  setOptimWeights(settings.optim_weights);

  tolerance = OptimTolerance{};
  if (settings.optim_atol.has_value()) tolerance.atol = settings.optim_atol.value();
  if (settings.optim_rtol.has_value()) tolerance.rtol = settings.optim_rtol.value();
  if (settings.optim_ftol.has_value()) tolerance.ftol = settings.optim_ftol.value();
  if (settings.optim_inftol.has_value()) tolerance.inftol = settings.optim_inftol.value();
  if (settings.optim_maxiter.has_value()) tolerance.maxiter = settings.optim_maxiter.value();

  if (settings.optim_regul.has_value()) optim_regul = settings.optim_regul.value();

  penalty = OptimPenalty{};
  if (settings.optim_penalty.has_value()) penalty.penalty = settings.optim_penalty.value();
  if (settings.optim_penalty_param.has_value()) penalty.penalty_param = settings.optim_penalty_param.value();
  if (settings.optim_penalty_dpdm.has_value()) penalty.penalty_dpdm = settings.optim_penalty_dpdm.value();
  if (settings.optim_penalty_energy.has_value()) penalty.penalty_energy = settings.optim_penalty_energy.value();
  if (settings.optim_penalty_variation.has_value()) penalty.penalty_variation = settings.optim_penalty_variation.value();

  if (settings.optim_regul_tik0.has_value()) {
    optim_regul_tik0 = settings.optim_regul_tik0.value();
  } else if (settings.optim_regul_interpolate.has_value()) {
    // Handle deprecated optim_regul_interpolate logic
    optim_regul_tik0 = settings.optim_regul_interpolate.value();
    logOutputToRank0(mpi_rank, "# Warning: 'optim_regul_interpolate' is deprecated. Please use 'optim_regul_tik0' instead.\n");
  }

  // Output parameters
  if (settings.datadir.has_value()) datadir = settings.datadir.value();
  output_to_write = parseIndexedToVectorVector(settings.indexed_output);
  if (settings.output_frequency.has_value()) output_frequency = settings.output_frequency.value();
  if (settings.optim_monitor_frequency.has_value()) optim_monitor_frequency = settings.optim_monitor_frequency.value();
  if (settings.runtype.has_value()) runtype = settings.runtype.value();
  if (settings.usematfree.has_value()) usematfree = settings.usematfree.value();
  if (settings.linearsolver_type.has_value()) linearsolver_type = settings.linearsolver_type.value();
  if (settings.linearsolver_maxiter.has_value()) linearsolver_maxiter = settings.linearsolver_maxiter.value();
  if (settings.timestepper_type.has_value()) timestepper_type = settings.timestepper_type.value();
  setRandSeed(settings.rand_seed);

  // Finalize interdependent settings, then validate
  finalize();
  validate();
}

Config::~Config(){}


void Config::finalize() {
  // Hamiltonian file + matrix-free compatibility check
  if ((hamiltonian_file_Hsys.has_value() || hamiltonian_file_Hc.has_value()) && usematfree) {
    if (!quietmode) {
      logOutputToRank0(mpi_rank, "# Warning: Matrix-free solver cannot be used when Hamiltonian is read from file. Switching to sparse-matrix version.\n");
    }
    usematfree = false;
  }
}

void Config::validate() const {
  if (ntime <= 0) {
    exitWithError(mpi_rank, "ERROR: ntime must be positive, got " + std::to_string(ntime));
  }

  if (dt <= 0) {
    exitWithError(mpi_rank, "ERROR: dt must be positive, got " + std::to_string(dt));
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
}

Config Config::fromFile(int mpi_rank, const std::string& filename, std::stringstream* log, bool quietmode) {
  if (hasSuffix(filename, ".toml")) {
    return Config::fromToml(mpi_rank, filename, log, quietmode);
  } else {
    logOutputToRank0(mpi_rank, "# Warning: Config file does not have .toml extension. "
      "The deprecated .cfg format will be removed in future versions.\n");
    return Config::fromCfg(mpi_rank, filename, log, quietmode);
  }
}

Config Config::fromToml(int mpi_rank, const std::string& filename, std::stringstream* log, bool quietmode) {
  toml::table config = toml::parse_file(filename);
  return Config(mpi_rank, *log, quietmode, config);
}

Config Config::fromTomlString(int mpi_rank, const std::string& toml_content, std::stringstream* log, bool quietmode) {
  toml::table config = toml::parse(toml_content);
  return Config(mpi_rank, *log, quietmode, config);
}

Config Config::fromCfg(int mpi_rank, const std::string& filename, std::stringstream* log, bool quietmode) {
  CfgParser parser(mpi_rank, *log, quietmode);
  ConfigSettings settings = parser.parseFile(filename);
  return Config(mpi_rank, *log, quietmode, settings);
}

Config Config::fromCfgString(int mpi_rank, const std::string& cfg_content, std::stringstream* log, bool quietmode) {
  CfgParser parser(mpi_rank, *log, quietmode);
  ConfigSettings settings = parser.parseString(cfg_content);
  return Config(mpi_rank, *log, quietmode, settings);
}

namespace {
  template<typename T>
  std::string printVector(std::vector<T> vec) {
    std::string out = "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      out += std::to_string(vec[i]);
      if (i < vec.size() - 1) {
        out += ", ";
      }
    }
    out += "]";
    return out;
  }

  std::string print(const InitialCondition& initial_condition) {
    return std::visit([](const auto& opt) {
        return opt.toString();
    }, initial_condition);
  }
} //namespace

std::string ControlSegmentInitialization::toString() const {
  std::string str = "type = \"";
  str += enumToString(type, CONTROL_SEGMENT_INIT_TYPE_MAP);
  str += "\"\n";
  str += "amplitude = " + std::to_string(amplitude) + "\n" + "phase = " + std::to_string(phase);
  return str;
}

std::string FromFileInitialCondition::toString() const {
  return "{type = \"file\", filename = \"" + filename + "}";
}

std::string PureInitialCondition::toString() const {
  std::string out = "{type = \"pure\", levels = ";
  out += printVector(levels);
  out += "}";
  return out;
}

std::string OscillatorIDsInitialCondition::toString(std::string name) const {
  std::string out = "{type = \"" + name + "\", osc_IDs = ";
  out += printVector(osc_IDs);
  out += "}";
  return out;
}

void Config::printConfig() const {
  std::string delim = ", ";
  log << "# Configuration settings\n";
  log << "# =============================================\n\n";

  log << "[system]\n";
  log << "nlevels = " << printVector(nlevels) << "\n";
  log << "nessential = " << printVector(nessential) << "\n";
  log << "ntime = " << ntime << "\n";
  log << "dt = " << dt << "\n";
  log << "transfreq = " << printVector(transfreq) << "\n";
  log << "selfkerr = " << printVector(selfkerr) << "\n";
  log << "crosskerr = " << printVector(crosskerr) << "\n";
  log << "Jkl = " << printVector(Jkl) << "\n";
  log << "rotfreq = " << printVector(rotfreq) << "\n";
  log << "collapse_type = " << enumToString(collapse_type, LINDBLAD_TYPE_MAP) << "\n";
  log << "decay_time = " << printVector(decay_time) << "\n";
  log << "dephase_time = " << printVector(dephase_time) << "\n";
  log << "initial_condition = " << print(initial_condition) << "\n";

  for (size_t i = 0; i < apply_pipulse.size(); ++i) {
    for (const auto& segment : apply_pipulse[i]) {
      log << "apply_pipulse = " << i
          << ", " << segment.tstart
          << ", " << segment.tstop
          << ", " << segment.amp << "\n";
    }
  }
  if (hamiltonian_file_Hsys.has_value()) {
    log << "hamiltonian_file_Hsys = " << hamiltonian_file_Hsys.value() << "\n";
  }
  if (hamiltonian_file_Hc.has_value()) {
    log << "hamiltonian_file_Hc = " << hamiltonian_file_Hc.value() << "\n";
  }

  // Optimization Parameters
  log << "\n// Optimization Parameters\n";
  for (size_t i = 0; i < oscillator_optimization.size(); ++i) {
    if (!oscillator_optimization[i].control_segments.empty()) {
      const auto& seg = oscillator_optimization[i].control_segments[0];
      log << "control_segments" << i << " = "
          << enumToString(seg.type, CONTROL_TYPE_MAP);
      // Add segment-specific parameters
      if (std::holds_alternative<SplineParams>(seg.params)) {
        auto params = std::get<SplineParams>(seg.params);
        log << ", " << params.nspline << ", " << params.tstart << ", " << params.tstop;
      } else if (std::holds_alternative<SplineAmpParams>(seg.params)) {
        auto params = std::get<SplineAmpParams>(seg.params);
        log << ", " << params.nspline << ", " << params.scaling << ", " << params.tstart << ", " << params.tstop;
      } else if (std::holds_alternative<StepParams>(seg.params)) {
        auto params = std::get<StepParams>(seg.params);
        log << ", " << params.step_amp1 << ", " << params.step_amp2 << ", " << params.tramp << ", "
          << params.tstart << ", " << params.tstop;
      }
      log << "\n";
    }
  }
  log << "control_enforceBC = " << (control_enforceBC ? "true" : "false") << "\n";
  if (control_initialization_file.has_value()) {
    log << "[optimization.control_initilization_file]\nfilename = " << control_initialization_file.value() << "\n";
  } else {
    for (size_t i = 0; i < oscillator_optimization.size(); ++i) {
      if (!oscillator_optimization[i].control_initializations.empty()) {
        const auto& init = oscillator_optimization[i].control_initializations[0];
        log << "[[optimization.control_initialization]]" << i << " = " << init.toString() << "\n";
      }
    }
  }
  for (size_t i = 0; i < oscillator_optimization.size(); ++i) {
    if (!oscillator_optimization[i].control_bounds.empty()) {
      log << "control_bounds" << i << " = " << printVector(oscillator_optimization[i].control_bounds) << "\n";
    }
  }
  for (size_t i = 0; i < oscillator_optimization.size(); ++i) {
    if (!oscillator_optimization[i].carrier_frequencies.empty()) {
      log << "carrier_frequency" << i << " = " << printVector(oscillator_optimization[i].carrier_frequencies) << "\n";
    }
  }

  log << "optim_target = " << toString(optim_target) << "\n";
  log << "gate_rot_freq = " << printVector(gate_rot_freq) << "\n";
  log << "optim_objective = " << enumToString(optim_objective, OBJECTIVE_TYPE_MAP) << "\n";
  log << "optim_weights = " << printVector(optim_weights) << "\n";
  log << "optim_atol = " << tolerance.atol << "\n";
  log << "optim_rtol = " << tolerance.rtol << "\n";
  log << "optim_ftol = " << tolerance.ftol << "\n";
  log << "optim_inftol = " << tolerance.inftol << "\n";
  log << "optim_maxiter = " << tolerance.maxiter << "\n";
  log << "optim_regul = " << optim_regul << "\n";
  log << "optim_penalty = " << penalty.penalty << "\n";
  log << "optim_penalty_param = " << penalty.penalty_param << "\n";
  log << "optim_penalty_dpdm = " << penalty.penalty_dpdm << "\n";
  log << "optim_penalty_energy = " << penalty.penalty_energy << "\n";
  log << "optim_penalty_variation = " << penalty.penalty_variation << "\n";
  log << "optim_regul_tik0 = " << (optim_regul_tik0 ? "true" : "false") << "\n";

  // Output and runtypes
  log << "\n// Output and runtypes\n";
  log << "datadir = " << datadir << "\n";

  for (size_t i = 0; i < output_to_write.size(); ++i) {
    if (!output_to_write[i].empty()) {
      log << "output" << i << " = ";
      for (size_t j = 0; j < output_to_write[i].size(); ++j) {
        log << enumToString(output_to_write[i][j], OUTPUT_TYPE_MAP);
        if (j < output_to_write[i].size() - 1) log << ", ";
      }
      log << "\n";
    }
  }

  log << "output_frequency = " << output_frequency << "\n";
  log << "optim_monitor_frequency = " << optim_monitor_frequency << "\n";
  log << "runtype = " << enumToString(runtype, RUN_TYPE_MAP) << "\n";
  log << "usematfree = " << (usematfree ? "true" : "false") << "\n";
  log << "linearsolver_type = " << enumToString(linearsolver_type, LINEAR_SOLVER_TYPE_MAP) << "\n";
  log << "linearsolver_maxiter = " << linearsolver_maxiter << "\n";
  log << "timestepper = " << enumToString(timestepper_type, TIME_STEPPER_TYPE_MAP) << "\n";
  log << "rand_seed = " << rand_seed << "\n";

  log << "# =============================================\n\n";
}

// Template helper implementation
template<typename T>
std::vector<std::vector<T>> Config::parseIndexedToVectorVector(const std::optional<std::map<int, std::vector<T>>>& indexed_map) const {
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

InitialCondition Config::parseInitialCondition(const InitialConditionConfig& config) const {
  auto opt_type = parseEnum(config.type, INITCOND_TYPE_MAP);

  if (!opt_type.has_value()) {
    exitWithError(mpi_rank, "ERROR: initial condition type is required.");
  }
  InitialConditionType type = opt_type.value();

    /* Sanity check for Schrodinger solver initial conditions */
  if (collapse_type == LindbladType::NONE){
    if (type == InitialConditionType::ENSEMBLE ||
        type == InitialConditionType::THREESTATES ||
        type == InitialConditionType::NPLUSONE ){
          exitWithError(mpi_rank, "\n\n ERROR for initial condition setting: \n When running Schroedingers solver"
            " (collapse_type == NONE), the initial condition needs to be either 'pure' or 'from file' or 'diagonal' or 'basis'."
            " Note that 'diagonal' and 'basis' in the Schroedinger case are the same (all unit vectors).\n\n");
    }
  }

  // If no params are given for BASIS, ENSEMBLE, or DIAGONAL, default to all oscillators
  auto init_cond_IDs = config.osc_IDs.value_or(std::vector<size_t>{});
  if (!config.osc_IDs.has_value() && (type == InitialConditionType::BASIS ||
      type == InitialConditionType::ENSEMBLE ||
      type == InitialConditionType::DIAGONAL)) {
    for (size_t i = 0; i < nlevels.size(); i++) {
      init_cond_IDs.push_back(i);
    }
  }

  switch (type) {
    case InitialConditionType::FROMFILE:
      if (!config.filename.has_value()) {
        exitWithError(mpi_rank, "ERROR: initialcondition of type FROMFILE must have a filename");
      }
      return FromFileInitialCondition{config.filename.value()};
    case InitialConditionType::PURE:
      if (!config.levels.has_value()) {
        exitWithError(mpi_rank, "ERROR: initialcondition of type PURE must have 'levels'");
      }
      if (config.levels.value().size() != nlevels.size()) {
        exitWithError(mpi_rank, "ERROR: initialcondition of type PURE must have exactly " +
          std::to_string(nlevels.size()) + " parameters, got " + std::to_string(config.levels.value().size()));
      }
      for (size_t k=0; k < config.levels.value().size(); k++) {
        if (config.levels.value()[k] >= nlevels[k]){
          exitWithError(mpi_rank, "ERROR in config setting. The requested pure state initialization "
            + std::to_string(config.levels.value()[k]) + " exceeds the number of allowed levels for that oscillator ("
            + std::to_string(nlevels[k]) + ").\n");
        }
      }
      return PureInitialCondition{config.levels.value()};

    case InitialConditionType::BASIS:
      if (collapse_type == LindbladType::NONE) {
        // DIAGONAL and BASIS initial conditions in the Schroedinger case are the same. Overwrite it to DIAGONAL
        return DiagonalInitialCondition{init_cond_IDs};
      }
      return BasisInitialCondition{init_cond_IDs};

    case InitialConditionType::ENSEMBLE:
      if (init_cond_IDs.back() >= nlevels.size()) {
        exitWithError(mpi_rank, "ERROR: Last element in initialcondition params exceeds number of oscillators");
      }

      for (size_t i = 1; i < init_cond_IDs.size()-1; i++){
        if (init_cond_IDs[i]+1 != init_cond_IDs[i+1]) {
          exitWithError(mpi_rank, "ERROR: List of oscillators for ensemble initialization should be consecutive!\n");
        }
      }
      return EnsembleInitialCondition{init_cond_IDs};

    case InitialConditionType::DIAGONAL:
      return DiagonalInitialCondition{init_cond_IDs};

    case InitialConditionType::THREESTATES:
      return ThreeStatesInitialCondition{};

    case InitialConditionType::NPLUSONE:
      return NPlusOneInitialCondition{};

    case InitialConditionType::PERFORMANCE:
      return PerformanceInitialCondition{};
  }
}

// Conversion helper implementations
InitialCondition Config::parseInitialCondition(const std::optional<InitialConditionConfig>& config) const {
  if (!config.has_value()) {
    // Default: BasisInitialCondition with all oscillators
    std::vector<size_t> all_oscillators;
    for (size_t i = 0; i < nlevels.size(); i++) {
      all_oscillators.push_back(i);
    }
    return BasisInitialCondition{all_oscillators};
  }

  return parseInitialCondition(config.value());
}

std::vector<std::vector<PiPulseSegment>> Config::parsePiPulses(const std::optional<std::vector<PiPulseConfig>>& pulses) const {
  auto apply_pipulse = std::vector<std::vector<PiPulseSegment>>(nlevels.size());

  if (!pulses.has_value()) {
    return apply_pipulse;
  }
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
  return apply_pipulse;
}

std::vector<std::vector<ControlSegment>> Config::parseControlSegments(const std::optional<std::map<int, std::vector<ControlSegmentConfig>>>& segments_opt) const {
  std::vector<ControlSegment> default_segments = {{ControlType::BSPLINE, SplineParams{10, 0.0, ntime * dt}}};

  if (!segments_opt.has_value()) {
    return std::vector<std::vector<ControlSegment>>(nlevels.size(), default_segments);
  }
  const auto segments = segments_opt.value();
  auto parsed_segments = std::vector<std::vector<ControlSegment>>(nlevels.size());
  for (size_t i = 0; i < parsed_segments.size(); i++) {
    if (segments.find(static_cast<int>(i)) != segments.end()) {
      auto parsed = parseOscControlSegments(segments.at(i));
      parsed_segments[i] = parsed;
      default_segments = parsed;
    } else {
      parsed_segments[i] = default_segments;
    }
  }
  return parsed_segments;
}

std::vector<ControlSegment> Config::parseOscControlSegments(const std::vector<ControlSegmentConfig>& segments) const {
  std::vector<ControlSegment> control_segs = std::vector<ControlSegment>();

  for (const auto& seg_config : segments) {
    control_segs.push_back(parseControlSegment(seg_config));
  }
  return control_segs;
}

ControlSegment Config::parseControlSegment(const ControlSegmentConfig& seg_config) const {
  const auto& params = seg_config.parameters;

  // Create appropriate params variant based on type
  ControlSegment segment;
  segment.type = seg_config.control_type;

  if (seg_config.control_type == ControlType::BSPLINE ||
      seg_config.control_type == ControlType::BSPLINE0) {
    SplineParams spline_params;
    assert(params.size() >= 1); // nspline is required, should be validated in CfgParser
    spline_params.nspline = static_cast<size_t>(params[0]);
    spline_params.tstart = params.size() > 1 ? params[1] : 0.0;
    spline_params.tstop = params.size() > 2 ? params[2] : ntime * dt;
    segment.params = spline_params;
  } else if (seg_config.control_type == ControlType::BSPLINEAMP) {
    SplineAmpParams spline_amp_params;
    assert(params.size() >= 2); // nspline and scaling are required, should be validated in CfgParser
    spline_amp_params.nspline = static_cast<size_t>(params[0]);
    spline_amp_params.scaling = static_cast<double>(params[1]);
    spline_amp_params.tstart = params.size() > 2 ? params[2] : 0.0;
    spline_amp_params.tstop = params.size() > 3 ? params[3] : ntime * dt;
    segment.params = spline_amp_params;
  } else if (seg_config.control_type == ControlType::STEP) {
    StepParams step_params;
    assert(params.size() >= 3); // step_amp1, step_amp2, tramp are required, should be validated in CfgParser
    step_params.step_amp1 = static_cast<double>(params[0]);
    step_params.step_amp2 = static_cast<double>(params[1]);
    step_params.tramp = static_cast<double>(params[2]);
    step_params.tstart = params.size() > 3 ? params[3] : 0.0;
    step_params.tstop = params.size() > 4 ? params[4] : ntime * dt;
    segment.params = step_params;
  }

  return segment;
}

ControlSegment Config::parseControlSegment(const toml::table& table) const {
  ControlSegment segment;

  std::string type_str = validators::field<std::string>(table, "type").required().get();
  std::optional<ControlType> type = parseEnum(type_str, CONTROL_TYPE_MAP);
  if (!type.has_value()) {
    exitWithError(mpi_rank, "Unrecognized type '" + type_str + "' in control segment.");
  }
  segment.type = *type;

  switch (*type) {
  case ControlType::BSPLINE: {
    SplineParams spline_params;
    spline_params.nspline = validators::field<size_t>(table, "num").required().get();
    spline_params.tstart = validators::field<double>(table, "tstart").get_or(0.0);
    spline_params.tstop = validators::field<double>(table, "tstop").get_or(ntime * dt);
    segment.params = spline_params;
    break;
  }
  case ControlType::BSPLINE0: {
    SplineParams spline_params;
    spline_params.nspline = validators::field<size_t>(table, "num").required().get();
    spline_params.tstart = validators::field<double>(table, "tstart").get_or(0.0);
    spline_params.tstop = validators::field<double>(table, "tstop").get_or(ntime * dt);
    segment.params = spline_params;
    break;
  }
  case ControlType::BSPLINEAMP: {
    SplineAmpParams spline_amp_params;
    spline_amp_params.nspline = validators::field<size_t>(table, "num").required().get();
    spline_amp_params.scaling = validators::field<double>(table, "scaling").required().get();
    spline_amp_params.tstart = validators::field<double>(table, "tstart").get_or(0.0);
    spline_amp_params.tstop = validators::field<double>(table, "tstop").get_or(ntime * dt);
    segment.params = spline_amp_params;
    break;
  }
  case ControlType::STEP:
    StepParams step_params;
    step_params.step_amp1 = validators::field<double>(table, "step_amp1").required().get();
    step_params.step_amp2 = validators::field<double>(table, "step_amp2").required().get();
    step_params.tramp = validators::field<double>(table, "tramp").required().get();
    step_params.tstart = validators::field<double>(table, "tstart").get_or(0.0);
    step_params.tstop = validators::field<double>(table, "tstop").get_or(ntime * dt);
    segment.params = step_params;
    break;
  case ControlType::NONE:
    exitWithError(mpi_rank, "Unexpected control type " + type_str);
  }

  return segment;
}

std::vector<std::vector<ControlSegmentInitialization>> Config::parseControlInitializations(const std::optional<std::map<int, std::vector<ControlInitializationConfig>>>& init_configs) const {
  ControlSegmentInitialization default_init = ControlSegmentInitialization{ControlSegmentInitType::CONSTANT, 0.0, 0.0};

  std::vector<std::vector<ControlSegmentInitialization>> control_initializations(nlevels.size());
  for (size_t i = 0; i < nlevels.size(); i++) {
    if (!init_configs.has_value() || init_configs->find(static_cast<int>(i)) == init_configs->end()) {
      control_initializations[i] = {default_init};
      continue;
    }
    for (const auto& init_config : init_configs->at(static_cast<int>(i))) {
      ControlSegmentInitialization init = ControlSegmentInitialization{init_config.init_seg_type, init_config.amplitude.value(), init_config.phase.value_or(0.0)};

      default_init = init;
      control_initializations[i].push_back(init);
    }
  }
  return control_initializations;
}

ControlSegmentInitialization Config::parseControlInitialization(const toml::table& table) const {
  std::string type_str = validators::field<std::string>(table, "type").required().get();

  std::optional<ControlSegmentInitType> type = parseEnum(type_str, CONTROL_SEGMENT_INIT_TYPE_MAP);
  if (!type.has_value()) {
    exitWithError(mpi_rank, "Unrecognized type '" + type_str + "' in control initialization.");
  }
  return ControlSegmentInitialization {
    type.value(),
    validators::field<double>(table, "amplitude").required().get(),
    validators::field<double>(table, "phase").get_or(0.0)
  };
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

size_t Config::computeNumInitialConditions() const {
  size_t n_initial_conditions = 0;
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
  else if (std::holds_alternative<DiagonalInitialCondition>(initial_condition)) {
    /* Compute ninit = dim(subsystem defined by list of oscil IDs) */
    const auto& diag_init = std::get<DiagonalInitialCondition>(initial_condition);
    const auto& osc_IDs = diag_init.osc_IDs;

    n_initial_conditions = 1;
    for (size_t oscilID : osc_IDs) {
      if (oscilID < nessential.size()) n_initial_conditions *= nessential[oscilID];
    }
  }
  else if (std::holds_alternative<BasisInitialCondition>(initial_condition)) {
    /* Compute ninit = dim(subsystem defined by list of oscil IDs) */
    const auto& basis_init = std::get<BasisInitialCondition>(initial_condition);
    const auto& osc_IDs = basis_init.osc_IDs;

    n_initial_conditions = 1;
    for (size_t oscilID : osc_IDs) {
      if (oscilID < nessential.size()) n_initial_conditions *= nessential[oscilID];
    }
    // if Schroedinger solver: ninit = N, do nothing.
    // else Lindblad solver: ninit = N^2
    if (collapse_type != LindbladType::NONE) {
      n_initial_conditions = (int) pow(n_initial_conditions, 2.0);
    }
  }
  if (!quietmode) {
    logOutputToRank0(mpi_rank, "Number of initial conditions: " + std::to_string(n_initial_conditions) + "\n");
  }
  return n_initial_conditions;
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

void Config::setRandSeed(std::optional<int> rand_seed_) {
  rand_seed = rand_seed_.value_or(-1);
  if (rand_seed < 0){
    std::random_device rd;
    rand_seed = rd();  // random non-reproducable seed
  }
}
