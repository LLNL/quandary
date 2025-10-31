#include <cstdio>
#include <fstream>
#include <vector>

#include "cfgparser.hpp"
#include "config.hpp"
#include "config_types.hpp"
#include "defs.hpp"
#include "util.hpp"

CfgParser::CfgParser(MPI_Comm comm, std::stringstream& logstream, bool quietmode) {
  // Initialize MPI and logging members
  this->comm = comm;
  this->log = &logstream;
  this->quietmode = quietmode;
  MPI_Comm_rank(comm, &mpi_rank);

  // Register config parameter setters
  // General options
  registerConfig("nlevels", nlevels);
  registerConfig("nessential", nessential);
  registerConfig("ntime", ntime);
  registerConfig("dt", dt);
  registerConfig("transfreq", transfreq);
  registerConfig("selfkerr", selfkerr);
  registerConfig("crosskerr", crosskerr);
  registerConfig("Jkl", Jkl);
  registerConfig("rotfreq", rotfreq);
  registerConfig("collapse_type", collapse_type);
  registerConfig("decay_time", decay_time);
  registerConfig("dephase_time", dephase_time);
  registerConfig("initialcondition", initialcondition);
  registerConfig("apply_pipulse", apply_pipulse);
  registerConfig("hamiltonian_file_Hsys", hamiltonian_file_Hsys);
  registerConfig("hamiltonian_file_Hc", hamiltonian_file_Hc);

  // Optimization options
  registerConfig("control_enforceBC", control_enforceBC);
  registerConfig("optim_target", optim_target);
  registerConfig("gate_rot_freq", gate_rot_freq);
  registerConfig("optim_objective", optim_objective);
  registerConfig("optim_weights", optim_weights);

  // Indexed settings (per-oscillator)
  registerIndexedConfig("control_segments", indexed_control_segments);
  registerIndexedConfig("control_initialization", indexed_control_init);
  registerIndexedConfig("control_bounds", indexed_control_bounds);
  registerIndexedConfig("carrier_frequency", indexed_carrier_frequencies);
  registerIndexedConfig("output", indexed_output);
  registerConfig("optim_atol", optim_atol);
  registerConfig("optim_rtol", optim_rtol);
  registerConfig("optim_ftol", optim_ftol);
  registerConfig("optim_inftol", optim_inftol);
  registerConfig("optim_maxiter", optim_maxiter);
  registerConfig("optim_regul", optim_regul);
  registerConfig("optim_penalty", optim_penalty);
  registerConfig("optim_penalty_param", optim_penalty_param);
  registerConfig("optim_penalty_dpdm", optim_penalty_dpdm);
  registerConfig("optim_penalty_energy", optim_penalty_energy);
  registerConfig("optim_penalty_variation", optim_penalty_variation);
  registerConfig("optim_regul_tik0", optim_regul_tik0);
  registerConfig("optim_regul_interpolate", optim_regul_interpolate);

  // Output and runtypes
  registerConfig("datadir", datadir);
  registerConfig("output", output);
  registerConfig("output_frequency", output_frequency);
  registerConfig("optim_monitor_frequency", optim_monitor_frequency);
  registerConfig("runtype", runtype);
  registerConfig("usematfree", usematfree);
  registerConfig("linearsolver_type", linearsolver_type);
  registerConfig("linearsolver_maxiter", linearsolver_maxiter);
  registerConfig("timestepper", timestepper);
  registerConfig("rand_seed", rand_seed);
}

std::vector<std::vector<double>> CfgParser::convertIndexedToVectorVector(
    const std::map<int, std::vector<double>>& indexed_map,
    size_t num_oscillators) {
  std::vector<std::vector<double>> result(num_oscillators);
  for (const auto& [osc_idx, values] : indexed_map) {
    if (static_cast<size_t>(osc_idx) < result.size()) {
      result[osc_idx] = values;
    }
  }
  return result;
}

std::vector<std::vector<OutputType>> CfgParser::convertIndexedToOutputVector(
    const std::map<int, std::vector<OutputType>>& indexed_map,
    size_t num_oscillators) {
  std::vector<std::vector<OutputType>> result(num_oscillators);
  for (const auto& [osc_idx, values] : indexed_map) {
    if (static_cast<size_t>(osc_idx) < result.size()) {
      result[osc_idx] = values;
    }
  }
  return result;
}

Config CfgParser::build() {
  return Config(
    comm,
    *log,
    quietmode,
    // General options
    nlevels,
    nessential,
    ntime,
    dt,
    transfreq,
    selfkerr,
    crosskerr,
    Jkl,
    rotfreq,
    collapse_type,
    decay_time,
    dephase_time,
    initialcondition,
    apply_pipulse,
    hamiltonian_file_Hsys,
    hamiltonian_file_Hc,
    // Indexed control parameters
    indexed_control_segments,
    control_enforceBC,
    indexed_control_init,
    indexed_control_bounds,
    indexed_carrier_frequencies,
    // Optimization parameters
    optim_target,
    gate_rot_freq,
    optim_objective,
    optim_weights,
    optim_atol,
    optim_rtol,
    optim_ftol,
    optim_inftol,
    optim_maxiter,
    optim_regul,
    optim_penalty,
    optim_penalty_param,
    optim_penalty_dpdm,
    optim_penalty_energy,
    optim_penalty_variation,
    // Use deprecated form if optim_regul_tik0 not set
    optim_regul_tik0.has_value() ? optim_regul_tik0 : optim_regul_interpolate,
    // Output parameters
    datadir,
    indexed_output,
    output_frequency,
    optim_monitor_frequency,
    runtype,
    usematfree,
    linearsolver_type,
    linearsolver_maxiter,
    timestepper,
    rand_seed
  );
}

namespace {

std::string trimWhitespace(std::string s) {
  s.erase(std::remove_if(s.begin(), s.end(),
    [](unsigned char c) { return std::isspace(c); }), s.end());
  return s;
}

bool isComment(const std::string& line) {
  return line.size() > 0 && (line[0] == '#' || line[0] == '/');
}

bool isValidControlType(const std::string& str) {
  return CONTROL_TYPE_MAP.find(toLower(str)) != CONTROL_TYPE_MAP.end();
}

bool isValidControlInitializationType(const std::string& str) {
  return CONTROL_INITIALIZATION_TYPE_MAP.find(toLower(str)) != CONTROL_INITIALIZATION_TYPE_MAP.end();
}

} // namespace

std::vector<std::string> CfgParser::split(const std::string& str, char delimiter) {
  std::vector<std::string> result;
  std::stringstream ss(str);
  std::string item;
  while (std::getline(ss, item, delimiter)) {
    if (!item.empty()) {
      result.push_back(item);
    }
  }
  return result;
}

void CfgParser::applyConfigLine(const std::string& line) {
  std::string trimmedLine = trimWhitespace(line);
  if (!trimmedLine.empty() && !isComment(trimmedLine)) {
    int pos = trimmedLine.find('=');
    std::string key = trimmedLine.substr(0, pos);
    std::string value = trimmedLine.substr(pos + 1);

    // First try exact key match
    if (setters.count(key)) {
      try {
        setters[key](value);
      } catch (const std::exception& e) {
        logErrorToRank0(mpi_rank, "Error parsing '" + key + "': " + e.what());
      }
    } else {
      // Try to handle indexed settings (e.g., control_segments0, output1)
      bool handled = handleIndexedSetting(key, value);
      if (!handled) {
        logErrorToRank0(mpi_rank, "Unknown option '" + key + "'");
      }
    }
  }
}

bool CfgParser::handleIndexedSetting(const std::string& key, const std::string& value) {
  // Check if key ends with a digit (indexed setting)
  if (key.empty() || !std::isdigit(key.back())) {
    return false;
  }

  // Find where the index starts
  size_t index_pos = key.find_last_not_of("0123456789") + 1;
  if (index_pos == std::string::npos || index_pos >= key.length()) {
    return false;
  }

  std::string base_key = key.substr(0, index_pos);
  int index = std::stoi(key.substr(index_pos));

  if (indexed_setters.count(base_key)) {
    try {
      indexed_setters[base_key](index, value);
      return true;
    } catch (const std::exception& e) {
      logErrorToRank0(mpi_rank, "Error parsing indexed setting '" + key + "': " + e.what());
    }
  }

  return false;
}

void CfgParser::loadFromFile(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    logErrorToRank0(mpi_rank, "Unable to read the file " + filename);
    exit(1);
  }

  loadFromStream(file);
  file.close();
}

void CfgParser::loadFromString(const std::string& config_content) {
  std::istringstream stream(config_content);
  loadFromStream(stream);
}

// Struct converter implementations
template<>
InitialConditionConfig CfgParser::convertFromString<InitialConditionConfig>(const std::string& str) {
  auto parts = split(str);
  if (parts.empty()) {
    logErrorToRank0(mpi_rank, "ERROR: Empty initialcondition specification");
    exit(1);
  }

  InitialConditionConfig config;
  config.type = convertFromString<InitialConditionType>(parts[0]);

  if (config.type == InitialConditionType::FROMFILE) {
    if (parts.size() < 2) {
      logErrorToRank0(mpi_rank, "ERROR: initialcondition of type FROMFILE must have a filename");
    }
    config.filename = parts[1];
  } else {
    // Parse remaining parameters as integers
    for (size_t i = 1; i < parts.size(); ++i) {
      config.params.push_back(convertFromString<int>(parts[i]));
    }
  }

  return config;
}

template<>
OptimTargetConfig CfgParser::convertFromString<OptimTargetConfig>(const std::string& str) {
  auto parts = split(str);
  if (parts.empty()) {
    exitWithError(mpi_rank, "ERROR: optim_target must have at least a target type specified.");
  }

  OptimTargetConfig config;
  config.target_type = convertFromString<TargetType>(parts[0]);

  switch (config.target_type) {
    case TargetType::GATE:
      if (parts.size() < 2) {
        exitWithError(mpi_rank, "Target type 'gate' requires a gate name.");
      }
      config.gate_type = convertFromString<GateType>(parts[1]);
      if (config.gate_type == GateType::FILE) {
        if (parts.size() < 3) {
          exitWithError(mpi_rank, "ERROR: Gate type 'file' requires a filename.");
        }
        config.gate_file = parts[2];
      }
      break;
    case TargetType::PURE:
      for (size_t i = 1; i < parts.size(); ++i) {
        config.levels.push_back(convertFromString<int>(parts[i]));
      }
      break;
    case TargetType::FROMFILE:
      if (parts.size() < 2) {
        exitWithError(mpi_rank, "ERROR: Gate type 'file' requires a filename.");
      }
      config.filename = parts[1];
      break;
  }

  return config;
}

template<>
std::vector<PiPulseConfig> CfgParser::convertFromString<std::vector<PiPulseConfig>>(const std::string& str) {
  auto parts = split(str);
  if (parts.size() % 4 != 0) {
    exitWithError(mpi_rank, "ERROR: PiPulse vector requires multiples of 4 parameters: oscil_id, tstart, tstop, amp");
  }

  std::vector<PiPulseConfig> configs;
  for (size_t i = 0; i < parts.size(); i += 4) {
    PiPulseConfig config;
    config.oscil_id = convertFromString<size_t>(parts[i]);
    config.tstart = convertFromString<double>(parts[i+1]);
    config.tstop = convertFromString<double>(parts[i+2]);
    config.amp = convertFromString<double>(parts[i+3]);
    configs.push_back(config);
  }

  return configs;
}

template<>
std::vector<ControlSegmentConfig> CfgParser::convertFromString<std::vector<ControlSegmentConfig>>(const std::string& str) {
  const auto parts = split(str);

  std::vector<ControlSegmentConfig> segments;
  size_t i = 0;

  while (i < parts.size()) {
    if (!isValidControlType(parts[i])) {
      exitWithError(mpi_rank, "ERROR: Expected control type, got: " + parts[i]);
    }

    ControlSegmentConfig segment;
    segment.control_type = convertFromString<ControlType>(parts[i++]);

    // Parse parameters until next ControlType or end
    while (i < parts.size() && !isValidControlType(parts[i])) {
      segment.parameters.push_back(convertFromString<double>(parts[i++]));
    }

    // Validate minimum parameter count
    size_t min_params = 1;
    switch (segment.control_type) {
      case ControlType::STEP:
        min_params = 3; // step_amp1, step_amp2, tramp
        break;
      case ControlType::BSPLINE:
      case ControlType::BSPLINE0:
        min_params = 1; // num_basis_functions
        break;
      case ControlType::BSPLINEAMP:
        min_params = 2; // nspline, scaling
        break;
      case ControlType::NONE:
        exitWithError(mpi_rank, "ERROR: Control segment type NONE is not valid for configuration.");
        break;
    }

    if (segment.parameters.size() < min_params) {
      exitWithError(mpi_rank, "ERROR: Control type requires at least " + std::to_string(min_params) +
        " parameters, got " + std::to_string(segment.parameters.size()));
    }

    segments.push_back(segment);
  }

  return segments;
}

template<>
std::vector<ControlInitializationConfig> CfgParser::convertFromString<std::vector<ControlInitializationConfig>>(const std::string& str) {
  const auto parts = split(str);

  std::vector<ControlInitializationConfig> initializations;
  size_t i = 0;

  while (i < parts.size()) {
    if (!isValidControlInitializationType(parts[i])) {
      exitWithError(mpi_rank, "ERROR: Expected control initialization type, got: " + parts[i]);
    }

    ControlInitializationConfig initialization;
    initialization.init_type = convertFromString<ControlInitializationType>(parts[i++]);

    // Validate minimum parameter count
    if (parts.size() <= 1) {
      exitWithError(mpi_rank, "ERROR: Expected control_initialization to have a type and at least one parameter.");
    }

    switch (initialization.init_type) {
      case ControlInitializationType::FILE:
        initialization.filename = parts[i];
        initializations.push_back(initialization);
        return initializations;
      case ControlInitializationType::CONSTANT:
      case ControlInitializationType::RANDOM:
        initialization.amplitude = convertFromString<double>(parts[i++]);
        if (i < parts.size() && !isValidControlInitializationType(parts[i])) {
          initialization.phase = convertFromString<double>(parts[i++]);
        }
        break;
    }

    initializations.push_back(initialization);
  }

  return initializations;
}
