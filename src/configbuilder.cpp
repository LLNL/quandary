#include <fstream>

#include "configbuilder.hpp"
#include "config.hpp"
#include "defs.hpp"

ConfigBuilder::ConfigBuilder(MPI_Comm comm, std::stringstream& logstream, bool quietmode) {
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

std::vector<std::vector<double>> ConfigBuilder::convertIndexedToVectorVector(
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

std::vector<std::vector<OutputType>> ConfigBuilder::convertIndexedToOutputVector(
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

Config ConfigBuilder::build() {
  // Extract and convert struct-based config to primitive types

  // Extract initial condition data
  InitialConditionType initial_condition_type = initialcondition
    ? initialcondition->type
    : InitialConditionType::BASIS;

  int n_initial_conditions = 1; // Default
  std::vector<size_t> initial_condition_IDs;
  std::string initial_condition_file;

  if (initialcondition) {
    // Convert int params to size_t IDs
    for (int param : initialcondition->params) {
      initial_condition_IDs.push_back(static_cast<size_t>(param));
    }
    if (initialcondition->filename) {
      initial_condition_file = *initialcondition->filename;
    }
  }

  // Extract optimization target data
  TargetType optim_target_type = optim_target
    ? optim_target->target_type
    : TargetType::PURE;

  std::string optim_target_file;
  GateType optim_target_gate_type = GateType::NONE;
  std::string optim_target_gate_file;
  std::vector<size_t> optim_target_purestate_levels;

  if (optim_target) {
    if (optim_target->filename) {
      optim_target_file = *optim_target->filename;
    }
    if (optim_target->gate_type) {
      optim_target_gate_type = *optim_target->gate_type;
    }
    // Convert int levels to size_t
    for (int level : optim_target->levels) {
      optim_target_purestate_levels.push_back(static_cast<size_t>(level));
    }
  }

  // Convert PiPulseConfig to PiPulseSegment format
  std::vector<std::vector<PiPulseSegment>> converted_apply_pipulse;
  if (apply_pipulse) {
    // Determine number of oscillators from nlevels
    size_t num_oscillators = nlevels ? nlevels->size() : 1;
    converted_apply_pipulse.resize(num_oscillators);

    for (const auto& pulse_config : *apply_pipulse) {
      if (pulse_config.oscil_id >= 0 &&
          static_cast<size_t>(pulse_config.oscil_id) < num_oscillators) {
        PiPulseSegment segment;
        segment.tstart = pulse_config.tstart;
        segment.tstop = pulse_config.tstop;
        segment.amp = pulse_config.amp;
        converted_apply_pipulse[pulse_config.oscil_id].push_back(segment);
      }
    }
  }

  // Convert ControlSegmentConfig to ControlSegment format using indexed settings
  std::vector<std::vector<ControlSegment>> converted_control_segments;
  size_t num_oscillators = nlevels ? nlevels->size() :
                           (indexed_control_segments.empty() ? 1 :
                            indexed_control_segments.rbegin()->first + 1);
  converted_control_segments.resize(num_oscillators);

  // Process indexed control segments (control_segments0, control_segments1, etc.)
  for (const auto& [osc_idx, seg_config] : indexed_control_segments) {
    if (static_cast<size_t>(osc_idx) >= converted_control_segments.size()) {
      converted_control_segments.resize(osc_idx + 1);
    }

    ControlSegment segment;
    segment.type = seg_config.control_type;

    // Create appropriate params variant based on type
    if (seg_config.control_type == ControlType::BSPLINE ||
        seg_config.control_type == ControlType::BSPLINE0) {
      SplineParams params;
      params.nspline = seg_config.num_basis_functions.value_or(10);
      params.tstart = seg_config.tstart.value_or(0.0);
      params.tstop = seg_config.tstop.value_or(ntime.value_or(1000) * dt.value_or(0.1));
      segment.params = params;
    } else if (seg_config.control_type == ControlType::BSPLINEAMP) {
      SplineAmpParams params;
      params.nspline = seg_config.num_basis_functions.value_or(10);
      params.tstart = seg_config.tstart.value_or(0.0);
      params.tstop = seg_config.tstop.value_or(ntime.value_or(1000) * dt.value_or(0.1));
      params.scaling = seg_config.scaling.value_or(1.0);
      segment.params = params;
    } else if (seg_config.control_type == ControlType::STEP) {
      StepParams params;
      params.step_amp1 = seg_config.amplitude_1.value_or(0.0);
      params.step_amp2 = seg_config.amplitude_2.value_or(0.0);
      params.tramp = 0.0;
      params.tstart = seg_config.tstart.value_or(0.0);
      params.tstop = seg_config.tstop.value_or(ntime.value_or(1000) * dt.value_or(0.1));
      segment.params = params;
    }

    converted_control_segments[osc_idx].push_back(segment);
  }

  // Convert ControlInitializationConfig to ControlSegmentInitialization format using indexed settings
  std::vector<std::vector<ControlSegmentInitialization>> converted_control_initializations;
  converted_control_initializations.resize(num_oscillators);

  // Process indexed control initializations (control_initialization0, control_initialization1, etc.)
  for (const auto& [osc_idx, init_config] : indexed_control_init) {
    if (static_cast<size_t>(osc_idx) >= converted_control_initializations.size()) {
      converted_control_initializations.resize(osc_idx + 1);
    }

    ControlSegmentInitialization init;
    init.type = init_config.init_type;
    init.amplitude = init_config.amplitude.value_or(0.0);
    init.phase = init_config.phase.value_or(0.0);

    converted_control_initializations[osc_idx].push_back(init);
  }

  // Apply defaults for required fields
  std::vector<size_t> final_nlevels = nlevels.value_or(std::vector<size_t>{2});
  std::vector<size_t> final_nessential = nessential.value_or(final_nlevels);
  int final_ntime = ntime.value_or(1000);
  double final_dt = dt.value_or(0.1);

  return Config(
    comm,
    mpi_rank,
    log,
    quietmode,
    // System parameters
    final_nlevels,
    final_nessential,
    final_ntime,
    final_dt,
    transfreq.value_or(std::vector<double>{}),
    selfkerr.value_or(std::vector<double>{}),
    crosskerr.value_or(std::vector<double>{}),
    Jkl.value_or(std::vector<double>{}),
    rotfreq.value_or(std::vector<double>{}),
    collapse_type.value_or(LindbladType::NONE),
    decay_time.value_or(std::vector<double>{}),
    dephase_time.value_or(std::vector<double>{}),
    initial_condition_type,
    n_initial_conditions,
    initial_condition_IDs,
    initial_condition_file,
    converted_apply_pipulse,
    // Control parameters
    converted_control_segments,
    control_enforceBC.value_or(false),
    converted_control_initializations,
    std::optional<std::string>{}, // control_initialization_file
    convertIndexedToVectorVector(indexed_control_bounds, num_oscillators),
    convertIndexedToVectorVector(indexed_carrier_frequencies, num_oscillators),
    // Optimization parameters
    optim_target_type,
    optim_target_file,
    optim_target_gate_type,
    optim_target_gate_file,
    optim_target_purestate_levels,
    gate_rot_freq.value_or(std::vector<double>{}),
    optim_objective.value_or(ObjectiveType::JFROBENIUS),
    optim_weights.value_or(std::vector<double>{}),
    // Construct OptimTolerance struct
    OptimTolerance{
      optim_atol.value_or(1e-8),
      optim_rtol.value_or(1e-4),
      optim_ftol.value_or(1e-8),
      optim_inftol.value_or(1e-5),
      optim_maxiter.value_or(200)
    },
    optim_regul.value_or(1e-4),
    // Construct OptimPenalty struct
    OptimPenalty{
      optim_penalty.value_or(0.0),
      optim_penalty_param.value_or(0.5),
      optim_penalty_dpdm.value_or(0.0),
      optim_penalty_energy.value_or(0.0),
      optim_penalty_variation.value_or(0.01)
    },
    optim_regul_tik0.value_or(false),
    // Output parameters
    datadir.value_or("./data_out"),
    convertIndexedToOutputVector(indexed_output, num_oscillators),
    output_frequency.value_or(1),
    optim_monitor_frequency.value_or(10),
    runtype.value_or(RunType::SIMULATION),
    usematfree.value_or(false),
    linearsolver_type.value_or(LinearSolverType::GMRES),
    linearsolver_maxiter.value_or(10),
    timestepper.value_or(TimeStepperType::IMR),
    rand_seed.value_or(1234),
    hamiltonian_file_Hsys.value_or(""),
    hamiltonian_file_Hc.value_or("")
  );
}

// void Config::validate() {
//   if ((!hamiltonian_file_Hsys.empty() || !hamiltonian_file_Hc.empty()) && usematfree) {
//     if (!quietmode) {
//       std::string message = "# Warning: Matrix-free solver can not be used when Hamiltonian is read from file. Switching to sparse-matrix version.\n";
//       logOutputToRank0(mpi_rank, message);
//     }
//     usematfree = false;
//   }

//   /* Sanity check for Schrodinger solver initial conditions */
//   if (collapse_type == LindbladType::NONE){
//     if (initial_condition_type == InitialConditionType::ENSEMBLE ||
//         initial_condition_type == InitialConditionType::THREESTATES ||
//         initial_condition_type == InitialConditionType::NPLUSONE ){
//           printf("\n\n ERROR for initial condition setting: \n When running Schroedingers solver (collapse_type == NONE), the initial condition needs to be either 'pure' or 'from file' or 'diagonal' or 'basis'. Note that 'diagonal' and 'basis' in the Schroedinger case are the same (all unit vectors).\n\n");
//           exit(1);
//     } else if (initial_condition_type == InitialConditionType::BASIS) {
//       // DIAGONAL and BASIS initial conditions in the Schroedinger case are the same. Overwrite it to DIAGONAL
//       initial_condition_type = InitialConditionType::DIAGONAL;
//     }
//   }

//     // Validate initial conditions
//     if (initial_condition_type == InitialConditionType::PURE) {
//       if (initial_condition_IDs.size() != nlevels.size()) {
//         std::string message = "ERROR during pure-state initialization: List of IDs must contain"
//           + std::to_string(nlevels.size()) + "elements!\n";
//         logErrorToRank0(mpi_rank, message);
//         exit(1);
//       }
//       for (size_t k=0; k < initial_condition_IDs.size(); k++) {
//         if (initial_condition_IDs[k] > nlevels[k]-1){
//           std::string message = "ERROR in config setting. The requested pure state initialization |"
//             + std::to_string(initial_condition_IDs[k])
//             + "> exceeds the number of allowed levels for that oscillator ("
//             + std::to_string(nlevels[k]-1) + ").\n";
//           logErrorToRank0(mpi_rank, message);
//           exit(1);
//         }
//         assert(initial_condition_IDs[k] < nlevels[k]);
//       }
//     } else if (initial_condition_type == InitialConditionType::ENSEMBLE) {
//       // Sanity check for the list in initcond_IDs!
//       assert(initial_condition_IDs.size() >= 1); // at least one element
//       assert(initial_condition_IDs[initial_condition_IDs.size()-1] < nlevels.size()); // last element can't exceed total number of oscillators
//       for (size_t i=0; i < initial_condition_IDs.size()-1; i++){ // list should be consecutive!
//         if (initial_condition_IDs[i]+1 != initial_condition_IDs[i+1]) {
//           logErrorToRank0(mpi_rank, "ERROR: List of oscillators for ensemble initialization should be consecutive!\n");
//           exit(1);
//         }
//       }
//     }
// }

// void Config::setNEssential(const std::string& nessential_str) {
//   /* Get the number of essential levels per oscillator.
//    * Default: same as number of levels */
//    // TODO must be called after nlevels is set.
//   nessential = nlevels;

//   std::vector<int> read_nessential = convertFromString<std::vector<int>>(nessential_str);
//   /* Overwrite if config option is given */
//   if (read_nessential[0] > -1) {
//     for (size_t iosc = 0; iosc<nlevels.size(); iosc++){
//       if (iosc < read_nessential.size()) nessential[iosc] = read_nessential[iosc];
//       else                               nessential[iosc] = read_nessential[read_nessential.size()-1];
//       if (nessential[iosc] > nlevels[iosc]) nessential[iosc] = nlevels[iosc];
//     }
//   }
// }

// void Config::setInitialConditions(const std::string& init_cond_str, LindbladType collapse_type_) {
//   std::vector<std::string> init_conds = convertFromString<std::vector<std::string>>(init_cond_str);

//   if(init_conds.empty()) {
//     logErrorToRank0(mpi_rank, "\n\n ERROR: Missing initial conditions.\n");
//     exit(1);
//   }

//   initial_condition_type = convertFromString<InitialConditionType>(init_conds[0]);
//   n_initial_conditions = 1;
//   switch(initial_condition_type) {
//     case InitialConditionType::FROMFILE:
//     case InitialConditionType::PURE:
//     case InitialConditionType::PERFORMANCE:
//     case InitialConditionType::ENSEMBLE:
//       n_initial_conditions = 1;
//       break;
//     case InitialConditionType::THREESTATES:
//       n_initial_conditions = 3;
//       break;
//     case InitialConditionType::NPLUSONE: {
//       // compute system dimension N
//       n_initial_conditions = 1;
//       for (size_t i=0; i<nlevels.size(); i++){
//         n_initial_conditions *= nlevels[i];
//       }
//       n_initial_conditions +=1;
//       break;
//     }
//     case InitialConditionType::DIAGONAL:
//     case InitialConditionType::BASIS: {
//       /* Compute n_initial_conditions = dim(subsystem defined by list of oscil IDs) */
//       n_initial_conditions = 1;
//       if (init_conds.size() < 2) {
//         for (size_t j=0; j<nlevels.size(); j++) init_conds.push_back(std::to_string(j));
//       }
//       for (size_t i = 1; i<init_cond_str.size(); i++){
//         size_t oscilID = std::stoi(init_conds[i]);
//         if (oscilID < nessential.size()) n_initial_conditions *= nessential[oscilID];
//       }
//       if (initial_condition_type == InitialConditionType::BASIS) {
//         // if Schroedinger solver: n_initial_conditions = N, do nothing.
//         // else Lindblad solver: n_initial_conditions = N^2
//         if (collapse_type_ != LindbladType::NONE) n_initial_conditions = (int) pow(n_initial_conditions,2.0);
//       }
//       break;
//     }

//     if (initial_condition_type == InitialConditionType::FROMFILE) {
//       if (init_conds.size() != 2) {
//         logErrorToRank0(mpi_rank, "Initial condition type 'file' requires a filename.");
//         exit(1);
//       }
//       initial_condition_file = init_conds[1];
//     } else {
//       if (init_conds.size() < 2) {
//         for (size_t j=0; j<nlevels.size(); j++)
//           initial_condition_IDs.push_back(j); // Default: all oscillators
//       } else {
//         for (size_t i=1; i<init_conds.size(); i++) {
//           initial_condition_IDs.push_back(std::stoi(init_conds[i])); // Use config option, if given.
//         }
//       }
//     }
//   }

//   if (!quietmode) {
//     logOutputToRank0(mpi_rank, "Number of initial conditions: " + std::to_string(n_initial_conditions));
//   }
// }

// void Config::setRandSeed(int rand_seed_) {
//   rand_seed = rand_seed_;
//   if (rand_seed_ < 0){
//     std::random_device rd;
//     rand_seed = rd();  // random non-reproducable seed
//   }
// }

// void Config::setApplyPiPulse(const std::string& value) {
//   std::vector<std::string> pipulse_str = convertFromString<std::vector<std::string>>(value);

//   if (pipulse_str.size() % 4 != 0) {
//     std::string message = "Wrong pi-pulse configuration. Number of elements must be multiple of 4!\n";
//     message += "apply_pipulse config option: <oscilID>, <tstart>, <tstop>, <amp>, <anotherOscilID>, <anotherTstart>, <anotherTstop>, <anotherAmp> ...\n";
//     logErrorToRank0(mpi_rank, message);
//     exit(1);
//   }
//   apply_pipulse.resize(nlevels.size());

//   size_t k=0;
//   while (k < pipulse_str.size()) {
//     // Set pipulse for this oscillator
//     size_t pipulse_id = std::stoi(pipulse_str[k+0]);
//     double tstart = std::stod(pipulse_str[k+1]);
//     double tstop = std::stod(pipulse_str[k+2]);
//     double amp = std::stod(pipulse_str[k+3]);
//     PiPulseSegment pipulse = {tstart, tstop, amp};
//     apply_pipulse[pipulse_id].push_back(pipulse);

//     std::ostringstream message;
//     message << "Applying PiPulse to oscillator " << pipulse_id << " in [" << pipulse.tstart << ","
//       << pipulse.tstop << "]: |p+iq|=" << pipulse.amp;
//     logOutputToRank0(mpi_rank, message.str());

//     // Set zero control for all other oscillators during this pipulse
//     for (size_t i=0; i<nlevels.size(); i++){
//       if (i != pipulse_id) {
//         apply_pipulse[i].push_back({tstart, tstop, 0.0});
//       }
//     }
//     k+=4;
//   }
// }

// void Config::setOptimTarget(const std::string& value) {
//   std::vector<std::string> target_str = Config::split(value);

//   if (target_str.empty()) {
//     logErrorToRank0(mpi_rank, "No optimization target specified.");
//     exit(1);
//   }

//   optim_target_type = convertFromString<TargetType>(target_str[0]);

//   if (optim_target_type == TargetType::FROMFILE) {
//     if (target_str.size() < 2) {
//       logErrorToRank0(mpi_rank, "Target type 'file' requires a filename.");
//       exit(1);
//     }
//     optim_target_file = target_str[1];
//   } else if (optim_target_type == TargetType::GATE) {
//     if (target_str.size() < 2) {
//       logErrorToRank0(mpi_rank, "Target type 'gate' requires a gate name.");
//       exit(1);
//     }
//     optim_target_gate_type = convertFromString<GateType>(target_str[1]);

//     if (optim_target_gate_type == GateType::FILE) {
//       if (target_str.size() < 3) {
//         logErrorToRank0(mpi_rank, "Gate type 'file' requires a filename.");
//         exit(1);
//       }
//       optim_target_gate_file = target_str[2];
//     }
//   } else if (optim_target_type == TargetType::PURE) {
//     if (target_str.size() < 2) {
//       logOutputToRank0(mpi_rank, "# Warning: You want to prepare a pure state, but didn't specify which one. Taking default: ground-state |0...0> \n");
//       optim_target_purestate_levels = std::vector<size_t>(nlevels.size(), 0);
//     } else {
//       for (size_t i = 1; i < target_str.size(); i++) {
//         optim_target_purestate_levels.push_back(convertFromString<size_t>(target_str[i]));
//       }
//       optim_target_purestate_levels.resize(nlevels.size(), nlevels.back());
//       for (size_t i = 0; i < nlevels.size(); i++) {
//         if (optim_target_purestate_levels[i] >= nlevels[i]) {
//           logErrorToRank0(mpi_rank, "ERROR in config setting. The requested pure state target |" + std::to_string(optim_target_purestate_levels[i]) +
//             "> exceeds the number of modeled levels for that oscillator (" + std::to_string(nlevels[i]) + ").\n");
//           exit(1);
//         }
//       }
//     }
//   }
// }

// void Config::setControlSegments(const std::string& value) {
//   // if not set tstart should be 0.0 and tstop should be nt * dt
//   switch (controlsegments.type) {
//     case ControlType::STEP: {
//       if (controlsegments.size() <= idstr+2){
//         printf("ERROR: Wrong setting for control segments: Step Amplitudes or tramp not found.\n");
//         exit(1);
//       }
//       break;
//     }
//     case ControlType::BSPLINE: {
//       if (controlsegments.size() <= idstr){
//         printf("ERROR: Wrong setting for control segments: Number of splines not found.\n");
//         exit(1);
//       }
//     }
// }

// void Config::setControlInitialization(const std::string& value) {
//     // phase should be zero if not set
//     // default should be constant and either 1 (if Control is Step) or 0 (else)
//     // Set a default if initialization string is not given for this segment
//     if (controlinitializations.size() < idini+2) {
//       controlinitializations.push_back("constant");
//       if (basisfunctions[seg]->getType() == ControlType::STEP)
//         controlinitializations.push_back("1.0");
//       else
//         controlinitializations.push_back("0.0");
//     }
// }

namespace {

std::string trimWhitespace(std::string s) {
  s.erase(std::remove_if(s.begin(), s.end(),
    [](unsigned char c) { return std::isspace(c); }), s.end());
  return s;
}

bool isComment(const std::string& line) {
  return line.size() > 0 && (line[0] == '#' || line[0] == '/');
}

} // namespace

std::vector<std::string> ConfigBuilder::split(const std::string& str, char delimiter) {
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

void ConfigBuilder::applyConfigLine(const std::string& line) {
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

bool ConfigBuilder::handleIndexedSetting(const std::string& key, const std::string& value) {
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

  // Use the unified indexed setters pattern
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

void ConfigBuilder::loadFromFile(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    logErrorToRank0(mpi_rank, "Unable to read the file " + filename);
    exit(1);
  }

  loadFromStream(file);
  file.close();
}

void ConfigBuilder::loadFromString(const std::string& config_content) {
  std::istringstream stream(config_content);
  loadFromStream(stream);
}

// Struct converter implementations
template<>
InitialConditionConfig ConfigBuilder::convertFromString<InitialConditionConfig>(const std::string& str) {
  auto parts = split(str);
  if (parts.empty()) {
    logErrorToRank0(mpi_rank, "ERROR: Empty initialcondition specification");
    exit(1);
  }

  InitialConditionConfig config;
  config.type = convertFromString<InitialConditionType>(parts[0]);

  // Parse remaining parameters as integers
  for (size_t i = 1; i < parts.size(); ++i) {
    config.params.push_back(convertFromString<int>(parts[i]));
  }

  return config;
}

template<>
OptimTargetConfig ConfigBuilder::convertFromString<OptimTargetConfig>(const std::string& str) {
  auto parts = split(str);
  if (parts.empty()) {
    logErrorToRank0(mpi_rank, "ERROR: Empty optim_target specification");
    exit(1);
  }

  OptimTargetConfig config;
  config.target_type = convertFromString<TargetType>(parts[0]);

  if (parts.size() > 1) {
    if (config.target_type == TargetType::GATE) {
      config.gate_type = convertFromString<GateType>(parts[1]);
    } else if (config.target_type == TargetType::FROMFILE) {
      config.filename = parts[1];
    } else if (config.target_type == TargetType::PURE) {
      // Parse pure state levels
      for (size_t i = 1; i < parts.size(); ++i) {
        config.levels.push_back(convertFromString<int>(parts[i]));
      }
    }
  }

  return config;
}

template<>
PiPulseConfig ConfigBuilder::convertFromString<PiPulseConfig>(const std::string& str) {
  auto parts = split(str);
  if (parts.size() != 4) {
    logErrorToRank0(mpi_rank, "ERROR: PiPulse requires 4 parameters: oscil_id, tstart, tstop, amp");
    exit(1);
  }

  PiPulseConfig config;
  config.oscil_id = convertFromString<int>(parts[0]);
  config.tstart = convertFromString<double>(parts[1]);
  config.tstop = convertFromString<double>(parts[2]);
  config.amp = convertFromString<double>(parts[3]);

  return config;
}

template<>
ControlSegmentConfig ConfigBuilder::convertFromString<ControlSegmentConfig>(const std::string& str) {
  auto parts = split(str);
  if (parts.size() < 2) {
    logErrorToRank0(mpi_rank, "ERROR: ControlSegment requires at least control_type and num_basis_functions");
    exit(1);
  }

  ControlSegmentConfig config;
  config.control_type = convertFromString<ControlType>(parts[0]);
  config.num_basis_functions = convertFromString<int>(parts[1]); // TODO

  // Parse optional parameters based on control type and available parts
  if (parts.size() > 2) {
    if (config.control_type == ControlType::BSPLINEAMP) {
      config.scaling = convertFromString<double>(parts[2]);
      if (parts.size() > 3) {
        config.tstart = convertFromString<double>(parts[3]);
        if (parts.size() > 4) {
          config.tstop = convertFromString<double>(parts[4]);
        }
      }
    } else {
      // For other control types, parameters 2 and 3 are tstart, tstop
      config.tstart = convertFromString<double>(parts[2]);
      if (parts.size() > 3) {
        config.tstop = convertFromString<double>(parts[3]);
      }
    }
  }

  return config;
}

template<>
ControlInitializationConfig ConfigBuilder::convertFromString<ControlInitializationConfig>(const std::string& str) {
  auto parts = split(str);
  if (parts.empty()) {
    logErrorToRank0(mpi_rank, "ERROR: Empty control_initialization specification");
    exit(1);
  }

  ControlInitializationConfig config;
  config.init_type = convertFromString<ControlInitializationType>(parts[0]);

  if (parts.size() > 1) {
    if (config.init_type == ControlInitializationType::FILE) {
      config.filename = parts[1];
    } else {
      config.amplitude = convertFromString<double>(parts[1]);
      if (parts.size() > 2) {
        config.phase = convertFromString<double>(parts[2]);
      }
    }
  }

  return config;
}
