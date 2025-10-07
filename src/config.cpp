#include "config.hpp"
#include "defs.hpp"
#include "util.hpp"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>


Config::Config(MPI_Comm comm_, std::stringstream& logstream, bool quietmode_)
{
  comm = comm_;
  MPI_Comm_rank(comm, &mpi_rank);
  log = &logstream;
  quietmode = quietmode_;

  registerVector("nlevels", nlevels);
  registerVector("nessential", nessential);
  registerScalar("ntime", ntime);
  registerScalar("dt", dt);
  registerAndFillVector("transfreq", transfreq, nlevels.size(), std::vector<double>(nlevels.size(), 1e20));
  registerAndFillVector("selfkerr", selfkerr, nlevels.size(), std::vector<double>(nlevels.size(), 0.0));

  size_t coupling_size = (nlevels.size()-1) * nlevels.size() / 2;
  registerAndFillVector("crosskerr", rotfreq, coupling_size, std::vector<double>(nlevels.size(), 0.0));
  registerAndFillVector("Jkl", Jkl, coupling_size, std::vector<double>(nlevels.size(), 0.0));

  registerAndFillVector("rotfreq", rotfreq, nlevels.size(), std::vector<double>(nlevels.size(), 0.0));
  registerScalar("collapse_type", collapse_type);
  registerAndFillVector("decay_time", decay_time, nlevels.size(), std::vector<double>(nlevels.size(), 0.0));
  registerAndFillVector("dephase_time", dephase_time, nlevels.size(), std::vector<double>(nlevels.size(), 0.0));
  setters["initialcondition"] = [this](const std::string& val) { setInitialConditions(val, collapse_type); };
  setters["apply_pipulse"] = [this](const std::string& val) { setApplyPiPulse(val); };

  registerVectorOfVectors("carrier_frequency", carrier_frequencies, nlevels.size(), 0.0);

  // TODO check if default is for all oscillators or only the first one
  const std::string default_seg_str = "spline, 10, 0.0, " + std::to_string(ntime * dt);
  // registerVectorOfVectors("control_segments", control_segments, nlevels.size(), default_seg_str);
  // setters["control_segments"] = [this](const std::string& val) { setControlSegments(val); };
  registerScalar("control_enforceBC", control_enforceBC);
  // setters["control_initialization"] = [this](const std::string& val) { setControlInitialization(val, control_type); };
  registerVectorOfVectors("control_bounds", control_bounds, nlevels.size(), 10000.0); // TODO default should be last value if shorter

  setters["optim_target"] = [this](const std::string& val) { setOptimTarget(val); };
  registerAndFillVector("gate_rot_freq", gate_rot_freq, nlevels.size(), std::vector<double>(nlevels.size(), 0.0));
  registerScalar("optim_objective", optim_objective);
  registerAndFillVector("optim_weights", optim_weights, n_initial_conditions, std::vector<double>(n_initial_conditions, 1.0));
  registerScalar("optim_atol", optim_atol);
  registerScalar("optim_rtol", optim_rtol);
  registerScalar("optim_ftol", optim_ftol);
  registerScalar("optim_inftol", optim_inftol);
  registerScalar("optim_maxiter", optim_maxiter);
  registerScalar("optim_regul", optim_regul);
  registerScalar("optim_penalty", optim_penalty);
  registerScalar("optim_penalty_param", optim_penalty_param);
  registerScalar("optim_penalty_dpdm", optim_penalty_dpdm);
  registerScalar("optim_penalty_energy", optim_penalty_energy);
  registerScalar("optim_penalty_variation", optim_penalty_variation);
  registerScalar("optim_regul_interpolate", optim_regul_interpolate);

  registerScalar("datadir", datadir);
  registerVectorOfVectors("output", output, nlevels.size(), {});
  registerScalar("output_frequency", output_frequency);
  registerScalar("optim_monitor_frequency", optim_monitor_frequency);
  registerScalar("run_type", runtype);
  registerScalar("usematfree", usematfree);
  registerScalar("linearsolver_type", linearsolver_type);
  registerScalar("linearsolver_maxiter", linearsolver_maxiter);
  registerScalar("timestepper", timestepper_type);
  setters["rand_seed"] = [this](const std::string& val) { setRandSeed(std::stoi(val)); };
  registerScalar("hamiltonian_file_Hsys", hamiltonian_file_Hsys);
  registerScalar("hamiltonian_file_Hc", hamiltonian_file_Hc);
}

Config::~Config(){}

void Config::validate() {
  if ((!hamiltonian_file_Hsys.empty() || !hamiltonian_file_Hc.empty()) && usematfree) {
    if (!quietmode) {
      std::string message = "# Warning: Matrix-free solver can not be used when Hamiltonian is read from file. Switching to sparse-matrix version.\n";
      logOutputToRank0(mpi_rank, message);
    }
    usematfree = false;
  }

  /* Sanity check for Schrodinger solver initial conditions */
  if (collapse_type == LindbladType::NONE){
    if (initial_condition_type == InitialConditionType::ENSEMBLE ||
        initial_condition_type == InitialConditionType::THREESTATES ||
        initial_condition_type == InitialConditionType::NPLUSONE ){
          printf("\n\n ERROR for initial condition setting: \n When running Schroedingers solver (collapse_type == NONE), the initial condition needs to be either 'pure' or 'from file' or 'diagonal' or 'basis'. Note that 'diagonal' and 'basis' in the Schroedinger case are the same (all unit vectors).\n\n");
          exit(1);
    } else if (initial_condition_type == InitialConditionType::BASIS) {
      // DIAGONAL and BASIS initial conditions in the Schroedinger case are the same. Overwrite it to DIAGONAL
      initial_condition_type = InitialConditionType::DIAGONAL;  
    }
  }

    // Validate initial conditions
    if (initial_condition_type == InitialConditionType::PURE) { 
      if (initial_condition_IDs.size() != nlevels.size()) {
        std::string message = "ERROR during pure-state initialization: List of IDs must contain"
          + std::to_string(nlevels.size()) + "elements!\n";
        logErrorToRank0(mpi_rank, message);
        exit(1);
      }
      for (size_t k=0; k < initial_condition_IDs.size(); k++) {
        if (initial_condition_IDs[k] > nlevels[k]-1){
          std::string message = "ERROR in config setting. The requested pure state initialization |"
            + std::to_string(initial_condition_IDs[k])
            + "> exceeds the number of allowed levels for that oscillator ("
            + std::to_string(nlevels[k]-1) + ").\n";
          logErrorToRank0(mpi_rank, message);
          exit(1);
        }
        assert(initial_condition_IDs[k] < nlevels[k]);
      }
    } else if (initial_condition_type == InitialConditionType::ENSEMBLE) {
      // Sanity check for the list in initcond_IDs!
      assert(initial_condition_IDs.size() >= 1); // at least one element 
      assert(initial_condition_IDs[initial_condition_IDs.size()-1] < nlevels.size()); // last element can't exceed total number of oscillators
      for (size_t i=0; i < initial_condition_IDs.size()-1; i++){ // list should be consecutive!
        if (initial_condition_IDs[i]+1 != initial_condition_IDs[i+1]) {
          logErrorToRank0(mpi_rank, "ERROR: List of oscillators for ensemble initialization should be consecutive!\n");
          exit(1);
        }
      }
    }
}

void Config::setNEssential(const std::string& nessential_str) {
  /* Get the number of essential levels per oscillator.
   * Default: same as number of levels */
   // TODO must be called after nlevels is set.
  nessential = nlevels;

  std::vector<int> read_nessential = convertFromString<std::vector<int>>(nessential_str);
  /* Overwrite if config option is given */
  if (read_nessential[0] > -1) {
    for (size_t iosc = 0; iosc<nlevels.size(); iosc++){
      if (iosc < read_nessential.size()) nessential[iosc] = read_nessential[iosc];
      else                               nessential[iosc] = read_nessential[read_nessential.size()-1];
      if (nessential[iosc] > nlevels[iosc]) nessential[iosc] = nlevels[iosc];
    }
  }
}

void Config::setInitialConditions(const std::string& init_cond_str, LindbladType collapse_type_) {
  std::vector<std::string> init_conds = convertFromString<std::vector<std::string>>(init_cond_str);

  if(init_conds.empty()) {
    logErrorToRank0(mpi_rank, "\n\n ERROR: Missing initial conditions.\n");
    exit(1);
  }

  initial_condition_type = convertFromString<InitialConditionType>(init_conds[0]);
  n_initial_conditions = 1;
  switch(initial_condition_type) {
    case InitialConditionType::FROMFILE:
    case InitialConditionType::PURE:
    case InitialConditionType::PERFORMANCE:
    case InitialConditionType::ENSEMBLE:
      n_initial_conditions = 1;
      break;
    case InitialConditionType::THREESTATES:
      n_initial_conditions = 3;
      break;
    case InitialConditionType::NPLUSONE: {
      // compute system dimension N
      n_initial_conditions = 1;
      for (size_t i=0; i<nlevels.size(); i++){
        n_initial_conditions *= nlevels[i];
      }
      n_initial_conditions +=1;
      break;
    }
    case InitialConditionType::DIAGONAL:
    case InitialConditionType::BASIS: {
      /* Compute n_initial_conditions = dim(subsystem defined by list of oscil IDs) */
      n_initial_conditions = 1;
      if (init_conds.size() < 2) {
        for (size_t j=0; j<nlevels.size(); j++) init_conds.push_back(std::to_string(j));
      }
      for (size_t i = 1; i<init_cond_str.size(); i++){
        size_t oscilID = std::stoi(init_conds[i]);
        if (oscilID < nessential.size()) n_initial_conditions *= nessential[oscilID];
      }
      if (initial_condition_type == InitialConditionType::BASIS) {
        // if Schroedinger solver: n_initial_conditions = N, do nothing.
        // else Lindblad solver: n_initial_conditions = N^2
        if (collapse_type_ != LindbladType::NONE) n_initial_conditions = (int) pow(n_initial_conditions,2.0);
      }
      break;
    }
    
    if (initial_condition_type == InitialConditionType::FROMFILE) {
      if (init_conds.size() != 2) {
        logErrorToRank0(mpi_rank, "Initial condition type 'file' requires a filename.");
        exit(1);
      }
      initial_condition_file = init_conds[1];
    } else {
      if (init_conds.size() < 2) {
        for (size_t j=0; j<nlevels.size(); j++)
          initial_condition_IDs.push_back(j); // Default: all oscillators
      } else {
        for (size_t i=1; i<init_conds.size(); i++) {
          initial_condition_IDs.push_back(std::stoi(init_conds[i])); // Use config option, if given.
        }
      }
    }
  }

  if (!quietmode) {
    logOutputToRank0(mpi_rank, "Number of initial conditions: " + std::to_string(n_initial_conditions));
  }
}

void Config::setRandSeed(int rand_seed_) {
  rand_seed = rand_seed_;
  if (rand_seed_ < 0){
    std::random_device rd;
    rand_seed = rd();  // random non-reproducable seed
  }
}

void Config::setApplyPiPulse(const std::string& value) {
  std::vector<std::string> pipulse_str = convertFromString<std::vector<std::string>>(value);

  if (pipulse_str.size() % 4 != 0) {
    std::string message = "Wrong pi-pulse configuration. Number of elements must be multiple of 4!\n";
    message += "apply_pipulse config option: <oscilID>, <tstart>, <tstop>, <amp>, <anotherOscilID>, <anotherTstart>, <anotherTstop>, <anotherAmp> ...\n";
    logErrorToRank0(mpi_rank, message);
    exit(1);
  }
  apply_pipulse.resize(nlevels.size());

  size_t k=0;
  while (k < pipulse_str.size()) {
    // Set pipulse for this oscillator
    size_t pipulse_id = std::stoi(pipulse_str[k+0]);
    PiPulse& pipulse = apply_pipulse[pipulse_id];
    pipulse.tstart.push_back(std::stod(pipulse_str[k+1]));
    pipulse.tstop.push_back(std::stod(pipulse_str[k+2]));
    pipulse.amp.push_back(std::stod(pipulse_str[k+3]));

    std::ostringstream message;
    message << "Applying PiPulse to oscillator " << pipulse_id << " in [" << pipulse.tstart.back() << ","
      << pipulse.tstop.back() << "]: |p+iq|=" << pipulse.amp.back();
    logOutputToRank0(mpi_rank, message.str());

    // Set zero control for all other oscillators during this pipulse
    for (size_t i=0; i<nlevels.size(); i++) {
      if (i != pipulse_id) {
        pipulse.tstart.push_back(stod(pipulse_str[k+1]));
        pipulse.tstop.push_back(stod(pipulse_str[k+2]));
        pipulse.amp.push_back(0.0);
      }
    }
    k+=4;
  }
}

void Config::setOptimTarget(const std::string& value) {
  std::vector<std::string> target_str = Config::split(value);

  if (target_str.empty()) {
    logErrorToRank0(mpi_rank, "No optimization target specified.");
    exit(1);
  }

  optim_target_type = convertFromString<TargetType>(target_str[0]);

  if (optim_target_type == TargetType::FROMFILE) {
    if (target_str.size() < 2) {
      logErrorToRank0(mpi_rank, "Target type 'file' requires a filename.");
      exit(1);
    }
    optim_target_file = target_str[1];
  } else if (optim_target_type == TargetType::GATE) {
    if (target_str.size() < 2) {
      logErrorToRank0(mpi_rank, "Target type 'gate' requires a gate name.");
      exit(1);
    }
    optim_target_gate_type = convertFromString<GateType>(target_str[1]);

    if (optim_target_gate_type == GateType::FILE) {
      if (target_str.size() < 3) {
        logErrorToRank0(mpi_rank, "Gate type 'file' requires a filename.");
        exit(1);
      }
      optim_target_gate_file = target_str[2];
    }
  } else if (optim_target_type == TargetType::PURE) {
    if (target_str.size() < 2) {
      logOutputToRank0(mpi_rank, "# Warning: You want to prepare a pure state, but didn't specify which one. Taking default: ground-state |0...0> \n");
      optim_target_purestate_levels = std::vector<size_t>(nlevels.size(), 0);
    } else {
      for (size_t i = 1; i < target_str.size(); i++) {
        optim_target_purestate_levels.push_back(convertFromString<size_t>(target_str[i]));
      }
      optim_target_purestate_levels.resize(nlevels.size(), nlevels.back());
      for (size_t i = 0; i < nlevels.size(); i++) {
        if (optim_target_purestate_levels[i] >= nlevels[i]) {
          logErrorToRank0(mpi_rank, "ERROR in config setting. The requested pure state target |" + std::to_string(optim_target_purestate_levels[i]) +
            "> exceeds the number of modeled levels for that oscillator (" + std::to_string(nlevels[i]) + ").\n");
          exit(1);
        }
      }
    }
  }
}

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

std::vector<std::string> Config::split(const std::string& str, char delimiter) {
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

Config Config::createFromFile(const std::string& filename, MPI_Comm comm, std::stringstream& logstream, bool quietmode) {
  Config config = Config(comm, logstream, quietmode);
  config.loadFromFile(filename);
  config.validate();
  return config;
}

void Config::loadFromFile(const std::string& filename) {
  std::string line;
  std::ifstream file;
  file.open(filename.c_str());
  if (!file.is_open()) {
    logErrorToRank0(mpi_rank, "Unable to read the file " + filename);
    abort();
  }

  while (getline(file, line)) {
    applyConfigLine(line);
  }
  file.close();
}

void Config::applyConfigLine(const std::string& line) {
  std::string trimmedLine = trimWhitespace(line);
  if (!isComment(trimmedLine)) {
    int pos = trimmedLine.find('=');
    std::string key = trimmedLine.substr(0, pos);
    std::string value = trimmedLine.substr(pos + 1);

    if (setters.count(key)) {
      try {
        setters[key](value);
      } catch (const std::exception& e) {
        logErrorToRank0(mpi_rank, "Error parsing '" + key + "': " + e.what());
      }
    } else {
      logErrorToRank0(mpi_rank, "Unknown option '" + key + "'");
    }
  }
}

void Config::printConfig() const {
  std::cout << "Configuration:\n";

  std::cout << "  nlevels = ";
  for (size_t i = 0; i < nlevels.size(); ++i) {
    std::cout << nlevels[i];
    if (i < nlevels.size() - 1) std::cout << ", ";
  }
  std::cout << "\n";

  std::cout << "  nessential = ";
  for (size_t i = 0; i < nessential.size(); ++i) {
    std::cout << nessential[i];
    if (i < nessential.size() - 1) std::cout << ", ";
  }
  std::cout << "\n";

  std::cout << "  ntime = " << ntime << "\n";
  std::cout << "  dt = " << dt << "\n";
  std::cout << "\n";
}
