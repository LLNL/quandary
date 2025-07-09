#include "config.hpp"
#include "util.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <random>


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
  // registerVector("transfreq", transfreq);
  // registerVector("selfkerr", selfkerr);
  // registerVector("crosskerr", crosskerr);
  // registerVector("Jkl", Jkl);
  // registerVector("rotfreq", rotfreq);
  // registerScalar("collapse_type", collapse_type);
  // registerVector("decay_time", decay_time);
  // registerVector("dephase_time", dephase_time);
  // registerScalar("initialcondition", initialcondition);
  // registerVector("apply_pipulse", apply_pipulse);

  // registerVector("control_segments", control_segments);
  // registerScalar("control_enforceBC", control_enforceBC);
  // registerVector("control_initialization", control_initialization);
  // registerVector("control_bounds", control_bounds);
  // registerVector("carrier_frequencies", carrier_frequencies);
  // registerVector("optim_target", optim_target);
  // registerVector("gate_rot_freq", gate_rot_freq);
  // registerScalar("optim_objective", optim_objective);
  // registerVector("optim_weights", optim_weights);
  // registerScalar("optim_atol", optim_atol);
  // registerScalar("optim_rtol", optim_rtol);
  // registerScalar("optim_ftol", optim_ftol);
  // registerScalar("optim_inftol", optim_inftol);
  // registerScalar("optim_maxiter", optim_maxiter);
  // registerScalar("optim_regul", optim_regul);
  // registerScalar("optim_penalty", optim_penalty);
  // registerScalar("optim_penalty_param", optim_penalty_param);
  // registerScalar("optim_penalty_dpdm", optim_penalty_dpdm);
  // registerScalar("optim_penalty_energy", optim_penalty_energy);
  // registerScalar("optim_penalty_variation", optim_penalty_variation);
  // registerScalar("optim_regul_tik0", optim_regul_tik0);

  // registerScalar("datadir", datadir);
  // registerVector("output", output);
  // registerScalar("output_frequency", output_frequency);
  // registerScalar("optim_monitor_frequency", optim_monitor_frequency);
  // registerScalar("runtype", runtype);
  // registerScalar("usematfree", usematfree);
  // registerScalar("linearsolver_type", linearsolver_type);
  // registerScalar("linearsolver_maxiter", linearsolver_maxiter);
  // registerScalar("timestepper", timestepper);
  // registerScalar("rand_seed", rand_seed);
  // setters["rand_seed"] = [this](const std::string& val) { setRandSeed(val); };
}

Config::~Config(){}

void Config::setNLevels(const std::vector<int>& nlevels_) {
  nlevels = nlevels_;
}

void Config::setRandSeed(int rand_seed_) {
  rand_seed = rand_seed_;
  if (rand_seed_ < 0){
    std::random_device rd;
    rand_seed = rd();  // random non-reproducable seed
  }
}

std::string trimWhitespace(std::string s) {
  s.erase(std::remove_if(s.begin(), s.end(),
    [](unsigned char c) { return std::isspace(c); }), s.end());
  return s;
}

bool isComment(const std::string& line) {
  return line.size() > 0 && (line[0] == '#' || line[0] == '/');
}

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
