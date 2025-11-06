#pragma once

#include "defs.hpp"
#include <map>
#include <optional>
#include <string>
#include <vector>

/**
 * @brief Common configuration structures shared between CfgParser and Config
 *
 * These structs represent parsed configuration data in an intermediate form,
 * used by CfgParser for parsing and by Config for initialization.
 */

struct InitialConditionConfig {
  InitialConditionType type;  ///< Type of initial condition
  std::vector<size_t> params;    ///< Additional parameters
  std::optional<std::string> filename;  ///< Filename (if type is FILE)
};

struct OptimTargetConfig {
  std::string target_type;               ///< Target type (gate, pure, file)
  std::optional<std::string> gate_type;    ///< Gate type (for "gate, cnot")
  std::optional<std::string> filename;  ///< Filename (for "file, path.dat")
  std::optional<std::string> gate_file; ///< Gate filename (for "gate, file, path.dat")
  std::optional<std::vector<size_t>> levels;           ///< Pure state levels (for "pure, 0, 1")
};

struct PiPulseConfig {
  size_t oscil_id;  ///< Oscillator ID
  double tstart;    ///< Start time
  double tstop;     ///< Stop time
  double amp;       ///< Amplitude
};

struct ControlSegmentConfig {
  ControlType control_type;        ///< Type of control segment
  std::vector<double> parameters;  ///< Parameters for control segment
};

struct ControlInitializationConfig {
  ControlInitializationType init_type;  ///< Type of initialization
  std::optional<double> amplitude;      ///< Initial amplitude
  std::optional<double> phase;          ///< Initial phase (optional)
  std::optional<std::string> filename;  ///< Filename (for file init)
};

/**
 * @brief Configuration settings passed to Config constructor.
 *
 * Contains all optional configuration parameters that can be provided
 * to configure a Config object. Used by CfgParser to pass settings.
 */
struct ConfigSettings {
  // General parameters
  std::optional<std::vector<size_t>> nlevels;
  std::optional<std::vector<size_t>> nessential;
  std::optional<size_t> ntime;
  std::optional<double> dt;
  std::optional<std::vector<double>> transfreq;
  std::optional<std::vector<double>> selfkerr;
  std::optional<std::vector<double>> crosskerr;
  std::optional<std::vector<double>> Jkl;
  std::optional<std::vector<double>> rotfreq;
  std::optional<LindbladType> collapse_type;
  std::optional<std::vector<double>> decay_time;
  std::optional<std::vector<double>> dephase_time;
  std::optional<InitialConditionConfig> initialcondition;
  std::optional<std::vector<PiPulseConfig>> apply_pipulse;
  std::optional<std::string> hamiltonian_file_Hsys;
  std::optional<std::string> hamiltonian_file_Hc;

  // Control and optimization parameters
  std::optional<std::map<int, std::vector<ControlSegmentConfig>>> indexed_control_segments;
  std::optional<bool> control_enforceBC;
  std::optional<std::map<int, std::vector<ControlInitializationConfig>>> indexed_control_init;
  std::optional<std::map<int, std::vector<double>>> indexed_control_bounds;
  std::optional<std::map<int, std::vector<double>>> indexed_carrier_frequencies;
  std::optional<OptimTargetConfig> optim_target;
  std::optional<std::vector<double>> gate_rot_freq;
  std::optional<ObjectiveType> optim_objective;
  std::optional<std::vector<double>> optim_weights;
  std::optional<double> optim_atol;
  std::optional<double> optim_rtol;
  std::optional<double> optim_ftol;
  std::optional<double> optim_inftol;
  std::optional<size_t> optim_maxiter;
  std::optional<double> optim_regul;
  std::optional<double> optim_penalty;
  std::optional<double> optim_penalty_param;
  std::optional<double> optim_penalty_dpdm;
  std::optional<double> optim_penalty_energy;
  std::optional<double> optim_penalty_variation;
  std::optional<bool> optim_regul_tik0;
  std::optional<bool> optim_regul_interpolate; // deprecated

  // Output parameters
  std::optional<std::string> datadir;
  std::optional<std::map<int, std::vector<OutputType>>> indexed_output;
  std::optional<size_t> output_frequency;
  std::optional<size_t> optim_monitor_frequency;
  std::optional<RunType> runtype;
  std::optional<bool> usematfree;
  std::optional<LinearSolverType> linearsolver_type;
  std::optional<size_t> linearsolver_maxiter;
  std::optional<TimeStepperType> timestepper_type;
  std::optional<int> rand_seed;
};
