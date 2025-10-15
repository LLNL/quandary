#pragma once

#include "defs.hpp"
#include <optional>
#include <string>
#include <vector>

/**
 * @brief Common configuration structures shared between ConfigBuilder and Config
 *
 * These structs represent parsed configuration data in an intermediate form,
 * used by ConfigBuilder for parsing and by Config for initialization.
 */

struct InitialConditionConfig {
  InitialConditionType type;  ///< Type of initial condition
  std::vector<int> params;    ///< Additional integer parameters
  std::optional<std::string> filename;  ///< Filename (if type is FILE)
};

struct OptimTargetConfig {
  TargetType target_type;               ///< Target type (gate, pure, file)
  std::optional<GateType> gate_type;    ///< Gate type (for "gate, cnot")
  std::optional<std::string> filename;  ///< Filename (for "file, path.dat")
  std::optional<std::string> gate_file; ///< Gate filename (for "gate, file, path.dat")
  std::vector<int> levels;              ///< Pure state levels (for "pure, 0, 1")
};

struct PiPulseConfig {
  int oscil_id;     ///< Oscillator ID
  double tstart;    ///< Start time
  double tstop;     ///< Stop time
  double amp;       ///< Amplitude
};

struct ControlSegmentConfig {
  ControlType control_type;             ///< Type of control segment
  std::optional<int> num_basis_functions; ///< Number of basis functions
  std::optional<double> tstart;         ///< Start time (optional)
  std::optional<double> tstop;          ///< Stop time (optional)
  std::optional<double> scaling;        ///< Scaling factor (for spline_amplitude)
  std::optional<double> amplitude_1;    ///< Amplitude (for step)
  std::optional<double> amplitude_2;    ///< Stop time (for step)
};

struct ControlInitializationConfig {
  ControlInitializationType init_type;  ///< Type of initialization
  std::optional<double> amplitude;      ///< Initial amplitude
  std::optional<double> phase;          ///< Initial phase (optional)
  std::optional<std::string> filename;  ///< Filename (for file init)
};
