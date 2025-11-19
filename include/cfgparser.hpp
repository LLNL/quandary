#pragma once

#include <petsc.h>

#include <cstddef>
#include <set>
#include <type_traits>
#include <vector>

#include "config_types.hpp"
#include "defs.hpp"
#include "mpi_logger.hpp"
#include "util.hpp"

namespace {
// Vector detection
template <typename T>
struct is_vector : std::false_type {};
template <typename T>
struct is_vector<std::vector<T>> : std::true_type {};
template <typename T>
inline constexpr bool is_vector_v = is_vector<T>::value;
} // namespace

/**
 * @brief Configuration parser that converts raw config to validated Config
 *
 * Handles the logic for applying defaults, parsing mixed-type strings,
 * and validating configuration parameters.
 */
class CfgParser {
 private:
  std::unordered_map<std::string, std::function<void(const std::string&)>> setters; ///< Setters from config string
  std::unordered_map<std::string, std::function<void(int, const std::string&)>>
      indexed_setters; ///< Setters for indexed config strings

  const MPILogger& logger;

  // Configuration settings storage
  ParsedConfigData settings; ///< All configuration settings in one place
  std::optional<bool> optim_regul_interpolate; ///< Deprecated version of optim_regul_tik0

 public:
  CfgParser(const MPILogger& logger);
  ParsedConfigData parseFile(const std::string& filename);
  ParsedConfigData parseString(const std::string& config_content);

 private:
  std::vector<std::string> split(const std::string& str, char delimiter = ',');
  void applyConfigLine(const std::string& line);
  bool handleIndexedSetting(const std::string& key, const std::string& value);

  template <typename StreamType>
  void loadFromStream(StreamType& stream) {
    std::string line;
    while (getline(stream, line)) {
      applyConfigLine(line);
    }
  }

  std::vector<std::vector<double>> convertIndexedToVectorVector(const std::map<int, std::vector<double>>& indexed_map,
                                                                size_t num_oscillators);
  std::vector<std::vector<OutputType>> convertIndexedToOutputVector(
      const std::map<int, std::vector<OutputType>>& indexed_map, size_t num_oscillators);

  template <typename T>
  void registerConfig(const std::string& key, std::optional<T>& member) {
    setters[key] = [this, &member](const std::string& value) { member = convertFromString<T>(value); };
  }

  template <typename T>
  void registerIndexedConfig(const std::string& base_key, std::optional<std::map<int, T>>& storage) {
    indexed_setters[base_key] = [this, &storage](int index, const std::string& value) {
      if (!storage.has_value()) {
        storage = std::map<int, T>{};
      }
      (*storage)[index] = convertFromString<T>(value);
    };
  }

  template <typename T>
  T convertFromString(const std::string& str) {
    if constexpr (std::is_same_v<T, std::string>) {
      return str;
    } else if constexpr (std::is_same_v<T, bool>) {
      const std::set<std::string> trueValues = {"true", "yes", "1"};
      std::string lowerStr = str;
      std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(), ::tolower);
      return trueValues.find(lowerStr) != trueValues.end();
    } else if constexpr (std::is_same_v<T, int>) {
      return std::stoi(str);
    } else if constexpr (std::is_same_v<T, size_t>) {
      return static_cast<size_t>(std::stoul(str));
    } else if constexpr (std::is_same_v<T, double>) {
      return std::stod(str);
    } else if constexpr (is_vector_v<T>) {
      return parseVector<T>(str);
    } else {
      static_assert(false, "Unsupported type for convertFromString");
    }
  }

  template <typename VectorType>
  VectorType parseVector(const std::string& str) {
    using ElementType = typename VectorType::value_type;
    VectorType vec;
    auto parts = split(str);
    vec.reserve(parts.size());
    for (const auto& part : parts) {
      vec.push_back(convertFromString<ElementType>(part));
    }
    return vec;
  }

  // Enum converter specializations
  template <>
  RunType convertFromString<RunType>(const std::string& str);

  template <>
  LindbladType convertFromString<LindbladType>(const std::string& str);

  template <>
  LinearSolverType convertFromString<LinearSolverType>(const std::string& str);

  template <>
  TimeStepperType convertFromString<TimeStepperType>(const std::string& str);

  template <>
  TargetType convertFromString<TargetType>(const std::string& str);

  template <>
  InitialConditionType convertFromString<InitialConditionType>(const std::string& str);

  template <>
  GateType convertFromString<GateType>(const std::string& str);

  template <>
  OutputType convertFromString<OutputType>(const std::string& str);

  template <>
  ObjectiveType convertFromString<ObjectiveType>(const std::string& str);

  template <>
  ControlType convertFromString<ControlType>(const std::string& str);

  template <>
  ControlSegmentInitType convertFromString<ControlSegmentInitType>(const std::string& str);

  // Struct converters
  template <>
  InitialConditionData convertFromString<InitialConditionData>(const std::string& str);

  template <>
  OptimTargetData convertFromString<OptimTargetData>(const std::string& str);

  template <>
  std::vector<PiPulseData> convertFromString<std::vector<PiPulseData>>(const std::string& str);

  template <>
  std::vector<ControlSegmentData> convertFromString<std::vector<ControlSegmentData>>(const std::string& str);

  template <>
  std::vector<ControlInitializationData> convertFromString<std::vector<ControlInitializationData>>(
      const std::string& str);
};
