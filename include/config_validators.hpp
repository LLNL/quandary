#pragma once

#include <sstream>
#include <stdexcept>
#include <string>
#include <toml++/toml.hpp>
#include <vector>

/**
 * @brief Validation utilities for TOML configuration parsing.
 *
 * Provides chainable validators for type-safe TOML field extraction with
 * built-in validation (required, positive, ranges, custom predicates).
 */
namespace validators {

// Helper to get readable type names for error messages
template <typename T>
std::string getTypeName() {
  if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int64_t> || std::is_same_v<T, size_t>) {
    return "integer";
  } else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, float>) {
    return "number";
  } else if constexpr (std::is_same_v<T, std::string>) {
    return "string";
  } else if constexpr (std::is_same_v<T, bool>) {
    return "boolean";
  } else {
    return "unknown type";
  }
}

/**
 * @brief Exception thrown when configuration validation fails.
 */
class ValidationError : public std::runtime_error {
 public:
  ValidationError(const std::string& field, const std::string& message)
      : std::runtime_error("Validation error for field '" + field + "': " + message) {}
};

/**
 * @brief Chainable validator for scalar TOML fields.
 *
 * Provides type-safe extraction and validation of scalar values from TOML configuration.
 * Supports chaining multiple validation rules (required, range checks, etc.).
 *
 * Example usage:
 * @code
 * int value = validators::field<int>(config, "port")
 *               .required()
 *               .greaterThan(0)
 *               .value();
 * @endcode
 *
 * @tparam T Type of field to validate (int, double, string, bool, etc.)
 */
template <typename T>
class Validator {
 private:
  const toml::table& config;
  std::string key;
  bool is_required = false;
  std::optional<T> greater_than;
  std::optional<T> greater_than_equal;

 public:
  Validator(const toml::table& config_, const std::string& key_) : config(config_), key(key_) {}

  /**
   * @brief Marks field as required (will error if missing).
   *
   * @return Reference to this validator for chaining
   */
  Validator& required() {
    is_required = true;
    return *this;
  }

  /**
   * @brief Requires value to be strictly greater than threshold.
   *
   * @param greater_than_ Threshold value (exclusive)
   * @return Reference to this validator for chaining
   */
  Validator& greaterThan(T greater_than_) {
    greater_than = greater_than_;
    return *this;
  }

  /**
   * @brief Requires value to be greater than or equal to threshold.
   *
   * @param greater_than_equal_ Threshold value (inclusive)
   * @return Reference to this validator for chaining
   */
  Validator& greaterThanEqual(T greater_than_equal_) {
    greater_than_equal = greater_than_equal_;
    return *this;
  }

  /**
   * @brief Requires value to be strictly positive (> 0).
   *
   * @return Reference to this validator for chaining
   */
  Validator& positive() {
    greaterThan(T{0});
    return *this;
  }

 private:
  std::optional<T> extractValue() {
    // If key doesn't exist, return nullopt
    if (!config.contains(key)) {
      return std::nullopt;
    }

    // Key exists, try to extract value with type checking
    auto val = config[key].template value<T>();
    if (!val) {
      // Key exists but wrong type - always an error
      throw ValidationError(key, "wrong type (expected " + getTypeName<T>() + ")");
    }

    return val;
  }

  T validateValue(T result) {
    if (greater_than && result <= *greater_than) {
      std::ostringstream oss;
      oss << "must be > " << *greater_than << ", got " << result;
      throw ValidationError(key, oss.str());
    }

    if (greater_than_equal && result < *greater_than_equal) {
      std::ostringstream oss;
      oss << "must be >= " << *greater_than_equal << ", got " << result;
      throw ValidationError(key, oss.str());
    }

    return result;
  }

 public:
  /**
   * @brief Extracts and validates the field value.
   *
   * Throws ValidationError if the field is required and missing,
   * or if any validation rules fail.
   *
   * @return The validated field value
   * @throws ValidationError If validation fails
   */
  T value() {
    auto val = extractValue();

    if (!val && is_required) {
      throw ValidationError(key, "field is required");
    }
    if (!val) {
      throw ValidationError(key, "field not found");
    }

    return validateValue(*val);
  }

  /**
   * @brief Extracts field value or returns default if missing.
   *
   * If the field is present, validates it (may throw). If absent,
   * returns the provided default without validation.
   *
   * @param default_value_ Default value to use if field is missing
   * @return The field value or default
   * @throws ValidationError If field exists but validation fails
   */
  T valueOr(T default_value_) {
    auto val = extractValue();
    if (!val) return default_value_; // Key doesn't exist - use default

    return validateValue(*val); // Key exists - validate it (will throw on wrong type)
  }
};

/**
 * @brief Chainable validator for vector/array TOML fields.
 *
 * Provides type-safe extraction and validation of array values from TOML configuration.
 * Supports validation of both the array itself (length) and its elements (values).
 *
 * Example usage:
 * @code
 * auto values = validators::vectorField<double>(config, "frequencies")
 *                 .required()
 *                 .minLength(1)
 *                 .positive()
 *                 .value();
 * @endcode
 *
 * @tparam T Element type of the vector (int, double, string, etc.)
 */
template <typename T>
class VectorValidator {
 private:
  const toml::table& config;
  std::string key;
  bool is_required = false;
  std::optional<size_t> min_length;
  std::optional<size_t> max_length;
  bool is_positive = false;

 public:
  VectorValidator(const toml::table& config_, const std::string& key_) : config(config_), key(key_) {}

  /**
   * @brief Marks field as required (will error if missing).
   *
   * @return Reference to this validator for chaining
   */
  VectorValidator& required() {
    is_required = true;
    return *this;
  }

  /**
   * @brief Requires minimum vector length.
   *
   * @param min_len_ Minimum number of elements (inclusive)
   * @return Reference to this validator for chaining
   */
  VectorValidator& minLength(size_t min_len_) {
    min_length = min_len_;
    return *this;
  }

  /**
   * @brief Requires all vector elements to be strictly positive (> 0).
   *
   * @return Reference to this validator for chaining
   */
  VectorValidator& positive() {
    is_positive = true;
    return *this;
  }

 private:
  std::optional<std::vector<T>> extractVector() {
    // If key doesn't exist, return nullopt
    if (!config.contains(key)) {
      return std::nullopt;
    }

    // Key exists, check if it's an array
    auto* arr = config[key].as_array();
    if (!arr) {
      // Key exists but wrong type - always an error
      throw ValidationError(key, "wrong type (expected array)");
    }

    // Extract and validate array elements
    std::vector<T> result;
    for (size_t i = 0; i < arr->size(); ++i) {
      auto val = arr->at(i).template value<T>();
      if (!val) {
        std::ostringstream oss;
        oss << "element [" << i << "] wrong type (expected " << getTypeName<T>() << ")";
        throw ValidationError(key, oss.str());
      }
      result.push_back(*val);
    }

    return result;
  }

  std::vector<T> validateVector(std::vector<T> result) {
    if (min_length && result.size() < *min_length) {
      std::ostringstream oss;
      oss << "must have at least " << *min_length << " elements, got " << result.size();
      throw ValidationError(key, oss.str());
    }

    if (max_length && result.size() > *max_length) {
      std::ostringstream oss;
      oss << "must have at most " << *max_length << " elements, got " << result.size();
      throw ValidationError(key, oss.str());
    }

    for (size_t i = 0; i < result.size(); ++i) {
      T& element = result[i];

      if (is_positive && element <= T{0}) {
        std::ostringstream oss;
        oss << "element [" << i << "] must be positive, got " << element;
        throw ValidationError(key, oss.str());
      }
    }

    return result;
  }

 public:
  /**
   * @brief Extracts and validates the vector field.
   *
   * @return The validated vector
   * @throws ValidationError If validation fails
   */
  std::vector<T> value() {
    auto val = extractVector();

    if (!val && is_required) {
      throw ValidationError(key, "field is required");
    }
    if (!val) {
      throw ValidationError(key, "field not found");
    }

    return validateVector(*val);
  }

  /**
   * @brief Extracts vector or returns default if missing.
   *
   * @param default_value_ Default vector to use if field is missing
   * @return The field vector or default
   * @throws ValidationError If field exists but validation fails
   */
  std::vector<T> valueOr(const std::vector<T>& default_value_) {
    auto val = extractVector();
    if (!val) return default_value_; // Key doesn't exist - use default

    return validateVector(*val); // Key exists - validate it (will throw on wrong type)
  }
};

/**
 * @brief Creates a scalar field validator.
 *
 * Helper function to start a validation chain for scalar fields.
 *
 * @tparam T Type of field to validate
 * @param config_ TOML table containing the field
 * @param key_ Name of the field to validate
 * @return A Validator for chaining validation rules
 */
template <typename T>
Validator<T> field(const toml::table& config_, const std::string& key_) {
  return Validator<T>(config_, key_);
}

/**
 * @brief Creates a vector field validator.
 *
 * Helper function to start a validation chain for array/vector fields.
 *
 * @tparam T Element type of the vector
 * @param config_ TOML table containing the field
 * @param key_ Name of the field to validate
 * @return A VectorValidator for chaining validation rules
 */
template <typename T>
VectorValidator<T> vectorField(const toml::table& config_, const std::string& key_) {
  return VectorValidator<T>(config_, key_);
}

/**
 * @brief Extracts an optional vector from a TOML node.
 *
 * Helper for extracting vectors when the validator API doesn't fit well
 * (e.g., nested structures or conditional parsing). If the node is not
 * an array or contains type mismatches, returns nullopt.
 *
 * @tparam T Element type of the vector
 * @param node TOML node that may contain an array
 * @return Vector if node is a valid array with matching types, nullopt otherwise
 */
template <typename T>
std::optional<std::vector<T>> getOptionalVector(const toml::node_view<toml::node>& node) {
  auto* arr = node.as_array();
  if (!arr) return std::nullopt;

  std::vector<T> result;
  for (size_t i = 0; i < arr->size(); ++i) {
    auto val = arr->at(i).template value<T>();
    if (!val) return std::nullopt; // Type mismatch in array element
    result.push_back(*val);
  }

  return result;
}

/**
 * @brief Extracts a required table from a TOML configuration.
 *
 * Validates that the specified key exists and contains a table.
 *
 * @param config Parent TOML table
 * @param key Name of the table field
 * @param logger Logger for error reporting
 * @return Reference to the table
 * @throws Exits via logger if table is missing or wrong type
 */
inline const toml::table& getRequiredTable(const toml::table& config, const std::string& key) {
  if (!config.contains(key)) {
    throw ValidationError(key, "table is required");
  }

  auto* table = config[key].as_table();
  if (!table) {
    throw ValidationError(key, "must be a table");
  }

  return *table;
}

} // namespace validators
