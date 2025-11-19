#pragma once

#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <toml++/toml.hpp>
#include <vector>

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

// Custom exception for validation errors
class ValidationError : public std::runtime_error {
 public:
  ValidationError(const std::string& field, const std::string& message)
      : std::runtime_error("Validation error for field '" + field + "': " + message) {}
};

// Chainable validator builder
template <typename T>
class Validator {
 private:
  const toml::table& config;
  std::string key;
  bool is_required = false;
  std::optional<T> greater_than;
  std::optional<T> greater_than_equal;
  std::vector<std::function<bool(const T&)>> predicates;
  std::vector<std::string> error_messages;

 public:
  Validator(const toml::table& config_, const std::string& key_) : config(config_), key(key_) {}

  Validator& required() {
    is_required = true;
    return *this;
  }

  Validator& greaterThan(T greater_than_) {
    greater_than = greater_than_;
    return *this;
  }

  Validator& greaterThanEqual(T greater_than_equal_) {
    greater_than_equal = greater_than_equal_;
    return *this;
  }

  Validator& positive() {
    greaterThan(T{0});
    return *this;
  }

  Validator& custom(std::function<bool(const T&)> predicate_, const std::string& error_msg_) {
    predicates.push_back(predicate_);
    error_messages.push_back(error_msg_);
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

    for (size_t i = 0; i < predicates.size(); ++i) {
      if (!predicates[i](result)) {
        throw ValidationError(key, error_messages[i]);
      }
    }

    return result;
  }

 public:
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

  T valueOr(T default_value_) {
    auto val = extractValue();
    if (!val) return default_value_; // Key doesn't exist - use default

    return validateValue(*val); // Key exists - validate it (will throw on wrong type)
  }
};

// Vector validator specialization
template <typename T>
class VectorValidator {
 private:
  const toml::table& config;
  std::string key;
  bool is_required = false;
  std::optional<size_t> min_length;
  std::optional<size_t> max_length;
  std::optional<T> min_value;
  bool is_positive = false;

 public:
  VectorValidator(const toml::table& config_, const std::string& key_) : config(config_), key(key_) {}

  VectorValidator& required() {
    is_required = true;
    return *this;
  }

  VectorValidator& minLength(size_t min_len_) {
    min_length = min_len_;
    return *this;
  }

  VectorValidator& maxLength(size_t max_len_) {
    max_length = max_len_;
    return *this;
  }

  VectorValidator& positive() {
    is_positive = true;
    return *this;
  }

  VectorValidator& minValue(T min_val_) {
    min_value = min_val_;
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

      if (min_value && element < *min_value) {
        std::ostringstream oss;
        oss << "element [" << i << "] must be >= " << *min_value << ", got " << element;
        throw ValidationError(key, oss.str());
      }
    }

    return result;
  }

 public:
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

  std::vector<T> valueOr(const std::vector<T>& default_value_) {
    auto val = extractVector();
    if (!val) return default_value_; // Key doesn't exist - use default

    return validateVector(*val); // Key exists - validate it (will throw on wrong type)
  }
};

// Helper functions to start validation chains
template <typename T>
Validator<T> field(const toml::table& config_, const std::string& key_) {
  return Validator<T>(config_, key_);
}

template <typename T>
VectorValidator<T> vectorField(const toml::table& config_, const std::string& key_) {
  return VectorValidator<T>(config_, key_);
}

} // namespace validators
