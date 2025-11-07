#pragma once

#include <toml++/toml.hpp>

#include <vector>
#include <string>
#include <functional>
#include <sstream>
#include <stdexcept>

// TODO make naming camel case
namespace validators {

// Helper to get readable type names for error messages
template<typename T>
std::string get_type_name() {
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
template<typename T>
class Validator {
private:
  const toml::table& config_;
  std::string key_;
  std::optional<T> value_;
  bool required_ = false;
  std::optional<T> greaterThan_;
  std::optional<T> greaterThanEqual_;
  std::vector<std::function<bool(const T&)>> predicates_;
  std::vector<std::string> error_messages_;

public:
  Validator(const toml::table& config, const std::string& key)
    : config_(config), key_(key) {}

  Validator& required() {
    required_ = true;
    return *this;
  }

  Validator& greater_than(T greaterThan) {
    greaterThan_ = greaterThan;
    return *this;
  }

  Validator& greater_than_equal(T greaterThanEqual) {
    greaterThanEqual_ = greaterThanEqual;
    return *this;
  }

  Validator& positive() {
    greater_than(T{0});
    return *this;
  }

  Validator& custom(std::function<bool(const T&)> predicate, const std::string& error_msg) {
    predicates_.push_back(predicate);
    error_messages_.push_back(error_msg);
    return *this;
  }

private:
  std::optional<T> extract_value() {
    // If key doesn't exist, return nullopt
    if (!config_.contains(key_)) {
      return std::nullopt;
    }

    // Key exists, try to extract value with type checking
    auto val = config_[key_].template value<T>();
    if (!val) {
      // Key exists but wrong type - always an error
      throw ValidationError(key_, "wrong type (expected " + get_type_name<T>() + ")");
    }

    return val;
  }

  T validate_value(T result) {
    if (greaterThan_ && result <= *greaterThan_) {
      std::ostringstream oss;
      oss << "must be > " << *greaterThan_ << ", got " << result;
      throw ValidationError(key_, oss.str());
    }

    if (greaterThanEqual_ && result < *greaterThanEqual_) {
      std::ostringstream oss;
      oss << "must be >= " << *greaterThanEqual_ << ", got " << result;
      throw ValidationError(key_, oss.str());
    }

    for (size_t i = 0; i < predicates_.size(); ++i) {
      if (!predicates_[i](result)) {
        throw ValidationError(key_, error_messages_[i]);
      }
    }

    return result;
  }

public:
// TODO value and value_or is more like optionals
  T get() {
    auto val = extract_value();

    if (!val && required_) {
      throw ValidationError(key_, "field is required");
    }
    if (!val) {
      throw ValidationError(key_, "field not found");
    }

    return validate_value(*val);
  }

  T get_or(T default_value) {
    auto val = extract_value();
    if (!val) return default_value;  // Key doesn't exist - use default

    return validate_value(*val);  // Key exists - validate it (will throw on wrong type)
  }
};

// Vector validator specialization
template<typename T>
class VectorValidator {
private:
  const toml::table& config_;
  std::string key_;
  bool required_ = false;
  std::optional<size_t> min_length_;
  std::optional<size_t> max_length_;
  std::optional<T> min_value_;
  bool positive_ = false;

public:
  VectorValidator(const toml::table& config, const std::string& key)
    : config_(config), key_(key) {}

  VectorValidator& required() {
    required_ = true;
    return *this;
  }

  VectorValidator& min_length(size_t min_len) {
    min_length_ = min_len;
    return *this;
  }

  VectorValidator& max_length(size_t max_len) {
    max_length_ = max_len;
    return *this;
  }

  VectorValidator& positive() {
    positive_ = true;
    return *this;
  }

  VectorValidator& min_value(T min_val) {
    min_value_ = min_val;
    return *this;
  }

private:
  std::optional<std::vector<T>> extract_vector() {
    // If key doesn't exist, return nullopt
    if (!config_.contains(key_)) {
      return std::nullopt;
    }

    // Key exists, check if it's an array
    auto* arr = config_[key_].as_array();
    if (!arr) {
      // Key exists but wrong type - always an error
      throw ValidationError(key_, "wrong type (expected array)");
    }

    // Extract and validate array elements
    std::vector<T> result;
    for (size_t i = 0; i < arr->size(); ++i) {
      auto val = arr->at(i).template value<T>();
      if (!val) {
        std::ostringstream oss;
        oss << "element [" << i << "] wrong type (expected " << get_type_name<T>() << ")";
        throw ValidationError(key_, oss.str());
      }
      result.push_back(*val);
    }

    return result;
  }

  std::vector<T> validate_vector(std::vector<T> result) {
    if (min_length_ && result.size() < *min_length_) {
      std::ostringstream oss;
      oss << "must have at least " << *min_length_ << " elements, got " << result.size();
      throw ValidationError(key_, oss.str());
    }

    if (max_length_ && result.size() > *max_length_) {
      std::ostringstream oss;
      oss << "must have at most " << *max_length_ << " elements, got " << result.size();
      throw ValidationError(key_, oss.str());
    }

    for (size_t i = 0; i < result.size(); ++i) {
      T& element = result[i];

      if (positive_ && element <= T{0}) {
        std::ostringstream oss;
        oss << "element [" << i << "] must be positive, got " << element;
        throw ValidationError(key_, oss.str());
      }

      if (min_value_ && element < *min_value_) {
        std::ostringstream oss;
        oss << "element [" << i << "] must be >= " << *min_value_ << ", got " << element;
        throw ValidationError(key_, oss.str());
      }
    }

    return result;
  }

public:
  std::vector<T> get() {
    auto val = extract_vector();

    if (!val && required_) {
      throw ValidationError(key_, "field is required");
    }
    if (!val) {
      throw ValidationError(key_, "field not found");
    }

    return validate_vector(*val);
  }

  std::vector<T> get_or(const std::vector<T>& default_value) {
    auto val = extract_vector();
    if (!val) return default_value;  // Key doesn't exist - use default

    return validate_vector(*val);  // Key exists - validate it (will throw on wrong type)
  }
};

// Helper functions to start validation chains
template<typename T>
Validator<T> field(const toml::table& config, const std::string& key) {
  return Validator<T>(config, key);
}

template<typename T>
VectorValidator<T> vector_field(const toml::table& config, const std::string& key) {
  return VectorValidator<T>(config, key);
}

} // namespace validators
