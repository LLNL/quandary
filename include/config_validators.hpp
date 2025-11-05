#pragma once

#include <toml++/toml.hpp>

#include <vector>
#include <string>
#include <functional>
#include <sstream>
#include <stdexcept>

namespace validators {

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
  std::optional<T> min_;
  std::vector<std::function<bool(const T&)>> predicates_;
  std::vector<std::string> error_messages_;

public:
  Validator(const toml::table& config, const std::string& key)
    : config_(config), key_(key) {}

  Validator& required() {
    required_ = true;
    return *this;
  }

  Validator& min(T min_val) {
    min_ = min_val;
    return *this;
  }

  Validator& positive() {
    return min(T{1});
  }

  Validator& custom(std::function<bool(const T&)> predicate, const std::string& error_msg) {
    predicates_.push_back(predicate);
    error_messages_.push_back(error_msg);
    return *this;
  }

private:
  T validate_value(T result) {
    if (min_ && result < *min_) {
      std::ostringstream oss;
      oss << "must be >= " << *min_ << ", got " << result;
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
  T get() {
    auto val = config_[key_].template value<T>();

    if (!val && required_) {
      throw ValidationError(key_, "field is required");
    }
    if (!val) {
      throw ValidationError(key_, "field not found or wrong type");
    }

    return validate_value(*val);
  }

  T get_or(T default_value) {
    auto val = config_[key_].template value<T>();
    if (!val) return default_value;

    return validate_value(*val);
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

  std::vector<T> get() {
    auto* arr = config_[key_].as_array();

    if (!arr && required_) {
      throw ValidationError(key_, "field is required");
    }
    if (!arr) {
      throw ValidationError(key_, "must be an array");
    }

    if (min_length_ && arr->size() < *min_length_) {
      std::ostringstream oss;
      oss << "must have at least " << *min_length_ << " elements, got " << arr->size();
      throw ValidationError(key_, oss.str());
    }

    if (max_length_ && arr->size() > *max_length_) {
      std::ostringstream oss;
      oss << "must have at most " << *max_length_ << " elements, got " << arr->size();
      throw ValidationError(key_, oss.str());
    }

    std::vector<T> result;
    for (size_t i = 0; i < arr->size(); ++i) {
      auto val = arr->at(i).template value<T>();
      if (!val) {
        std::ostringstream oss;
        oss << "element [" << i << "] has wrong type";
        throw ValidationError(key_, oss.str());
      }

      T element = *val;

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


      result.push_back(element);
    }

    return result;
  }

  std::vector<T> get_or(const std::vector<T>& default_value) {
    auto* arr = config_[key_].as_array();
    if (!arr) return default_value;

    // Apply same validation logic as get()
    return get();
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
