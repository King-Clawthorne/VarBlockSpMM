#pragma once

#include "varblockspmm/vbsr.hpp"
#include <cuda_runtime.h>

#include <charconv>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace vbsr::bench {

inline constexpr int warmup_count = 5;
inline constexpr unsigned input_seed = 9123;
inline constexpr double absolute_tolerance = 5e-4;
inline constexpr double relative_tolerance = 5e-5;

void check_cuda(cudaError_t status);

// Audit arguments must consume the entire token. In particular, reject negative
// unsigned seeds and numeric prefixes such as "32junk" before allocating memory.
template <class T> T parse_integer(std::string_view text, std::string_view name) {
  T value{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc{} || end != text.data() + text.size()) {
    throw std::invalid_argument("invalid " + std::string(name) + ": " + std::string(text));
  }
  return value;
}

void validate_panel_width(int rhs_width);
void validate_repetitions(int repetitions);
bool parse_locality(std::string_view locality);
Distribution parse_distribution(std::string_view name);
size_t panel_elements(int64_t scalar_rows, int rhs_width);

// Noncopyable owners also clean up when a later allocation or CUDA call fails.
template <class T> class DeviceBuffer {
public:
  explicit DeviceBuffer(size_t count = 0) : count_(count) {
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
      throw std::overflow_error("device buffer size overflow");
    }
    if (count != 0) {
      check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    }
  }
  ~DeviceBuffer() { cudaFree(data_); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* data() const { return data_; }
  size_t size() const { return count_; }
  void upload(std::span<const T> source) {
    if (source.size() != count_) {
      throw std::invalid_argument("device upload size mismatch");
    }
    if (count_ != 0) {
      check_cuda(cudaMemcpy(data_, source.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
    }
  }

private:
  T* data_ = nullptr;
  size_t count_ = 0;
};

class Event {
public:
  Event() { check_cuda(cudaEventCreate(&event_)); }
  ~Event() { cudaEventDestroy(event_); }
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  cudaEvent_t get() const { return event_; }

private:
  cudaEvent_t event_{};
};

std::vector<float> make_input(size_t count);
void print_environment(const HostMatrix* matrix = nullptr);
void print_timing_header();
void verify_probes(const HostMatrix& matrix, const std::vector<float>& input, const float* output,
                   int rhs_width);
void verify_output(const std::vector<float>& reference, const float* output);

// Keep warmup indices, event placement, validation, and CSV schema consistent
// across both audit executables. Raw timings belong to the executable that ran.
void measure(std::string_view name, const std::function<void(int)>& operation,
             const std::function<void()>& validate, int repetitions, int position);

} // namespace vbsr::bench
