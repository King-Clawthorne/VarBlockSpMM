// Application-specific comparison for the transport trace. The specialized
// plan exploits the repeated block operators that a hand-written discontinuous
// Galerkin code would exploit. Every method runs the same dependent trace, to
// the same physical time, and passes the same numerical checks.
#include "dg_specialized.hpp"
#include "dg_fused.cuh"
#include "application_data.hpp"
#include "support.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

namespace b = vbsr::bench;

std::vector<float> read_panel(const std::string& name, size_t count) {
  std::ifstream f(name, std::ios::binary | std::ios::ate);
  if (!f || f.tellg() != std::streamoff(count * sizeof(float)))
    throw std::runtime_error("invalid panel file: " + name);
  f.seekg(0);
  std::vector<float> result(count);
  f.read(reinterpret_cast<char*>(result.data()), std::streamsize(count * sizeof(float)));
  if (!f)
    throw std::runtime_error("panel read failed");
  return result;
}

int main(int argc, char** argv) {
  try {
    if (argc != 6)
      throw std::invalid_argument("usage: dg input-base rhs steps reps order");
    std::string base = argv[1];
    int rhs = b::parse_integer<int>(argv[2], "rhs"), steps = b::parse_integer<int>(argv[3], "steps");
    int reps = b::parse_integer<int>(argv[4], "reps");
    auto seed = b::parse_integer<unsigned>(argv[5], "order");
    b::validate_panel_width(rhs);
    b::validate_repetitions(reps);
    if (steps <= 0 || steps > 1024)
      throw std::invalid_argument("invalid step count");
    auto host = b::load_application(base + ".bin").packed;
    auto size = b::panel_elements(host.scalar_rows(), rhs);
    auto initial = read_panel(base + ".input", size), reference = read_panel(base + ".reference", size);
    auto exact = read_panel(base + ".exact", size);
    auto weights = read_panel(base + ".weights", size_t(host.scalar_rows()));
    b::DeviceBuffer<float> original(size), x(size), y(size);
    original.upload(initial);
    bool identity_rejected = false;
    try {
      b::verify_output(reference, original.data());
    } catch (const std::runtime_error&) {
      identity_rejected = true;
    }
    if (!identity_rejected)
      throw std::runtime_error("unchanged-state negative control was accepted");
    std::cerr << "identity_negative_control,rejected\n";
    auto reset = [&] {
      b::check_cuda(cudaMemcpyAsync(x.data(), original.data(), size * sizeof(float),
                                    cudaMemcpyDeviceToDevice));
    };
    float* result = steps % 2 ? y.data() : x.data();
    auto validate = [&] {
      b::verify_output(reference, result);
      std::vector<float> actual(size);
      b::check_cuda(cudaMemcpy(actual.data(), result, size * sizeof(float), cudaMemcpyDeviceToHost));
      double error = 0, norm = 0;
      for (size_t i = 0; i < size; ++i) {
        double w = weights[i % weights.size()], difference = double(actual[i]) - exact[i];
        if (!(w > 0) || !std::isfinite(w) || !std::isfinite(exact[i]))
          throw std::runtime_error("invalid analytic reference or mass weight");
        error += w * difference * difference;
        norm += w * double(exact[i]) * exact[i];
      }
      double relative = std::sqrt(error / norm);
      if (!std::isfinite(relative) || relative > 2e-3)
        throw std::runtime_error("GPU transport analytic L2 check failed");
      std::cerr << "analytic_relative_l2," << relative << '\n';
    };
    {
      // Report the structure the specialized comparator exploits, once, before
      // any timing. A campaign record without this line is incomplete. The
      // alternating trace holds one plan per direction, and each owns its
      // operators and pointer arrays, so report the timed total as well.
      vbsr::bench::DgSpecializedPlan even(host, rhs, x.data(), y.data());
      vbsr::bench::DgSpecializedPlan odd(host, rhs, y.data(), x.data());
      std::cerr << "specialized_structure," << even.distinct_operators() << ','
                << even.launch_count() << ',' << even.operator_bytes() << ','
                << even.assembled_bytes() << ',' << even.storage_bytes() << ','
                << even.storage_bytes() + odd.storage_bytes() << '\n';
    }
    std::vector<std::string> methods = {"direct", "dg_specialized", "dg_fused",
                                        "dg_fused_copies", "grouped", "bsr8"};
    std::shuffle(methods.begin(), methods.end(), std::mt19937(seed));
    b::print_environment(&host);
    b::print_timing_header();
    for (int pos = 0; pos < int(methods.size()); ++pos) {
      auto name = methods[pos];
      auto started = std::chrono::steady_clock::now();
      auto run = [&](auto&& execute) {
        auto operation = [&](int) {
          reset();
          b::check_cuda(cudaMemsetAsync(y.data(), 0xff, size * sizeof(float)));
          for (int step = 0; step < steps; ++step)
            execute(step, step % 2 ? y.data() : x.data(), step % 2 ? x.data() : y.data());
        };
        operation(0);
        b::check_cuda(cudaDeviceSynchronize());
        auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        std::cerr << "startup_trace_ms," << name << ',' << ms << '\n';
        validate();
        // Each observation reaches the same physical time, T=1/128.
        b::measure(name, operation, validate, reps, pos);
      };
      if (name == "direct") {
        vbsr::Matrix matrix(host);
        vbsr::Plan plan(matrix.device_view(), {rhs});
        run([&](int, const float* a, float* c) { plan.execute(a, c); });
      } else if (name == "dg_specialized") {
        // The trace alternates buffers, so bind one plan per direction and
        // keep every pointer array cached, as the MAGMA comparator does.
        vbsr::bench::DgSpecializedPlan even(host, rhs, x.data(), y.data());
        vbsr::bench::DgSpecializedPlan odd(host, rhs, y.data(), x.data());
        run([&](int step, const float*, float*) { (step % 2 ? odd : even).execute(); });
      } else if (name == "dg_fused") {
        vbsr::bench::DgFusedPlan plan(host, rhs);
        run([&](int, const float* a, float* c) { plan.execute(a, c); });
      } else if (name == "dg_fused_copies") {
        // Matched control: same kernel, private operator copies per element.
        vbsr::bench::DgFusedPlan plan(host, rhs, false);
        run([&](int, const float* a, float* c) { plan.execute(a, c); });
      } else if (name == "grouped") {
        vbsr::GroupedGemmPlan plan(host, rhs, true);
        run([&](int, const float* a, float* c) { plan.execute(a, c); });
      } else {
        vbsr::ScalarCsrPlan plan(host, rhs, 0, false, true, 8);
        run([&](int, const float* a, float* c) { plan.execute(a, c); });
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
