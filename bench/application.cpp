#include "application_data.hpp"
#include "compact_csr.hpp"
#include "dense.hpp"
#include "support.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

namespace {
namespace bench = vbsr::bench;

struct Arguments {
    std::filesystem::path input_path;
    int rhs_width;
    unsigned order_seed;
    int repetitions;
};

Arguments parse_arguments(int count, char** values) {
    if (count != 5) {
        throw std::invalid_argument("usage: application input.bin rhs order_seed reps");
    }
    Arguments args{values[1], bench::parse_integer<int>(values[2], "RHS width"),
                   bench::parse_integer<unsigned>(values[3], "order seed"),
                   bench::parse_integer<int>(values[4], "repetitions")};
    bench::validate_panel_width(args.rhs_width);
    bench::validate_repetitions(args.repetitions);
    return args;
}

enum class Method { Direct, CsrOne, CsrTwo, CsrThree, GroupedCached, BsrEight, Dense };
std::string_view method_name(Method method) {
    switch (method) {
    case Method::Direct:
        return "direct";
    case Method::CsrOne:
        return "compact_alg1_pre";
    case Method::CsrTwo:
        return "compact_alg2";
    case Method::CsrThree:
        return "compact_alg3_pre";
    case Method::GroupedCached:
        return "grouped_cached";
    case Method::BsrEight:
        return "bsr8";
    case Method::Dense:
        return "dense";
    }
    throw std::invalid_argument("unknown application method");
}

class ApplicationAudit {
  public:
    explicit ApplicationAudit(const Arguments& args)
        : args_(args), data_(bench::load_application(args.input_path)),
          host_input_(
              bench::make_input(bench::panel_elements(data_.packed.scalar_cols(), args.rhs_width))),
          input_(host_input_.size()),
          output_(bench::panel_elements(data_.packed.scalar_rows(), args.rhs_width)),
          reference_(data_.compact.reference(host_input_, data_.packed.scalar_rows(),
                                             data_.packed.scalar_cols(), args.rhs_width)) {
        input_.upload(host_input_);
    }

    void run() {
        bench::print_environment();
        std::vector<Method> methods = {Method::Direct,   Method::CsrOne,        Method::CsrTwo,
                                       Method::CsrThree, Method::GroupedCached, Method::BsrEight,
                                       Method::Dense};
        std::mt19937 random_engine(args_.order_seed);
        std::shuffle(methods.begin(), methods.end(), random_engine);
        bench::print_timing_header();
        for (size_t position = 0; position < methods.size(); ++position)
            run_method(methods[position], int(position));
    }

  private:
    Arguments args_;
    bench::ApplicationData data_;
    std::vector<float> host_input_;
    bench::DeviceBuffer<float> input_, output_;
    std::vector<float> reference_;

    void measure(Method method, int position, const std::function<void(int)>& operation) {
        bench::check_cuda(cudaMemset(output_.data(), 0xff, output_.size() * sizeof(float)));
        bench::measure(
            method_name(method), operation,
            [&] { bench::verify_output(reference_, output_.data()); }, args_.repetitions, position);
    }

    void setup_record(Method method, std::chrono::steady_clock::time_point start, size_t bytes) {
        bench::check_cuda(cudaDeviceSynchronize());
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        std::cerr << "setup," << method_name(method) << ',' << ms << ',' << bytes << '\n';
    }

    void measure_sparse(Method method, int position, bench::CsrAlgorithm algorithm) {
        auto start = std::chrono::steady_clock::now();
        bench::CompactCsrPlan plan(data_.compact, data_.packed.scalar_rows(),
                                   data_.packed.scalar_cols(), args_.rhs_width, input_.data(),
                                   output_.data(), algorithm);
        plan.execute();
        setup_record(method, start, plan.storage_bytes());
        measure(method, position, [&](int) { plan.execute(); });
    }

    void run_method(Method method, int position) {
        switch (method) {
        case Method::Direct: {
            auto start = std::chrono::steady_clock::now();
            vbsr::Matrix device(data_.packed);
            vbsr::Plan plan(device, {args_.rhs_width, vbsr::Kernel::RowOwned});
            plan.execute(input_.data(), output_.data());
            setup_record(method, start, device.storage_bytes());
            measure(method, position, [&](int) { plan.execute(input_.data(), output_.data()); });
            break;
        }
        case Method::GroupedCached: {
            auto start = std::chrono::steady_clock::now();
            vbsr::GroupedGemmPlan plan(data_.packed, args_.rhs_width, true);
            plan.execute(input_.data(), output_.data());
            setup_record(method, start, plan.storage_bytes());
            measure(method, position, [&](int) { plan.execute(input_.data(), output_.data()); });
            break;
        }
        case Method::BsrEight: {
            auto start = std::chrono::steady_clock::now();
            vbsr::ScalarCsrPlan plan(data_.packed, args_.rhs_width, 0, false, true, 8);
            plan.execute(input_.data(), output_.data());
            setup_record(method, start, plan.storage_bytes());
            measure(method, position, [&](int) { plan.execute(input_.data(), output_.data()); });
            break;
        }
        case Method::Dense: {
            auto start = std::chrono::steady_clock::now();
            bench::DensePlan plan(data_.compact, data_.packed.scalar_rows(),
                                  data_.packed.scalar_cols(), args_.rhs_width, input_.data(),
                                  output_.data());
            plan.execute();
            setup_record(method, start, plan.storage_bytes());
            measure(method, position, [&](int) { plan.execute(); });
            break;
        }
        case Method::CsrOne:
            measure_sparse(method, position, bench::CsrAlgorithm::One);
            break;
        case Method::CsrTwo:
            measure_sparse(method, position, bench::CsrAlgorithm::Two);
            break;
        case Method::CsrThree:
            measure_sparse(method, position, bench::CsrAlgorithm::Three);
            break;
        }
    }
};
}

int main(int argc, char** argv) {
    try {
        ApplicationAudit audit(parse_arguments(argc, argv));
        audit.run();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
