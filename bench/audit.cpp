#include "support.hpp"
#include <algorithm>
#include <iostream>
#include <random>

namespace {
namespace bench = vbsr::bench;

struct Arguments {
  vbsr::GeneratorOptions generator;
  int rhs_width;
  unsigned order_seed;
  bool irregular;
  int repetitions;
};

Arguments parse_arguments(int count, char** values) {
  if (count != 10) {
    throw std::invalid_argument(
        "usage: audit rows degree rhs distribution locality seed order_seed irregular reps");
  }
  Arguments args;
  args.generator.block_rows = args.generator.block_cols =
      bench::parse_integer<int>(values[1], "rows");
  args.generator.degree = bench::parse_integer<int>(values[2], "degree");
  args.rhs_width = bench::parse_integer<int>(values[3], "RHS width");
  args.generator.distribution = bench::parse_distribution(values[4]);
  args.generator.local_columns = bench::parse_locality(values[5]);
  args.generator.seed = bench::parse_integer<uint64_t>(values[6], "matrix seed");
  args.order_seed = bench::parse_integer<unsigned>(values[7], "order seed");
  int irregular = bench::parse_integer<int>(values[8], "irregular flag");
  if (irregular != 0 && irregular != 1) {
    throw std::invalid_argument("irregular flag must be 0 or 1");
  }
  args.irregular = irregular != 0;
  args.repetitions = bench::parse_integer<int>(values[9], "repetitions");
  bench::validate_panel_width(args.rhs_width);
  bench::validate_repetitions(args.repetitions);
  return args;
}

vbsr::HostMatrix thin_rows(const vbsr::HostMatrix& source) {
  auto matrix = source;
  matrix.row_ptr = {0};
  matrix.block_col.clear();
  matrix.values.clear();
  matrix.value_off = {0};
  for (int row = 0; row < source.block_rows; ++row) {
    int count = source.row_ptr[row + 1] - source.row_ptr[row];
    if (row % 4 == 0)
      count = 0;
    if (row % 4 == 1)
      count = 1;
    for (int slot = 0; slot < count; ++slot) {
      int block_index = source.row_ptr[row] + slot;
      matrix.block_col.push_back(source.block_col[block_index]);
      matrix.values.insert(matrix.values.end(),
                           source.values.begin() + source.value_off[block_index],
                           source.values.begin() + source.value_off[block_index + 1]);
      matrix.value_off.push_back(matrix.values.size());
    }
    matrix.row_ptr.push_back(int(matrix.block_col.size()));
  }
  matrix.validate();
  return matrix;
}

vbsr::HostMatrix make_matrix(const Arguments& args) {
  auto matrix = vbsr::generate(args.generator);
  return args.irregular ? thin_rows(matrix) : matrix;
}

enum class Method {
  Direct,
  CsrDefault,
  CsrOne,
  CsrTwo,
  CsrThree,
  GroupedCached,
  GroupedChanging,
  DirectChanging,
  FixedBsr,
  BsrEight
};
std::string_view method_name(Method method) {
  switch (method) {
  case Method::Direct:
    return "direct";
  case Method::CsrDefault:
    return "csr_default";
  case Method::CsrOne:
    return "csr_alg1_pre";
  case Method::CsrTwo:
    return "csr_alg2";
  case Method::CsrThree:
    return "csr_alg3_pre";
  case Method::GroupedCached:
    return "grouped_cached";
  case Method::GroupedChanging:
    return "grouped_changing";
  case Method::DirectChanging:
    return "direct_changing";
  case Method::FixedBsr:
    return "bsr32";
  case Method::BsrEight:
    return "bsr8";
  }
  throw std::invalid_argument("unknown audit method");
}

class SyntheticAudit {
public:
  explicit SyntheticAudit(const Arguments& args)
      : args_(args), matrix_(make_matrix(args)), device_(matrix_),
        host_input_(
            bench::make_input(bench::panel_elements(matrix_.scalar_cols(), args.rhs_width))),
        input_(host_input_.size()), alternate_input_(host_input_.size()),
        output_(bench::panel_elements(matrix_.scalar_rows(), args.rhs_width)),
        alternate_output_(output_.size()) {
    input_.upload(host_input_);
    alternate_input_.upload(host_input_);
  }

  void run() {
    bench::print_environment(&matrix_);
    // Preserve the published initial order and random engine so each order seed
    // still selects the same sequence of methods.
    std::vector<Method> methods = {
        Method::Direct,   Method::CsrDefault,    Method::CsrOne,          Method::CsrTwo,
        Method::CsrThree, Method::GroupedCached, Method::GroupedChanging, Method::DirectChanging,
        Method::BsrEight};
    if (args_.generator.distribution == vbsr::Distribution::Uniform)
      methods.push_back(Method::FixedBsr);
    std::mt19937 random_engine(args_.order_seed);
    std::shuffle(methods.begin(), methods.end(), random_engine);
    bench::print_timing_header();
    for (size_t position = 0; position < methods.size(); ++position)
      run_method(methods[position], int(position));
  }

private:
  Arguments args_;
  vbsr::HostMatrix matrix_;
  vbsr::Matrix device_;
  std::vector<float> host_input_;
  bench::DeviceBuffer<float> input_, alternate_input_, output_, alternate_output_;

  void validate() const {
    bench::verify_probes(matrix_, host_input_, output_.data(), args_.rhs_width);
  }

  template <class Plan> void measure_plan(Method method, Plan& plan, int position, bool changing) {
    bench::check_cuda(cudaMemset(output_.data(), 0xff, output_.size() * sizeof(float)));
    bench::check_cuda(cudaMemset(alternate_output_.data(), 0xff, alternate_output_.size() * sizeof(float)));
    bench::measure(
        method_name(method),
        [&](int iteration) {
          const bool alternate = changing && iteration % 2 != 0;
          plan.execute(alternate ? alternate_input_.data() : input_.data(),
                       alternate ? alternate_output_.data() : output_.data());
        },
        [&] {
          validate();
          if (changing)
            bench::verify_probes(matrix_, host_input_, alternate_output_.data(), args_.rhs_width);
        },
        args_.repetitions, position);
  }

  void measure_sparse(Method method, int position, int algorithm, bool preprocess,
                      bool fixed_bsr = false, int block_size = 32) {
    vbsr::ScalarCsrPlan plan(matrix_, args_.rhs_width, algorithm, preprocess, fixed_bsr, block_size);
    measure_plan(method, plan, position, false);
  }

  void run_method(Method method, int position) {
    switch (method) {
    case Method::Direct:
    case Method::DirectChanging: {
      vbsr::Plan plan(device_, {args_.rhs_width, vbsr::Kernel::RowOwned});
      measure_plan(method, plan, position, method == Method::DirectChanging);
      break;
    }
    case Method::GroupedCached:
    case Method::GroupedChanging: {
      vbsr::GroupedGemmPlan plan(matrix_, args_.rhs_width, true);
      measure_plan(method, plan, position, method == Method::GroupedChanging);
      break;
    }
    case Method::CsrDefault:
      measure_sparse(method, position, 0, false);
      break;
    case Method::CsrOne:
      measure_sparse(method, position, 1, true);
      break;
    case Method::CsrTwo:
      measure_sparse(method, position, 2, false);
      break;
    case Method::CsrThree:
      measure_sparse(method, position, 3, true);
      break;
    case Method::FixedBsr:
      measure_sparse(method, position, 0, false, true);
      break;
    case Method::BsrEight:
      measure_sparse(method, position, 0, false, true, 8);
      break;
    }
  }
};
} // namespace

int main(int argc, char** argv) {
  try {
    SyntheticAudit audit(parse_arguments(argc, argv));
    audit.run();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
