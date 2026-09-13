#include "magma.hpp"
#include "application_data.hpp"
#include <algorithm>
#include <iostream>
#include <random>
#include <numeric>
#include <magma_v2.h>

int main(int argc, char** argv) {
  namespace b = vbsr::bench;
  try {
    bool verify_only = argc > 1 && std::string(argv[argc-1]) == "--verify-only";
    if (verify_only) --argc;
    if (argc < 9 || argc > 11) throw std::invalid_argument("usage: relevance rows degree rhs shape seed reps order input-file-or-none [locality] [batch-size] [--verify-only]");
    int rows = b::parse_integer<int>(argv[1], "rows"), degree = b::parse_integer<int>(argv[2], "degree");
    int rhs = b::parse_integer<int>(argv[3], "rhs"), shape = b::parse_integer<int>(argv[4], "shape");
    unsigned seed = b::parse_integer<unsigned>(argv[5], "seed"), order = b::parse_integer<unsigned>(argv[7], "order");
    int reps = b::parse_integer<int>(argv[6], "reps");
    int batch = argc == 11 ? b::parse_integer<int>(argv[10], "batch-size") : 1;
    if (batch != 1 && batch != 8) throw std::invalid_argument("batch-size must be 1 or 8");
    b::validate_panel_width(rhs); b::validate_repetitions(reps);
    vbsr::HostMatrix host;
    if (std::string(argv[8]) != "none") host = b::load_application(argv[8]).packed;
    else {
      if (degree < 1 || degree > rows) throw std::invalid_argument("degree must be in [1,rows]");
      auto distribution = shape==-3 ? vbsr::Distribution::Uniform : shape==-2 ? vbsr::Distribution::LowVariance : shape==-1 ? vbsr::Distribution::HighVariance : vbsr::Distribution::Bimodal;
      bool local=argc>=10 ? b::parse_locality(argv[9]) : false;
      host = vbsr::generate({rows,rows,std::min(degree,16),rhs,distribution,local,seed});
      if (shape>0) {
        if (shape < 8 || shape > 64 || shape % 8) throw std::invalid_argument("shape must be 0 or a supported size");
        std::fill(host.row_size.begin(),host.row_size.end(),shape);
        std::fill(host.col_size.begin(),host.col_size.end(),shape);
        // Each degree selects a nested prefix of the same row permutation.
        // Sorting restores the public format's canonical column order.
        host.block_col.clear(); host.row_ptr={0}; host.value_off={0};
        for (int r=0;r<rows;++r) {
          std::vector<int> columns(rows);
          std::iota(columns.begin(),columns.end(),0);
          std::shuffle(columns.begin(),columns.end(),std::mt19937(seed*65537u+unsigned(r)));
          columns.resize(degree); std::sort(columns.begin(),columns.end());
          host.block_col.insert(host.block_col.end(),columns.begin(),columns.end());
          host.row_ptr.push_back(int(host.block_col.size()));
        }
        host.value_off.resize(host.block_col.size()+1);
        for (int i=0;i<=rows;++i) host.row_scalar_off[i]=host.col_scalar_off[i]=int64_t(i)*shape;
        for (size_t j=0;j<host.value_off.size();++j) host.value_off[j]=int64_t(j)*shape*shape;
        host.values=b::make_input(size_t(host.value_off.back()));
      }
    }
    host.validate();
    if (magma_init() != MAGMA_SUCCESS) throw std::runtime_error("MAGMA initialization failed");
    {
      vbsr::Matrix matrix(host);
      auto input=b::make_input(b::panel_elements(host.scalar_cols(),rhs));
      b::DeviceBuffer<float> db(input.size()), dc(b::panel_elements(host.scalar_rows(),rhs));
      db.upload(input);
      std::vector<float> reference;
      if (verify_only || host.values.size() <= 2000000) reference=vbsr::cpu_reference(host,input,rhs);
      auto validate=[&] {
        if (!reference.empty()) b::verify_output(reference,dc.data());
        else b::verify_probes(host,input,dc.data(),rhs);
      };
      b::print_environment(&host);
      if (!verify_only) b::print_timing_header();
      auto run=[&](const std::string& name, const auto& operation, int pos) {
        if (verify_only) {
          operation(0);
          b::check_cuda(cudaDeviceSynchronize());
          validate();
          std::cout << "PASS full-output " << name << " elements=" << reference.size() << '\n';
        } else b::measure(name,operation,validate,reps,pos,batch);
      };
      std::vector<std::string> methods={"direct","magma_slots","magma_reduce","bsr8","csr1","csr2","csr3","grouped"};
      if (shape==32 || shape==-3) methods.push_back("bsr32");
      std::shuffle(methods.begin(),methods.end(),std::mt19937(order));
      for (int pos=0;pos<int(methods.size());++pos) {
        const auto& name=methods[pos];
        b::check_cuda(cudaMemset(dc.data(),0xff,dc.size()*sizeof(float)));
        if (name=="direct") {
          vbsr::Plan plan(matrix.device_view(),{rhs});
          run(name,[&](int){plan.execute(db.data(),dc.data());},pos);
        } else if (name.starts_with("magma")) {
          b::MagmaPlan plan(host,matrix.device_view(),db.data(),dc.data(),rhs,name=="magma_reduce");
          run(name,[&](int){plan.execute();},pos);
        } else if (name=="grouped") {
          vbsr::GroupedGemmPlan plan(host,rhs,true);
          run(name,[&](int){plan.execute(db.data(),dc.data());},pos);
        } else {
          bool bsr=name.starts_with("bsr");
          int alg=bsr ? 0 : name.back()-'0';
          vbsr::ScalarCsrPlan plan(host,rhs,alg,alg==1||alg==3,bsr,name=="bsr32"?32:8);
          run(name,[&](int){plan.execute(db.data(),dc.data());},pos);
        }
      }
    }
    magma_finalize();
    return 0;
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
