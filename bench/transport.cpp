#include "magma.hpp"
#include "application_data.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <magma_v2.h>

namespace b = vbsr::bench;
std::vector<float> read_panel(const std::string& name, size_t count) {
  std::ifstream f(name,std::ios::binary|std::ios::ate);
  if (!f || f.tellg()!=std::streamoff(count*sizeof(float))) throw std::runtime_error("invalid panel file: "+name);
  f.seekg(0); std::vector<float> result(count);
  f.read(reinterpret_cast<char*>(result.data()),std::streamsize(count*sizeof(float)));
  if (!f) throw std::runtime_error("panel read failed");
  return result;
}
int main(int argc,char** argv) {
  try {
    if(argc!=6) throw std::invalid_argument("usage: transport input-base rhs steps reps order");
    std::string base=argv[1];
    int rhs=b::parse_integer<int>(argv[2],"rhs"), steps=b::parse_integer<int>(argv[3],"steps");
    int reps=b::parse_integer<int>(argv[4],"reps");
    auto seed=b::parse_integer<unsigned>(argv[5],"order");
    b::validate_panel_width(rhs); b::validate_repetitions(reps);
    if(steps<=0 || steps>1024) throw std::invalid_argument("invalid step count");
    auto host=b::load_application(base+".bin").packed;
    auto size=b::panel_elements(host.scalar_rows(),rhs);
    auto initial=read_panel(base+".input",size), reference=read_panel(base+".reference",size);
    if(magma_init()!=MAGMA_SUCCESS) throw std::runtime_error("MAGMA init failed");
    {
      b::DeviceBuffer<float> original(size), x(size), y(size);
      original.upload(initial);
      auto reset=[&] { b::check_cuda(cudaMemcpyAsync(x.data(),original.data(),size*sizeof(float),cudaMemcpyDeviceToDevice)); };
      float* result=steps%2 ? y.data() : x.data();
      auto validate=[&] { b::verify_output(reference,result); };
      std::vector<std::string> methods={"direct","magma_slots","magma_reduce","bsr8","csr1","csr2","csr3","grouped"};
      std::shuffle(methods.begin(),methods.end(),std::mt19937(seed));
      b::print_environment(&host); b::print_timing_header();
      for(int pos=0;pos<int(methods.size());++pos) {
        auto name=methods[pos];
        auto started=std::chrono::steady_clock::now();
        auto run=[&](auto&& execute) {
          auto operation=[&](int) {
            reset();
            for(int step=0;step<steps;++step)
              execute(step,step%2?y.data():x.data(),step%2?x.data():y.data());
          };
          operation(0); b::check_cuda(cudaDeviceSynchronize());
          auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
          std::cerr<<"startup_trace_ms,"<<name<<","<<ms<<"\n";
          validate();
          // Each timing observation is a complete dependent 32-step trace.
          b::measure(name,operation,validate,reps,pos);
        };
        if(name=="direct") {
          vbsr::Matrix matrix(host); vbsr::Plan plan(matrix.device_view(),{rhs});
          run([&](int,const float* a,float* c){plan.execute(a,c);});
        } else if(name.starts_with("magma")) {
          vbsr::Matrix matrix(host);
          b::MagmaPlan even(host,matrix.device_view(),x.data(),y.data(),rhs,name=="magma_reduce");
          b::MagmaPlan odd(host,matrix.device_view(),y.data(),x.data(),rhs,name=="magma_reduce");
          run([&](int step,const float*,float*){ (step%2?odd:even).execute(); });
        } else if(name=="grouped") {
          vbsr::GroupedGemmPlan plan(host,rhs,true);
          run([&](int,const float* a,float* c){plan.execute(a,c);});
        } else {
          bool bsr=name=="bsr8"; int alg=bsr?0:name.back()-'0';
          vbsr::ScalarCsrPlan plan(host,rhs,alg,alg==1||alg==3,bsr,8);
          run([&](int,const float* a,float* c){plan.execute(a,c);});
        }
      }
    }
    magma_finalize(); return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
