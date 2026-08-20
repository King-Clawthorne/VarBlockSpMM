#include "varblockspmm/vbsr.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct Args{int rows=1024,degree=8,rhs=32,reps=50,warmup=10;uint64_t seed=1;bool local=true;vbsr::Distribution dist=vbsr::Distribution::HighVariance;};
static Args parse(int argc,char**argv){Args a;for(int i=1;i<argc;i++){std::string k=argv[i];auto value=[&](){if(++i>=argc)throw std::runtime_error("missing value for "+k);return std::string(argv[i]);};if(k=="--rows")a.rows=std::stoi(value());else if(k=="--degree")a.degree=std::stoi(value());else if(k=="--rhs")a.rhs=std::stoi(value());else if(k=="--reps")a.reps=std::stoi(value());else if(k=="--warmup")a.warmup=std::stoi(value());else if(k=="--seed")a.seed=std::stoull(value());else if(k=="--locality")a.local=value()=="local";else if(k=="--distribution"){auto v=value();if(v=="uniform")a.dist=vbsr::Distribution::Uniform;else if(v=="low")a.dist=vbsr::Distribution::LowVariance;else if(v=="high")a.dist=vbsr::Distribution::HighVariance;else if(v=="bimodal")a.dist=vbsr::Distribution::Bimodal;else throw std::runtime_error("unknown distribution");}else throw std::runtime_error("unknown argument: "+k);}return a;}
static const char* name(vbsr::Distribution d){switch(d){case vbsr::Distribution::Uniform:return"uniform";case vbsr::Distribution::LowVariance:return"low";case vbsr::Distribution::HighVariance:return"high";default:return"bimodal";}}
struct Result{double gpu_med,gpu_p95,host_med,host_p95;};
static Result time_call(const std::function<void()>&f,int warmup,int reps){for(int i=0;i<warmup;i++)f();cudaDeviceSynchronize();std::vector<double>gpu,host;cudaEvent_t begin,end;cudaEventCreate(&begin);cudaEventCreate(&end);for(int i=0;i<reps;i++){auto h0=std::chrono::steady_clock::now();cudaEventRecord(begin);f();cudaEventRecord(end);cudaEventSynchronize(end);auto h1=std::chrono::steady_clock::now();float ms;cudaEventElapsedTime(&ms,begin,end);gpu.push_back(ms);host.push_back(std::chrono::duration<double,std::milli>(h1-h0).count());}cudaEventDestroy(begin);cudaEventDestroy(end);std::sort(gpu.begin(),gpu.end());std::sort(host.begin(),host.end());auto p95=[&](const std::vector<double>&v){return v[std::min(v.size()-1,size_t(v.size()*0.95))];};return{gpu[gpu.size()/2],p95(gpu),host[host.size()/2],p95(host)};}

int main(int argc,char**argv){try{
  auto x=parse(argc,argv);vbsr::GeneratorOptions o;o.block_rows=x.rows;o.block_cols=x.rows;o.degree=x.degree;o.rhs_width=x.rhs;o.distribution=x.dist;o.local_columns=x.local;o.seed=x.seed;auto h=vbsr::generate(o);vbsr::Matrix a(h);std::vector<float>b(h.scalar_cols()*x.rhs,0.01f);float *db,*dc;cudaMalloc(&db,b.size()*sizeof(float));cudaMalloc(&dc,h.scalar_rows()*x.rhs*sizeof(float));cudaMemcpy(db,b.data(),b.size()*sizeof(float),cudaMemcpyHostToDevice);vbsr::Plan direct(a,{x.rhs,vbsr::Kernel::RowOwned});vbsr::ScalarCsrPlan csr(h,x.rhs);vbsr::GroupedGemmPlan grouped(h,x.rhs);double flops=0;for(int i=0;i<h.block_rows;i++)for(int p=h.row_ptr[i];p<h.row_ptr[i+1];p++)flops+=2.0*h.row_size[i]*h.col_size[h.block_col[p]]*x.rhs;
  std::cout<<"method,block_rows,degree,distribution,locality,rhs,seed,gpu_median_ms,gpu_p95_ms,hot_median_ms,hot_p95_ms,useful_gflops,launches,workspace_bytes\n";
  auto emit=[&](const char*m,const std::function<void()>&f,int launches,size_t ws){auto r=time_call(f,x.warmup,x.reps);std::cout<<m<<','<<x.rows<<','<<x.degree<<','<<name(x.dist)<<','<<(x.local?"local":"random")<<','<<x.rhs<<','<<x.seed<<','<<std::fixed<<std::setprecision(5)<<r.gpu_med<<','<<r.gpu_p95<<','<<r.host_med<<','<<r.host_p95<<','<<flops/(r.gpu_med*1e6)<<','<<launches<<','<<ws<<'\n';};
  emit("row_owned_hybrid",[&]{direct.execute(db,dc);},1,0);
  emit("row_owned_scalar",[&]{vbsr::launch_row_owned_scalar(a.device_view(),db,dc,x.rhs,0);},1,0);
  emit("scalar_csr_cusparse",[&]{csr.execute(db,dc);},1,csr.workspace_bytes());
  emit("slot_grouped_cublas",[&]{grouped.execute(db,dc);},grouped.launch_count(),0);
  cudaFree(db);cudaFree(dc);return 0;
}catch(const std::exception&e){std::cerr<<"error: "<<e.what()<<"\n";return 2;}}
