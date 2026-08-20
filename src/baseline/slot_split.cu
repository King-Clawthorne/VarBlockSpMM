#include "varblockspmm/vbsr.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace vbsr {
static void cuda_check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void blas_check(cublasStatus_t e){if(e!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS failure: "+std::to_string(int(e)));}

struct GroupedGemmPlan::Impl {
  struct Slot {
    std::vector<cublasOperation_t> ta,tb;
    std::vector<int> m,n,k,lda,ldb,ldc,group_size;
    std::vector<float> alpha,beta;
    std::vector<const float*> A;
    std::vector<int64_t> b_off,c_off;
    const float** dA{}; const float** dB{}; float** dC{};
  };
  int rhs{}; int64_t rows{},cols{}; float* values{}; cublasHandle_t handle{};
  std::vector<Slot> slots;
  ~Impl(){for(auto&s:slots){cudaFree(s.dA);cudaFree(s.dB);cudaFree(s.dC);}if(handle)cublasDestroy(handle);cudaFree(values);}
};

GroupedGemmPlan::GroupedGemmPlan(const HostMatrix& a,int rhs):impl_(new Impl) {
  a.validate(); if(rhs!=8&&rhs!=16&&rhs!=32&&rhs!=64)throw std::invalid_argument("invalid rhs width");
  impl_->rhs=rhs;impl_->rows=a.scalar_rows();impl_->cols=a.scalar_cols();
  cuda_check(cudaMalloc(&impl_->values,a.values.size()*sizeof(float)));
  cuda_check(cudaMemcpy(impl_->values,a.values.data(),a.values.size()*sizeof(float),cudaMemcpyHostToDevice));
  blas_check(cublasCreate(&impl_->handle));
  int max_degree=0;for(int i=0;i<a.block_rows;i++)max_degree=std::max(max_degree,a.row_ptr[i+1]-a.row_ptr[i]);
  impl_->slots.resize(max_degree);
  for(int slot=0;slot<max_degree;slot++) {
    using Key=std::pair<int,int>; std::map<Key,std::vector<std::pair<int,int>>> groups;
    for(int i=0;i<a.block_rows;i++){int p=a.row_ptr[i]+slot;if(p<a.row_ptr[i+1])groups[{a.row_size[i],a.col_size[a.block_col[p]]}].push_back({i,p});}
    auto& s=impl_->slots[slot];
    for(const auto& entry:groups){int r=entry.first.first,q=entry.first.second;s.ta.push_back(CUBLAS_OP_N);s.tb.push_back(CUBLAS_OP_N);s.m.push_back(r);s.n.push_back(rhs);s.k.push_back(q);s.lda.push_back(r);s.ldb.push_back(int(a.scalar_cols()));s.ldc.push_back(int(a.scalar_rows()));s.alpha.push_back(1.0f);s.beta.push_back(slot?1.0f:0.0f);s.group_size.push_back(int(entry.second.size()));for(auto [i,p]:entry.second){int j=a.block_col[p];s.A.push_back(impl_->values+a.value_off[p]);s.b_off.push_back(a.col_scalar_off[j]);s.c_off.push_back(a.row_scalar_off[i]);}}
    cuda_check(cudaMalloc(&s.dA,s.A.size()*sizeof(float*)));cuda_check(cudaMalloc(&s.dB,s.A.size()*sizeof(float*)));cuda_check(cudaMalloc(&s.dC,s.A.size()*sizeof(float*)));cuda_check(cudaMemcpy(s.dA,s.A.data(),s.A.size()*sizeof(float*),cudaMemcpyHostToDevice));
  }
}

GroupedGemmPlan::~GroupedGemmPlan()=default;
int GroupedGemmPlan::launch_count()const{return int(impl_->slots.size());}
void GroupedGemmPlan::execute(const float* B,float* C,cudaStream_t stream){blas_check(cublasSetStream(impl_->handle,stream));cuda_check(cudaMemsetAsync(C,0,impl_->rows*impl_->rhs*sizeof(float),stream));for(auto& s:impl_->slots){std::vector<const float*> bp(s.A.size());std::vector<float*> cp(s.A.size());for(size_t i=0;i<bp.size();i++){bp[i]=B+s.b_off[i];cp[i]=C+s.c_off[i];}cuda_check(cudaMemcpyAsync(s.dB,bp.data(),bp.size()*sizeof(float*),cudaMemcpyHostToDevice,stream));cuda_check(cudaMemcpyAsync(s.dC,cp.data(),cp.size()*sizeof(float*),cudaMemcpyHostToDevice,stream));blas_check(cublasSgemmGroupedBatched(impl_->handle,s.ta.data(),s.tb.data(),s.m.data(),s.n.data(),s.k.data(),s.alpha.data(),s.dA,s.lda.data(),s.dB,s.ldb.data(),s.beta.data(),s.dC,s.ldc.data(),int(s.group_size.size()),s.group_size.data()));}}

void slot_split_baseline(const HostMatrix&a,const float*B,float*C,int rhs,cudaStream_t stream){GroupedGemmPlan p(a,rhs);p.execute(B,C,stream);cuda_check(cudaStreamSynchronize(stream));}
}
