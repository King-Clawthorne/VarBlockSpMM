#include "varblockspmm/vbsr.hpp"
#include <cusparse.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace vbsr {
static void cuda_check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void sparse_check(cusparseStatus_t e){if(e!=CUSPARSE_STATUS_SUCCESS)throw std::runtime_error("cuSPARSE failure: "+std::to_string(int(e)));}
struct ScalarCsrPlan::Impl {
  int rhs{}; int64_t rows{},cols{},nnz{}; int32_t *rp{},*ci{};float* values{};void* workspace{};size_t bytes{};
  cusparseHandle_t handle{};cusparseSpMatDescr_t mat{};cusparseDnMatDescr_t bd{},cd{};
  ~Impl(){cudaFree(workspace);if(bd)cusparseDestroyDnMat(bd);if(cd)cusparseDestroyDnMat(cd);if(mat)cusparseDestroySpMat(mat);if(handle)cusparseDestroy(handle);cudaFree(rp);cudaFree(ci);cudaFree(values);}
};
ScalarCsrPlan::ScalarCsrPlan(const HostMatrix&a,int rhs):impl_(new Impl){a.validate();if(rhs!=8&&rhs!=16&&rhs!=32&&rhs!=64)throw std::invalid_argument("invalid rhs width");impl_->rhs=rhs;impl_->rows=a.scalar_rows();impl_->cols=a.scalar_cols();std::vector<int32_t>rp(a.scalar_rows()+1),ci;std::vector<float>v;int32_t nz=0;for(int i=0;i<a.block_rows;i++)for(int x=0;x<a.row_size[i];x++){rp[a.row_scalar_off[i]+x]=nz;for(int p=a.row_ptr[i];p<a.row_ptr[i+1];p++){int j=a.block_col[p],q=a.col_size[j],r=a.row_size[i];for(int z=0;z<q;z++){ci.push_back(int32_t(a.col_scalar_off[j]+z));v.push_back(a.values[a.value_off[p]+x+z*r]);++nz;}}}rp.back()=nz;impl_->nnz=nz;cuda_check(cudaMalloc(&impl_->rp,rp.size()*sizeof(int32_t)));cuda_check(cudaMalloc(&impl_->ci,ci.size()*sizeof(int32_t)));cuda_check(cudaMalloc(&impl_->values,v.size()*sizeof(float)));cuda_check(cudaMemcpy(impl_->rp,rp.data(),rp.size()*sizeof(int32_t),cudaMemcpyHostToDevice));cuda_check(cudaMemcpy(impl_->ci,ci.data(),ci.size()*sizeof(int32_t),cudaMemcpyHostToDevice));cuda_check(cudaMemcpy(impl_->values,v.data(),v.size()*sizeof(float),cudaMemcpyHostToDevice));sparse_check(cusparseCreate(&impl_->handle));sparse_check(cusparseCreateCsr(&impl_->mat,impl_->rows,impl_->cols,nz,impl_->rp,impl_->ci,impl_->values,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_32F));}
ScalarCsrPlan::~ScalarCsrPlan()=default;
size_t ScalarCsrPlan::workspace_bytes()const{return impl_->bytes;}
void ScalarCsrPlan::execute(const float*B,float*C,cudaStream_t stream){sparse_check(cusparseSetStream(impl_->handle,stream));float one=1,zero=0;if(!impl_->bd){sparse_check(cusparseCreateDnMat(&impl_->bd,impl_->cols,impl_->rhs,impl_->cols,const_cast<float*>(B),CUDA_R_32F,CUSPARSE_ORDER_COL));sparse_check(cusparseCreateDnMat(&impl_->cd,impl_->rows,impl_->rhs,impl_->rows,C,CUDA_R_32F,CUSPARSE_ORDER_COL));size_t need=0;sparse_check(cusparseSpMM_bufferSize(impl_->handle,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,impl_->mat,impl_->bd,&zero,impl_->cd,CUDA_R_32F,CUSPARSE_SPMM_ALG_DEFAULT,&need));if(need){cuda_check(cudaMalloc(&impl_->workspace,need));impl_->bytes=need;}}else{sparse_check(cusparseDnMatSetValues(impl_->bd,const_cast<float*>(B)));sparse_check(cusparseDnMatSetValues(impl_->cd,C));}sparse_check(cusparseSpMM(impl_->handle,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,&one,impl_->mat,impl_->bd,&zero,impl_->cd,CUDA_R_32F,CUSPARSE_SPMM_ALG_DEFAULT,impl_->workspace));}
void cusparse_scalar_baseline(const HostMatrix&a,const float*B,float*C,int rhs,cudaStream_t stream){ScalarCsrPlan p(a,rhs);p.execute(B,C,stream);cuda_check(cudaStreamSynchronize(stream));}
}
