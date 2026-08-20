#include "varblockspmm/vbsr.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace vbsr {
static void ck(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void HostMatrix::validate() const {
  if(block_rows<=0 || block_cols<=0) throw std::invalid_argument("positive block dimensions required");
  if(row_ptr.size()!=size_t(block_rows+1) || row_size.size()!=size_t(block_rows) || col_size.size()!=size_t(block_cols)) throw std::invalid_argument("metadata size mismatch");
  if(row_scalar_off.size()!=size_t(block_rows+1) || col_scalar_off.size()!=size_t(block_cols+1)) throw std::invalid_argument("scalar offsets size mismatch");
  if(row_ptr.front()!=0 || row_ptr.back()<0 || size_t(row_ptr.back())!=block_col.size()) throw std::invalid_argument("bad row_ptr");
  if(value_off.size()!=block_col.size()+1 || value_off.front()!=0 || size_t(value_off.back())!=values.size()) throw std::invalid_argument("bad value offsets");
  for(int i=0;i<block_rows;i++) if(row_ptr[i]>row_ptr[i+1] || row_size[i]<=0 || row_scalar_off[i+1]-row_scalar_off[i]!=row_size[i]) throw std::invalid_argument("bad block row");
  for(int j=0;j<block_cols;j++) if(col_size[j]<=0 || col_scalar_off[j+1]-col_scalar_off[j]!=col_size[j]) throw std::invalid_argument("bad block column");
  for(size_t k=0;k<block_col.size();k++) { int j=block_col[k]; if(j<0||j>=block_cols) throw std::invalid_argument("block column out of range"); int i=0; while(row_ptr[i+1]<=int(k)) ++i; if(value_off[k+1]-value_off[k]!=int64_t(row_size[i])*col_size[j]) throw std::invalid_argument("block payload mismatch"); }
}
template<class T> static T* copy(const std::vector<T>& v){ T* p=nullptr; ck(cudaMalloc(&p,v.size()*sizeof(T))); ck(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice)); return p; }
Matrix::Matrix(const HostMatrix& h){ h.validate(); block_rows_=h.block_rows; block_cols_=h.block_cols; nnzb_=int(h.block_col.size()); rows_=h.scalar_rows(); cols_=h.scalar_cols(); row_ptr_=copy(h.row_ptr); block_col_=copy(h.block_col); row_size_=copy(h.row_size); col_size_=copy(h.col_size); row_off_=copy(h.row_scalar_off); col_off_=copy(h.col_scalar_off); value_off_=copy(h.value_off); values_=copy(h.values); }
void Matrix::release(){ cudaFree(row_ptr_);cudaFree(block_col_);cudaFree(row_size_);cudaFree(col_size_);cudaFree(row_off_);cudaFree(col_off_);cudaFree(value_off_);cudaFree(values_); row_ptr_=nullptr; }
Matrix::~Matrix(){ release(); }
Matrix::Matrix(Matrix&& o) noexcept { *this=std::move(o); }
Matrix& Matrix::operator=(Matrix&& o) noexcept { if(this!=&o){release(); block_rows_=o.block_rows_;block_cols_=o.block_cols_;nnzb_=o.nnzb_;rows_=o.rows_;cols_=o.cols_;row_ptr_=o.row_ptr_;block_col_=o.block_col_;row_size_=o.row_size_;col_size_=o.col_size_;row_off_=o.row_off_;col_off_=o.col_off_;value_off_=o.value_off_;values_=o.values_;o.row_ptr_=nullptr;o.block_col_=nullptr;o.row_size_=nullptr;o.col_size_=nullptr;o.row_off_=nullptr;o.col_off_=nullptr;o.value_off_=nullptr;o.values_=nullptr;} return *this; }
DeviceMatrix Matrix::device_view() const { return {block_rows_,block_cols_,nnzb_,rows_,cols_,row_ptr_,block_col_,row_size_,col_size_,row_off_,col_off_,value_off_,values_}; }
std::vector<float> cpu_reference(const HostMatrix& a,const std::vector<float>& b,int n){ if(b.size()!=size_t(a.scalar_cols()*n)) throw std::invalid_argument("B size"); std::vector<float> c(a.scalar_rows()*n,0); for(int i=0;i<a.block_rows;i++) for(int k=a.row_ptr[i];k<a.row_ptr[i+1];k++){int j=a.block_col[k], r=a.row_size[i], q=a.col_size[j]; for(int x=0;x<r;x++) for(int y=0;y<n;y++){double s=0;for(int z=0;z<q;z++)s+=double(a.values[a.value_off[k]+x+z*r])*b[(a.col_scalar_off[j]+z)+int64_t(y)*a.scalar_cols()];c[(a.row_scalar_off[i]+x)+int64_t(y)*a.scalar_rows()]+=float(s);}} return c; }
}
