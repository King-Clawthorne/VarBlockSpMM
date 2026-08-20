#include "varblockspmm/vbsr.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace vbsr {
template<int RHS>
__global__ void row_owned_scalar(DeviceMatrix a,const float* __restrict__ b,float* __restrict__ c) {
  const int i=blockIdx.x,r=a.row_size[i],stride=blockDim.x*gridDim.y;
  for(int t=blockIdx.y*blockDim.x+threadIdx.x;t<r*RHS;t+=stride) {
    const int x=t%r,y=t/r;float acc=0.0f;
    #pragma unroll 1
    for(int p=a.row_ptr[i];p<a.row_ptr[i+1];p++) {
      const int j=a.block_col[p],q=a.col_size[j];const float* av=a.values+a.value_off[p]+x;const float* bv=b+a.col_scalar_off[j]+int64_t(y)*a.scalar_cols;
      #pragma unroll 4
      for(int z=0;z<q;z++)acc=fmaf(av[z*r],bv[z],acc);
    }
    c[a.row_scalar_off[i]+x+int64_t(y)*a.scalar_rows]=acc;
  }
}

template<int RHS,int VEC>
__global__ void row_owned_ilp(DeviceMatrix a,const float* __restrict__ b,float* __restrict__ c) {
  const int i=blockIdx.x,r=a.row_size[i],groups=RHS/VEC,stride=blockDim.x*gridDim.y;
  for(int t=blockIdx.y*blockDim.x+threadIdx.x;t<r*groups;t+=stride) {
    const int x=t%r,y0=(t/r)*VEC;float acc[VEC]={};
    #pragma unroll 1
    for(int p=a.row_ptr[i];p<a.row_ptr[i+1];p++) {
      const int j=a.block_col[p],q=a.col_size[j];const float* av=a.values+a.value_off[p]+x;const float* bv=b+a.col_scalar_off[j]+int64_t(y0)*a.scalar_cols;
      #pragma unroll 4
      for(int z=0;z<q;z++){const float avalue=av[z*r];
        #pragma unroll
        for(int v=0;v<VEC;v++)acc[v]=fmaf(avalue,bv[z+int64_t(v)*a.scalar_cols],acc[v]);
      }
    }
    #pragma unroll
    for(int v=0;v<VEC;v++)c[a.row_scalar_off[i]+x+int64_t(y0+v)*a.scalar_rows]=acc[v];
  }
}

template<int RHS>static void launch_scalar(DeviceMatrix a,const float*b,float*c,cudaStream_t s){constexpr int threads=256;constexpr int y=(64*RHS+threads-1)/threads;row_owned_scalar<RHS><<<dim3(a.block_rows,y),threads,0,s>>>(a,b,c);}
template<int RHS>static void launch_ilp(DeviceMatrix a,const float*b,float*c,cudaStream_t s){constexpr int threads=256;constexpr int vec=4;constexpr int y=(64*(RHS/vec)+threads-1)/threads;row_owned_ilp<RHS,vec><<<dim3(a.block_rows,y),threads,0,s>>>(a,b,c);}
void launch_row_owned_scalar(DeviceMatrix a,const float*b,float*c,int n,cudaStream_t s){switch(n){case 8:launch_scalar<8>(a,b,c,s);break;case 16:launch_scalar<16>(a,b,c,s);break;case 32:launch_scalar<32>(a,b,c,s);break;case 64:launch_scalar<64>(a,b,c,s);break;default:throw std::invalid_argument("unsupported rhs width");}auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void launch_row_owned(DeviceMatrix a,const float*b,float*c,int n,cudaStream_t s){switch(n){case 8:launch_scalar<8>(a,b,c,s);break;case 16:launch_ilp<16>(a,b,c,s);break;case 32:launch_ilp<32>(a,b,c,s);break;case 64:launch_ilp<64>(a,b,c,s);break;default:throw std::invalid_argument("unsupported rhs width");}auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
}
