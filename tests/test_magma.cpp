#include "magma.hpp"
#include <magma_v2.h>
#include <iostream>
int main() {
  namespace b=vbsr::bench;
  try {
    if(magma_init()!=MAGMA_SUCCESS) throw std::runtime_error("MAGMA init failed");
    struct Stream {
      cudaStream_t value{};
      Stream() { b::check_cuda(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking)); }
      ~Stream() { cudaStreamDestroy(value); }
    } stream;
    for(int rhs:{8,16,32,64}) for(bool empty:{false,true}) {
      vbsr::HostMatrix h;
      h.block_rows=h.block_cols=8; h.row_ptr={0};h.value_off={0};
      h.row_scalar_off=h.col_scalar_off={0};
      for(int r=0;r<8;++r) {
        h.row_size.push_back((r+1)*8);h.col_size.push_back((r+1)*8);
        h.row_scalar_off.push_back(h.row_scalar_off.back()+(r+1)*8);
        h.col_scalar_off.push_back(h.col_scalar_off.back()+(r+1)*8);
      }
      for(int r=0;r<8;++r) {
        for(int c=0;c<8 && !(empty && r==0);++c) {
          h.block_col.push_back(c);h.value_off.push_back(h.value_off.back()+h.row_size[r]*h.col_size[c]);
        }
        h.row_ptr.push_back(int(h.block_col.size()));
      }
      h.values=b::make_input(size_t(h.value_off.back()));h.validate();
      vbsr::Matrix matrix(h);
      auto input=b::make_input(b::panel_elements(h.scalar_cols(),rhs));
      auto reference=vbsr::cpu_reference(h,input,rhs);
      b::DeviceBuffer<float> x(input.size()),y(reference.size());x.upload(input);
      b::check_cuda(cudaStreamSynchronize(nullptr));
      for(bool reduce:{false,true}) {
        b::check_cuda(cudaMemsetAsync(y.data(),0xff,y.size()*sizeof(float),stream.value));
        b::check_cuda(cudaStreamSynchronize(stream.value));
        b::MagmaPlan plan(h,matrix.device_view(),x.data(),y.data(),rhs,reduce,stream.value);
        plan.execute();b::check_cuda(cudaStreamSynchronize(stream.value));b::verify_output(reference,y.data());
      }
    }
    magma_finalize();std::cout<<"PASS: both MAGMA compositions, all 64 shapes, four RHS widths, empty rows\n";return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
