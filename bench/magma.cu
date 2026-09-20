#include "magma.hpp"
#include <algorithm>
#include <climits>
#include <magma_v2.h>

extern "C" void magmablas_sgemm_vbatched_core(magma_trans_t, magma_trans_t, magma_int_t,
                                              magma_int_t, magma_int_t, magma_int_t*, magma_int_t*,
                                              magma_int_t*, float, const float* const*, magma_int_t,
                                              magma_int_t, magma_int_t*, const float* const*,
                                              magma_int_t, magma_int_t, magma_int_t*, float,
                                              float**, magma_int_t, magma_int_t, magma_int_t*,
                                              magma_int_t, magma_queue_t);

namespace vbsr::bench {
namespace {
__global__ void sum_products(DeviceMatrix a, const float* products, float* c, int rhs) {
    int row = blockIdx.x;
    int h = a.row_size[row];
    for (int x = threadIdx.x; x < h * rhs; x += blockDim.x) {
        int r = x % h, col = x / h;
        float value = 0;
        for (int j = a.row_ptr[row]; j < a.row_ptr[row + 1]; ++j)
            value += products[int64_t(j) * 64 * rhs + r + col * 64];
        c[a.row_scalar_off[row] + r + int64_t(col) * a.scalar_rows] = value;
    }
}
struct Batch {
    DeviceBuffer<int> m, n, k, lda, ldb, ldc;
    DeviceBuffer<const float*> a, b;
    DeviceBuffer<float*> c;
    int max_m{}, max_k{}, count{};
    explicit Batch(int count)
        : m(count), n(count), k(count), lda(count), ldb(count), ldc(count), a(count), b(count),
          c(count), count(count) {}
};
}
struct MagmaPlan::Impl {
    std::vector<std::unique_ptr<Batch>> batches;
    DeviceBuffer<float> products;
    DeviceMatrix matrix;
    float* output;
    int rhs;
    bool reduce, empty;
    cudaStream_t stream;
    magma_queue_t queue{};
    Impl(const HostMatrix& h, DeviceMatrix d, const float* b, float* c, int width, bool reduction,
         cudaStream_t s)
        : products(reduction ? size_t(d.nnzb) * 64 * width : 0), matrix(d), output(c), rhs(width),
          reduce(reduction), empty(false), stream(s) {
        h.validate();
        validate_panel_width(rhs);
        if (d.scalar_rows > INT_MAX || d.scalar_cols > INT_MAX)
            throw std::invalid_argument("MAGMA leading dimension exceeds int32");
        int degree = 0;
        std::vector<int> owner(d.nnzb);
        for (int r = 0; r < h.block_rows; ++r) {
            degree = std::max(degree, h.row_ptr[r + 1] - h.row_ptr[r]);
            empty |= h.row_ptr[r + 1] == h.row_ptr[r];
            for (int j = h.row_ptr[r]; j < h.row_ptr[r + 1]; ++j)
                owner[j] = r;
        }
        for (int slot = 0; slot < (reduce ? 1 : degree); ++slot) {
            std::vector<int> ids;
            if (reduce) {
                for (int j = 0; j < d.nnzb; ++j)
                    ids.push_back(j);
            } else
                for (int r = 0; r < h.block_rows; ++r)
                    if (h.row_ptr[r] + slot < h.row_ptr[r + 1])
                        ids.push_back(h.row_ptr[r] + slot);
            if (ids.empty())
                continue;
            auto batch = std::make_unique<Batch>(int(ids.size()));
            std::vector<int> m, n, k, lda, ldb, ldc;
            std::vector<const float*> ap, bp;
            std::vector<float*> cp;
            for (int j : ids) {
                int r = owner[j], col = h.block_col[j];
                m.push_back(h.row_size[r]);
                n.push_back(rhs);
                k.push_back(h.col_size[col]);
                lda.push_back(h.row_size[r]);
                ldb.push_back(int(d.scalar_cols));
                ldc.push_back(reduce ? 64 : int(d.scalar_rows));
                ap.push_back(d.values + h.value_off[j]);
                bp.push_back(b + h.col_scalar_off[col]);
                cp.push_back(reduce ? products.data() + int64_t(j) * 64 * rhs
                                    : c + h.row_scalar_off[r]);
                batch->max_m = std::max(batch->max_m, m.back());
                batch->max_k = std::max(batch->max_k, k.back());
            }
            batch->m.upload(m);
            batch->n.upload(n);
            batch->k.upload(k);
            batch->lda.upload(lda);
            batch->ldb.upload(ldb);
            batch->ldc.upload(ldc);
            batch->a.upload(ap);
            batch->b.upload(bp);
            batch->c.upload(cp);
            batches.push_back(std::move(batch));
        }

        check_cuda(cudaStreamSynchronize(nullptr));
        int device;
        check_cuda(cudaGetDevice(&device));
        magma_queue_create_from_cuda(device, stream, nullptr, nullptr, &queue);
        if (!queue)
            throw std::runtime_error("MAGMA queue construction failed");
    }
    ~Impl() {
        if (queue)
            magma_queue_destroy(queue);
    }
};
MagmaPlan::MagmaPlan(const HostMatrix& h, DeviceMatrix d, const float* b, float* c, int rhs,
                     bool reduce, cudaStream_t stream)
    : impl_(std::make_unique<Impl>(h, d, b, c, rhs, reduce, stream)) {}
MagmaPlan::~MagmaPlan() = default;
size_t MagmaPlan::workspace_bytes() const { return impl_->products.size() * sizeof(float); }
void MagmaPlan::execute() {
    auto& p = *impl_;
    if (!p.reduce && p.empty)
        check_cuda(cudaMemsetAsync(p.output, 0,
                                   size_t(p.matrix.scalar_rows) * p.rhs * sizeof(float), p.stream));
    for (size_t slot = 0; slot < p.batches.size(); ++slot) {
        auto& b = *p.batches[slot];

        magmablas_sgemm_vbatched_core(MagmaNoTrans, MagmaNoTrans, b.max_m, p.rhs, b.max_k,
                                      b.m.data(), b.n.data(), b.k.data(), 1.0f, b.a.data(), 0, 0,
                                      b.lda.data(), b.b.data(), 0, 0, b.ldb.data(),
                                      (p.reduce || slot == 0) ? 0.0f : 1.0f, b.c.data(), 0, 0,
                                      b.ldc.data(), b.count, p.queue);
    }
    if (p.reduce)
        sum_products<<<p.matrix.block_rows, 128, 0, p.stream>>>(p.matrix, p.products.data(),
                                                                p.output, p.rhs);
    check_cuda(cudaGetLastError());
}
}
