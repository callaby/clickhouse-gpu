#include <stdexcept>
#include <cassert>
//#include <cub/cub.cuh>

#include <Common/Cuda/CudaSafeCall.h>

#include <Columns/Cuda/CudaColumnString.h>


namespace DB
{

CudaColumnString::CudaColumnString(size_t max_str_num_,size_t max_sz_) : max_str_num(max_str_num_), max_sz(max_sz_)
{
    if ((max_str_num_ == 0)||(max_sz_ == 0)) throw std::logic_error("CudaColumnString: try to create zero size buffer");
    cudaError_t err;
    err = cudaMalloc( (void**)&buf, max_sz*sizeof(char) );
    if (err != cudaSuccess) throw std::runtime_error("CudaColumnString: failed to alloc cuda memory for strings");
    err = cudaMalloc( (void**)&lens, max_str_num*sizeof(UInt32) );
    if (err != cudaSuccess) {
        CUDA_SAFE_CALL_NOTHROW( cudaFree(buf) );
        throw std::runtime_error("CudaColumnString: failed to alloc cuda memory for lengths buffer");
    }
    err = cudaMalloc( (void**)&offsets, max_str_num*sizeof(UInt32) );
    if (err != cudaSuccess) {
        CUDA_SAFE_CALL_NOTHROW( cudaFree(buf) ); 
        CUDA_SAFE_CALL_NOTHROW( cudaFree(lens) );
        throw std::runtime_error("CudaColumnString: failed to alloc cuda memory for offsets buffer");
    }
    /*buf4_sz = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, buf4_sz, lens, offsets, max_str_num);
    err = cudaMalloc( (void**)&tmp_buf4, buf4_sz );*/
    err = cudaMalloc( (void**)&offsets64, max_str_num*sizeof(UInt64) );
    if (err != cudaSuccess) {
        CUDA_SAFE_CALL_NOTHROW( cudaFree(buf) ); 
        CUDA_SAFE_CALL_NOTHROW( cudaFree(lens) ); 
        CUDA_SAFE_CALL_NOTHROW( cudaFree(offsets) ); 
        //throw std::runtime_error("CudaColumnString: failed to alloc cuda memory for additional buffer");
        throw std::runtime_error("CudaColumnString: failed to alloc cuda memory for offsets64 buffer");
    }
}


void CudaColumnString::setSize(size_t str_num_, size_t sz_)
{
    str_num = str_num_; sz = sz_;
}

__global__ void kerCalcLengths(UInt32 block_begin, UInt32 block_size, 
    UInt32 block_offset, UInt32 block_buf_size, UInt64 *offsets64, 
    UInt32 *lens, UInt32 *offsets)
{
    UInt32  i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < block_size)) return;   
    UInt32  local_offset,
            local_offset_next = offsets64[block_begin + i];
    if (i > 0) 
        local_offset = offsets64[block_begin + i-1];
    else
        local_offset = 0;

    offsets[block_begin + i] = local_offset + block_offset;
    lens[block_begin + i] = local_offset_next - local_offset;
    //offsets[block_begin + i] = block_begin + i;
    //lens[block_begin + i] = 1;
}

void CudaColumnString::calcLengths(const std::vector<UInt32> &blocks_sizes, 
    const std::vector<UInt32> &blocks_buf_sizes, cudaStream_t stream)
{
    assert(blocks_sizes.size() == blocks_buf_sizes.size());
    UInt32  block_begin = 0, block_offset = 0;
    for (size_t i = 0;i < blocks_sizes.size();++i) 
    {
        kerCalcLengths<<<(blocks_sizes[i]/256)+1,256,0,stream>>>(
            block_begin, blocks_sizes[i], block_offset, blocks_buf_sizes[i], 
            offsets64, lens, offsets);
        block_begin += blocks_sizes[i]; 
        block_offset += blocks_buf_sizes[i];
    }
}


CudaColumnString::~CudaColumnString()
{
    CUDA_SAFE_CALL_NOTHROW( cudaFree(buf) );
    CUDA_SAFE_CALL_NOTHROW( cudaFree(lens) );
    CUDA_SAFE_CALL_NOTHROW( cudaFree(offsets) );
    CUDA_SAFE_CALL_NOTHROW( cudaFree(offsets64) );
    //CUDA_SAFE_CALL_NOTHROW( cudaFree(tmp_buf4) );
}

}
