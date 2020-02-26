#pragma once

#include <memory>
#include <cstdint>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include <Core/Cuda/Types.h>

/// TODO make some kind of simple cuda/host buffer object with once defined size and min alloc/free capabilities

namespace DB
{

/** Analog of ColumnString for GPU storage
  */
class CudaColumnString
{
public:
    CudaColumnString(size_t max_str_num_,size_t max_sz_);

    bool        empty()const { return (str_num == 0)||(sz == 0); }
    size_t      getStrNum()const { return str_num; }
    size_t      getBufSz()const { return sz; }
    char        *getBuf()const { return buf; }
    UInt32      *getLens()const { return lens; }
    UInt32      *getOffsets()const { return offsets; }
    UInt64      *getOffsets64()const { return offsets64; }
    void        setSize(size_t str_num_, size_t sz_);
    /// it is nonblocking wrt host method
    void        calcLengths(const std::vector<UInt32> &blocks_sizes, 
        const std::vector<UInt32> &blocks_buf_sizes, cudaStream_t stream = 0);

    ~CudaColumnString();
protected:
    //TODO decide about size_t here (simply take UInt32?)
    size_t      str_num, max_str_num;
    size_t      sz, max_sz;
    char        *buf;
    UInt32      *lens, *offsets;
    /// that is initial offsets
    UInt64      *offsets64;
    /// tmp_buf4 is used for exclusive scan
    //size_t      buf4_sz;
    //char        *tmp_buf4;
};

using CudaColumnStringPtr = std::shared_ptr<CudaColumnString>;


}
