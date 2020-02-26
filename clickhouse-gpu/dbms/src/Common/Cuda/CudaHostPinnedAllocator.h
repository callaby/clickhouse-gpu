#pragma once

#include <Common/Cuda/CudaHostPinnedMemPool.h>

class CudaHostPinnedAllocator
{
public:
    /// Allocate memory range.
    void * alloc(size_t size, size_t alignment = 8)
    {
        return CudaHostPinnedMemPool::instance().alloc(size, alignment);
    }

    /// Free memory range.
    void free(void * buf, size_t size)
    {
        CudaHostPinnedMemPool::instance().free(buf);
    }
};
