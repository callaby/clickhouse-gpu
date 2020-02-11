// Copyright 2016-2020 NVIDIA
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//        http://www.apache.org/licenses/LICENSE-2.0
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

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

    /** Enlarge memory range.
      * Data from old range is moved to the beginning of new range.
      * Address of memory range could change.
      */
    void * realloc(void * buf, size_t old_size, size_t new_size, size_t alignment = 8)
    {
        return CudaHostPinnedMemPool::instance().realloc(buf, old_size, new_size, alignment);
    }
};
