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

#include <memory>
#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <boost/noncopyable.hpp>

#include <Common/Cuda/CudaSafeCall.h>
#include <Common/Cuda/CudaHostPinnedAllocator.h>

//TODO make it non-copyable
template<class T, typename TAllocator = CudaHostPinnedAllocator>
class CudaHostPinnedArray : private boost::noncopyable, private TAllocator
{
public:
    typedef T   ValueType;

    CudaHostPinnedArray(size_t sz_);

    bool    empty()const { return sz == 0; }
    size_t  getSize()const { return sz; }
    size_t  getMemSize()const { return sz*sizeof(T); }
    T       *getData()const { return d; }
    T       &operator[](size_t i) { return d[i]; }
    const T &operator[](size_t i)const { return d[i]; }

    ~CudaHostPinnedArray();
protected:
    size_t  sz;
    T       *d;
};

template<class T, typename TAllocator>
CudaHostPinnedArray<T,TAllocator>::CudaHostPinnedArray(size_t sz_) : sz(sz_)
{
    d = (T*)TAllocator::alloc(sz*sizeof(T));
    //CUDA_SAFE_CALL( cudaMallocHost((void**)&d, sz*sizeof(T)) );
}

template<class T, typename TAllocator>
CudaHostPinnedArray<T,TAllocator>::~CudaHostPinnedArray()
{
    TAllocator::free(d, sz*sizeof(T));
    //CUDA_SAFE_CALL_NOTHROW( cudaFreeHost(d) );
}

template<class T, typename TAllocator = CudaHostPinnedAllocator>
using CudaHostPinnedArrayPtr = std::shared_ptr<CudaHostPinnedArray<T,TAllocator>>;
