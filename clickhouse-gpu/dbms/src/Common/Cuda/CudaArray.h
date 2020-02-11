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
#include <boost/noncopyable.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#include <Common/Cuda/CudaSafeCall.h>

template<class T>
class CudaArray : private boost::noncopyable
{
public:
    typedef T   ValueType;

    CudaArray(size_t sz_);

    //ISSUE __device__?
    bool    empty()const { return sz == 0; }
    size_t  getSize()const { return sz; }
    size_t  getMemSize()const { return sz*sizeof(T); }
    T       *getData()const { return d; }

    ~CudaArray();
protected:
    size_t  sz;
    T       *d;
};

template<class T>
CudaArray<T>::CudaArray(size_t sz_) : sz(sz_)
{
    CUDA_SAFE_CALL( cudaMalloc((void**)&d, sz*sizeof(T)) );
}

template<class T>
CudaArray<T>::~CudaArray()
{
    CUDA_SAFE_CALL_NOTHROW( cudaFree(d) );
}

template<class T>
using CudaArrayPtr = std::shared_ptr<CudaArray<T>>;
