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

#include <string>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>

#include <Core/Cuda/Types.h>
#include <Columns/Cuda/CudaColumnString.h>

namespace DB
{

using CudaAggregateDataPtr = char*;
//using AggregateDataPtr = char*;

class ICudaAggregateFunction
{
public:
    typedef UInt32      CudaSizeType;
    typedef UInt64      ResultType;

    virtual size_t      cudaSizeOfData() const = 0;
    virtual bool        isDataNeeded() const = 0;
    virtual void        cudaInitAggregateData(CudaSizeType places_num, CudaAggregateDataPtr places, cudaStream_t stream = 0) const = 0;
    virtual size_t      cudaSizeOfAddBulkInternalBuf(CudaSizeType max_elements_num) = 0;
    virtual void        cudaAddBulk(CudaAggregateDataPtr places, CudaColumnStringPtr str_column,
                                    CudaSizeType elements_num, CudaSizeType *res_buckets, 
                                    char *tmp_buf, cudaStream_t stream = 0) const = 0;
    virtual void        cudaMergeBulk(CudaAggregateDataPtr places, CudaSizeType elements_num,
                                      CudaAggregateDataPtr places_from, CudaSizeType *res_buckets, 
                                      cudaStream_t stream = 0) const = 0;
    virtual ResultType  getResult(CudaAggregateDataPtr place) const = 0;

    virtual ~ICudaAggregateFunction() {}
};

using CudaAggregateFunctionPtr = std::shared_ptr<ICudaAggregateFunction>;


}
