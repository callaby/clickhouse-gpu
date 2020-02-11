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

#include <Core/Cuda/Types.h>
#include <Common/Cuda/CudaArray.h>

#include <Columns/Cuda/CudaColumnString.h>

#include <AggregateFunctions/Cuda/ICudaAggregateFunction.h>

namespace DB
{

struct CudaAggregateFunctionCountData
{
    UInt64 count = 0;
};

//using AggregateFunctionCountData = CudaAggregateFunctionCountData;

class CudaAggregateFunctionCount final : public ICudaAggregateFunction
{
    typedef ICudaAggregateFunction::CudaSizeType        CudaSizeType;
    typedef ICudaAggregateFunction::ResultType          ResultType;

public:
    size_t      cudaSizeOfData() const override
    {
        return sizeof(CudaAggregateFunctionCountData);
    }
    bool        isDataNeeded() const override
    {
        return false;
    }
    /// TODO move to cu file
    void        cudaInitAggregateData(CudaSizeType places_num, CudaAggregateDataPtr places, cudaStream_t stream = 0) const override
    {
        stream = stream;    /// because of 'unused argument'
        CUDA_SAFE_CALL( cudaMemset( places, 0, sizeof(CudaAggregateFunctionCountData)*places_num ) );
    }
    size_t      cudaSizeOfAddBulkInternalBuf(CudaSizeType max_elements_num) override
    {
        max_elements_num = max_elements_num;
        return 0;
    }
    void        cudaAddBulk(CudaAggregateDataPtr places, CudaColumnStringPtr str_column,
                            CudaSizeType elements_num, CudaSizeType *res_buckets, 
                            char *tmp_buf, cudaStream_t stream = 0) const override;
    void        cudaMergeBulk(CudaAggregateDataPtr places, CudaSizeType elements_num,
                              CudaAggregateDataPtr places_from, CudaSizeType *res_buckets, 
                              cudaStream_t stream = 0) const override;

    ResultType  getResult(CudaAggregateDataPtr place) const override
    {
        return ((CudaAggregateFunctionCountData*)place)->count;
    }

    virtual ~CudaAggregateFunctionCount() override {}
};


}
