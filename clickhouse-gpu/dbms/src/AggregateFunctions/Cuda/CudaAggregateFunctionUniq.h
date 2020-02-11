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
#include <Common/Cuda/CudaHyperLogLogWithSmallSetOptimization.h>

#include <Columns/Cuda/CudaColumnString.h>

#include <AggregateFunctions/Cuda/ICudaAggregateFunction.h>


namespace DB
{

/// the only supported 'type'(T) is String

struct CudaAggregateFunctionUniqHLL12Data
{
    /// temporaly without SmallSet correction
    //using Set = CudaHyperLogLogCounter<12>;
    using Set = CudaHyperLogLogWithSmallSetOptimization<UInt64, 16, 12>;
    Set set;

    __device__ __host__ void initNonzeroData()
    {
        set.initNonzeroData();
    }
    __device__ __host__ CudaAggregateFunctionUniqHLL12Data()
    {
    }
};

//using AggregateFunctionUniqHLL12Data = CudaAggregateFunctionUniqHLL12Data;

/// The only supported Data here is CudaAggregateFunctionUniqHLL12Data
template <typename T, typename Data>
class CudaAggregateFunctionUniq final : public ICudaAggregateFunction
{
    typedef ICudaAggregateFunction::CudaSizeType        CudaSizeType;
public:
    size_t  cudaSizeOfData() const override;
    void    cudaAddBulk(CudaAggregateDataPtr places, const CudaColumnString *str_column,
        CudaSizeType elements_num, CudaSizeType *res_buckets, cudaStream_t stream = 0) const override;

    virtual ~CudaAggregateFunctionUniq() override {}
private:
    //static_assert();
};

/// The only supported Data here is CudaAggregateFunctionUniqHLL12Data
template <>
class CudaAggregateFunctionUniq<String, CudaAggregateFunctionUniqHLL12Data> final : public ICudaAggregateFunction
{
    typedef ICudaAggregateFunction::CudaSizeType        CudaSizeType;
    typedef ICudaAggregateFunction::ResultType          ResultType;

public:
    size_t      cudaSizeOfData() const override
    {
        return sizeof(CudaAggregateFunctionUniqHLL12Data);
    }
    bool        isDataNeeded() const override
    {
        return true;
    }
    void        cudaInitAggregateData(CudaSizeType places_num, CudaAggregateDataPtr places, cudaStream_t stream = 0) const override;
    size_t      cudaSizeOfAddBulkInternalBuf(CudaSizeType max_elements_num) override
    {
        return sizeof(UInt64)*max_elements_num;
        //hashes = CudaArrayPtr<UInt64>(new CudaArray<UInt64>(max_elements_num));
    }
    void        cudaAddBulk(CudaAggregateDataPtr places, CudaColumnStringPtr str_column,
                            CudaSizeType elements_num, CudaSizeType *res_buckets, 
                            char *tmp_buf, cudaStream_t stream = 0) const override;
    void        cudaMergeBulk(CudaAggregateDataPtr places, CudaSizeType elements_num,
                              CudaAggregateDataPtr places_from, CudaSizeType *res_buckets, 
                              cudaStream_t stream = 0) const override;

    ResultType  getResult(CudaAggregateDataPtr place) const override
    {
        return ((CudaAggregateFunctionUniqHLL12Data*)place)->set.size();
    }

    virtual ~CudaAggregateFunctionUniq() override {}
};


}
