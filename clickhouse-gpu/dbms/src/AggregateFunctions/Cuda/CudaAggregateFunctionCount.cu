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

#include <Common/Cuda/CudaAtomics.cuh>
#include <Common/Cuda/CudaSafeCall.h>

#include <AggregateFunctions/Cuda/CudaAggregateFunctionCount.h>

namespace DB
{

__global__ void  kerCudaAddBulkCount(CudaAggregateFunctionCountData *places, 
    ICudaAggregateFunction::CudaSizeType elements_num, 
    ICudaAggregateFunction::CudaSizeType *res_buckets)
{
    ICudaAggregateFunction::CudaSizeType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < elements_num)) return;
    ICudaAggregateFunction::CudaSizeType    res_bucket = res_buckets[i];
    cuda_details::atomicAdd(&(places[res_bucket].count), (UInt64)1);
}


__global__ void  kerCudaMergeBulkCount(CudaAggregateFunctionCountData *places, ICudaAggregateFunction::CudaSizeType elements_num,
    CudaAggregateFunctionCountData *places_from, ICudaAggregateFunction::CudaSizeType *res_buckets)
{
    ICudaAggregateFunction::CudaSizeType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < elements_num)) return;
    ICudaAggregateFunction::CudaSizeType    res_bucket = res_buckets[i];
    if (res_bucket == ~((ICudaAggregateFunction::CudaSizeType)0)) return;
    cuda_details::atomicAdd(&(places[res_bucket].count), places_from[i].count);
}


void CudaAggregateFunctionCount::cudaAddBulk(CudaAggregateDataPtr places, CudaColumnStringPtr str_column,
    CudaSizeType elements_num, CudaSizeType *res_buckets, 
    char *tmp_buf, cudaStream_t stream) const
{
    /// TODO Strange unused parameter error
    tmp_buf = tmp_buf; str_column = str_column;
    kerCudaAddBulkCount<<<(elements_num/256)+1,256,0,stream>>>(
        (CudaAggregateFunctionCountData*)places, elements_num, res_buckets);
}


void CudaAggregateFunctionCount::cudaMergeBulk(CudaAggregateDataPtr places, CudaSizeType elements_num,
    CudaAggregateDataPtr places_from, CudaSizeType *res_buckets, 
    cudaStream_t stream) const
{
    kerCudaMergeBulkCount<<<(elements_num/256)+1,256,0,stream>>>(
        (CudaAggregateFunctionCountData*)places, elements_num, 
        (CudaAggregateFunctionCountData*)places_from, res_buckets);
}

/*class CudaAggregateFunctionCount final : public ICudaAggregateFunction
{
public:

    ResultType  getResult(AggregateDataPtr place) const override
    {
        return ((CudaAggregateFunctionCountData*)place)->count;
    }
};*/


}
