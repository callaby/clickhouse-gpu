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
//#include <Common/Cuda/City_Hash/city.h>
#include <Common/Cuda/cudaCalcMurmurHash64.h>
#include <Common/Cuda/cudaCalcCityHash64.h>

#include <AggregateFunctions/Cuda/CudaAggregateFunctionUniq.h>


namespace DB
{

/// the only supported 'type'(T) is String

__global__ void  kerCudaInitAggregateData(ICudaAggregateFunction::CudaSizeType places_num,
    CudaAggregateFunctionUniqHLL12Data *places)
{
    ICudaAggregateFunction::CudaSizeType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < places_num)) return;
    (places + i)->initNonzeroData();
    //new (places + i) CudaAggregateFunctionUniqHLL12Data();
}

__global__ void  kerCudaAddBulk(CudaAggregateFunctionUniqHLL12Data *places, 
        ICudaAggregateFunction::CudaSizeType elements_num, const UInt64 *hashes, 
        ICudaAggregateFunction::CudaSizeType *res_buckets)
{
    ICudaAggregateFunction::CudaSizeType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < elements_num)) return;
    UInt64                              hash = hashes[i];
    ICudaAggregateFunction::CudaSizeType    res_bucket = res_buckets[i];
    places[res_bucket].set.insert(hash);
}

__global__ void  kerCudaMergeBulk(CudaAggregateFunctionUniqHLL12Data *places, ICudaAggregateFunction::CudaSizeType elements_num,
        CudaAggregateFunctionUniqHLL12Data *places_from, ICudaAggregateFunction::CudaSizeType *res_buckets)
{
    ICudaAggregateFunction::CudaSizeType i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < elements_num)) return;
    ICudaAggregateFunction::CudaSizeType    res_bucket = res_buckets[i];
    if (res_bucket == ~((ICudaAggregateFunction::CudaSizeType)0)) return;
    places[res_bucket].set.merge(places_from[i].set);
}

/// The only supported Data here is CudaAggregateFunctionUniqHLL12Data

void CudaAggregateFunctionUniq<String, CudaAggregateFunctionUniqHLL12Data>::cudaInitAggregateData(
    CudaSizeType places_num, CudaAggregateDataPtr places, cudaStream_t stream) const
{
    CUDA_SAFE_CALL( cudaMemset( places, 0, sizeof(CudaAggregateFunctionUniqHLL12Data)*places_num ) );
    kerCudaInitAggregateData<<<(places_num/256)+1,256,0,stream>>>(places_num, (CudaAggregateFunctionUniqHLL12Data*)places);
}

void CudaAggregateFunctionUniq<String, CudaAggregateFunctionUniqHLL12Data>::cudaAddBulk(
    CudaAggregateDataPtr places, CudaColumnStringPtr str_column,
    CudaSizeType elements_num, CudaSizeType *res_buckets, 
    char *tmp_buf, cudaStream_t stream) const
{
    cudaCalcCityHash64(elements_num, str_column->getBuf(), false, str_column->getLens(), 
        str_column->getOffsets(), (UInt64*)tmp_buf, stream);

    kerCudaAddBulk<<<(elements_num/256)+1,256,0,stream>>>(
        (CudaAggregateFunctionUniqHLL12Data*)places, elements_num, (UInt64*)tmp_buf, res_buckets);
}

void CudaAggregateFunctionUniq<String, CudaAggregateFunctionUniqHLL12Data>::cudaMergeBulk(
    CudaAggregateDataPtr places, CudaSizeType elements_num,
    CudaAggregateDataPtr places_from, CudaSizeType *res_buckets, 
    cudaStream_t stream) const
{
    kerCudaMergeBulk<<<(elements_num/256)+1,256,0,stream>>>(
        (CudaAggregateFunctionUniqHLL12Data*)places, elements_num, 
        (CudaAggregateFunctionUniqHLL12Data*)places_from, res_buckets);
}

}
