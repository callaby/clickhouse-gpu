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
#include <boost/noncopyable.hpp>
#include <cuda.h>
#include <cuda_runtime.h>

#include <Core/Cuda/Types.h>
#include <Common/Cuda/CudaArray.h>
#include <Common/Cuda/CudaHostPinnedArray.h>

#define DBMS_CUDA_EMPTY_HASH_VAL  ((HashType)0)
#define DBMS_CUDA_EMPTY_LEN_VAL   ((SizeType)0)

class CudaStringsHashTable;
using CudaStringsHashTablePtr = std::shared_ptr<CudaStringsHashTable>;

class CudaStringsHashTable : private boost::noncopyable
{
public:
    typedef DB::UInt32      SizeType;
    typedef DB::UInt64      HashType;
    typedef char*           Pointer;

    CudaStringsHashTable(SizeType buckets_num_, SizeType str_buf_max_sz_/*, SizeType add_max_str_num_*/);
    void        erase(cudaStream_t stream = 0);
    void        addData(SizeType str_num, char *buf, SizeType *offsets, SizeType *lens,
                    SizeType *res_buckets, cudaStream_t stream = 0);
    SizeType    getBucketsNum()const { return buckets_num; }
    SizeType    getStrBufSz()const { return str_buf_sz; }
    char        *getStrBuf()const { return str_buf->getData(); }
    SizeType    *getLens()const { return lens->getData(); }
    SizeType    *getOffsets()const { return offsets->getData(); }
    void        calcOffsets(cudaStream_t stream = 0);
    void        mergeToOtherTable(CudaStringsHashTablePtr table, SizeType *res_buckets, cudaStream_t stream = 0);
private:
    SizeType                            buckets_num, str_buf_sz, str_buf_max_sz;
    CudaArrayPtr<HashType>              hashes;
    CudaArrayPtr<SizeType>              lens, offsets;
    /// Pointers to strings
    CudaArrayPtr<Pointer>               pointers;
    /// buffer to calculate offsets of newly added strings
    CudaArrayPtr<SizeType>              buf1 ,buf2;
    /// buf4 is for exclusive_scan temporal data when cub is used
    size_t                              buf4_sz;
    CudaArrayPtr<char>                  buf4;
    CudaArrayPtr<SizeType>              cuda_total_added_num;
    CudaHostPinnedArrayPtr<SizeType>    host_total_added_num;
    CudaHostPinnedArrayPtr<bool>        host_table_is_full_flag;
    CudaArrayPtr<char>                  str_buf;
};
