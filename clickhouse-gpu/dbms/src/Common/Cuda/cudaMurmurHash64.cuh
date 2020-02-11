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
#include <Common/Cuda/cudaReadUnaligned.cuh>

inline __device__ DB::UInt64    cudaMurmurHash64(const char *s, DB::UInt32 len, unsigned int seed)
{
    const DB::UInt64 m = 0xc6a4a7935bd1e995;
    const int r = 47;

    DB::UInt64 h = seed ^ (len * m);

    const char      *data = s;
    const char      *end = data + len;
    bool            is_first_read = true;
    DB::UInt64        tmp_buf;

    while(data != end)
    {
        bool     is_complete_uint64 = (end - data >= 8);
        DB::UInt64 k = cudaReadStringUnaligned64(is_first_read, tmp_buf, data, end);

        if (is_complete_uint64) {
            k *= m; 
            k ^= k >> r; 
            k *= m; 
        }
        
        h ^= k;
        h *= m; 
    }
 
    h ^= h >> r;
    h *= m;
    h ^= h >> r;

    return h;
}