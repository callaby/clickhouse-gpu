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


#include <cstdio>

#include "cudaReadUnaligned.cuh"
#include "cudaMurmurHash64.cuh"
#include "cudaCalcMurmurHash64.h"

__global__ void kerCalcHash(DB::UInt32 str_num, char *arr, DB::UInt32 *begs, bool interpret_as_lengths, DB::UInt32 *lens, unsigned int seed, DB::UInt64 *res_hash)
{
    DB::UInt32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (!(i < str_num)) return;

    DB::UInt32 len = lens[i], beg = begs[i];
    if (!interpret_as_lengths) --len;

    DB::UInt64 h = cudaMurmurHash64(&(arr[beg]), len, seed);

    /// TODO make it optional
    if (h == 0xFFFFFFFFFFFFFFFF) h = 0x0000000000000000;

    res_hash[i] = h;
}

void cudaCalcMurmurHash64(DB::UInt32 str_num, char *buf, bool interpret_as_lengths, DB::UInt32 *lens, DB::UInt32 *offsets, unsigned int seed, DB::UInt64 *res_hash, cudaStream_t stream)
{
    kerCalcHash<<<(str_num/256)+1,256,0,stream>>>(str_num, buf, offsets, interpret_as_lengths, lens, seed, res_hash);
}