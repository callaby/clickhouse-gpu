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
#include <Common/Cuda/cudaMurmurHash64.cuh>

template<unsigned int seed = 1>
struct CudaStringMurmurHash64
{
    typedef     DB::UInt64    result_type;

    result_type     operator()(const char *s, DB::UInt32 len)const
    {
        return cudaMurmurHash64(s, len, seed);
    }
};