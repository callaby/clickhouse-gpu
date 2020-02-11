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

/// These functions can be overloaded for custom types.
namespace CudaZeroTraits
{

template <typename T>
__device__ __host__ bool check(const T x) { return x == 0; }

template <typename T>
__device__ __host__ void set(T & x) { x = 0; }

/// Returns 'sample' of zero object
template <typename T>
__device__ __host__ T    zero() { return (T)0; }

};