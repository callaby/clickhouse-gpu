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

#include <iostream>
#include <stdexcept>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

#define __STR_HELPER(x) #x
#define __STR(x) __STR_HELPER(x)

#define CUDA_SAFE_CALL_E(X, EXCEPTION) if ((X) != cudaSuccess) throw EXCEPTION
#define CUDA_SAFE_CALL(X)                                                                                                                                                                       \
    do {                                                                                                                                                                                        \
        cudaError_t cuda_res = (X);                                                                                                                                                             \
        if (cuda_res != cudaSuccess) throw std::runtime_error(std::string("CUDA_SAFE_CALL " __FILE__ " " __STR(__LINE__) " : " #X " failed: ") + std::string(cudaGetErrorString(cuda_res)));    \
    } while (0)

#define CUDA_SAFE_CALL_NOTHROW(X)                                                                   \
    do {                                                                                            \
        cudaError_t cuda_res = (X);                                                                 \
        if (cuda_res != cudaSuccess) std::cout <<                                                   \
            std::string("CUDA_SAFE_CALL " __FILE__ " " __STR(__LINE__) " : " #X " failed: ") <<     \
            std::string(cudaGetErrorString(cuda_res));                                              \
    } while (0)
