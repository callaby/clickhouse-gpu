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
#include <vector>

#include <Core/Cuda/Types.h>
#include <Common/Cuda/CudaHostPinnedArray.h>

namespace DB
{

//TODO make it non-copyable
class CudaHostStringsBuffer
{
public:
    CudaHostStringsBuffer(size_t max_str_num_, size_t max_sz_,
        bool has_lens_ = true, bool has_offsets_ = true, bool has_offsets64_ = true);

    bool                        empty()const { return (str_num == 0)||(sz == 0); }
    size_t                      getStrNum()const { return str_num; }
    size_t                      getBufSz()const { return sz; }
    size_t                      getMaxStrNum()const { return max_str_num; }
    size_t                      getBufMaxSz()const { return max_sz; }
    char                        *getBuf()const { return buf->getData(); }
    UInt32                      *getLens()const { return lens->getData(); }
    UInt32                      *getOffsets()const { return offsets->getData(); }
    UInt64                      *getOffsets64()const { return offsets64->getData(); }
    const std::vector<UInt32>   &getBlocksSizes()const { return blocks_sizes; }
    const std::vector<UInt32>   &getBlocksBufSizes()const { return blocks_buf_sizes; }
    bool                        hasSpace(size_t str_num_, size_t str_buf_sz_)const;
    void                        addData(size_t str_num_, size_t str_buf_sz_, 
                                        const char *str_buf_, const UInt64 *offsets_, 
                                        size_t memcpy_threads_num_ = 1);
    void                        setSize(size_t str_num_, size_t sz_);
    void                        reset();
protected:
    bool                                has_lens, has_offsets, has_offsets64;
    size_t                              str_num, max_str_num;
    size_t                              sz, max_sz;
    CudaHostPinnedArrayPtr<char>        buf;
    CudaHostPinnedArrayPtr<UInt32>      lens, offsets;
    CudaHostPinnedArrayPtr<UInt64>      offsets64;
    std::vector<UInt32>                 blocks_sizes, blocks_buf_sizes;
};

using CudaHostStringsBufferPtr = std::shared_ptr<CudaHostStringsBuffer>;

}

