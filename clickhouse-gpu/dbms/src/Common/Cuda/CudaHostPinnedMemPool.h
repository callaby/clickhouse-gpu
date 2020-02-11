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

#include <cstddef>
#include <mutex>

#include <ext/singleton.h>

#include <Common/Cuda/SinglyLinkedList.h>
#include <Common/Cuda/CudaSafeCall.h>

class CudaHostPinnedMemPool final : public ext::singleton<CudaHostPinnedMemPool>
{
private:
    struct FreeHeader 
    {
        std::size_t blockSize;
    };
    struct AllocationHeader 
    {
        std::size_t blockSize;
        char padding;
    };
    
    typedef SinglyLinkedList<FreeHeader>::Node Node;

    std::size_t m_totalSize;
    std::size_t m_used;   
    std::size_t m_peak;
    
    void* m_start_ptr = nullptr;
    SinglyLinkedList<FreeHeader> m_freeList;

    /// protects whole structure during alloc free realloc
    std::mutex  mtx;
public:
    CudaHostPinnedMemPool();

    ~CudaHostPinnedMemPool();

    void *alloc(std::size_t size, std::size_t alignment = 8);
    void free(void* ptr);
    void *realloc(void * buf, size_t old_size, size_t new_size, size_t alignment = 8);

    void init(std::size_t totalSize);
    void reset();
private:
    //CudaHostPinnedMemPool(CudaHostPinnedMemPool &freeListAllocator);

    void coalescence(Node* prevBlock, Node * freeBlock);

    void find(std::size_t size, std::size_t alignment, std::size_t& padding, Node*& previousNode, Node*& foundNode);
};

