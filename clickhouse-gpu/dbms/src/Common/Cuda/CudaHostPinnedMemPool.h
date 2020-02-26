#pragma once

#include <cstddef>

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

public:
    CudaHostPinnedMemPool();

    ~CudaHostPinnedMemPool();

    void* alloc(std::size_t size, const std::size_t alignment = 0);

    void free(void* ptr);

    void init(const std::size_t totalSize);

    void reset();
private:
    //CudaHostPinnedMemPool(CudaHostPinnedMemPool &freeListAllocator);

    void coalescence(Node* prevBlock, Node * freeBlock);

    void find(const std::size_t size, const std::size_t alignment, std::size_t& padding, Node*& previousNode, Node*& foundNode);
};

