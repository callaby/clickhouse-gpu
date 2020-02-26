#include <Common/Cuda/cudaInitDevice.h>
#include <Common/Cuda/CudaHostPinnedMemPool.h>

void cudaInitDevice(int dev_number, size_t pinned_pool_size)
{   
    printf("cudaInitDevice: dev_number = %d, pinned_pool_size = %d", dev_number, pinned_pool_size);
    fflush(stdout);
    CUDA_SAFE_CALL(cudaSetDevice(dev_number));
    CudaHostPinnedMemPool::instance().init(pinned_pool_size);
}
