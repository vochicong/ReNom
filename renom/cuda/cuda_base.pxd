from libc.stdint cimport uintptr_t

ctypedef uintptr_t cudaStream_ptr

cdef extern from "cuda_runtime.h":
    ctypedef enum cudaError_t:
        cudaSuccess = 0,
        cudaErrorMissingConfiguration = 1
        cudaErrorMemoryAllocation = 2,
        cudaErrorInitializationError = 3

    ctypedef struct CUevent_st:
      pass
    ctypedef struct CUstream_st:
      pass

    ctypedef CUevent_st * cudaEvent_t
    ctypedef CUstream_st *  cudaStream_t

    ctypedef enum cudaMemcpyKind:
        cudaMemcpyHostToHost,
        cudaMemcpyHostToDevice,
        cudaMemcpyDeviceToHost,
        cudaMemcpyDeviceToDevice
    ctypedef int size_t
    ctypedef struct cudaDeviceProp:
        char * name,
        size_t totalGlobalMem,
        size_t sharedMemPerBlock,
        int regsPerBlock,
        int warpSize,
        size_t memPitch,
        int maxThreadsPerBlock,
        int maxThreadsDim[3],
        int maxGridSize[3],
        size_t totalConstMem,
        int major,
        int minor,
        int clockRate,
        size_t textureAlignment,
        int deviceOverlap,
        int multiProcessorCount,
        int kernelExecTimeoutEnabled,
        int integrated,
        int canMapHostMemory,
        int computeMode,
        int concurrentKernels,
        int ECCEnabled,
        int pciBusID,
        int pciDeviceID,
        int tccDriver,

    cudaError_t cudaMemset(void * devPtr, int value, size_t size)
    cudaError_t cudaMemcpy(void * dst, const void * src, int size, cudaMemcpyKind kind)
    cudaError_t cudaMemcpyAsync(void * dst, const void * src, int count, cudaMemcpyKind kind, cudaStream_t stream)
    cudaError_t cudaMemcpyPeer(void * dst, int dstDevice, const void * src, int srcDevice, size_t count)

    cudaError_t cudaHostAlloc(void ** ptr, size_t size, int flags)
    cudaError_t cudaMallocHost(void ** ptr, size_t size)
    cudaError_t cudaFreeHost(void * ptr)
    cudaError_t cudaHostRegister(void * ptr, size_t size, int flags)
    cudaError_t cudaHostUnregister(void *ptr)
    cudaError_t cudaEventCreate(cudaEvent_t * event)
    cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
    cudaError_t cudaEventSynchronize(cudaEvent_t)
    cudaError_t cudaMalloc(void ** ptr, size_t size)
    #cudaError_t cudaMallocHost(void ** ptr, size_t size, unsigned int flags)
    cudaError_t cudaSetDevice(int size)
    cudaError_t cudaGetDevice(int *device)
    cudaError_t cudaDeviceSynchronize()
    cudaError_t cudaFree(void * ptr)
    const char * cudaGetErrorString(cudaError_t erorr)

    cudaError_t cudaStreamCreate(cudaStream_t * pStream)
    cudaError_t cudaStreamDestroy(cudaStream_t stream)
    cudaError_t cudaStreamSynchronize(cudaStream_t stream)

    cudaError_t cudaGetDeviceProperties(cudaDeviceProp * prop, int device)
    cudaError_t cudaGetDeviceCount(int * count)
    cudaError_t cudaDeviceCanAccessPeer( int* canAccessPeer, int  device, int  peerDevice)


cdef extern from "cuda.h":
    ctypedef enum cudaError_enum:
        CUDA_SUCCESS,

    ctypedef enum CUlimit:
        CU_LIMIT_STACK_SIZE, CU_LIMIT_PRINTF_FIFO_SIZE, CU_LIMIT_MALLOC_HEAP_SIZE,
        CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH, CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT,
        CU_LIMIT_MAX

    ctypedef cudaError_enum CUresult
    ctypedef int CUcontext
    ctypedef int CUdevice

    CUresult cuCtxCreate(CUcontext * pctx, unsigned int flags, CUdevice dev)
    CUresult cuCtxDestroy(CUcontext ctx)
    CUresult cuCtxGetDevice(CUdevice * device)
    CUresult cuGetErrorString(CUresult error, const char ** pStr)

    CUresult cuCtxSetLimit (CUlimit limit, size_t value)
    CUresult cuCtxGetLimit (size_t *value, CUlimit limit)

    CUresult cuInit(unsigned int)


cdef extern from "nvToolsExtCudaRt.h":
  void nvtxNameCudaStreamA(cudaStream_t stream, const char* name)
