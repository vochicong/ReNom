
cdef extern from "cuda_runtime.h":
    ctypedef enum cudaError_t:
        cudaError
    ctypedef enum cudaMemcpyKind:
        cudaMemcpyHostToHost,
        cudaMemcpyHostToDevice,
        cudaMemcpyDeviceToHost,
        cudaMemcpyDeviceToDevice
    ctypedef int cudaStream_t
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

    cudaError_t cudaMalloc(void ** ptr, size_t size)
    cudaError_t cudaMallocHost(void ** ptr, size_t size, unsigned int flags)
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


cdef extern from "cuda.h":
    ctypedef enum cudaError_enum:
        CUDA_SUCCESS,
    ctypedef cudaError_enum CUresult
    ctypedef int CUcontext
    ctypedef int CUdevice

    CUresult cuCtxCreate(CUcontext * pctx, unsigned int flags, CUdevice dev)
    CUresult cuCtxDestroy(CUcontext ctx)
    CUresult cuCtxGetDevice(CUdevice * device)
    CUresult cuGetErrorString(CUresult error, const char ** pStr)

