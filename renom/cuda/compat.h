#include <cudnn.h>

cudnnStatus_t cudnnSetConvolution2dDescriptor_9(
        cudnnConvolutionDescriptor_t convDesc,
        int pad_h,      // zero-padding height
        int pad_w,      // zero-padding width
        int u,          // vertical filter stride
        int v,          // horizontal filter stride
        int upscalex,   // upscale the input in x-direction
        int upscaley,   // upscale the input in y-direction
        cudnnConvolutionMode_t mode,
        cudnnDataType_t dataType);
