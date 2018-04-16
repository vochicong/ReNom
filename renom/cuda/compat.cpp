#include "compat.h"

cudnnStatus_t cudnnSetConvolution2dDescriptor_9(
        cudnnConvolutionDescriptor_t convDesc,
        int pad_h,    // zero-padding height
        int pad_w,    // zero-padding width
        int u,        // vertical filter stride
        int v,        // horizontal filter stride
        int upscalex,  // upscale the input in x-direction
        int upscaley,  // upscale the input in y-direction
        cudnnConvolutionMode_t mode,
        cudnnDataType_t dataType) {


#if CUDNN_MAJOR>=7

    return cudnnSetConvolution2dDescriptor(
            convDesc,
            pad_h,    // zero-padding height
            pad_w,    // zero-padding width
            u,        // vertical filter stride
            v,        // horizontal filter stride
            upscalex,  // upscale the input in x-direction
            upscaley,  // upscale the input in y-direction
            mode,
            dataType);
#else

    return cudnnSetConvolution2dDescriptor_v5(
            convDesc,
            pad_h,    // zero-padding height
            pad_w,    // zero-padding width
            u,        // vertical filter stride
            v,        // horizontal filter stride
            upscalex,  // upscale the input in x-direction
            upscaley,  // upscale the input in y-direction
            mode,
            dataType);
#endif



}
