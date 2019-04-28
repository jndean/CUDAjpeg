#ifndef PIXELTRANSFORMGPU_H
#define PIXELTRANSFORMGPU_H

#include<format.h>

__global__ void iDCT_GPU(int* in,
			 unsigned char *out,
			 int stride,
			 int samples_x, int samples_y,
			 int num_DCT_blocks);
__global__ void iDCT_GPU_warp_shuffle(int* in,
			 unsigned char *out,
			 int stride,
			 int samples_x, int samples_y,
			 int num_DCT_blocks);
__host__ void upsampleChannelGPU(JPGReader* jpg, ColourChannel* channel);
__host__ void upsampleAndColourTransformGPU(JPGReader* jpg);


#endif // PIXELTRANSFORMGPU_H //
