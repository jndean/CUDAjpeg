#ifndef PIXELTRANSFORMGPU_H
#define PIXELTRANSFORMGPU_H


#define W1 2841
#define W2 2676
#define W3 2408
#define W5 1609
#define W6 1108
#define W7 565


__global__ void iDCT_GPU(int* in,
			 unsigned char *out,
			 int stride,
			 int samples_x, int samples_y,
			 int num_DCT_blocks);
/*__host__ void upsampleChannel(JPGReader* jpg, ColourChannel* channel);
  __host__ void upsampleAndColourTransform(JPGReader* jpg);*/


#endif // PIXELTRANSFORMGPU_H //
