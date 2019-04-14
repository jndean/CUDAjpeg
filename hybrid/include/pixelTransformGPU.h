#ifndef PIXELTRANSFORMGPU_H
#define PIXELTRANSFORMGPU_H


#define W1 2841
#define W2 2676
#define W3 2408
#define W5 1609
#define W6 1108
#define W7 565


__global__ void iDCT_rows_GPU(int* D, int num_DCT_blocks);
__global__ void iDCT_cols_GPU(const int* D, unsigned char *out, int stride);
/*__host__ void upsampleChannel(JPG* jpg, ColourChannel* channel);
  __host__ void upsampleAndColourTransform(JPG* jpg);*/


#endif // PIXELTRANSFORMGPU_H //
