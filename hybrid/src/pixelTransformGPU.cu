#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<cuda_runtime.h>

#include<pixelTransformGPU.h>
#include<format.h>


__device__ inline unsigned char clip(const int x) {
  return (x < 0) ? 0 : ((x > 0xFF) ? 0xFF : (unsigned char) x);
}


__global__ void iDCT_GPU(int* in,
			 unsigned char *out,
			 int stride,
			 int samples_x, int samples_y,
			 int num_DCT_blocks){
  
   // 8 DCT blocks per thread block, 8 threads per DCT block //
  int block_index = (blockIdx.x << 3) + (threadIdx.x >> 3);
  if (block_index >= num_DCT_blocks) return;
  int thread_in_block = threadIdx.x & 7;
  in += (block_index << 6) + (thread_in_block << 3);

  // --------------- Do a single row in the DCT block --------------- //
  int x0, x1, x2, x3, x4, x5, x6, x7, x8;
  if (!((x1 = in[4] << 11)
        | (x2 = in[6])
        | (x3 = in[2])
        | (x4 = in[1])
        | (x5 = in[7])
        | (x6 = in[5])
        | (x7 = in[3]))){
    
    in[0] = in[1] = in[2] = in[3] = in[4] = in[5] = in[6] = in[7] = in[0] << 3;

  } else{
    x0 = (in[0] << 11) + 128;
    x8 = W7 * (x4 + x5);
    x4 = x8 + (W1 - W7) * x4;
    x5 = x8 - (W1 + W7) * x5;
    x8 = W3 * (x6 + x7);
    x6 = x8 - (W3 - W5) * x6;
    x7 = x8 - (W3 + W5) * x7;
    x8 = x0 + x1;
    x0 -= x1;
    x1 = W6 * (x3 + x2);
    x2 = x1 - (W2 + W6) * x2;
    x3 = x1 + (W2 - W6) * x3;
    x1 = x4 + x6;
    x4 -= x6;
    x6 = x5 + x7;
    x5 -= x7;
    x7 = x8 + x3;
    x8 -= x3;
    x3 = x0 + x2;
    x0 -= x2;
    x2 = (181 * (x4 + x5) + 128) >> 8;
    x4 = (181 * (x4 - x5) + 128) >> 8;
    in[0] = (x7 + x1) >> 8;
    in[1] = (x3 + x2) >> 8;
    in[2] = (x0 + x4) >> 8;
    in[3] = (x8 + x6) >> 8;
    in[4] = (x8 - x6) >> 8;
    in[5] = (x0 - x4) >> 8;
    in[6] = (x3 - x2) >> 8;
    in[7] = (x7 - x1) >> 8;
  }

  // Make sure other rows within this DCT block are finished //
  // TODO: move fo cooperative group per DCT block rather than syncing a whole warp? //
  __syncwarp();
  
  // -------------------- Do a single column --------------------//
  in -= thread_in_block * 7;
  
  int blocks_per_outer_block = samples_x * samples_y;
  int blocks_per_row = samples_y * (stride >> 3); 
  int outer_block_y = block_index / blocks_per_row;
  int remaining_blocks = block_index % blocks_per_row;
  int outer_block_x = remaining_blocks / blocks_per_outer_block;
  remaining_blocks = remaining_blocks % blocks_per_outer_block;
  int inner_block_y = remaining_blocks / samples_x;
  int inner_block_x = remaining_blocks % samples_x;
  int block_x = outer_block_x * samples_x + inner_block_x;
  int block_y = outer_block_y * samples_y + inner_block_y;
  out += (block_y * stride + block_x) << 3;
  out += thread_in_block;
  
    
  if (!((x1 = in[8*4] << 8)
        | (x2 = in[8*6])
        | (x3 = in[8*2])
        | (x4 = in[8*1])
        | (x5 = in[8*7])
        | (x6 = in[8*5])
        | (x7 = in[8*3]))){
    x1 = clip(((in[0] + 32) >> 6) + 128);
    for (x0 = 8;  x0;  --x0) {
      *out = (unsigned char) x1;
      out += stride;
    }
    return;
  }
  
  x0 = (in[0] << 8) + 8192;
  x8 = W7 * (x4 + x5) + 4;
  x4 = (x8 + (W1 - W7) * x4) >> 3;
  x5 = (x8 - (W1 + W7) * x5) >> 3;
  x8 = W3 * (x6 + x7) + 4;
  x6 = (x8 - (W3 - W5) * x6) >> 3;
  x7 = (x8 - (W3 + W5) * x7) >> 3;
  x8 = x0 + x1;
  x0 -= x1;
  x1 = W6 * (x3 + x2) + 4;
  x2 = (x1 - (W2 + W6) * x2) >> 3;
  x3 = (x1 + (W2 - W6) * x3) >> 3;
  x1 = x4 + x6;
  x4 -= x6;
  x6 = x5 + x7;
  x5 -= x7;
  x7 = x8 + x3;
  x8 -= x3;
  x3 = x0 + x2;
  x0 -= x2;
  x2 = (181 * (x4 + x5) + 128) >> 8;
  x4 = (181 * (x4 - x5) + 128) >> 8;
  *out = clip(((x7 + x1) >> 14) + 128);  out += stride;
  *out = clip(((x3 + x2) >> 14) + 128);  out += stride;
  *out = clip(((x0 + x4) >> 14) + 128);  out += stride;
  *out = clip(((x8 + x6) >> 14) + 128);  out += stride;
  *out = clip(((x8 - x6) >> 14) + 128);  out += stride;
  *out = clip(((x0 - x4) >> 14) + 128);  out += stride;
  *out = clip(((x3 - x2) >> 14) + 128);  out += stride;
  *out = clip(((x7 - x1) >> 14) + 128);
}

/*
__host__ void upsampleChannel(JPG* jpg, ColourChannel* channel) {
    int x, y, xshift = 0, yshift = 0;
    unsigned char *out, *lout;
    while (channel->width < jpg->width) { channel->width <<= 1; ++xshift; }
    while (channel->height < jpg->height) { channel->height <<= 1; ++yshift; }
    out = (unsigned char*) malloc(channel->width * channel->height);
    if (!out) THROW(OOM_ERROR);
    
    for (y = 0, lout = out;  y < channel->height;  ++y, lout += channel->width) {
        unsigned char *lin = &channel->pixels[(y >> yshift) * channel->stride];
        for (x = 0;  x < channel->width;  ++x)
            lout[x] = lin[x >> xshift];
    }
    
    channel->stride = channel->width;
    free(channel->pixels);
    channel->pixels = out;
}


__host__ void upsampleAndColourTransform(JPG* jpg) {
  int i;
  ColourChannel* channel;
  for (i = 0, channel = &jpg->channels[0];  i < jpg->num_channels;  ++i, ++channel) {
    if ((channel->width < jpg->width) || (channel->height < jpg->height))
      upsampleChannel(jpg, channel);
    if ((channel->width < jpg->width) || (channel->height < jpg->height)){
      fprintf(stderr, "Logical error in upscale?\n");
      THROW(SYNTAX_ERROR);
    }
  }
  if (jpg->num_channels == 3) {
    // convert to RGB //
    unsigned char *prgb = jpg->pixels;
    const unsigned char *py  = jpg->channels[0].pixels;
    const unsigned char *pcb = jpg->channels[1].pixels;
    const unsigned char *pcr = jpg->channels[2].pixels;
    for (int yy = jpg->height;  yy;  --yy) {
      for (int x = 0;  x < jpg->width;  ++x) {
	register int y = py[x] << 8;
	register int cb = pcb[x] - 128;
	register int cr = pcr[x] - 128;
	*prgb++ = clip((y            + 359 * cr + 128) >> 8);
	*prgb++ = clip((y -  88 * cb - 183 * cr + 128) >> 8);
	*prgb++ = clip((y + 454 * cb            + 128) >> 8);
      }
      py += jpg->channels[0].stride;
      pcb += jpg->channels[1].stride;
      pcr += jpg->channels[2].stride;
    }
  } else if (jpg->channels[0].width != jpg->channels[0].stride) {
    // grayscale -> only remove stride
    ColourChannel *channel = &jpg->channels[0];
    unsigned char *pin = &channel->pixels[channel->stride];
    unsigned char *pout = &channel->pixels[channel->width];
    for (int y = channel->height - 1;  y;  --y) {
      memcpy(pout, pin, channel->width);
      pin += channel->stride;
      pout += channel->width;
    }
    channel->stride = channel->width;
  }
}
*/