#ifndef ENTROPYRLEDECODEGPU_H
#define ENTROPYRLEDECODEGPU_H

#define LUMINANCE 0
#define CHROMINANCE 1


typedef struct _HuffmanDecode_args {
  unsigned char *in;
  int num_threads;
  DhtVlc *vlc_tables[4];
  unsigned char *run_lengths[2];
  short *raw_values[4], *jump_lengths[4];
} HuffmanDecode_args;

template <int num_channel_types>
__global__ void huffmanDecode_kernel(HuffmanDecode_args args);


typedef struct _DecodeBlockLengths_args {
  int num_positions;
  short *dc_jumps[2], *ac_jumps[2];
  unsigned char *run_lengths[2];
  short *out_lengths[2];
} DecodeBlockLengths_args;

template <int num_channel_types>
__global__ void decodeBlockLengths_kernel(DecodeBlockLengths_args args);


typedef struct _ReduceBlockLengthsStart_args {
  int num_positions;
  short *lengths_in[2];
  int *lengths_out;
} ReduceBlockLengthsStart_args;

template <int num_lum_samples, int num_chrom_samples>
__global__ void reduceBlockLengthsStart_kernel(ReduceBlockLengthsStart_args args);

typedef struct _ReduceBlockLengthsStep_args {
  int num_positions, step;
  int *lengths_in, *lengths_out;
} ReduceBlockLengthsStep_args;

template <int num_lum_samples, int num_chrom_samples>
__global__ void reduceBlockLengthsStep_kernel(ReduceBlockLengthsStep_args args);


/*
typedef struct _ReduceJumpsAC_args {
  int num_positions;
  short *jumps_in[2], *jumps_out[2];
  unsigned char *lengths_in[2], *lengths_out[2];
} ReduceJumpsAC_args;

template <int num_channel_types>
__global__ void reduceJumpsAC_kernel(ReduceJumpsAC_args);
*/

#endif // ENTROPYRLEDECODEGPU_H //
