#ifndef ENTROPYRLEDECODEGPU_H
#define ENTROPYRLEDECODEGPU_H

__global__ void huffmanDecode_kernel(unsigned char* in, int num_threads, short* out_vals,
				     unsigned char* out_bit_lengths, unsigned char* out_run_lengths,
				     DhtVlc* vlc_table);

#endif // ENTROPYRLEDECODEGPU_H //
