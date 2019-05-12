#include<stdio.h>

#include<format.h>
#include<entropyRLEdecodeGPU.h>


#define EOB_MARKER 20


__global__ void huffmanDecode_kernel(unsigned char* in, int num_threads, short* out_vals,
				     unsigned char* out_bit_lengths, unsigned char* out_run_lengths,
				     DhtVlc* vlc_table) {
    int pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= num_threads) return;
    int byte_pos = pos >> 3;
    int bit_pos = pos & 7;
    // Read in (at least) 4 bytes from byte_pos
    unsigned int bits = *((int*) (in + byte_pos));

    // Decode huffman symbol using lookup table //
    unsigned int lookup_key = (bits >> ((sizeof(unsigned int) * 8) - 16 - bit_pos)) & 0xFFFF;
    DhtVlc vlc = vlc_table[lookup_key];
    unsigned char num_symbol_bits = vlc.num_bits;
    unsigned char num_value_bits = vlc.tuple & 0x0F;
    unsigned char run_length = ((vlc.tuple >> 4) & 0x0F) + 1;
    int total_num_bits = num_symbol_bits + num_value_bits;
    out_bit_lengths[pos] = total_num_bits;
    out_run_lengths[pos] = run_length;
    
    if (num_value_bits) {
      // Get the value following the runlength tuple //
      short value = bits >> ((sizeof(unsigned int) * 8) - bit_pos - total_num_bits);
      value &= (1 << num_value_bits) - 1;
      if (value < (1 << (num_value_bits - 1)))
	value += ((-1) << num_value_bits) + 1;
      out_vals[pos] = value;
      
    } else {
      // Flag an EOB marker with an impossible value of runlength. //
      // The special 16-consecutive-zeros case is covered by setting value=0 //
      if (run_length == 1) out_run_lengths[pos] = EOB_MARKER;
      else if (run_length == 16) out_vals[pos] = 0;
      else printf("GPU huffman decode error\n");
      // Eventually might add a way to handle an error here?
    }
}