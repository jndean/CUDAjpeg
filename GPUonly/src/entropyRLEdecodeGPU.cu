#include<stdio.h>

#include<format.h>
#include<entropyRLEdecodeGPU.h>



template <int num_channel_types>
__global__ void huffmanDecode_kernel(HuffmanDecode_args args) {
  
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= args.num_threads) return;
  int byte_pos = pos >> 3;
  int bit_pos = pos & 7;
  // Read in 5 bytes from byte_pos
  long long int bits =
    (((long long int) args.in[byte_pos + 0]) << 32) |
    (((long long int) args.in[byte_pos + 1]) << 24) |
    (((long long int) args.in[byte_pos + 2]) << 16) |
    (((long long int) args.in[byte_pos + 3]) << 8 ) |
    (((long long int) args.in[byte_pos + 4]));
  
  // Decode huffman symbol using lookup table //
  unsigned int lookup_key = (bits >> (24 - bit_pos)) & 0xFFFF;

  for (int tid = 0; tid < num_channel_types; tid++) {
    
    // ----- DC Section ----- //
    int table_id = 2 * tid;
    DhtVlc vlc = args.vlc_tables[table_id][lookup_key];
    unsigned char num_symbol_bits = vlc.num_bits;
    unsigned char num_value_bits = vlc.tuple & 0x0F;
    short total_num_bits = num_symbol_bits + num_value_bits;
    args.jump_lengths[table_id][pos] = total_num_bits;
      
    // Invalid symbol marked by 0 //
    if (!num_symbol_bits) {
      args.jump_lengths[table_id][pos] = 0;
      continue;
    }
    
    if (num_value_bits) {
      short value = bits >> (40 - bit_pos - total_num_bits);
      value &= (1 << num_value_bits) - 1;
      if (value < (1 << (num_value_bits - 1)))
	value += ((-1) << num_value_bits) + 1;
      args.raw_values[table_id][pos] = value;
    } else 
      args.raw_values[table_id][pos] = 0;

    
    // ----- AC Section ----- //
    table_id = 2 * tid + 1;  
    vlc = args.vlc_tables[table_id][lookup_key];
    num_symbol_bits = vlc.num_bits;
    num_value_bits = vlc.tuple & 0x0F;
    total_num_bits = num_symbol_bits + num_value_bits;
    unsigned char run_length = ((vlc.tuple >> 4) & 0x0F) + 1;
      
    // Invalid symbol marked by 0 //
    if (!num_symbol_bits) {
      args.jump_lengths[table_id][pos] = 0;
      continue;
    }
    
    if (num_value_bits) {
      // Get the value following the runlength tuple //
      short value = bits >> (40 - bit_pos - total_num_bits);
      value &= (1 << num_value_bits) - 1;
      if (value < (1 << (num_value_bits - 1)))
	value += ((-1) << num_value_bits) + 1;
      args.raw_values[table_id][pos] = value;
      args.run_lengths[table_id][pos] = run_length;
      args.jump_lengths[table_id][pos] = total_num_bits;
      
    } else {
      // Handle special cases //
      if (run_length == 1) {
	// EOB marked by flipping the sign of jump length //
	args.jump_lengths[table_id][pos] = -total_num_bits;
      
      } else if (run_length == 16) {
	// The 16-consecutive-zeros case is covered by setting value=0 //
	args.raw_values[table_id][pos] = 0;
	args.jump_lengths[table_id][pos] = total_num_bits;
	args.run_lengths[table_id][pos] = run_length;
      }
      else printf("GPU huffman decode error\n");
      // Eventually might add a way to handle an error here?
    }
   
  }
}
template __global__ void huffmanDecode_kernel<1>(HuffmanDecode_args args);
template __global__ void huffmanDecode_kernel<2>(HuffmanDecode_args args);



template <int num_channel_types>
__global__ void reduceJumpsAC_kernel(ReduceJumpsAC_args args) {
  /* 
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= args.num_positions) return;

  for (int channel_id = 0; channel_id < num_channel_types; channel_id++){
    
    // 'jump' accumulates the block length in bits (from the file) //
    // 'block_len' accumulates the block length in DCT coefficients //
    unsigned char block_len = args.lengths_in[channel_id][pos];
    short jump = args.jumps_in[channel_id][pos];

    if (block_len >= SMALLEST_MARKER_VAL) {
      args.lengths_out[channel_id][pos] = block_len;
      args.jumps_out[channel_id][pos] = 0;
      continue;
    }
    
    int next_pos = pos + jump;
    if (next_pos >= args.num_positions) {
      args.lengths_out[channel_id][pos] = INVALID_SYMBOL_MARKER;
      continue;
    }

    short next_jump = args.jumps_in[channel_id][next_pos];
    unsigned char next_block_len = args.lengths_in[channel_id][pos];
    if (next_block_len == INVALID_SYMBOL_MARKER) {
      args.lengths_out[channel_id][pos] = INVALID_SYMBOL_MARKER;
      continue;
    } else if (next_block_len == EOB_MARKER) {
      args.jumps_out[channel_id][pos] = jump;
      args.lengths_out[channel_id][pos] = block_len;
      continue;
    }

    unsigned char both_block_len = block_len + next_block_len;
    if (both_block_len >= 64) {
      args.lengths_out[channel_id][pos] = INVALID_SYMBOL_MARKER;
      continue;
    }
    args.lengths_out[channel_id][pos] = both_block_len;
    
    short double_jump = jump + next_jump;
    args.jumps_out[channel_id][pos] = double_jump;
    } */
}
template __global__ void reduceJumpsAC_kernel<1>(ReduceJumpsAC_args args);
template __global__ void reduceJumpsAC_kernel<2>(ReduceJumpsAC_args args);



template <int num_channel_types>
__global__ void reduceJumpsDC_kernel(ReduceJumpsDC_args args) {
  
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= args.num_positions) return;
  /*
  for (int channel_id = 0; channel_id < num_channel_types; channel_id++) {

    short dc_jump = args.dc_jumps[channel_id][pos];
    unsigned char dc_block_len = args.dc_lengths_in[channel_id][pos];
    if (args.block_lengths[channel_id][pos] > 0) 
      continue;

    int ac_pos = pos + dc_jump;
    if (ac_pos >= args.num_positions) {
      args.block_lengths[pos] = INVALID_BLOCK_MARKER;
      continue;
    }
    
    unsigned char ac_length = args.ac_lengths[channel_id][ac_pos];
    if (ac_length == INVALID_SYMBOL_MARKER) {
      args.block_lengths[pos] = INVALID_BLOCK_MARKER;
      continue;
    }

    if (ac_length == EOB_MARKER) {
      
    }

    }*/
}