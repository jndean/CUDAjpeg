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
    if (!num_symbol_bits) 
      args.jump_lengths[table_id][pos] = 0;
    
    else if (num_value_bits) {
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

    if (pos == 963) {
      printf("HERE symbits %d, valbits %d, runlen %d\n",
	     num_symbol_bits, num_value_bits, run_length);
    }
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
      args.run_lengths[tid][pos] = run_length;
      args.jump_lengths[table_id][pos] = total_num_bits;
      
    } else {
      
      if (run_length == 1) {
	// EOB marked by flipping the sign of jump length //
	args.jump_lengths[table_id][pos] = -total_num_bits;
      
      } else if (run_length == 16) {
	// The 16-consecutive-zeros case is covered by setting value=0 //
	args.raw_values[table_id][pos] = 0;
	args.jump_lengths[table_id][pos] = total_num_bits;
	args.run_lengths[tid][pos] = run_length;
      }
      else printf("GPU huffman decode error\n");
      // Eventually might add a way to handle an error here?
    }
   
  }
}
template __global__ void huffmanDecode_kernel<1>(HuffmanDecode_args args);
template __global__ void huffmanDecode_kernel<2>(HuffmanDecode_args args);



template <int num_channel_types>
__global__ void decodeBlockLengths_kernel(DecodeBlockLengths_args args) {

  int position, block_len;
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= args.num_positions) return;

  for (int channel_id = 0; channel_id < num_channel_types; channel_id++) {

    // Decode the DC term //
    short dc_jump = args.dc_jumps[channel_id][pos];
    if (dc_jump == 0)
      goto invalid_block;

    position = pos + dc_jump;
    if (position >= args.num_positions)
      goto invalid_block;
    block_len = 1;

    // Decode subsequent AC terms //
    while (block_len < 64) {
      short ac_jump = args.ac_jumps[channel_id][position];
      unsigned char run_len = args.run_lengths[channel_id][position];

      if (ac_jump <= 0) {
	if (ac_jump == 0)
	  goto invalid_block;
	position -= ac_jump;
	if (position >= args.num_positions)   // rm these ifs later?
	  goto invalid_block;
	break; // EOB marker
      }
      
      position += ac_jump;
      block_len += run_len;
      if (position >= args.num_positions)
	goto invalid_block;
    }

    if (block_len <= 64) {
      args.out_lengths[channel_id][pos] = position - pos;
      continue;
    } else
      goto invalid_block;
    
  invalid_block:
    args.out_lengths[channel_id][pos] = 0;
  }
  
}
template __global__ void decodeBlockLengths_kernel<1>(DecodeBlockLengths_args args);
template __global__ void decodeBlockLengths_kernel<2>(DecodeBlockLengths_args args);


/*
template <int num_channel_types>
__global__ void reduceJumpsAC_kernel(ReduceJumpsAC_args args) {
  
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= args.num_positions) return;

  for (int channel_id = 0; channel_id < num_channel_types; channel_id++){
    
    // 'jump' accumulates the block length in bits (from the file) //
    // 'block_len' accumulates the block length in DCT coefficients //
    unsigned char block_len = args.lengths_in[channel_id][pos];
    short jump = args.jumps_in[channel_id][pos];

    if (jump <= 0) {
      if (jump == 0) 
	args.jumps_out[channel_id][pos] = 0; // Preserve invalid block
      else {
	args.jumps_out[channel_id][pos] = jump; // Preserve EOB
	args.lengths_out[channel_id][pos] = block_len;
      }
      continue;
    }
    
    int next_pos = pos + jump;
    if (next_pos >= args.num_positions) {
      args.jumps_out[channel_id][pos] = 0; // Mark invalid block
      continue;
    }

    short next_jump = args.jumps_in[channel_id][next_pos];
    unsigned char next_block_len = args.lengths_in[channel_id][pos];

    if (next_jump == 0) {
      args.jumps_out[channel_id][pos] = 0; // Mark invalid block
      continue;
    }

    unsigned char out_block_len = block_len + next_block_len;
    // Propogate negative sign (EOB) on jumps
    short out_jump = (next_jump < 0) ? next_jump - jump : next_jump + jump;

    if (out_block_len > 64
	args.jumps_out[channel_id][pos] = double_jump; 
	args.lengths_out[channel_id][pos] = double_block_len;
     
    
  }
}
template __global__ void reduceJumpsAC_kernel<1>(ReduceJumpsAC_args args);
template __global__ void reduceJumpsAC_kernel<2>(ReduceJumpsAC_args args);
*/
