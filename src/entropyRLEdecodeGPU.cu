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
    block_len = 1;

    // Decode subsequent AC terms //
    while (block_len < 64) {
      
      if (position > args.num_positions)
	goto invalid_block;
      
      short ac_jump = args.ac_jumps[channel_id][position];
      unsigned char run_len = args.run_lengths[channel_id][position];

      if (ac_jump <= 0) {
	if (ac_jump == 0)
	  goto invalid_block;
	position -= ac_jump;
	break; // EOB marker
      }
      
      position += ac_jump;
      block_len += run_len;
    }

    if ((block_len <= 64) && (position <= args.num_positions)) {
      args.out_lengths[channel_id][pos] = position - pos;
      continue; // Done
    }
    
  invalid_block:
    args.out_lengths[channel_id][pos] = 0;
  }
  
}
template __global__ void decodeBlockLengths_kernel<1>(DecodeBlockLengths_args args);
template __global__ void decodeBlockLengths_kernel<2>(DecodeBlockLengths_args args);



template <int num_lum_samples, int num_chrom_samples>
__global__ void reduceBlockLengthsStart_kernel(ReduceBlockLengthsStart_args args) {

  // TODO: Could try a version of this where no length validity checks are made,
  // everything is always summed? Possibly wouldn't be good, since it would
  // increase the number of writes to save on minimal compute and branching costs
  
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= args.num_positions) return;

  // We can't know what kind of block is in the current posiiton,
  // nor what kind of block follows, so we do EVERY possibility.

  // Compute values to be written out//
  int lumlum_length_out = 0, lumchrom_length_out = 0;
  int chromchrom_length_out = 0, chromlum_length_out = 0;
  short length = args.lengths_in[LUMINANCE][pos];
  int next_pos = pos + length;
  if (length && (next_pos < args.num_positions)) {
    // Lum->Lum //
    short next_lum_length = args.lengths_in[LUMINANCE][next_pos];
    // TODO: this bound check maybe unnecessary if the length sum doesn't have to have the last block
    if (next_lum_length && (next_pos + next_lum_length <= args.num_positions)) 
      lumlum_length_out = length + next_lum_length;
    // Lum->Chrom //
    if (num_chrom_samples) {
      short next_chrom_length = args.lengths_in[CHROMINANCE][next_pos];
      if (next_chrom_length && (next_pos + next_chrom_length <= args.num_positions)) 
	lumchrom_length_out = length + next_chrom_length;
    }
  }

  if (num_chrom_samples) {
    length = args.lengths_in[CHROMINANCE][pos];
    next_pos = pos + length;
    if (length && (next_pos < args.num_positions)) {
      // Chrom->Chrom //
      short next_chrom_length = args.lengths_in[CHROMINANCE][next_pos];
      if (next_chrom_length && (next_pos + next_chrom_length <= args.num_positions)) 
	chromchrom_length_out = length + next_chrom_length;
      // Chrom->Lum //
      short next_lum_length = args.lengths_in[LUMINANCE][next_pos];
      if (next_lum_length && (next_pos + next_lum_length <= args.num_positions))
	chromlum_length_out = length + next_lum_length;
    }
  }
  
  // Do the writes //
  __syncwarp(); // Writes should be coalesced, does this actually help guarentee that?
  int *out_ptr = &args.lengths_out[pos];
  if (!num_chrom_samples) {
    // No Chomrinance implies a single Luminance sample per block
    *out_ptr = lumlum_length_out;
  } else {
    for (int i = 0; i < num_lum_samples - 1; i++, out_ptr+=args.num_positions)
      *out_ptr = lumlum_length_out;
    *out_ptr = lumchrom_length_out; out_ptr += args.num_positions;
    *out_ptr = chromchrom_length_out; out_ptr += args.num_positions;
    *out_ptr = chromlum_length_out; 
  }  
}
template __global__ void reduceBlockLengthsStart_kernel<4,2>(ReduceBlockLengthsStart_args args);
template __global__ void reduceBlockLengthsStart_kernel<2,2>(ReduceBlockLengthsStart_args args);
template __global__ void reduceBlockLengthsStart_kernel<1,2>(ReduceBlockLengthsStart_args args);
template __global__ void reduceBlockLengthsStart_kernel<1,0>(ReduceBlockLengthsStart_args args);


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
