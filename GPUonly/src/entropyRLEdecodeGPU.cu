#include<stdio.h>

#include<format.h>
#include<entropyRLEdecodeGPU.h>



__host__ void restartMarkerScan(JPGReader *jpg) {
  // Ensure there is memory for the positions output //

  printf("Done restart marker scan\n");
}


__global__ void huffmanDecode_kernel(unsigned char* in,
				     int num_threads,
				     DeviceAddressBook addresses) {
  
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos >= num_threads) return;
  int byte_pos = pos >> 3;
  int bit_pos = pos & 7;
  // Read in 5 bytes from byte_pos
  unsigned long long int bits =
    (((unsigned long long int) in[byte_pos + 0]) << 32) |
    (((unsigned long long int) in[byte_pos + 1]) << 24) |
    (((unsigned long long int) in[byte_pos + 2]) << 16) |
    (((unsigned long long int) in[byte_pos + 3]) << 8 ) |
    (((unsigned long long int) in[byte_pos + 4]));
  
  // Decode huffman symbol using lookup table //
  unsigned int lookup_key = (bits >> (24 - bit_pos)) & 0xFFFF;
  
  for (int table_id = 0; table_id < 4; table_id++) {
    DhtVlc vlc = addresses.vlc_tables[(table_id << 16) + lookup_key];
    unsigned char num_symbol_bits = vlc.num_bits;
    unsigned char num_value_bits = vlc.tuple & 0x0F;
    unsigned char run_length = ((vlc.tuple >> 4) & 0x0F) + 1;
    unsigned char total_num_bits = num_symbol_bits + num_value_bits;
    // if (!num_symbol_bits) invalid symbol, handle it later
    addresses.jump_lengths[table_id][pos] = total_num_bits;
    addresses.run_lengths[table_id][pos] = run_length;
    
    if (num_value_bits) {
      // Get the value following the runlength tuple //
      short value = bits >> (64 - bit_pos - total_num_bits);
      value &= (1 << num_value_bits) - 1;
      if (value < (1 << (num_value_bits - 1)))
	value += ((-1) << num_value_bits) + 1;
      addresses.raw_values[table_id][pos] = value;
      
    } else {
      // Flag an EOB marker with an impossible value of runlength. //
      // The special 16-consecutive-zeros case is covered by setting value=0 //
      if (run_length == 1) addresses.run_lengths[table_id][pos] = EOB_MARKER;
      else if (run_length == 16) addresses.raw_values[table_id][pos] = 0;
      else printf("GPU huffman decode error\n");
      // Eventually might add a way to handle an error here?
    }
  }
}