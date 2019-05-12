#include<stdio.h>
#include<string.h>
#include<time.h>

#include<format.h>
#include<decodeScanGPU.h>
#include<entropyRLEdecodeGPU.h>



__host__ static void unstuffBuf(JPGReader *jpg) {
  clock_t start_t = clock();
  unsigned char next_val, *src, *dst, *buf = jpg->pos;
  for (src=buf, dst=buf; src < jpg->end;) {
    if((*dst++ = *src++) != 0xFF) continue;
    // Remove byte stuffing //
    if ((next_val = *src++) == 0x00) continue;
    // But keep restart markers //
    if (0xD0 == (next_val & 0xF8)) *dst++ = next_val;
  }
  jpg->end = dst;
  // Put an EOF marker at the new end
  *dst++ = 0xFF;
  *dst++ = 0xD9;
  clock_t end_t = clock();
  jpg->time += end_t - start_t;
}


// This only shows the bits, but doesn't move past them //
__host__ static int showBits(JPGReader* jpg, int num_bits) {
  if(!num_bits) return 0;

  while (jpg->num_bufbits < num_bits){
    unsigned char newbyte = (jpg->pos >= jpg->end) ? 0xFF : *jpg->pos++;
    jpg->bufbits = (jpg->bufbits << 8) | newbyte;
    jpg->num_bufbits += 8;
  }
  return (jpg->bufbits >> (jpg->num_bufbits - num_bits)) & ((1 << num_bits) - 1);
}


// Show the bits AND move past them //
__host__ static int getBits(JPGReader* jpg, int num_bits) {
  int res = showBits(jpg, num_bits);
  jpg->num_bufbits -= num_bits;
  return res;
}


__host__ static int getVLC(JPGReader* jpg, DhtVlc* vlc_table, unsigned char* code) {
  int symbol = showBits(jpg, 16);
  DhtVlc vlc = vlc_table[symbol];
  if(!vlc.num_bits) {
    jpg->error = SYNTAX_ERROR;
    return 0;
  }
  jpg->num_bufbits -= vlc.num_bits;  
  if(code) *code = vlc.tuple;
  unsigned char num_bits = vlc.tuple & 0x0F;
  if (!num_bits) return 0;
  int value = getBits(jpg, num_bits);
  if (value < (1 << (num_bits - 1))) {
    value += ((-1) << num_bits) + 1;
    /*short tmp = value;
    tmp += ((-1) << num_bits) + 1;
    value = tmp;*/
  }
  return value;  
}


__host__ static void decodeBlock(JPGReader* jpg, ColourChannel* channel) {
  unsigned char code = 0;
  int value, coef = 0;
  int* block = channel->working_space_pos;

  // Read DC value //
  channel->dc_cumulative_val += getVLC(jpg, &jpg->vlc_tables[channel->dc_id][0], NULL);
  block[0] = (channel->dc_cumulative_val) * jpg->dq_tables[channel->dq_id][0];
  // Read  AC values //
  do {
    value = getVLC(jpg, &jpg->vlc_tables[channel->ac_id][0], &code);
    if (!code) break; // EOB marker //
    if (!(code & 0x0F) && (code != 0xF0)) THROW(SYNTAX_ERROR);
    coef += (code >> 4) + 1;
    if (coef > 63) THROW(SYNTAX_ERROR);
    block[(int)deZigZag[coef]] = value * jpg->dq_tables[channel->dq_id][coef];
  } while(coef < 63);

  channel->working_space_pos += 64;
}


__host__ void decodeScanGPU(JPGReader* jpg) {
  unsigned char *pos = jpg->pos;
  unsigned int header_len = read16(pos);
  if (pos + header_len > jpg->end) THROW(SYNTAX_ERROR);
  pos += 2;
  
  // Read segment header //
  if (header_len < (4 + 2 * jpg->num_channels)) THROW(SYNTAX_ERROR);
  if (*(pos++) != jpg->num_channels) THROW(UNSUPPORTED_ERROR);
  int i;
  ColourChannel *channel;
  for(i = 0, channel=jpg->channels; i<jpg->num_channels; i++, channel++, pos+=2){
    if (pos[0] != channel->id) THROW(SYNTAX_ERROR);
    if (pos[1] & 0xEE) THROW(SYNTAX_ERROR);
    channel->dc_id = pos[1] >> 4;
    channel->ac_id = (pos[1] & 1) | 2;
  }
  if (pos[0] || (pos[1] != 63) || pos[2]) THROW(UNSUPPORTED_ERROR);
  pos = jpg->pos = jpg->pos + header_len;

  // Remove byte stuffing //
  unstuffBuf(jpg);

  // Do the decode scan //
  int restart_interval = jpg->restart_interval;
  if (!restart_interval) {
    
    /*jpg->device_pos = jpg->device_file_buf.mem + (jpg->pos - jpg->buf);
    int num_threads = (jpg->end - jpg->pos) * 8; // One thread per bit-position
    int threads_per_block = 256;
    int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;
    huffmanDecode_kernel<<<num_blocks, threads_per_block>>>(jpg->device_pos,
								num_threads,
								jpg->device_buf_values.mem,
								jpg->device_jump_lengths.mem,
								jpg->device_run_lengths.mem,
								jpg->device_vlc_tables);
								if (cudaGetLastError() != cudaSuccess) THROW(CUDA_KERNEL_LAUNCH_ERROR);*/
    
    
    for (int block_y = 0; block_y < jpg->num_blocks_y; block_y++){
      for (int block_x = 0; block_x < jpg->num_blocks_x; block_x++){
	// Loop over all channels //
	for (i = 0, channel = jpg->channels; i < jpg->num_channels; i++, channel++){
	  // Loop over samples in block //
	  for (int sample_y = 0; sample_y < channel->samples_y; ++sample_y){
	    for (int sample_x = 0; sample_x < channel->samples_x; ++sample_x){
	      decodeBlock(jpg, channel);
	      if (jpg->error) return;
	    }}}}}
    
  } else {

    int restart_count = restart_interval;
    int next_restart_index = 0;
    
    // Loop over all blocks
    for (int block_y = 0; block_y < jpg->num_blocks_y; block_y++){
      for (int block_x = 0; block_x < jpg->num_blocks_x; block_x++){

	// Loop over all channels //
	for (i = 0, channel = jpg->channels; i < jpg->num_channels; i++, channel++){

	  // Loop over samples in block //
	  for (int sample_y = 0; sample_y < channel->samples_y; ++sample_y){
	    for (int sample_x = 0; sample_x < channel->samples_x; ++sample_x){
	      decodeBlock(jpg, channel);
	      if (jpg->error) return;
	    }
	  }
	}

	if (restart_interval && !(--restart_count) && (jpg->pos < jpg->end)){
	  // Byte align //
	  jpg->num_bufbits &= 0xF8;
	  i = getBits(jpg, 16);
	  if (((i & 0xFFF8) != 0xFFD0) || ((i & 7) != next_restart_index))
	    THROW(SYNTAX_ERROR);
	  next_restart_index = (next_restart_index + 1) & 7;
	  restart_count = restart_interval;
	  for (i = 0; i < 3; i++)
	    jpg->channels[i].dc_cumulative_val = 0;
	}
      }
    }
  }
}
