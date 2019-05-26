#include<stdio.h>
#include<string.h>
#include<time.h>

#include<format.h>
#include<decodeScanGPU.h>
#include<entropyRLEdecodeGPU.h>



__host__ static void unstuffBuf(JPGReader *jpg) {
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

static int tmpval;

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
  if (!num_bits) {tmpval += vlc.num_bits; return 0;}
  int value = getBits(jpg, num_bits);
  if (value < (1 << (num_bits - 1))) {
    value += ((-1) << num_bits) + 1;
    /*short tmp = value;
    tmp += ((-1) << num_bits) + 1;
    value = tmp;*/
  }
  printf("key %d\tsymbits %d\tvalbits %d\tval %d\tpos %d\n", (symbol >> 16) & 0xFF, vlc.num_bits, num_bits, value, tmpval);
  tmpval += vlc.num_bits + num_bits;
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
    
    jpg->device_pos = jpg->device_file_buf.mem + (jpg->pos - jpg->buf);
    int num_threads = (jpg->end - jpg->pos) * 8; // One thread per bit-position
    int threads_per_block = 128;
    int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;
    clock_t start = clock();
    huffmanDecode_kernel<<<num_blocks, threads_per_block>>>(jpg->device_pos,
							    num_threads,
							    jpg->deviceAddresses);
    if (cudaGetLastError() != cudaSuccess) THROW(CUDA_KERNEL_LAUNCH_ERROR);
    cudaDeviceSynchronize();
    clock_t end = clock();
    jpg->time += end - start;


    /*unsigned char tmp_ac_run_len[800];
    unsigned char tmp_dc_run_len[800];
    unsigned char tmp_ac_jump_len[800];
    unsigned char tmp_dc_jump_len[800];
    unsigned char Ctmp_ac_run_len[800];
    unsigned char Ctmp_dc_run_len[800];
    unsigned char Ctmp_ac_jump_len[800];
    unsigned char Ctmp_dc_jump_len[800];
    cudaMemcpy(&tmp_ac_run_len, jpg->device_run_lengths[jpg->channels[0].ac_id].mem,
	       800, cudaMemcpyDeviceToHost);    
    cudaMemcpy(&tmp_dc_run_len, jpg->device_run_lengths[jpg->channels[0].dc_id].mem,
	       800, cudaMemcpyDeviceToHost);
    cudaMemcpy(&tmp_ac_jump_len, jpg->device_jump_lengths[jpg->channels[0].ac_id].mem,
	       800, cudaMemcpyDeviceToHost);    
    cudaMemcpy(&tmp_dc_jump_len, jpg->device_jump_lengths[jpg->channels[0].dc_id].mem,
	       800, cudaMemcpyDeviceToHost);
    cudaMemcpy(&Ctmp_ac_run_len, jpg->device_run_lengths[jpg->channels[1].ac_id].mem,
	       800, cudaMemcpyDeviceToHost);    
    cudaMemcpy(&Ctmp_dc_run_len, jpg->device_run_lengths[jpg->channels[1].dc_id].mem,
	       800, cudaMemcpyDeviceToHost);
    cudaMemcpy(&Ctmp_ac_jump_len, jpg->device_jump_lengths[jpg->channels[1].ac_id].mem,
	       800, cudaMemcpyDeviceToHost);    
    cudaMemcpy(&Ctmp_dc_jump_len, jpg->device_jump_lengths[jpg->channels[1].dc_id].mem,
	       800, cudaMemcpyDeviceToHost);
    
       
    int position = 0;
    int coeff = 1;
    printf("%d, %d, (pos %d) (jump %d)\n", 0, tmp_dc_run_len[position], position,
	   tmp_dc_jump_len[position]);
    //coeff += tmp_dc_run_len[0];
    position += tmp_dc_jump_len[0];
    for(i = 0; i < 60; i++){
      printf("%d, %d, (pos %d)\n", coeff, tmp_ac_run_len[position], position);
      if (20 == tmp_ac_run_len[position]) {
	position += tmp_ac_jump_len[position];
	break;
      }
      coeff += tmp_ac_run_len[position];
      position += tmp_ac_jump_len[position];
    }
    
    coeff = 1;
    printf("%d, %d, (pos %d) (jump %d)\n", 0, Ctmp_dc_run_len[position], position,
	   Ctmp_dc_jump_len[position]);
    position += Ctmp_dc_jump_len[position];
    for(i = 0; (i < 60 && coeff != 64); i++){
      printf("%d, %d, (pos %d) (jump %d)\n", coeff, Ctmp_ac_run_len[position], position,
	   Ctmp_ac_jump_len[position]);
      if (20 == Ctmp_ac_run_len[position]) {
	position += Ctmp_ac_jump_len[position];
	break;
      }
      coeff += Ctmp_ac_run_len[position];
      position += Ctmp_ac_jump_len[position];
    }

    cudaMemcpy(&Ctmp_ac_run_len, jpg->device_run_lengths[jpg->channels[2].ac_id].mem,
	       800, cudaMemcpyDeviceToHost);    
    cudaMemcpy(&Ctmp_dc_run_len, jpg->device_run_lengths[jpg->channels[2].dc_id].mem,
	       800, cudaMemcpyDeviceToHost);
    cudaMemcpy(&Ctmp_ac_jump_len, jpg->device_jump_lengths[jpg->channels[2].ac_id].mem,
	       800, cudaMemcpyDeviceToHost);    
    cudaMemcpy(&Ctmp_dc_jump_len, jpg->device_jump_lengths[jpg->channels[2].dc_id].mem,
	       800, cudaMemcpyDeviceToHost);
    coeff = 1;
    printf("%d, %d, (pos %d) (jump %d)\n", 0, Ctmp_dc_run_len[position], position,
	   Ctmp_dc_jump_len[position]);
    position += Ctmp_dc_jump_len[position];
    for(i = 0; (i < 60 && coeff != 64); i++){
      printf("%d, %d, (pos %d) (jump %d)\n", coeff, Ctmp_ac_run_len[position], position,
	   Ctmp_ac_jump_len[position]);
      if (20 == Ctmp_ac_run_len[position]) {
	position += Ctmp_ac_jump_len[position];
	break;
      }
      coeff += Ctmp_ac_run_len[position];
      position += Ctmp_ac_jump_len[position];
    }
    printf("\n--------------------\n");
    tmpval = 0;*/
    
    for (int block_y = 0; block_y < jpg->num_blocks_y; block_y++) {
      for (int block_x = 0; block_x < jpg->num_blocks_x; block_x++) {
	// Loop over all channels //
	for (i = 0, channel = jpg->channels; i < jpg->num_channels; i++, channel++) {
	  // Loop over samples in block //
	  for (int sample_y = 0; sample_y < channel->samples_y; ++sample_y) {
	    for (int sample_x = 0; sample_x < channel->samples_x; ++sample_x) {
	      decodeBlock(jpg, channel);
	      //THROW(PROGRAMMER_ERROR);
	      if (jpg->error) return;
	    }
	  }
	}
	THROW(PROGRAMMER_ERROR);
      }
    }
    
  } else {

    restartMarkerScan(jpg);
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
