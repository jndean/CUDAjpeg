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
static int tmpcoeff;

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
  if (!num_bits) {
    //printf("pos %d\tcoef %d\tsymbits %d\t EOB!\n", tmpval, tmpcoeff, vlc.num_bits);
    tmpval += vlc.num_bits;
    return 0;}
  int value = getBits(jpg, num_bits);
  if (value < (1 << (num_bits - 1))) {
    value += ((-1) << num_bits) + 1;
    /*short tmp = value;
    tmp += ((-1) << num_bits) + 1;
    value = tmp;*/
  }
  // printf("pos %d\tcoef %d\tsymbits %d\tvalbits %d\tval %d\n", tmpval, tmpcoeff,
  //	 vlc.num_bits, num_bits, value);
  tmpval += vlc.num_bits + num_bits;
  return value;  
}


__host__ static void decodeBlock(JPGReader* jpg, ColourChannel* channel) {
  unsigned char code = 0;
  int value, coef = 0;
  int* block = channel->working_space_pos;
  tmpcoeff = 0;

  // Read DC value //
  channel->dc_cumulative_val += getVLC(jpg, &jpg->vlc_tables[channel->dc_id][0], NULL);
  block[0] = (channel->dc_cumulative_val) * jpg->dq_tables[channel->dq_id][0];
  tmpcoeff += 1;
  // Read  AC values //
  do {
    value = getVLC(jpg, &jpg->vlc_tables[channel->ac_id][0], &code);
    if (!code) break; // EOB marker //
    if (!(code & 0x0F) && (code != 0xF0)) THROW(SYNTAX_ERROR);
    coef += (code >> 4) + 1;
    tmpcoeff = coef+1;
    if (coef > 63) THROW(SYNTAX_ERROR);
    block[(int)deZigZag[coef]] = value * jpg->dq_tables[channel->dq_id][coef];
  } while(coef < 63);

  channel->working_space_pos += 64;
}



__host__ void reshuffleHuffmanTables(JPGReader* jpg) {
  
  jpg->vlc_tables[0] = jpg->vlc_tables_backup[jpg->channels[0].dc_id];
  jpg->device_vlc_tables[0] = jpg->device_vlc_tables_backup[jpg->channels[0].dc_id];
  jpg->channels[0].dc_id = 0;
  jpg->vlc_tables[1] = jpg->vlc_tables_backup[jpg->channels[0].ac_id];
  jpg->device_vlc_tables[1] = jpg->device_vlc_tables_backup[jpg->channels[0].ac_id];
  jpg->channels[0].ac_id = 1;
  if (jpg->num_channels == 3) {
    jpg->vlc_tables[2] = jpg->vlc_tables_backup[jpg->channels[1].dc_id];
    jpg->device_vlc_tables[2] = jpg->device_vlc_tables_backup[jpg->channels[1].dc_id];
    jpg->channels[1].dc_id = jpg->channels[2].dc_id = 2;
    jpg->vlc_tables[3] = jpg->vlc_tables_backup[jpg->channels[1].ac_id];
    jpg->device_vlc_tables[3] = jpg->device_vlc_tables_backup[jpg->channels[1].ac_id];
    jpg->channels[1].ac_id = jpg->channels[2].ac_id = 3;
  }
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

  // Reshuffle the vlc tables so they're in a standardised order //
  reshuffleHuffmanTables(jpg);
  // Remove byte stuffing //
  unstuffBuf(jpg);
  cudaMemcpy(jpg->device_file_buf.mem, jpg->pos,
	     jpg->end - jpg->pos, cudaMemcpyHostToDevice);

  // Do the decode scan //
  int restart_interval = jpg->restart_interval;
  if (!restart_interval) {

    int num_threads = (jpg->end - jpg->pos) * 8; // One thread per bit-position
    int threads_per_block = 128;
    int num_blocks = (num_threads + threads_per_block - 1) / threads_per_block;

    HuffmanDecode_args huff_args;
    huff_args.in = jpg->device_file_buf.mem;
    huff_args.num_threads = num_threads;
    for (i = 0; i < 4; i++) {
      huff_args.vlc_tables[i] = jpg->device_vlc_tables[i];
      huff_args.jump_lengths[i] = jpg->device_jump_lengths[i].mem;
      huff_args.raw_values[i] = jpg->device_values[i].mem;
    }
    for (i = 0; i < 2; i++)
      huff_args.run_lengths[i] = jpg->device_run_lengths[i].mem;
    clock_t start = clock();
    if (jpg->num_channels == 3) 
      huffmanDecode_kernel<2><<<num_blocks, threads_per_block>>>(huff_args);
    else 
      huffmanDecode_kernel<1><<<num_blocks, threads_per_block>>>(huff_args);
    
    if (cudaGetLastError() != cudaSuccess) THROW(CUDA_KERNEL_LAUNCH_ERROR);

    DecodeBlockLengths_args blocklen_args;
    blocklen_args.num_positions = num_threads;
    for (i = 0; i < 2; i++) {
      blocklen_args.dc_jumps[i] = jpg->device_jump_lengths[i * 2].mem;
      blocklen_args.ac_jumps[i] = jpg->device_jump_lengths[i * 2 + 1].mem;
      blocklen_args.run_lengths[i] = jpg->device_run_lengths[i].mem;
      blocklen_args.out_lengths[i] = jpg->device_block_lengths[i].mem;
    }
    if (jpg->num_channels == 3) 
      decodeBlockLengths_kernel<2><<<num_blocks, threads_per_block>>>(blocklen_args);
    else 
      decodeBlockLengths_kernel<1><<<num_blocks, threads_per_block>>>(blocklen_args);
    
    if (cudaGetLastError() != cudaSuccess) THROW(CUDA_KERNEL_LAUNCH_ERROR);
    cudaDeviceSynchronize();

    clock_t end = clock();
    jpg->time += end - start;

#define TMPARRAYLEN 2000
    unsigned char tmp_ac_run_len[TMPARRAYLEN];
    short tmp_ac_jump_len[TMPARRAYLEN];
    short tmp_dc_jump_len[TMPARRAYLEN];
    short tmp_ac_val[TMPARRAYLEN];
    short tmp_dc_val[TMPARRAYLEN];
    unsigned char Ctmp_ac_run_len[TMPARRAYLEN];
    short Ctmp_ac_jump_len[TMPARRAYLEN];
    short Ctmp_dc_jump_len[TMPARRAYLEN];
    short Ctmp_ac_val[TMPARRAYLEN];
    short Ctmp_dc_val[TMPARRAYLEN];
    //unsigned char *Ctmp_ac_run_len = tmp_ac_run_len;
    //unsigned char *Ctmp_ac_jump_len = tmp_ac_jump_len;
    //unsigned char *Ctmp_dc_jump_len = tmp_dc_jump_len;
    cudaMemcpy(&tmp_ac_run_len, jpg->device_run_lengths[0].mem,
	       TMPARRAYLEN, cudaMemcpyDeviceToHost);    
    cudaMemcpy(&tmp_ac_jump_len, jpg->device_jump_lengths[jpg->channels[0].ac_id].mem,
	       TMPARRAYLEN*2, cudaMemcpyDeviceToHost);    
    cudaMemcpy(&tmp_dc_jump_len, jpg->device_jump_lengths[jpg->channels[0].dc_id].mem,
	       TMPARRAYLEN*2, cudaMemcpyDeviceToHost);
    cudaMemcpy(&tmp_ac_val, jpg->device_values[jpg->channels[0].ac_id].mem,
	       TMPARRAYLEN*2, cudaMemcpyDeviceToHost);    
    cudaMemcpy(&tmp_dc_val, jpg->device_values[jpg->channels[0].dc_id].mem,
	       TMPARRAYLEN*2, cudaMemcpyDeviceToHost);
    
    cudaMemcpy(&Ctmp_ac_run_len, jpg->device_run_lengths[1].mem,
	       TMPARRAYLEN, cudaMemcpyDeviceToHost); 
    cudaMemcpy(&Ctmp_ac_jump_len, jpg->device_jump_lengths[jpg->channels[1].ac_id].mem,
	       TMPARRAYLEN*2, cudaMemcpyDeviceToHost);    
    cudaMemcpy(&Ctmp_dc_jump_len, jpg->device_jump_lengths[jpg->channels[1].dc_id].mem,
	       TMPARRAYLEN*2, cudaMemcpyDeviceToHost);
    cudaMemcpy(&Ctmp_ac_val, jpg->device_values[jpg->channels[1].ac_id].mem,
	       TMPARRAYLEN*2, cudaMemcpyDeviceToHost);    
    cudaMemcpy(&Ctmp_dc_val, jpg->device_values[jpg->channels[1].dc_id].mem,
	       TMPARRAYLEN*2, cudaMemcpyDeviceToHost);

    int position = 0;
    int coeff = 0;
    /*printf("(pos: %d, coeff: %d) jump_len: %d, run_len: 1, val: %d\n",
	   position, coeff, tmp_dc_jump_len[position], tmp_dc_val[position]);
    coeff += 1;
    position += tmp_dc_jump_len[position];

    while (coeff < 64) {
      printf("(pos: %d, coeff: %d) jump_len: %d, run_len: %d, val: %d\n",
	     position, coeff, tmp_ac_jump_len[position], tmp_ac_run_len[position],
	     tmp_ac_val[position]);
      if (tmp_ac_jump_len[position] < 0)  { printf("   EOB!\n"); coeff += 64; }
      if (tmp_ac_jump_len[position] == 0) printf("Error\n");
      coeff += tmp_ac_run_len[position];
      position += abs(tmp_ac_jump_len[position]);
    }
    
    coeff = 0;
    printf("(pos: %d, coeff: %d) jump_len: %d, run_len: 1, val: %d\n",
	   position, coeff, Ctmp_dc_jump_len[position], Ctmp_dc_val[position]);
    coeff += 1;
    position += Ctmp_dc_jump_len[position];

    while (coeff < 64) {
      printf("(pos: %d, coeff: %d) jump_len: %d, run_len: %d, val: %d\n",
	     position, coeff, Ctmp_ac_jump_len[position], Ctmp_ac_run_len[position],
	     Ctmp_ac_val[position]);
      if (Ctmp_ac_jump_len[position] < 0)  { printf("   EOB!\n"); coeff += 64; }
      if (Ctmp_ac_jump_len[position] == 0) printf("Error\n");
      coeff += Ctmp_ac_run_len[position];
      position += abs(Ctmp_ac_jump_len[position]);
    }

    coeff = 0;
    printf("(pos: %d, coeff: %d) jump_len: %d, run_len: 1, val: %d\n",
	   position, coeff, Ctmp_dc_jump_len[position], Ctmp_dc_val[position]);
    coeff += 1;
    position += Ctmp_dc_jump_len[position];

    while (coeff < 64) {
      printf("(pos: %d, coeff: %d) jump_len: %d, run_len: %d, val: %d\n",
	     position, coeff, Ctmp_ac_jump_len[position], Ctmp_ac_run_len[position],
	     Ctmp_ac_val[position]);
      if (Ctmp_ac_jump_len[position] < 0)  { printf("   EOB!\n"); coeff += 64; }
      if (Ctmp_ac_jump_len[position] == 0) printf("Error\n");
      coeff += Ctmp_ac_run_len[position];
      position += abs(Ctmp_ac_jump_len[position]);
    }

    coeff = 0;
    printf("(pos: %d, coeff: %d) jump_len: %d, run_len: 1, val: %d\n",
	   position, coeff, tmp_dc_jump_len[position], tmp_dc_val[position]);
    coeff += 1;
    position += tmp_dc_jump_len[position];

    while (coeff < 64) {
      printf("(pos: %d, coeff: %d) jump_len: %d, run_len: %d, val: %d\n",
	     position, coeff, tmp_ac_jump_len[position], tmp_ac_run_len[position],
	     tmp_ac_val[position]);
      if (tmp_ac_jump_len[position] < 0)  { printf("   EOB!\n"); coeff += 64; }
      if (tmp_ac_jump_len[position] == 0) printf("Error\n");
      coeff += tmp_ac_run_len[position];
      position += abs(tmp_ac_jump_len[position]);
    }

    coeff = 0;
    printf("(pos: %d, coeff: %d) jump_len: %d, run_len: 1, val: %d\n",
	   position, coeff, Ctmp_dc_jump_len[position], Ctmp_dc_val[position]);
    coeff += 1;
    position += Ctmp_dc_jump_len[position];

    while (coeff < 64) {
      printf("(pos: %d, coeff: %d) jump_len: %d, run_len: %d, val: %d\n",
	     position, coeff, Ctmp_ac_jump_len[position], Ctmp_ac_run_len[position],
	     Ctmp_ac_val[position]);
      if (Ctmp_ac_jump_len[position] < 0)  { printf("   EOB!\n"); coeff += 64; }
      if (Ctmp_ac_jump_len[position] == 0) printf("Error\n");
      coeff += Ctmp_ac_run_len[position];
      position += abs(Ctmp_ac_jump_len[position]);
    }

     coeff = 0;
    printf("(pos: %d, coeff: %d) jump_len: %d, run_len: 1, val: %d\n",
	   position, coeff, Ctmp_dc_jump_len[position], Ctmp_dc_val[position]);
    coeff += 1;
    position += Ctmp_dc_jump_len[position];

    while (coeff < 64) {
      printf("(pos: %d, coeff: %d) jump_len: %d, run_len: %d, val: %d\n",
	     position, coeff, Ctmp_ac_jump_len[position], Ctmp_ac_run_len[position],
	     Ctmp_ac_val[position]);
      if (Ctmp_ac_jump_len[position] < 0)  { printf("   EOB!\n"); coeff += 64; }
      if (Ctmp_ac_jump_len[position] == 0) printf("Error\n");
      coeff += Ctmp_ac_run_len[position];
      position += abs(Ctmp_ac_jump_len[position]);
      }*/

    short* tmp_block_lengths = (short*) malloc(num_threads * sizeof(short));
    short* Ctmp_block_lengths = (short*) malloc(num_threads * sizeof(short));

    cudaMemcpy(tmp_block_lengths, jpg->device_block_lengths[0].mem,
	       num_threads * sizeof(short), cudaMemcpyDeviceToHost);
    cudaMemcpy(Ctmp_block_lengths, jpg->device_block_lengths[1].mem,
	       num_threads * sizeof(short), cudaMemcpyDeviceToHost);
  
    position = 0;
    for (int block_y = 0; block_y < jpg->num_blocks_y; block_y++) {
      for (int block_x = 0; block_x < jpg->num_blocks_x; block_x++) {

	for (int sample_y = 0; sample_y < jpg->channels[0].samples_y; ++sample_y)
	  for (int sample_x = 0; sample_x < jpg->channels[0].samples_x; ++sample_x) {
            int val = tmp_block_lengths[position];
	    if(val == 0) printf(" - Error at %d -\n", position);
	    position += val;
	  }
	for (int sample_y = 0; sample_y < jpg->channels[1].samples_y; ++sample_y)
	  for (int sample_x = 0; sample_x < jpg->channels[1].samples_x; ++sample_x) {
	    int val = Ctmp_block_lengths[position]; 
	    if(val == 0) printf(" - Error at %d -\n", position);
	    position += val;
	  }
	for (int sample_y = 0; sample_y < jpg->channels[2].samples_y; ++sample_y)
	  for (int sample_x = 0; sample_x < jpg->channels[2].samples_x; ++sample_x) {
	    int val = Ctmp_block_lengths[position];
	    if(val == 0) printf(" - Error at %d -\n", position);
	    position += val;
	  }
      }
    }

    printf("GPU final position: %d\n", position);
    
    
    free(tmp_block_lengths);
    free(Ctmp_block_lengths);

    tmpval = 0;
    
    for (int block_y = 0; block_y < jpg->num_blocks_y; block_y++) {
      for (int block_x = 0; block_x < jpg->num_blocks_x; block_x++) {
	for (i = 0, channel = jpg->channels; i < jpg->num_channels; i++, channel++) {
	  for (int sample_y = 0; sample_y < channel->samples_y; ++sample_y) {
	    for (int sample_x = 0; sample_x < channel->samples_x; ++sample_x) {
	      decodeBlock(jpg, channel);
	      //THROW(PROGRAMMER_ERROR);
	      if (jpg->error) return;
	    }
	  }
	}
	//if (tmpval > 1500) THROW(PROGRAMMER_ERROR);
      }
    }
    printf("CPU final position: %d\n", tmpval);
    printf("------------------------------\n");

    
    
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
