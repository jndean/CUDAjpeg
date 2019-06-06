#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include<format.h>
#include<decodeScan.h>
#include<decodeScanGPU.h>
#include<pixelTransformGPU.h>



__host__ JPGReader* newJPGReader() {
  JPGReader* reader;
  if (!(reader = (JPGReader*) calloc(1, sizeof(JPGReader))))
    goto error;
  //if (cudaMalloc(&reader->deviceAddresses.dq_tables, 4*64) != cudaSuccess)
  //  goto error;
  for (int i=0; i<4; i++) {
    if (!(reader->vlc_tables_backup[i] = (DhtVlc*) malloc(65536 * sizeof(DhtVlc))))
      goto error;
    if (cudaMalloc(&reader->device_vlc_tables_backup[i], 65536 * sizeof(DhtVlc)) != cudaSuccess)
      goto error;
  }
  return reader;
  
 error:
  delJPGReader(reader);
  return NULL;
}


__host__ void delJPGReader(JPGReader* reader) {
  if (!reader) return;
  if (reader->pixels) free(reader->pixels);
  if (reader->device_pixels.mem) cudaFree(reader->device_pixels.mem);
  if (reader->file_buf.mem) free(reader->file_buf.mem);
  if (reader->device_file_buf.mem) cudaFree(reader->device_file_buf.mem);
  //if (reader->deviceAddresses.dq_tables) cudaFree(reader->deviceAddresses.dq_tables);
  int i;
  for (i = 0; i < 4; i++) {
    if (reader->device_values[i].mem) cudaFree(reader->device_values[i].mem);
    if (reader->device_jump_lengths[i].mem) cudaFree(reader->device_jump_lengths[i].mem);
    if (reader->vlc_tables_backup[i]) free(reader->vlc_tables[i]);
    if (reader->device_vlc_tables_backup[i]) cudaFree(reader->device_vlc_tables_backup[i]);
  }
  for (i = 0; i < 2; i++) {
    if (reader->device_run_lengths[i].mem) cudaFree(reader->device_run_lengths[i].mem);
    if (reader->device_block_lengths[i].mem) cudaFree(reader->device_block_lengths[i].mem);
  }
  ColourChannel *c;
  for (i = 0, c = reader->channels; i < 3; i++, c++) {
    if (c->pixels.mem) free(c->pixels.mem);
    if (c->working_space.mem) free(c->working_space.mem);
    if (c->raw_pixels.mem) free(c->raw_pixels.mem);
    if (c->device_working_space.mem) cudaFree(c->device_working_space.mem);
    if (c->device_raw_pixels.mem) cudaFree(c->device_raw_pixels.mem);
  }
  free(reader);
}


__host__ void printError(JPGReader* reader) {
  printf("Code: ");
  switch(reader->error){
  case NO_ERROR: printf("NO_ERROR"); break;
  case SYNTAX_ERROR: printf("SYNTAX_ERROR"); break;
  case UNSUPPORTED_ERROR: printf("UNSUPPORTED_ERROR"); break;
  case OOM_ERROR: printf("OOM_ERROR"); break;
  case CUDA_MEM_ERROR: printf("CUDA_MEM_ERROR"); break;
  case FILE_ERROR: printf("FILE_ERROR"); break;
  case CUDA_KERNEL_LAUNCH_ERROR: printf("CUDA_KERNEL_LAUNCH_ERROR"); break;
  case PROGRAMMER_ERROR: printf("PROGRAMMER_ERROR"); break;
  default: printf("[Unrecognised error code: %d]", reader->error);
  }
  if (reader->error_func) printf(" Func: %s", reader->error_func);
  if (reader->error_file) printf(" File: %s", reader->error_file);
  if (reader->error_line) printf(" Line: %d", reader->error_line);
}


template <class ManagedMem> // will be one of Managed(UChar|Short|Int)Mem
__host__ int ensureMemSize(ManagedMem* mem, const unsigned int new_size, int mode) {
  mem->size = new_size;
  if (mem->mem) {
    if (new_size <= mem->max_size){
      if (mode == USE_CALLOC) memset(mem->mem, 0, new_size * sizeof(*mem->mem));
      return NO_ERROR;
    }
    if (mode == USE_CUDA_MALLOC) cudaFree(mem->mem);
    else free(mem->mem);
    mem->mem = NULL;
  }
  switch (mode){
  case USE_CALLOC:
    mem->mem = (decltype(mem->mem)) calloc(new_size, sizeof(*mem->mem));
    break;
  case USE_MALLOC:
    mem->mem = (decltype(mem->mem)) malloc(new_size * sizeof(*mem->mem));
    break;
  case USE_CUDA_MALLOC:
    if (cudaMalloc(&mem->mem, new_size * sizeof(*mem->mem)) != cudaSuccess)
      return CUDA_MEM_ERROR;
    break;
  default:
    return PROGRAMMER_ERROR;
  }
  if (!(mem->mem)) return OOM_ERROR;
  mem->max_size = new_size;
  return NO_ERROR;
}


__host__ int openJPG(JPGReader* reader, const char *filename) {
  FILE* f = NULL;
  int error_val = NO_ERROR;
  unsigned int size = 0, size8;
  
  // Read file //
  f = fopen(filename, "r");
  if (NULL == f) { error_val = FILE_ERROR; goto end; }
  fseek(f, 0, SEEK_END);
  size = ftell(f);
  size8 = size * 8;
  if (error_val = ensureMemSize(&reader->file_buf, size, USE_MALLOC)) goto end;
  reader->buf = reader->file_buf.mem;
  if (error_val = ensureMemSize(&reader->device_file_buf, size + 10, USE_CUDA_MALLOC))
    goto end;
  for (int i = 0; i < 4; i++) {
    if (error_val = ensureMemSize(&reader->device_values[i], size8, USE_CUDA_MALLOC))
      goto end;
    if (error_val = ensureMemSize(&reader->device_jump_lengths[i], size8, USE_CUDA_MALLOC))
      goto end;
  }
  for (int i = 0; i < 2; i++) {
    if (error_val = ensureMemSize(&reader->device_run_lengths[i], size8, USE_CUDA_MALLOC))
      goto end;
    if (error_val = ensureMemSize(&reader->device_block_lengths[i], size8, USE_CUDA_MALLOC))
      goto end;
  }
  fseek(f, 0, SEEK_SET);
  if(fread(reader->buf, 1, size, f) != size) {
    error_val = FILE_ERROR;
    goto end;
  }
  fclose(f);
  f=NULL;
  
  // Start copying input data to GPU //
  //cudaMemcpy(reader->device_file_buf.mem, reader->file_buf.mem,
  //	     size, cudaMemcpyHostToDevice);

  // Magics //
  if((reader->buf[0]      != 0xFF) || (reader->buf[1]      != 0xD8) ||
     (reader->buf[size-2] != 0xFF) || (reader->buf[size-1] != 0xD9)) {
      error_val = FILE_ERROR; goto end;
  }

  if (size < 6) {error_val = SYNTAX_ERROR; goto end;}
  reader->end = reader->buf + size - 2; // Leave out EOF marker, already checked
  reader->pos = reader->buf + 2;
  // Shows decodeDRI hasn't been run yet //
  reader->restart_interval = 0;
  // Shows decodeSOF hasn't been run yet //
  reader->num_blocks_x = reader->num_blocks_y = 0;
  reader->bufbits = 0;
  reader->num_bufbits = 0;
  reader->time = 0;
  reader->error = NO_ERROR;

  
  // Main format block parsing loop //
  while(!reader->error){
    if (reader->pos > reader->end) {
      reader->error = SYNTAX_ERROR;
      break;
    }
    if (reader->pos[0] != 0xFF) {
      reader->error = SYNTAX_ERROR;
      break;
    }
    
    reader->pos += 2;
    switch(reader->pos[-1]) {
    case 0xC0: decodeSOF(reader); break;
    case 0xC4: decodeDHT(reader); break;
    case 0xDB: decodeDQT(reader); break;
    case 0xDD: decodeDRI(reader); break;
    case 0xDA: decodeScanGPU(reader); break;
    case 0xFE: skipBlock(reader); break;
    case 0xD9: break;
    default:
      if((reader->pos[-1] & 0xF0) == 0xE0) skipBlock(reader);
      else reader->error = SYNTAX_ERROR;
    }
    
    // Finished //
    if (reader->pos[-1] == 0xD9) {
      iDCT_resample_colourTransform(reader);
      break;
    }
  }

  if(reader->error) error_val = reader->error;

 end:
  if (NULL != f) fclose(f);
  return error_val;
}


__host__ void writeJPG(JPGReader* reader, const char* filename){
  FILE* f = fopen(filename, "wb");
   if (!f) {
     printf("Couldn't open output file %s\n", filename);
     return;
   }
   fprintf(f, "P%d\n%d %d\n255\n", (reader->num_channels > 1) ? 6 : 5,
	   reader->width, reader->height);
   fwrite((reader->num_channels == 1) ? reader->channels[0].pixels.mem : reader->pixels,
	  1, reader->width * reader->height * reader->num_channels, f);
   fclose(f);
}


__host__ unsigned short read16(const unsigned char *pos) {
    return (pos[0] << 8) | pos[1];
}


__host__ void skipBlock(JPGReader* jpg){
  jpg->pos += read16(jpg->pos);
}


__host__ void decodeSOF(JPGReader* jpg){
  int error = NO_ERROR;
  unsigned char* block = jpg->pos;
  unsigned int block_len = read16(block);
  if(block_len < 9 || block + block_len >= jpg->end)
    THROW(SYNTAX_ERROR);
  if(block[2] != 8)
    THROW(UNSUPPORTED_ERROR);

  // Read image info //
  jpg->height = read16(block+3);
  jpg->width = read16(block+5);
  if(!jpg->width || !jpg->height)
    THROW(SYNTAX_ERROR);
  jpg->num_channels = block[7];
  if(jpg->num_channels != 1 && jpg->num_channels != 3)
    THROW(UNSUPPORTED_ERROR);

  // Read channel info //
  if (block_len < 8 + (jpg->num_channels * 3))
    THROW(SYNTAX_ERROR);
  block += 8;
  int i, samples_x_max = 0, samples_y_max = 0;
  ColourChannel *chan = jpg->channels;
  for(i = 0; i < jpg->num_channels; i++, chan++, block += 3){
    chan->id = block[0];
    chan->samples_x = block[1] >> 4;
    chan->samples_y = block[1] & 0xF;
    chan->dq_id = block[2];
    
    if(!chan->samples_x || !chan->samples_y || chan->dq_id > 3)
      THROW(SYNTAX_ERROR);
    if((chan->samples_x & (chan->samples_x - 1)) ||
       (chan->samples_y & (chan->samples_y - 1)))
      THROW(UNSUPPORTED_ERROR); // require power of two
    if(chan->samples_x > samples_x_max) samples_x_max = chan->samples_x;
    if(chan->samples_y > samples_y_max) samples_y_max = chan->samples_y;
  }
  
  if (jpg->num_channels == 1){
    jpg->channels[0].samples_x = samples_x_max = 1;
    jpg->channels[0].samples_y = samples_y_max = 1;
  }

  // Compute dimensions in blocks and allocate output space //
  jpg->block_size_x = samples_x_max << 3;
  jpg->block_size_y = samples_y_max << 3;
  jpg->num_blocks_x = (jpg->width + jpg->block_size_x -1) / jpg->block_size_x;
  jpg->num_blocks_y = (jpg->height + jpg->block_size_y -1) / jpg->block_size_y;

  for(i = 0, chan = jpg->channels; i < jpg->num_channels; i++, chan++){
    chan->width = (jpg->width * chan->samples_x + samples_x_max -1) / samples_x_max;
    chan->height = (jpg->height * chan->samples_y + samples_y_max -1) / samples_y_max;
    chan->stride = jpg->num_blocks_x * (chan->samples_x << 3);
    chan->dc_cumulative_val = 0;
    
    if(((chan->width < 3) && (chan->samples_x != samples_x_max)) ||
       ((chan->height < 3) && (chan->samples_y != samples_y_max)))
      THROW(UNSUPPORTED_ERROR);

    // If more is needed, allocate CPU and/or GPU memory //
    unsigned int chan_size = chan->stride * jpg->num_blocks_y * (chan->samples_y << 3);
    if (error = ensureMemSize(&chan->working_space, chan_size, USE_CALLOC))
      THROW(error);
    if (error = ensureMemSize(&chan->device_working_space, chan_size, USE_CUDA_MALLOC))
      THROW(error);
    if (error = ensureMemSize(&chan->device_raw_pixels, chan_size, USE_CUDA_MALLOC))
      THROW(error);
    if (error = ensureMemSize(&chan->raw_pixels, chan_size, USE_MALLOC))
      THROW(error);
    chan->working_space_pos = chan->working_space.mem;

    int out_width = chan->width, out_height = chan->height;
    while (out_width < jpg->width) out_width <<= 1;
    while (out_height < jpg->height) out_height <<= 1;
    int chan_out_size = out_width * out_height;
    if (error = ensureMemSize(&chan->pixels, chan_out_size, USE_MALLOC))
      THROW(error);
    if (error = ensureMemSize(&chan->device_pixels, chan_out_size, USE_CUDA_MALLOC))
      THROW(error);
  }

  // This is here and in decodeDRI, as their order isn't guarenteed //
  /*if (jpg->restart_interval) {
    int num_blocks = jpg->num_blocks_y * jpg->num_blocks_x * jpg->num_channels;
    int num_restarts = num_blocks / jpg->restart_interval;
    if (error = ensureMemSize(&jpg->restart_marker_positions, num_restarts + 1, USE_MALLOC))
      THROW(error);
      }*/
  
  if (jpg->num_channels == 3){
    int pixels_size = jpg->width * jpg->height * 3;
    if ((!jpg->pixels) || (jpg->max_pixels_size < pixels_size)){
      if (jpg->pixels) free(jpg->pixels);
      jpg->pixels = (unsigned char*) malloc(pixels_size);
      if (!jpg->pixels) THROW(OOM_ERROR);
      if (jpg->max_pixels_size < pixels_size)
	jpg->max_pixels_size = pixels_size;
    }

    int error;
    if (error = ensureMemSize(&jpg->device_pixels, pixels_size, USE_CUDA_MALLOC))
      THROW(error);

    if ((jpg->channels[1].width != jpg->channels[2].width) ||
	(jpg->channels[1].height != jpg->channels[2].height) ||
	(jpg->channels[0].width != jpg->width) ||
	(jpg->channels[0].height != jpg->height))
      THROW(UNSUPPORTED_ERROR); // I thought this never happened
  } 

  jpg->pos += block_len;
}


__host__ void decodeDHT(JPGReader* jpg){
  unsigned char* pos = jpg->pos;
  unsigned int block_len = read16(pos);
  unsigned char *block_end = pos + block_len;
  if(block_end > jpg->end) THROW(SYNTAX_ERROR);
  pos += 2;
    
  while(pos < block_end){
    unsigned char val = pos[0];
    if (val & 0xEC) THROW(SYNTAX_ERROR);
    if (val & 0x02) THROW(UNSUPPORTED_ERROR);
    unsigned char table_id = (val | (val >> 3)) & 3; // AC and DC
    DhtVlc *vlc = jpg->vlc_tables_backup[table_id];

    unsigned char *tuple = pos + 17;
    int remain = 65536, spread = 65536;
    for (int code_len = 1; code_len <= 16; code_len++){
      spread >>= 1;
      int count = pos[code_len];
      if (!count) continue;
      if (tuple + count > block_end) THROW(SYNTAX_ERROR);
      
      remain -= count << (16 - code_len);
      if (remain < 0) THROW(SYNTAX_ERROR);
      for(int i = 0; i < count; i++, tuple++){
	for(int j = spread; j; j--, vlc++){
	  vlc->num_bits = (unsigned char) code_len;
	  vlc->tuple = *tuple;
	}
      }
    }
    while(remain--){
      vlc->num_bits = 0;
      vlc++;
    }
    pos = tuple;

    cudaMemcpy(jpg->device_vlc_tables_backup[table_id], jpg->vlc_tables_backup[table_id],
	       65536 * sizeof(DhtVlc), cudaMemcpyHostToDevice);
  }
  
  if (pos != block_end) THROW(SYNTAX_ERROR);
  jpg->pos = block_end;
}



__host__ void decodeDRI(JPGReader *jpg) {
  unsigned int block_len = read16(jpg->pos);
  unsigned char *block_end = jpg->pos + block_len;
  if ((block_len < 2) || (block_end > jpg->end)) THROW(SYNTAX_ERROR);
  jpg->restart_interval = read16(jpg->pos + 2);

  // This is here and in decodeSOF, as their order isn't guarenteed //
  /*if (jpg->num_blocks_x) {
    int error;
    int num_blocks = jpg->num_blocks_y * jpg->num_blocks_x * jpg->num_channels;
    int num_restarts = num_blocks / jpg->restart_interval;
    if (error = ensureMemSize(&jpg->restart_marker_positions, num_restarts + 1, USE_MALLOC))
      THROW(error);
    //printf(" ## %d, %d, %d ## \n", num_blocks, jpg->restart_interval, num_restarts);
    }*/
  
  jpg->pos = block_end; 
}


__host__ void decodeDQT(JPGReader *jpg){
  unsigned int block_len = read16(jpg->pos);
  unsigned char *block_end = jpg->pos + block_len;
  if (block_end > jpg->end) THROW(SYNTAX_ERROR);
  unsigned char *pos = jpg->pos + 2;

  while(pos + 65 <= block_end){
    unsigned char table_id = pos[0];
    if (table_id & 0xFC) THROW(SYNTAX_ERROR);
    unsigned char *table = &jpg->dq_tables[table_id][0];
    memcpy(table, pos+1, 64);
    pos += 65;
  }
  if (pos != block_end) THROW(SYNTAX_ERROR);
  jpg->pos = block_end;
  
  //cudaMemcpy(jpg->deviceAddresses.dq_tables, jpg->dq_tables,
  //4 * 64, cudaMemcpyHostToDevice);
}
