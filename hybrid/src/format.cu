#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include<format.h>
#include<decodeScanCPU.h>
#include<pixelTransformCPU.h>



__host__ JPGReader* newJPGReader(){
  return (JPGReader*) calloc(1, sizeof(JPGReader));
}



__host__ int openJPG(JPGReader* reader, const char *filename){
  FILE* f = NULL;
  int error_val = NO_ERROR;
  unsigned int size = 0;
  
  // Read file //
  f = fopen(filename, "r");
  if (NULL == f) { error_val = FILE_ERROR; goto end; }
  fseek(f, 0, SEEK_END);
  size = ftell(f);
  if ((size > reader->buf_size) || (!reader->buf)) {
    if (reader->buf) free(reader->buf);
    reader->buf = (unsigned char *) malloc(size);
    if(!reader->buf) { error_val = OOM_ERROR; goto end; }
  }
  fseek(f, 0, SEEK_SET);
  if(fread(reader->buf, 1, size, f) != size)
    { error_val = FILE_ERROR; goto end; }
  fclose(f);
  f=NULL;

  // Magics //
  if((reader->buf[0]      != 0xFF) || (reader->buf[1]      != 0xD8) ||
     (reader->buf[size-2] != 0xFF) || (reader->buf[size-1] != 0xD9)) {
      error_val = FILE_ERROR; goto end;
  }

  if (size < 6) {error_val = SYNTAX_ERROR; goto end;}
  reader->buf_size = size;
  reader->end = reader->buf + size;
  reader->pos = reader->buf + 2;
  reader->restart_interval = 0;
  reader->bufbits = reader->num_bufbits = reader->time = 0;
  reader->error = NO_ERROR;

  
  // Main format block parsing loop //
  while(!reader->error){
    if (reader->pos > reader->end - 2) {
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
    case 0xDA: decodeScanCPU(reader); break;
    case 0xFE: skipBlock(reader); break;
    case 0xD9: break;
    default:
      if((reader->pos[-1] & 0xF0) == 0xE0) skipBlock(reader);
      else reader->error = SYNTAX_ERROR;
    }

    // Finished //
    if (reader->pos[-1] == 0xD9) {
      upsampleAndColourTransform(reader);
      break;
    }
  }

  if(reader->error) error_val = reader->error;

 end:
  if (NULL != f) fclose(f);
  return error_val;
}



__host__ void delJPGReader(JPGReader* reader){
  if (reader->buf) free(reader->buf);
  int i; ColourChannel *c;
  for(i = 0, c = reader->channels; i < 3; i++, c++){
    if (c->pixels) free(c->pixels);
    if (c->working_space) free(c->working_space);
  }
  if(reader->pixels) free(reader->pixels);
  free(reader);
}



__host__ void writeJPG(JPGReader* reader, const char* filename){
  FILE* f = fopen(filename, "wb");
   if (!f) {
     printf("Couldn't open output file %s\n", filename);
     return;
   }
   fprintf(f, "P%d\n%d %d\n255\n", (reader->num_channels > 1) ? 6 : 5,
	   reader->width, reader->height);
   fwrite((reader->num_channels == 1) ? reader->channels[0].pixels : reader->pixels,
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

    int chan_size = chan->stride * jpg->num_blocks_y * (chan->samples_y << 3);
    if ((!chan->pixels) || (chan_size > chan->size)){
      if (chan->pixels) free(chan->pixels);
      chan->pixels = (unsigned char*) malloc(chan_size);
    }

    if ((!chan->working_space) || (chan_size > chan->size)){
      if (chan->working_space) free(chan->working_space);
      chan->working_space = (int*) calloc(chan_size, sizeof(int));
    } else
      memset(chan->working_space, 0, chan_size * sizeof(int));

    chan->working_space_pos = chan->working_space;
    chan->size = chan_size;
    if(!chan->pixels || !chan->working_space) THROW(OOM_ERROR);
  }
  
  if(jpg->num_channels == 3){
    int pixels_size = jpg->width * jpg->height * 3;
    if ((!jpg->pixels) || (jpg->pixels_size < pixels_size)){
      if (jpg->pixels) free(jpg->pixels);
      jpg->pixels = (unsigned char*) malloc(pixels_size);
      jpg->pixels_size = pixels_size;
      if(!jpg->pixels) THROW(OOM_ERROR);
    }
  } 
    
  jpg->pos += block_len;
}


__host__ void decodeDHT(JPGReader* jpg){
  unsigned char* pos = jpg->pos;
  unsigned int block_len = read16(pos);
  unsigned char *block_end = pos + block_len;
  if(block_end >= jpg->end) THROW(SYNTAX_ERROR);
  pos += 2;
  
  while(pos < block_end){
    unsigned char val = pos[0];
    if (val & 0xEC) THROW(SYNTAX_ERROR);
    if (val & 0x02) THROW(UNSUPPORTED_ERROR);
    unsigned char table_id = (val | (val >> 3)) & 3; // AC and DC
    DhtVlc *vlc = &jpg->vlc_tables[table_id][0];

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
  }
  
  if (pos != block_end) THROW(SYNTAX_ERROR);
  jpg->pos = block_end;
}


__host__ void decodeDRI(JPGReader *jpg){
  unsigned int block_len = read16(jpg->pos);
  unsigned char *block_end = jpg->pos + block_len;
  if ((block_len < 2) || (block_end >= jpg->end)) THROW(SYNTAX_ERROR);
  jpg->restart_interval = read16(jpg->pos + 2);
  jpg->pos = block_end; 
}


__host__ void decodeDQT(JPGReader *jpg){
  unsigned int block_len = read16(jpg->pos);
  unsigned char *block_end = jpg->pos + block_len;
  if (block_end >= jpg->end) THROW(SYNTAX_ERROR);
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
}
