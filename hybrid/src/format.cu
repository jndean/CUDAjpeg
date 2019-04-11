#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include<format.h>


__host__ JPG* newJPG(const char* filename){
  FILE* f = NULL;
  unsigned int size = 0;
  
  JPG* out = (JPG*) calloc(1, sizeof(JPG));
  if (NULL == out) goto failure;

  // Read file //
  f = fopen(filename, "r");
  if (NULL == f) goto failure;
  fseek(f, 0, SEEK_END);
  size = ftell(f);
  out->buf = (unsigned char *) malloc(size);
  if(NULL == out->buf) goto failure;
  fseek(f, 0, SEEK_SET);
  if(fread(out->buf, 1, size, f) != size) goto failure;
  fclose(f);
  f=NULL;
  
  // Magics //
  if((out->buf[0]      != 0xFF) || (out->buf[1]      != 0xD8) ||
     (out->buf[size-2] != 0xFF) || (out->buf[size-1] != 0xD9))
    goto failure;

  if (size < 6) goto failure;
  out->size = size;
  out->end = out->buf + size;
  out->pos = out->buf + 2;
  out->error = NO_ERROR;
#ifdef DEBUG_TIMING
  out->time = 0;
#endif  
  return out;
  
 failure:
  if (NULL != f) fclose(f);
  if (NULL != out) delJPG(out);
  return NULL;
}


__host__ void delJPG(JPG* jpg){
  if (jpg->buf) free(jpg->buf);
  int i; ColourChannel *c;
  for(i = 0, c = jpg->channels; i < 3; i++, c++){
    if (c->pixels) free(c->pixels);
    if (c->working_space) free(c->working_space);
  }
  if(jpg->pixels) free(jpg->pixels);
  free(jpg);
}


__host__ void writeJPG(JPG* jpg, const char* filename){
  FILE* f = fopen(filename, "wb");
   if (!f) {
     printf("Couldn't open output file %s\n", filename);
     return;
   }
   fprintf(f, "P%d\n%d %d\n255\n",
	   (jpg->num_channels > 1) ? 6 : 5,
	   jpg->width, jpg->height);
   fwrite((jpg->num_channels == 1) ? jpg->channels[0].pixels : jpg->pixels,
	  1,
	  jpg->width * jpg->height * jpg->num_channels,
	  f);
   fclose(f);
}


__host__ unsigned short read16(const unsigned char *pos) {
    return (pos[0] << 8) | pos[1];
}


__host__ void skipBlock(JPG* jpg){
  jpg->pos += read16(jpg->pos);
}


__host__ void decodeSOF(JPG* jpg){
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
    
    if(((chan->width < 3) && (chan->samples_x != samples_x_max)) ||
       ((chan->height < 3) && (chan->samples_y != samples_y_max)))
      THROW(UNSUPPORTED_ERROR);

    int chan_size = chan->stride * jpg->num_blocks_y * (chan->samples_y << 3);
    chan->pixels = (unsigned char*) malloc(chan_size);
    chan->working_space = (int*) calloc(chan_size, sizeof(int));
    chan->working_space_pos = chan->working_space;
    if(!chan->pixels || !chan->working_space) THROW(OOM_ERROR);
  }
  if(jpg->num_channels == 3){
    jpg->pixels = (unsigned char*) malloc(jpg->width * jpg->height * 3);
    if(!jpg->pixels) THROW(OOM_ERROR);
  } 
    
  jpg->pos += block_len;
}


__host__ void decodeDHT(JPG* jpg){
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


__host__ void decodeDRI(JPG *jpg){
  unsigned int block_len = read16(jpg->pos);
  unsigned char *block_end = jpg->pos + block_len;
  if ((block_len < 2) || (block_end >= jpg->end)) THROW(SYNTAX_ERROR);
  jpg->restart_interval = read16(jpg->pos + 2);
  jpg->pos = block_end; 
}


__host__ void decodeDQT(JPG *jpg){
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
