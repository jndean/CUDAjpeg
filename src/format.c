#include<stdio.h>
#include<stdlib.h>

#include<format.h>
#include<utilities.h>

JPG* newJPG(const char* filename){
  JPG* out = calloc(1, sizeof(JPG));
  if (NULL == out) goto failure;

  // Read file //
  FILE* f = fopen(filename, "r");
  if (NULL == f) goto failure;
  fseek(f, 0, SEEK_END);
  unsigned int size = ftell(f);
  out->buf = malloc(size);
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
  return out;
  
 failure:
  if (NULL != f) fclose(f);
  if (NULL != out) delJPG(out);
  return NULL;
}

void delJPG(JPG* jpg){
  if (NULL != jpg->buf) free(jpg->buf);
  int i; ColourChannel *c;
  for(i = 0, c = jpg->channels; i < 3; i++, c++)
    if (c->pixels) free(c->pixels);
  if(jpg->pixels) free(jpg->pixels);
  free(jpg);
}


void decodeSOF(JPG* jpg){
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
    chan->qtid = block[2];
    
    if(!chan->samples_x || !chan->samples_y || chan->qtid > 3)
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
    
    chan->pixels = malloc(chan->stride * jpg->num_blocks_y * (chan->samples_y << 3));
    if(!chan->pixels) THROW(OOM_ERROR);
  }
  if(jpg->num_channels == 3){
    jpg->pixels = malloc(jpg->width * jpg->height * 3);
    if(!jpg->pixels) THROW(OOM_ERROR);
  } 
    
  jpg->pos += block_len;
}


void decodeDHT(JPG* jpg){
  unsigned char* block = jpg->pos;
  unsigned int remaining = read16(block);
  if(block + remaining >= jpg->end) THROW(SYNTAX_ERROR);

  static unsigned char counts[16];

  while(remaining > 16){
    unsigned char val = block[0];
    if (val & 0xEC) THROW(SYNTAX_ERROR);
    if (val & 0x02) THROW(UNSUPPORTED_ERROR);
    val = (val | (val >> 3)) & 3;
    for (int code_length = 1; code_length <= 16; code_length++)
      
  }
}
  
