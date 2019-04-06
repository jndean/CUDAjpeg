#include<stdio.h>
#include<string.h>

#include<format.h>
#include<huffman.h>
#include<utilities.h>


// This only shows the bits, but doesn't move past them //
int showBits(JPG* jpg, int num_bits){
  unsigned char newbyte;
  if(!num_bits) return 0;

  while (jpg->num_bufbits < num_bits){
    if(jpg->pos >= jpg->end){
      jpg->bufbits = (jpg->bufbits << 8) | 0xFF;
      jpg->num_bufbits += 8;
      continue;
    }
    newbyte = *jpg->pos++;
    jpg->bufbits = (jpg->bufbits << 8) | newbyte;
    jpg->num_bufbits += 8;
    if (newbyte != 0xFF)
      continue;
	
    if(jpg->pos >= jpg->end)
      goto overflow_error;
    
    // Handle byte stuffing //
    unsigned char follow_byte = *jpg->pos++;
    switch (follow_byte){
    case 0x00:
    case 0xFF:
    case 0xD9:
      break;
    default:
      if ((follow_byte & 0xF8) != 0xD0){
	printf("The follow_byte case that doesn't have to be 0x00?\n");
	goto overflow_error;
      } else {
	printf("The acceptable non-zero followbyte case?\n");
	jpg->bufbits = (jpg->bufbits << 8) | newbyte;
	jpg->num_bufbits += 8;
      }
    }
  }
  return (jpg->bufbits >> (jpg->num_bufbits - num_bits)) & ((1 << num_bits) - 1);

 overflow_error:
  printf("Huffman decode overflow?\n");
  jpg->error = SYNTAX_ERROR;
  return 0;
}


// Show the bits AND move past them //
int getBits(JPG* jpg, int num_bits){
  int res = showBits(jpg, num_bits);
  jpg->num_bufbits -= num_bits;
  return res;
}


int getVLC(JPG* jpg, DhtVlc* vlc_table, unsigned char* code){
  int symbol = showBits(jpg, 16);
  DhtVlc vlc = vlc_table[symbol];
  if(!vlc.num_bits){
    jpg->error = SYNTAX_ERROR;
    return 0;
  }
  jpg->num_bufbits -= vlc.num_bits;  
  if(code) *code = vlc.tuple;
  unsigned char num_bits = vlc.tuple & 0x0F;
  if (!num_bits) return 0;
  int value = getBits(jpg, num_bits);
  if (value < (1 << (num_bits - 1)))
    value += ((-1) << num_bits) + 1;
  return value;  
}


void decodeBlock(JPG* jpg, ColourChannel* channel, unsigned char* out){
  unsigned char code = 0;
  int value, coef = 0;
  int* block = jpg->block_space;
  memset(block, 0, 64 * sizeof(int));

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

  // ## Do iDCT here ## //
}
