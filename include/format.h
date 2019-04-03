#ifndef FORMAT_H
#define FORMAT_H


typedef struct _DhtVlc
{
  unsigned char tuple, num_bits;
} DhtVlc;


typedef struct _ColourChannel
{
  int id;
  int qt_id, ac_id, dc_id;
  int width, height;
  int samples_x, samples_y, stride;
  unsigned char *pixels;
  int dc_cumulative_val;
} ColourChannel;


typedef struct _JPG
{
  // Pointers to start, position, end of file buffer //
  unsigned char *buf, *pos, *end;
  // Size of file buffer //
  unsigned int size;
  // Dimensions of image in pixels //
  unsigned short width, height;
  // Number of 8x8 blocks //
  unsigned short num_blocks_x, num_blocks_y;
   // Sample scales //
  unsigned short block_size_x, block_size_y;
  unsigned char num_channels;
  int error;
  // Info on each ccolour channel //
  ColourChannel channels[3];
  // Output pixel array //
  unsigned char *pixels;
  // Lookup table for huffman decoding each kind of channel //
  DhtVlc vlc_tables[4][65536];
  // Dequantisation tables for each kind of channel //
  unsigned char dq_tables[4][64];
  // Bit flags for presence and use of each table //
  //int dq_table_available, dq_table_used;
  // RS interval //
  int restart_interval;
  // Buffer bits already read during huffman decode //
  unsigned int bufbits;
  unsigned char num_bufbits;
  // Working space for current block decode //
  int block_space[64];
} JPG;

  
void delJPG(JPG* jpg);
JPG* newJPG(const char* filename);

void decodeSOF(JPG* jpg);
void decodeDHT(JPG* jpg);
void decodeDQT(JPG *jpg);
void decodeDRI(JPG *jpg);
void decodeSOS(JPG* jpg);


#endif // FORMAT_H //
