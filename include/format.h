#ifndef FORMAT_H
#define FORMAT_H


typedef struct _DhtVlc
{
  unsigned char tuple, num_bits;
} DhtVlc;


typedef struct _ColourChannel
{
  int id;
  int dq_id, ac_id, dc_id;
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

static const char deZigZag[64] = { 0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18,
11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35,
42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45,
38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63 };


#endif // FORMAT_H //
