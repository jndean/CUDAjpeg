#ifndef FORMAT_H
#define FORMAT_H

#define NO_ERROR 0
#define SYNTAX_ERROR 1
#define UNSUPPORTED_ERROR 2
#define OOM_ERROR 3

#define THROW(e) do { jpg->error = e; return; } while (0)


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
  unsigned char *buf, *pos, *end;
  unsigned int size;
  unsigned short width, height;
  unsigned short num_blocks_x, num_blocks_y;
  unsigned short block_size_x, block_size_y;
  unsigned char num_channels;
  int error;
  ColourChannel channels[3];
  unsigned char *pixels;
  DhtVlc vlc_tables[4][65536];
  unsigned char dq_tables[4][64];
  int restart_interval;
  unsigned int bufbits;
  unsigned char num_bufbits;
  int block_space[64];
} JPG;


__host__ unsigned short read16(const unsigned char *pos);

__host__ void delJPG(JPG* jpg);
__host__ JPG* newJPG(const char* filename);
__host__ void writeJPG(JPG* jpg, const char* filename);
__host__ void skipBlock(JPG* jpg);
__host__ void decodeSOF(JPG* jpg);
__host__ void decodeDHT(JPG* jpg);
__host__ void decodeDQT(JPG *jpg);
__host__ void decodeDRI(JPG *jpg);

static const char deZigZag[64] = {
  0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48,
  41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15,
  23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};


#endif // FORMAT_H //
