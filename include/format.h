#ifndef FORMAT_H
#define FORMAT_H

#include<time.h>

// Error codes //
#define NO_ERROR 0
#define SYNTAX_ERROR 1
#define UNSUPPORTED_ERROR 2
#define OOM_ERROR 3
#define CUDA_MEM_ERROR 4
#define FILE_ERROR 5
#define CUDA_KERNEL_LAUNCH_ERROR 6
#define PROGRAMMER_ERROR -1

// Memory options for ensureMemSize //
#define USE_MALLOC 0
#define USE_CALLOC 1
#define USE_CUDA_MALLOC 2


#define THROW(e) do { \
    jpg->error = e; \
    jpg->error_line = __LINE__; \
    jpg->error_file = (char*)__FILE__; \
    jpg->error_func = (char*)__func__; \
    return; } while (0)


typedef struct _DhtVlc
{
  unsigned char tuple, num_bits;
} DhtVlc;


typedef struct _ManagedUCharMem
{
  unsigned char *mem;
  unsigned int size, max_size;
} ManagedUCharMem;

typedef struct _ManagedShortMem
{
  short *mem;
  unsigned int size, max_size;
} ManagedShortMem;

typedef struct _ManagedIntMem
{
  int *mem;
  unsigned int size, max_size;
} ManagedIntMem;


typedef struct _ColourChannel
{
  int id;
  int dq_id, ac_id, dc_id;
  int width, height;
  int samples_x, samples_y, stride, block_stride;
  int dc_cumulative_val;
  ManagedIntMem working_space;
  ManagedIntMem device_working_space;
  ManagedUCharMem device_raw_pixels;
  ManagedUCharMem raw_pixels;
  ManagedUCharMem device_pixels;
  ManagedUCharMem pixels;
  int* working_space_pos;
} ColourChannel;


typedef struct _JPGReader
{
  // File buffer pointers (None of these own the memory) //
  unsigned char *buf, *pos, *end;
  //unsigned int buf_max_size;
  unsigned int bufbits;
  unsigned char num_bufbits;
  // Imgage properties //
  unsigned short width, height;
  unsigned short num_blocks_x, num_blocks_y;
  unsigned short block_size_x, block_size_y;
  unsigned char num_channels;
  int restart_interval;
  ColourChannel channels[3];
  // Memory //
  unsigned char *pixels;
  int max_pixels_size;
  ManagedUCharMem device_pixels;
  DhtVlc *vlc_tables[4], *vlc_tables_backup[4];
  DhtVlc *device_vlc_tables[4], *device_vlc_tables_backup[4];
  unsigned char dq_tables[4][64];
  ManagedUCharMem file_buf, device_file_buf;
  ManagedShortMem device_values[4], device_jump_lengths[4];
  ManagedUCharMem device_run_lengths[2];
  ManagedShortMem device_block_lengths[2];
  ManagedIntMem device_reduced_block_lengths;
  char num_reduction_steps;
  // Debug and error handling //
  clock_t time;
  int error, error_line;
  char *error_file, *error_func;
} JPGReader;

__host__ unsigned short read16(const unsigned char *pos);
__host__ void delJPGReader(JPGReader* reader);
__host__ JPGReader* newJPGReader();
__host__ int openJPG(JPGReader* reader, const char* filename);
__host__ void writeJPG(JPGReader* jpg, const char* filename);
__host__ void printError(JPGReader* reader);
__host__ void skipBlock(JPGReader* jpg);
__host__ void decodeSOF(JPGReader* jpg);
__host__ void decodeDHT(JPGReader* jpg);
__host__ void decodeDQT(JPGReader* jpg);
__host__ void decodeDRI(JPGReader* jpg);

static const char deZigZag[64] = {
  0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48,
  41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15,
  23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};


#endif // FORMAT_H //
