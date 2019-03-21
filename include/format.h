#ifndef FORMAT_H
#define FORMAT_H


typedef struct _ColourChannel
{
  int id;
  int qtid, acit, dcid;
  int width, height;
  int samples_x, samples_y, stride;
  unsigned char *pixels;
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
} JPG;

void delJPG(JPG* jpg);
JPG* newJPG(const char* filename);

void decodeSOF(JPG* jpg);
void decodeDHT(JPG* jpg);


#endif // FORMAT_H //
