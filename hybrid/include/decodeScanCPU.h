#ifndef DECODESCANCPU_H
#define DECODESCANCPU_H

#include<format.h>

__host__ int showBits(JPG* jpg, int num_bits);
__host__ int getBits(JPG* jpg, int num_bits);
__host__ int getVLC(JPG* jpg, DhtVlc* vlc, unsigned char* code);
__host__ void decodeBlock(JPG* jpg, ColourChannel* c, unsigned char* out);
__host__ void decodeScanCPU(JPG* jpg);

#endif // DECODESCANCPU_H //
