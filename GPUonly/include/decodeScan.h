#ifndef DECODESCANCPU_H
#define DECODESCANCPU_H

#include<format.h>

__host__ int showBits(JPGReader* jpgreader, int num_bits);
__host__ int getBits(JPGReader* jpgreader, int num_bits);
__host__ int getVLC(JPGReader* jpgreader, DhtVlc* vlc, unsigned char* code);
__host__ void decodeBlock(JPGReader* jpgreader, ColourChannel* c);
__host__ void decodeScanCPU(JPGReader* jpgreader);

#endif // DECODESCANCPU_H //
