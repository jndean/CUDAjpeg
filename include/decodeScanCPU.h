#ifndef DECODESCANCPU_H
#define DECODESCANCPU_H

#include<format.h>

int showBits(JPG* jpg, int num_bits);
int getBits(JPG* jpg, int num_bits);
int getVLC(JPG* jpg, DhtVlc* vlc, unsigned char* code);
void decodeBlock(JPG* jpg, ColourChannel* c, unsigned char* out);
void decodeScanCPU(JPG* jpg);

#endif // DECODESCANCPU_H //
