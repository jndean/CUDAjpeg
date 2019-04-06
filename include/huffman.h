#ifndef HUFFMAN_H
#define HUFFMAN_H

int showBits(JPG* jpg, int num_bits);
int getBits(JPG* jpg, int num_bits);
int getVLC(JPG* jpg, DhtVlc* vlc, unsigned char* code);

void decodeBlock(JPG* jpg, ColourChannel* c, unsigned char* out);

#endif // HUFFMAN_H //
