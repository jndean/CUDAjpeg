#ifndef IDCT_H
#define IDCT_H


#define W1 2841
#define W2 2676
#define W3 2408
#define W5 1609
#define W6 1108
#define W7 565


void iDCT_row(int* D);
void iDCT_col(const int* D, unsigned char *out, int stride);
void upsampleChannel(JPG* jpg, ColourChannel* channel);
void upsampleAndColourTransform(JPG* jpg);


#endif // IDCT_H //
