#ifndef IDCT_H
#define IDCT_H


__host__ void iDCT_row(int* D);
__host__ void iDCT_col(const int* D, unsigned char *out, int stride);
__host__ void upsampleChannel(JPGReader* jpgreader, ColourChannel* channel);
__host__ void upsampleAndColourTransform(JPGReader* jpgreader);


#endif // IDCT_H //
