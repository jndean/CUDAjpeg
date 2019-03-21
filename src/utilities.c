#include<utilities.h>
#include<format.h>

#include<stdio.h>

unsigned short read16(const unsigned char *pos) {
    return (pos[0] << 8) | pos[1];
}

void skipBlock(JPG* jpg){
  jpg->pos += read16(jpg->pos);
}
