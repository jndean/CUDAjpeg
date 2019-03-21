#ifndef UTILITIES_H
#define UTILITIES_H

#include<format.h>


#define NO_ERROR 0
#define SYNTAX_ERROR 1
#define UNSUPPORTED_ERROR 2
#define OOM_ERROR 3

#define THROW(e) do { jpg->error = e; return; } while (0)


unsigned short read16(const unsigned char *pos);
void skipBlock(JPG* jpg);

#endif // UTILITIES_H //
