#include<stdio.h>
#include<stdlib.h>

#include<format.h>
#include<decodeScanCPU.h>
#include<pixelTransformCPU.h>


int main(int argc, char** argv){
  
  if(2 != argc) {
    fprintf(stderr, "USAGE: %s filename.jpg\n", argv[0]);
    return EXIT_FAILURE;
  }


  JPG* jpg = newJPG(argv[1]);
  if (NULL == jpg){
    printf("Failed to read %s\n", argv[1]);
    return EXIT_FAILURE;
  }

  // Main format block parsing loop //
  while(!jpg->error){
    if (jpg->pos > jpg->end - 2) {
      jpg->error = SYNTAX_ERROR;
      break;
    }
    if (jpg->pos[0] != 0xFF) {
      jpg->error = SYNTAX_ERROR;
      break;
    }
    
    jpg->pos += 2;
    switch(jpg->pos[-1]) {
    case 0xC0: decodeSOF(jpg); break;
    case 0xC4: decodeDHT(jpg); break;
    case 0xDB: decodeDQT(jpg); break;
    case 0xDD: decodeDRI(jpg); break;
    case 0xDA: decodeScanCPU(jpg); break;
    case 0xFE: skipBlock(jpg); break;
    case 0xD9: break;
    default:
      if((jpg->pos[-1] & 0xF0) == 0xE0) skipBlock(jpg);
      else jpg->error = SYNTAX_ERROR;
    }

    // Finished //
    if (jpg->pos[-1] == 0xD9) {
      upsampleAndColourTransform(jpg);
      break;
    }
  }

  if(jpg->error){
    fprintf(stderr, "Decode failed with error code %d\n", jpg->error);
    delJPG(jpg);
    return EXIT_FAILURE;
  }
  printf("Successful jpeg decode\n");

  char* filename = (jpg->num_channels == 1) ? "outfile.pgm" : "outfile.ppm";
  writeJPG(jpg, filename);
  
  delJPG(jpg);
  return EXIT_SUCCESS;
}
