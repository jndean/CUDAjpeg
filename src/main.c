#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include<utilities.h>
#include<format.h>




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
    //printf("%x\n", (unsigned int)(jpg->pos - jpg->buf));
    switch(jpg->pos[-1]) {
    case 0xC0: printf("DecodeSOF()\n");decodeSOF(jpg); break;
    case 0xC4: printf("DecodeDHT()\n"); skipBlock(jpg);  break;
    case 0xDB: printf("DecodeDQT()\n"); skipBlock(jpg);  break;
    case 0xDD: printf("DecodeDRI()\n"); skipBlock(jpg);  break;
    case 0xDA: printf("DecodeScan()\n"); skipBlock(jpg); break;
    case 0xFE: skipBlock(jpg); break;
    default:
      if((jpg->pos[-1] & 0xF0) == 0xE0) skipBlock(jpg);
      else jpg->error = SYNTAX_ERROR;
    }

    // Finished //
    if (jpg->pos[-1] == 0xD9) break;
  }

  if(jpg->error){
    fprintf(stderr, "Decode failed with error code %d\n", jpg->error);
    delJPG(jpg);
    return EXIT_FAILURE;
  }


  
  
  delJPG(jpg);

  return EXIT_SUCCESS;
}
