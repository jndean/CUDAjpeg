#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<time.h>

#include<format.h>
#include<decodeScan.h>




int main(int argc, char** argv){
  
  if(argc < 2) {
    fprintf(stderr, "USAGE: %s filename.jpg ...\n", argv[0]);
    return EXIT_FAILURE;
  }

  cudaSetDevice(1);
  cudaDeviceReset();

  JPGReader* reader = newJPGReader();
  if (!reader) {
    fprintf(stderr, "Unable to create jpgreader, likely malloc failure\n");
    return EXIT_FAILURE;
  }

  int error = openJPG(reader, argv[1]);
  if (!error) {
    const char* filename = (reader->num_channels == 1) ? "outfile.pgm" : "outfile.ppm";
    writeJPG(reader, filename);
  }
  
  
  clock_t cumulative_time = 0;
  int i, n = 50;
  double total_time = 0;
  for (i=0; i<n; i++){
    int filename_id = 1 + (i % (argc - 1));
    clock_t start = clock();
    error = openJPG(reader, argv[filename_id]);
    total_time += (clock() - start);
    if (error){
      printf("Failed to open jpg %s, error code: ", argv[filename_id]);
      printError(reader); printf("\n");
    }
    
    cumulative_time += reader->time;
  }

  delJPGReader(reader);
  
  double t_pi = 1000.0 * (double) total_time / (n * CLOCKS_PER_SEC);
  printf("%0.3lfms per image\n", t_pi);
  
  double t = 1000.0 * (double) cumulative_time / CLOCKS_PER_SEC / n;
  printf("DEBUG_TIME %0.4lfms, %0.3lf%%\n", t, 100*t/t_pi);
 

  cudaDeviceSynchronize();
  cudaDeviceReset();
  return EXIT_SUCCESS;
}
