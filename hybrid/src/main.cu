#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<time.h>

#include<format.h>
#include<decodeScanCPU.h>
#include<pixelTransformCPU.h>




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

  clock_t cumulative_time = 0;
  int i, n = 20;
  double total_time = 0;
  for (i=0; i<n; i++){
    int filename_id = 1 + (i % (argc - 1));
    clock_t start = clock();
    fflush(stdout);
    int error = openJPG(reader, argv[filename_id]);
    fflush(stdout);
    total_time += (clock() - start);
    if (error)
      printf("Failed to open jpg %s, error code %d\n", argv[filename_id], error);
    cumulative_time += reader->time;
  }

  fflush(stdout);

  double t_pi = 1000.0 * (double) total_time / (n * CLOCKS_PER_SEC);
  printf("%0.3lfms per image\n", t_pi);
  
  double t = 1000.0 * (double) cumulative_time / CLOCKS_PER_SEC / n;
  printf("DEBUG_TIME %lfms, %0.4lf%%\n", t, 100*t/t_pi);

  if(1){
    int error = openJPG(reader, argv[1]);
    if (!error) {
      const char* filename = (reader->num_channels == 1) ? "outfile.pgm" : "outfile.ppm";
      writeJPG(reader, filename);
    }
  }
  delJPGReader(reader);
  
  cudaDeviceReset();
  return EXIT_SUCCESS;
}
