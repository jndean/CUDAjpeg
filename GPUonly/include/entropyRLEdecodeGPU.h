#ifndef ENTROPYRLEDECODEGPU_H
#define ENTROPYRLEDECODEGPU_H

#define EOB_MARKER 20

__host__ void restartMarkerScan(JPGReader *jpg);
__global__ void huffmanDecode_kernel(unsigned char* in,
				     int num_threads,
				     DeviceAddressBook addresses);

#endif // ENTROPYRLEDECODEGPU_H //
