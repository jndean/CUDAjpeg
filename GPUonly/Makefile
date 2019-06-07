TARGET   = main

SRCDIR   = src
INCDIR   = include
OBJDIR   = obj
BINDIR   = bin

SOURCES  := $(wildcard $(SRCDIR)/*.cu)
INCLUDES := $(wildcard $(INCDIR)/*.h)
OBJECTS  := $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

CFLAGS   = -O3 -I${INCDIR} -std=c++11 -g

default: main.o format.o decodeScanGPU.o decodeScan.o pixelTransformGPU.o entropyRLEdecodeGPU.o
	nvcc ${CFLAGS} $^ -o ${TARGET}

%.o: $(SRCDIR)/%.cu
	nvcc $(CFLAGS) -c $< -o $@

clean:
	rm *.o *~ src/*~ include/*~
