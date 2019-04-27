TARGET   = main

SRCDIR   = src
INCDIR   = include
OBJDIR   = obj
BINDIR   = bin

SOURCES  := $(wildcard $(SRCDIR)/*.cu)
INCLUDES := $(wildcard $(INCDIR)/*.h)
OBJECTS  := $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

CFLAGS   = -O3 -I${INCDIR}

default: main.o format.o decodeScan.o pixelTransformCPU.o pixelTransformGPU.o
	nvcc ${CFLAGS} $^ -o ${TARGET}

decodeScanCPU.o: pixelTransformGPU.o

%.o: $(SRCDIR)/%.cu
	nvcc $(CFLAGS) -c $< -o $@

clean:
	rm *.o *~ src/*~ include/*~
