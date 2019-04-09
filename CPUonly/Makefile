TARGET   = main

SRCDIR   = src
INCDIR   = include
OBJDIR   = obj
BINDIR   = bin

SOURCES  := $(wildcard $(SRCDIR)/*.c)
INCLUDES := $(wildcard $(INCDIR)/*.h)
OBJECTS  := $(SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)

CFLAGS   = -std=c99 -Wall -I${INCDIR} -O3

default: main.o format.o decodeScanCPU.o pixelTransformCPU.o
	gcc ${CFLAGS} $^ -o ${TARGET}

%.o: $(SRCDIR)/%.c $(INCDIR)/%.h
	gcc $(CFLAGS) -c $< -o $@

clean:
	rm *.o *~ src/*~ include/*~
