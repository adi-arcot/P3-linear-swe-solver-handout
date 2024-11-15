CPP=CC

CFLAGS=-lm
COPTFLAGS=-O3 -ffast-math -flto -march=native -funroll-loops -ftree-vectorize -xhost
# -O3 -ffast-math -march=native -funroll-loops -flto -fprefetch-loop-arrays -fstrict-aliasing -floop-nest-optimize -ftree-vectorize -fassociative-math -fbranch-target-load-optimize2
# -flto
# may -ftree-vectorize -fbranch-target-load-optimize
# -flto -ffast-math -march=native -funroll-loops -ftree-vectorize 
#COPTFLAGS=-O3 -ffast-math -flto -march=native -funroll-loops -ftree-vectorize -xhost
MPIFLAGS=-DMPI

NVCC=nvcc
NVCCFLAGS=-DCUDA

PYTHON=python3

all: mpi gpu basic_serial

mpi: build/mpi
gpu: build/gpu
serial: build/serial
basic_serial: build/basic_serial

build/mpi: common/main.cpp common/scenarios.cpp mpi/mpi.cpp
	$(CPP) $^ -o $@ $(MPIFLAGS) $(CFLAGS) $(COPTFLAGS)

build/gpu: common/main.cpp common/scenarios.cpp gpu/gpu.cu
	$(NVCC) $^ -o $@ $(NVCCFLAGS)

build/serial: common/main.cpp common/scenarios.cpp serial/serial.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(COPTFLAGS)

build/basic_serial: common/main.cpp common/scenarios.cpp serial/basic_serial.cpp
	$(CPP) $^ -o $@ $(CFLAGS) $(COPTFLAGS)

.PHONY: clean

clean:
	rm -f build/*.out
	rm -f build/*.o
	rm -f build/*.gif