NVCC=nvcc
NVCCFLAGS=-I../include -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart -I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_cudaarithm -lopencv_imgproc -lopencv_cudawarping
DEBUGFLAGS=-g -G


SRC=$(wildcard *.cu)
OBJ=$(patsubst %.cu,%,$(SRC))

OUTPUT_DIR=../../cvGPUSpeedup_builds

all: $(OBJ)
%: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $(OUTPUT_DIR)/$@

clean:
	rm -f $(addprefix $(OUTPUT_DIR)/,$(OBJ))