NVCC=nvcc
NVCCFLAGS_ONLY_FK=-gencode arch=compute_86,code=sm_86 -I../include -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -lcudart
NVCCFLAGS_ONLY_CV=-I/usr/local/include/opencv4 -L/usr/local/lib -lopencv_core -lopencv_cudaarithm -lopencv_imgproc -lopencv_cudaimgproc -lopencv_cudawarping
NVCCFLAGS=
DEBUGFLAGS=


SRC=$(wildcard *.cu)
OBJ=$(patsubst %.cu,%,$(SRC))

OUTPUT_DIR=../../cvGPUSpeedup_builds

debug: DEBUGFLAGS += -g -G
debug: all

fk: NVCCFLAGS += $(NVCCFLAGS_ONLY_FK)
fk:
	$(NVCC) $(DEBUGFLAGS) $(NVCCFLAGS) testFusedKernel.cu -o $(OUTPUT_DIR)/testFusedKernel

all: NVCCFLAGS += $(NVCCFLAGS_ONLY_FK) $(NVCCFLAGS_ONLY_CV)
all: $(OBJ)
%: %.cu
	$(NVCC) $(DEBUGFLAGS) $(NVCCFLAGS) $< -o $(OUTPUT_DIR)/$@

clean:
	rm -f $(addprefix $(OUTPUT_DIR)/,$(OBJ))
