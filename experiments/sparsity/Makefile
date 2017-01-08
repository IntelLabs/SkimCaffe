CXX = icpc
CXXFLAGS = -fopenmp -DUSE_MKL
LDFLAGS = -LSpMP -lspmp -mkl

ifeq (yes, $(LIBXSMM))
  CXXFLAGS += -Ilibxsmm/include
  LDFLAGS += -Llibxsmm/lib -lxsmm
endif

ifeq (yes, $(DBG))
  CXXFLAGS += -O0 -g
  ifeq (yes, $(KNL))
    CXXFLAGS += -xMIC-AVX512
  endif
else
  CXXFLAGS += -O3 -DNDEBUG -DMKL_DIRECT_CALL
  ifeq (yes, $(KNL))
    CXXFLAGS += -xMIC-AVX512
  else
    CXXFLAGS += -xHost
  endif
endif

ifeq (yes, $(DBG))
  NVCC_FLAGS += -O0 -g
else
  NVCC_FLAGS += -O3
endif

csrmm_test: csrmm_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

csrmm_test.s: csrmm_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ -S $^ $(LDFLAGS)

winograd_test: winograd_test.cpp
	$(CXX) $(CXXFLAGS) -I../../include -I../../.build_release/src/ -I/usr/local/cuda/include -I/usr/local/cudnn/cudnnv4/cuda/include -o $@ $^ $(LDFLAGS) -lglog -L../../.build_release/lib -lcaffe

winograd_test_cuda: winograd_test.o winograd_test_cuda.o
	$(CXX) $(LDFLAGS) -o $@ $^ -lglog -L../../.build_release/lib -lcaffe -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver

winograd_test_cuda.o: winograd_test.cu winograd_test.h
	nvcc -ccbin=$(CXX) -Xcompiler -fPIC $(NVCC_FLAGS) -DCUDA -DUSE_MKL -I../../include -I../../.build_release/src/ -I/usr/local/cuda/include -I/usr/local/cudnn/cudnnv4/cuda/include -o $@ -c winograd_test.cu -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50

winograd_test.o: winograd_test.cpp
	$(CXX) $(CXXFLAGS) -DCUDA -I../../include -I../../.build_release/src/ -I/usr/local/cuda/include -I/usr/local/cudnn/cudnnv4/cuda/include -o $@ -c $^

clean:
	rm csrmm_test
