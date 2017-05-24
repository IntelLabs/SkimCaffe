CXX = icpc
CXXFLAGS = -fopenmp -DUSE_MKL -I../../src/
LDFLAGS = -L../../src/SpMP -lspmp -mkl

#SEP = yes
ifeq (yes, $(SEP))
  CXXFLAGS += -I$(HOME)/sep/include -DSEP
  LDFLAGS += -L$(HOME)/sep/bin64 -lprog_api
endif

#VTUNE = yes
ifeq (yes, $(VTUNE))
  CXXFLAGS += -I/swtools/intel/vtune_amplifier_xe/include -DVTUNE
  LDFLAGS += -L/swtools/intel/vtune_amplifier_xe/lib64 -littnotify
endif

ifeq (yes, $(SNIPER))
  CXXFLAGS += -I$(SNIPER_ROOT)/include -DSNIPER -g
  LDFLAGS += -L$(SNIPER_ROOT)/lib -lsim_api
endif

ifeq (yes, $(SDE))
  CXXFLAGS += -DSDE -fasm-blocks
endif

ifeq (yes, $(LIBXSMM))
  CXXFLAGS += -Ilibxsmm/include
  LDFLAGS += -Llibxsmm/lib -lxsmm
endif

ifeq (yes, $(DBG))
  CXXFLAGS += -O0 -g
  ifeq (yes, $(KNL))
    CXXFLAGS += -xMIC-AVX512
  else
    ifeq (yes, $(SKX))
      CXXFLAGS += -xCORE-AVX512
    else
      CXXFLAGS += -xHost
    endif
  endif
else
  CXXFLAGS += -O3 -DNDEBUG -DMKL_DIRECT_CALL
  ifeq (yes, $(KNL))
    CXXFLAGS += -xMIC-AVX512
  else
    ifeq (yes, $(SKX))
      CXXFLAGS += -xCORE-AVX512
    else
      CXXFLAGS += -xHost
    endif
  endif
endif

ifeq (yes, $(DBG))
  NVCC_FLAGS += -O0 -g
else
  NVCC_FLAGS += -O3
endif

all: sconv_test fc_test

sconv_test: sconv_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

fc_test: fc_test.cpp ../../include/caffe/util/csrmm.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

csrmm_test: csrmm_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

sconv_test.s: sconv_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ -S $^ $(LDFLAGS)

fc_test.s: fc_test.cpp ../../include/caffe/util/spgemm.hpp
	$(CXX) $(CXXFLAGS) -o $@ -S $< $(LDFLAGS)

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
	rm -f sconv_test fc_test csrmm_test
