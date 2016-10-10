CXX = icpc
CXXFLAGS = -fopenmp
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

csrmm_test: csrmm_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

csrmm_test.s: csrmm_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ -S $^ $(LDFLAGS)

clean:
	rm csrmm_test
