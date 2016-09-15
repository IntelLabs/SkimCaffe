CXX = icpc
CXXFLAGS = -fopenmp -I../../src/
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
    CXXFLAGS += -xHost
  endif
else
  CXXFLAGS += -O3 -DNDEBUG -DMKL_DIRECT_CALL
  ifeq (yes, $(KNL))
    CXXFLAGS += -xMIC-AVX512
  else
    CXXFLAGS += -xHost
  endif
endif

sconv_test: sconv_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

fc_test: fc_test.cpp ../../include/caffe/util/spgemm.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

csrmm_test: csrmm_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

sconv_test.s: sconv_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ -S $^ $(LDFLAGS)

fc_test.s: fc_test.cpp ../../include/caffe/util/spgemm.hpp
	$(CXX) $(CXXFLAGS) -o $@ -S $< $(LDFLAGS)

csrmm_test.s: csrmm_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ -S $^ $(LDFLAGS)

clean:
	rm -f sconv_test fc_test csrmm_test
