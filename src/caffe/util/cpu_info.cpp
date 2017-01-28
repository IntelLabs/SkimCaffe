/*
All modification made by Intel Corporation: © 2016 Intel Corporation

All contributions by the University of California:
Copyright (c) 2014, 2015, The Regents of the University of California (Regents)
All rights reserved.

All other contributions:
Copyright (c) 2014, 2015, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md


Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <glog/logging.h>

#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "caffe/util/cpu_info.hpp"

namespace caffe {
namespace cpu {

Processor::Processor() {
  processor = 0;
  physicalId = 0;
  siblings = 0;
  coreId = 0;
  cpuCores = 0;
  speedMHz = 0;
}

CpuInfo::CpuInfo() {
  loadContentFromFile("/proc/cpuinfo");
}

CpuInfo::CpuInfo(const char *content) {
  loadContent(content);
}

void CpuInfo::loadContentFromFile(const char *fileName) {
  std::ifstream file(fileName);
  std::string content(
    (std::istreambuf_iterator<char>(file)),
    (std::istreambuf_iterator<char>()));

  loadContent(content.c_str());
}

void CpuInfo::loadContent(const char *content) {
  size_t contentLength = strlen(content);
  char *contentCopy = new char[contentLength + 1];
  snprintf(contentCopy, contentLength + 1, "%s", content);

  parseLines(contentCopy);

  fileContentBegin = contentCopy;
  fileContentEnd = &contentCopy[contentLength];
  currentLine = NULL;
}

CpuInfo::~CpuInfo() {
  delete [] fileContentBegin;
}

void CpuInfo::parseLines(char *content) {
  for (; *content; content++) {
    if (*content == '\n') {
      *content = '\0';
    }
  }
}

const char *CpuInfo::getFirstLine() {
  currentLine = fileContentBegin < fileContentEnd ? fileContentBegin : NULL;
  return getNextLine();
}

const char *CpuInfo::getNextLine() {
  if (!currentLine) {
    return NULL;
  }

  const char *savedCurrentLine = currentLine;
  while (*(currentLine++)) {
  }

  if (currentLine >= fileContentEnd) {
    currentLine = NULL;
  }

  return savedCurrentLine;
}

Collection::Collection(CpuInfoInterface *cpuInfo) : cpuInfo(*cpuInfo) {
  totalNumberOfSockets = 0;
  totalNumberOfCpuCores = 0;
  currentProcessor = NULL;

  processors.reserve(96);

  parseCpuInfo();
  collectBasicCpuInformation();
}

unsigned Collection::getProcessorSpeedMHz() {
  return processors.size() ? processors[0].speedMHz : 0;
}

unsigned Collection::getTotalNumberOfSockets() {
  return totalNumberOfSockets;
}

unsigned Collection::getTotalNumberOfCpuCores() {
  return totalNumberOfCpuCores;
}

unsigned Collection::getNumberOfProcessors() {
  return processors.size();
}

const Processor &Collection::getProcessor(unsigned processorId) {
  return processors[processorId];
}

void Collection::parseCpuInfo() {
  const char *cpuInfoLine = cpuInfo.getFirstLine();
  for (; cpuInfoLine; cpuInfoLine = cpuInfo.getNextLine()) {
    parseCpuInfoLine(cpuInfoLine);
  }
}

void Collection::parseCpuInfoLine(const char *cpuInfoLine) {
  int delimiterPosition = strcspn(cpuInfoLine, ":");

  if (cpuInfoLine[delimiterPosition] == '\0') {
    currentProcessor = NULL;
  } else {
    parseValue(cpuInfoLine, &cpuInfoLine[delimiterPosition + 2]);
  }
}

void Collection::parseValue(const char *fieldName, const char *valueString) {
  if (!currentProcessor) {
    appendNewProcessor();
  }

  if (beginsWith(fieldName, "processor")) {
    currentProcessor->processor = parseInteger(valueString);
  }

  if (beginsWith(fieldName, "physical id")) {
    currentProcessor->physicalId = parseInteger(valueString);
  }

  if (beginsWith(fieldName, "siblings")) {
    currentProcessor->siblings = parseInteger(valueString);
  }

  if (beginsWith(fieldName, "core id")) {
    currentProcessor->coreId = parseInteger(valueString);
  }

  if (beginsWith(fieldName, "cpu cores")) {
    currentProcessor->cpuCores = parseInteger(valueString);
  }

  if (beginsWith(fieldName, "model name")) {
    currentProcessor->speedMHz = extractSpeedFromModelName(valueString);
  }
}

void Collection::appendNewProcessor() {
  processors.push_back(Processor());
  currentProcessor = &processors.back();
}

bool Collection::beginsWith(const char *lineBuffer, const char *text) const {
  while (*text) {
    if (*(lineBuffer++) != *(text++)) {
      return false;
    }
  }

  return true;
}

unsigned Collection::parseInteger(const char *text) const {
  return atol(text);
}

/* Function extracts CPU speed from model name. If unit is not set it is
   assumed that values below 100 are specified in GHz, otherwise MHz */
unsigned Collection::extractSpeedFromModelName(const char *text) const {
  text = strstr(text, "@");
  if (!text) {
    return 0;
  }

  char *unit;
  double speed = strtod(&text[1], &unit);

  while (isspace(*unit)) {
    unit++;
  }

  bool isMHz = !strncmp(unit, "MHz", 3);
  bool isGHz = !strncmp(unit, "GHz", 3);
  bool isGHzPossible = (speed < 100);

  if (isGHz || (isGHzPossible && !isMHz)) {
    return 1000 * speed + 0.5;
  } else {
    return speed + 0.5;
  }
}

void Collection::collectBasicCpuInformation() {
  std::set<unsigned> uniquePhysicalId;
  std::vector<Processor>::iterator processor = processors.begin();
  for (; processor != processors.end(); processor++) {
    uniquePhysicalId.insert(processor->physicalId);
    updateCpuInformation(*processor, uniquePhysicalId.size());
  }
}

void Collection::updateCpuInformation(const Processor &processor,
    unsigned numberOfUniquePhysicalId) {
  if (totalNumberOfSockets == numberOfUniquePhysicalId) {
    return;
  }

  totalNumberOfSockets = numberOfUniquePhysicalId;
  totalNumberOfCpuCores += processor.cpuCores;
}

#ifdef _OPENMP

/* The OpenMpManager class is responsible for determining a set of all of
   available CPU cores and delegating each core to perform other tasks. The
   first of available cores is delegated for background threads, while other
   remaining cores are dedicated for OpenMP threads. Each OpenMP thread owns
   one core for exclusive use. The number of OpenMP threads is then limited
   to the number of available cores minus one. The amount of CPU cores may
   be limited by system eg. when numactl was used. */

#include <omp.h>
#include <sched.h>

static const char *openMpEnvVars[] = {
  "OMP_CANCELLATION", "OMP_DISPLAY_ENV", "OMP_DEFAULT_DEVICE", "OMP_DYNAMIC",
  "OMP_MAX_ACTIVE_LEVELS", "OMP_MAX_TASK_PRIORITY", "OMP_NESTED",
  "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_STACKSIZE",
  "OMP_SCHEDULE", "OMP_THREAD_LIMIT", "OMP_WAIT_POLICY", "GOMP_CPU_AFFINITY",
  "GOMP_DEBUG", "GOMP_STACKSIZE", "GOMP_SPINCOUNT", "GOMP_RTEMS_THREAD_POOLS",
  "KMP_AFFINITY", "KMP_NUM_THREADS", "MIC_KMP_AFFINITY",
  "MIC_OMP_NUM_THREADS", "MIC_OMP_PROC_BIND", "PHI_KMP_AFFINITY",
  "PHI_OMP_NUM_THREADS", "PHI_KMP_PLACE_THREADS", "MKL_NUM_THREADS",
  "MKL_DYNAMIC", "MKL_DOMAIN_NUM_THREADS"
};

static const unsigned numberOfOpenMpEnvVars =
  sizeof(openMpEnvVars) / sizeof(openMpEnvVars[0]);

OpenMpManager::OpenMpManager(Collection *collection) :
                             mainThreadId(boost::this_thread::get_id()),
                             collection(*collection) {
  getOpenMpEnvVars();
  getCurrentCpuSet();
  getCurrentCoreSet();
}

OpenMpManager &OpenMpManager::getInstance() {
  static CpuInfo cpuInfo;
  static Collection collection(&cpuInfo);
  static OpenMpManager openMpManager(&collection);
  return openMpManager;
}

void OpenMpManager::setGpuEnabled() {
  OpenMpManager &openMpManager = getInstance();
  openMpManager.isGpuEnabled = true;
}

void OpenMpManager::setGpuDisabled() {
  OpenMpManager &openMpManager = getInstance();
  openMpManager.isGpuEnabled = false;
}

bool OpenMpManager::isMajorThread(boost::thread::id currentThread) {
  OpenMpManager &openMpManager = getInstance();
  return (boost::this_thread::get_id() == openMpManager.mainThreadId);
}

// Ideally bind given thread to secondary logical core, if
// only one thread exists then bind to primary one
void OpenMpManager::bindCurrentThreadToNonPrimaryCoreIfPossible() {
  OpenMpManager &openMpManager = getInstance();
  if (openMpManager.isThreadsBindAllowed()) {
    int totalNumberOfAvailableCores = CPU_COUNT(&openMpManager.currentCoreSet);
    int logicalCoreToBindTo = totalNumberOfAvailableCores > 1 ? 1 : 0;
    openMpManager.bindCurrentThreadToLogicalCoreCpus(logicalCoreToBindTo);
  }
}

void OpenMpManager::bindOpenMpThreads() {
  OpenMpManager &openMpManager = getInstance();

  if (!openMpManager.isThreadsBindAllowed())
    return;

  openMpManager.setOpenMpThreadNumberLimit();
  #pragma omp parallel
  {
    unsigned logicalCoreId = omp_get_thread_num();
    openMpManager.bindCurrentThreadToLogicalCoreCpu(logicalCoreId);
  }
}

void OpenMpManager::getOpenMpEnvVars() {
  isAnyOpenMpEnvVarSpecified = false;
  for (unsigned i = 0; i < numberOfOpenMpEnvVars; i++) {
    if (getenv(openMpEnvVars[i])) {
      isAnyOpenMpEnvVarSpecified = true;
    }
  }
}

void OpenMpManager::getCurrentCpuSet() {
  if (sched_getaffinity(0, sizeof(currentCpuSet), &currentCpuSet)) {
    getDefaultCpuSet(&currentCpuSet);
  }
}

void OpenMpManager::getDefaultCpuSet(cpu_set_t *defaultCpuSet) {
  CPU_ZERO(defaultCpuSet);
  unsigned numberOfProcessors = collection.getNumberOfProcessors();
  for (int processorId = 0; processorId < numberOfProcessors; processorId++) {
    CPU_SET(processorId, defaultCpuSet);
  }
}

/* Function getCurrentCoreSet() fills currentCoreSet variable with a set of
   available CPUs, where only one CPU per core is chosen. When multiple CPUs
   of single core are used, function is selecting only first one of all
   available. */

void OpenMpManager::getCurrentCoreSet() {
  unsigned numberOfProcessors = collection.getNumberOfProcessors();
  unsigned totalNumberOfCpuCores = collection.getTotalNumberOfCpuCores();

  cpu_set_t usedCoreSet;
  CPU_ZERO(&usedCoreSet);
  CPU_ZERO(&currentCoreSet);

  for (int processorId = 0; processorId < numberOfProcessors; processorId++) {
    if (CPU_ISSET(processorId, &currentCpuSet)) {
      unsigned coreId = processorId % totalNumberOfCpuCores;
      if (!CPU_ISSET(coreId, &usedCoreSet)) {
        CPU_SET(coreId, &usedCoreSet);
        CPU_SET(processorId, &currentCoreSet);
      }
    }
  }
}

void OpenMpManager::selectAllCoreCpus(cpu_set_t *set, unsigned physicalCoreId) {
  unsigned numberOfProcessors = collection.getNumberOfProcessors();
  unsigned totalNumberOfCpuCores = collection.getTotalNumberOfCpuCores();

  int processorId = physicalCoreId % totalNumberOfCpuCores;
  while (processorId < numberOfProcessors) {
    if (CPU_ISSET(processorId, &currentCpuSet)) {
      CPU_SET(processorId, set);
    }

    processorId += totalNumberOfCpuCores;
  }
}

unsigned OpenMpManager::getPhysicalCoreId(unsigned logicalCoreId) {
  unsigned numberOfProcessors = collection.getNumberOfProcessors();

  for (int processorId = 0; processorId < numberOfProcessors; processorId++) {
    if (CPU_ISSET(processorId, &currentCoreSet)) {
      if (!logicalCoreId--) {
        return processorId;
      }
    }
  }

  LOG(FATAL) << "This should never happen!";
  return 0;
}

bool OpenMpManager::isThreadsBindAllowed() {
  return !isAnyOpenMpEnvVarSpecified && !isGpuEnabled;
}

// Limit of threads to number of logical cores available
void OpenMpManager::setOpenMpThreadNumberLimit() {
  omp_set_num_threads(CPU_COUNT(&currentCoreSet));
}

void OpenMpManager::bindCurrentThreadToLogicalCoreCpu(unsigned logicalCoreId) {
  unsigned physicalCoreId = getPhysicalCoreId(logicalCoreId);

  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(physicalCoreId, &set);
  sched_setaffinity(0, sizeof(set), &set);
}

void OpenMpManager::bindCurrentThreadToLogicalCoreCpus(unsigned logicalCoreId) {
  unsigned physicalCoreId = getPhysicalCoreId(logicalCoreId);

  cpu_set_t set;
  CPU_ZERO(&set);
  selectAllCoreCpus(&set, physicalCoreId);
  sched_setaffinity(0, sizeof(set), &set);
}

void OpenMpManager::printVerboseInformation() {
  OpenMpManager &openMpManager = getInstance();

  LOG(INFO) << "Processor speed [MHz]: "
    << openMpManager.collection.getProcessorSpeedMHz();

  LOG(INFO) << "Total number of sockets: "
    << openMpManager.collection.getTotalNumberOfSockets();

  LOG(INFO) << "Total number of CPU cores: "
    << openMpManager.collection.getTotalNumberOfCpuCores();

  LOG(INFO) << "Total number of processors: "
    << openMpManager.collection.getNumberOfProcessors();

  LOG(INFO) << "GPU is used: "
    << (openMpManager.isGpuEnabled ? "yes" : "no");

  LOG(INFO) << "OpenMP environmental variables are specified: "
    << (openMpManager.isAnyOpenMpEnvVarSpecified ? "yes" : "no");

  LOG(INFO) << "OpenMP thread bind allowed: "
    << (openMpManager.isThreadsBindAllowed() ? "yes" : "no");

  LOG(INFO) << "Number of OpenMP threads: "
    << omp_get_max_threads();
}

unsigned OpenMpManager::getProcessorSpeedMHz() {
  OpenMpManager &openMpManager = getInstance();
  return openMpManager.collection.getProcessorSpeedMHz();
}

int OpenMpManager::getNumThreadGroups(int batchSize)
{
  int nthread_groups = omp_in_parallel() ? omp_get_num_threads() : omp_get_max_threads(); // TODO: may want to use 1 thread group per tile in KNL later
  if (nthread_groups > 2*batchSize) {
    nthread_groups = batchSize;
  }
  return nthread_groups;
}

static int getNumThreadsPerGroup(int batchSize)
{
  int nthreads = omp_in_parallel() ? omp_get_num_threads() : omp_get_max_threads();
  int nthread_groups = OpenMpManager::getNumThreadGroups(batchSize);
  return (nthreads + nthread_groups - 1)/nthread_groups;
}

int OpenMpManager::getThreadGroupNum(int batchSize)
{
  assert(omp_in_parallel());
  return omp_get_thread_num()/getNumThreadsPerGroup(batchSize);
}

int OpenMpManager::getThreadNumInGroup(int batchSize)
{
  assert(omp_in_parallel());
  return omp_get_thread_num()%getNumThreadsPerGroup(batchSize);
}

int OpenMpManager::getNumThreadsInGroup(int batchSize)
{
  int nthreads = omp_in_parallel() ? omp_get_num_threads() : omp_get_max_threads();
  int nthreads_per_group = getNumThreadsPerGroup(batchSize);
  int gid = OpenMpManager::getThreadGroupNum(batchSize);
  return std::min(nthreads_per_group, nthreads - nthreads_per_group*gid);
}

void OpenMpManager::getBatchThreadPartition(int *begin, int *end, int batchSize)
{
  int nthread_groups = OpenMpManager::getNumThreadGroups(batchSize);
  int gid = OpenMpManager::getThreadGroupNum(batchSize);

  int n_per_group = (batchSize + nthread_groups - 1)/nthread_groups;
  *begin = std::min(n_per_group*gid, batchSize);
  *end = std::min(*begin + n_per_group, batchSize);
}

void OpenMpManager::getSimpleGroupedThreadPartition(int *begin, int *end, int work, int batchSize)
{
  int nthreads_in_group = OpenMpManager::getNumThreadsInGroup(batchSize);
  int tid_in_group = OpenMpManager::getThreadNumInGroup(batchSize);

  int workPerThread = (work + nthreads_in_group - 1)/nthreads_in_group;

  *begin = std::min(workPerThread*tid_in_group, work);
  *end = std::min(*begin + workPerThread, work);
}

synk::Barrier *threadGroupBarriers[1024];

synk::Barrier **OpenMpManager::getThreadGroupBarriers(int batchSize)
{
  static bool initialized = false;

  if (!initialized) {
    int nthread_groups = OpenMpManager::getNumThreadGroups(batchSize);
    int nthreads_per_group = getNumThreadsPerGroup(batchSize);

    if (omp_in_parallel()) {
      int nthreads = omp_get_num_threads();

#pragma omp single
      {
        for (int gid = 0; gid < nthread_groups; ++gid) {
          threadGroupBarriers[gid] = new synk::Barrier(1, std::min(nthreads_per_group, nthreads - nthreads_per_group*gid));
        }
      }

      int gid = OpenMpManager::getThreadGroupNum(batchSize);
      int tid_in_group = OpenMpManager::getThreadNumInGroup(batchSize);

      threadGroupBarriers[gid]->init(tid_in_group);

#pragma omp single
      initialized = true;
    }
    else {
      int nthreads = omp_get_max_threads();

      for (int gid = 0; gid < nthread_groups; ++gid) {
        threadGroupBarriers[gid] = new synk::Barrier(1, std::min(nthreads_per_group, nthreads - nthreads_per_group*gid));
      }

      if (1 == omp_get_max_threads()) {
        threadGroupBarriers[0]->init(0);
      }
      else {
  #pragma omp parallel
        {
          int gid = OpenMpManager::getThreadGroupNum(batchSize);
          int tid_in_group = OpenMpManager::getThreadNumInGroup(batchSize);

          threadGroupBarriers[gid]->init(tid_in_group);
        }
      }

      initialized = true;
    }
  }

  return threadGroupBarriers;
}

void OpenMpManager::barrierGroup(int batchSize)
{
  assert(omp_in_parallel());

  int nthreads = omp_get_num_threads();
  int nthread_groups = OpenMpManager::getNumThreadGroups(batchSize);
  int gid = OpenMpManager::getThreadGroupNum(batchSize);
  int tid_in_group = OpenMpManager::getThreadNumInGroup(batchSize);

  if (nthread_groups != nthreads) threadGroupBarriers[gid]->wait(tid_in_group);
}

#endif  // _OPENMP

}  // namespace cpu
}  // namespace caffe
