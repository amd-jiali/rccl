/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */

#pragma once
#include <map>
#include <unordered_map>
#include <chrono>
#include <cstring>

#include <rccl/rccl.h>
#include <hip/hip_bfloat16.h>
#include "hip/hip_fp16.h"
#include "rccl_float8.h"

#include "info.h"
#include "recorder.h"

// NOTE: Parsing is based on this line logging collective information in enqueue.cc
// INFO(NCCL_COLL,"%s: opCount %lx sendbuff %p recvbuff %p count %zi datatype %d op %d \
                   root %d comm %p [nranks=%d] stream %p task %d globalrank %d",
//                info->opName, info->comm->opCount, info->sendbuff, info->recvbuff, info->count,
//                info->datatype, info->op, info->root, info->comm, info->comm->nRanks, info->stream,
//                info->comm->tasks.nTasksP2p + info->comm->tasks.nTasksColl,
//                info->comm->localRankToRank[info->comm->localRank]);

#define HIP_CALL(cmd)                                                   \
  do {                                                                  \
      hipError_t error = (cmd);                                         \
      if (error != hipSuccess) {                                        \
        printf("Encountered HIP error (%s) at line %d in file %s\n",    \
               hipGetErrorString(error), __LINE__, __FILE__);           \
        exit(-1);                                                       \
      }                                                                 \
  } while (0)

#define NCCL_CALL(cmd)                                          \
  do {                                                          \
    ncclResult_t res = cmd;                                     \
    if (res != ncclSuccess) {                                   \
      printf("NCCL failure %s:%d '%s'\n",                       \
             __FILE__,__LINE__,ncclGetErrorString(res));        \
    }                                                           \
  } while(0)

struct DeviceMemAllocation
{
  void*                 base = NULL;
  size_t                size = 0;
  int                   lastLineUsed = -1;
};

struct DeviceGraphInfo
{
  int                   depth = 0;
  std::unordered_set<int>
                        starts;
  int                   end = -1;
  hipStream_t           stream = NULL;
  hipGraph_t            graph = NULL;
  hipGraphExec_t        graphExec = NULL;

  std::vector<hipEvent_t>
                        events;
  int                   counter = 0;
};

class Replayer
{
 private:
  // rank specific info
  int                   myRank;
  int                   numGlobalRanks;
  /// int numGpusPerMpiRank;
  /// int localGpuOffset;                                     // First local GPU device idx for this MPI process
  /// int firstGlobalRank;                                    // First global rank for this MPI process
  std::ifstream         log;

  // Contextual info parsed from first pass, to assist replay later
  //  Communicator
  std::vector<uint64_t>                                 Ids; // all communicators (uniqueIDs) created from commInit, assuming called once only
  std::unordered_map<uint64_t, std::vector<int>>        idRankMap; // all ranks in the communicator created by an ID on this rank


  //  Memory allocation and lifespan
  std::unordered_map<void*, DeviceMemAllocation>        dMemMap;

  // Resources for replayer, mostly maps from pointer in log to resources in replay time
  std::unordered_map<uint64_t, ncclUniqueId>            idMap; // replayer uniqueID mapped to logged ones, for ID creator rank only
  std::unordered_map<ncclComm_t, ncclComm_t>            commMap; // replayer communicator mapped to the logged ones

  std::unordered_map<hipStream_t, std::pair<hipStream_t,int>>
                                                        streams; // replayer streams mapped to the logged ones // use using later?
  std::unordered_map<void*, void*>                      handleMap; // UBR handle
  std::unordered_map<unsigned long long, DeviceGraphInfo>
                                                        graphLife; // when does a graph (graphID) end and how many node it contains

  // auxiliary variables for replayer
  ncclUniqueId uniqueID;
  rccl::rcclCall_t lastCall;

 public:
  Replayer(const std::string& logname, int json_format, int rank, int size);
  void parse();
  void replay();
};