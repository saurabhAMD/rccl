/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "common.h"

__shared__ ncclShmemData ncclShmem;
#if __CUDA_ARCH__ < 700
  __shared__ ulong2 ncclShmemPerWarp[ncclShmemScratchWarpSize()*(NCCL_MAX_NTHREADS/WARP_SIZE)/sizeof(ulong2)];
#endif

struct RunWorkNop {
  __device__ void run(ncclWork *w) {}
};

__global__ void ncclDevKernel_Generic(struct ncclDevComm* comm, uint64_t channelMask, struct ncclWork* workHead) {
  ncclKernelMain<-1, RunWorkNop, false>(comm, channelMask, workHead);
}
#ifdef ENABLE_COLLTRACE
__global__ void ncclDevKernelDebug_Generic(struct ncclDevComm* comm, uint64_t channelMask, struct ncclWork* workHead) {
  ncclKernelMain<-1, RunWorkNop, true>(comm, channelMask, workHead);
}
#endif

__device__ void ncclDevFunc_Nop() {}
