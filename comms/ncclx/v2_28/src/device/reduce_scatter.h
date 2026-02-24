/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "device.h"
#include "collectives.h"
#include "primitives.h"
#include "ccd.cuh"

namespace {
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runRing(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclRing *ring = &ncclShmem.channel.ring;
    int const *ringRanks = ring->userRanks;
    const int nranks = ncclShmem.comm.nRanks;
    size_t count;
    size_t gridOffset;
    size_t channelCount;
    size_t chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count, &gridOffset, &channelCount, &chunkCount);
    size_t offset;
    size_t dataOffset;
    uint32_t nelem;
    int rankDest;

    // Coverity reports that the callee treats &ring->next as an array.  However, due to the use of
    // FanSymmetric<1>, only the first element is ever accessed, so it's fine.
    // coverity[callee_ptr_arith:FALSE]
    Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0>
      prims(tid, nthreads, &ring->prev, &ring->next, work->sendbuff, work->recvbuff, work->redOpArg);

    for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
      nelem = min(chunkCount, channelCount - elemOffset);

      dataOffset = gridOffset + elemOffset;
      /////////////// begin ReduceScatter steps ///////////////
      // step 0: push data to next GPU
      rankDest = ringRanks[nranks-1];
      offset = dataOffset + rankDest * count;
      prims.send(offset, nelem);

      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        rankDest = ringRanks[nranks-j];
        offset = dataOffset + rankDest * count;
        prims.recvReduceSend(offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final result
      rankDest = ringRanks[0];
      offset = dataOffset + rankDest * count;
      prims.recvReduceCopy(offset, dataOffset, nelem, /*postOp=*/true);
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Sparse helpers: thin wrappers around polling and signaling for the
  // header-based sparse ring protocol (runRingSparse).
  //////////////////////////////////////////////////////////////////////////////

  // Wait until the send FIFO has room: poll sendConn->head until head + NCCL_STEPS > step.
  // sendConn->head is written by the consumer (proxy or peer) to return credits.
  __device__ __forceinline__
  void sparseWaitSend(uint64_t* headPtr, uint64_t step) {
    int spins = 0;
    int abort = 0;
    while (ld_volatile_global(headPtr) + NCCL_STEPS <= step) {
      if (checkAbort(abort, 1, spins)) break;
    }
  }

  // Signal that data has been written to send slot: fence + advance tail + set connFifo.size.
  __device__ __forceinline__
  void sparsePostSend(uint64_t* tailPtr, uint64_t step, ncclConnFifo* connFifo, size_t totalBytes) {
    fence_acq_rel_sys();
    if (connFifo != nullptr) {
      connFifo[step % NCCL_STEPS].size = totalBytes;
    }
    st_relaxed_sys_global(tailPtr, step + 1);
  }

  // Wait until recv data is ready: poll recvConn->tail until tail > step.
  // recvConn->tail is written by the producer (proxy or peer) after data arrives.
  __device__ __forceinline__
  void sparseWaitRecv(uint64_t* tailPtr, uint64_t step) {
    int spins = 0;
    int abort = 0;
    while (ld_volatile_global(tailPtr) <= step) {
      if (checkAbort(abort, 1, spins)) break;
    }
  }

  // Return recv credits: advance recvConn->head.
  __device__ __forceinline__
  void sparsePostRecv(uint64_t* headPtr, uint64_t step) {
    st_relaxed_sys_global(headPtr, step + 1);
  }

  // Cooperative memcpy: all threads copy nbytes from src to dst using 16-byte loads.
  __device__ __forceinline__
  void sparseMemcpy(
    char* __restrict__ dst, const char* __restrict__ src,
    size_t nbytes, int tid, int nthreads
  ) {
    // Use int4 (16-byte) loads/stores for coalescing
    int4* dst4 = (int4*) dst;
    const int4* src4 = (const int4*) src;
    size_t n16 = nbytes / 16;
    for (size_t i = tid; i < n16; i += nthreads) {
      dst4[i] = src4[i];
    }
    // Handle remainder bytes (thread 0 only)
    size_t done = n16 * 16;
    if (tid == 0) {
      for (size_t i = done; i < nbytes; i++) {
        dst[i] = src[i];
      }
    }
  }

  // Cooperative element-wise reduce: dst[i] = RedOp(a[i], b[i])
  // Uses NCCL's Apply_Reduce which handles half, bf16, fp8, etc.
  template<typename T, typename RedOp>
  __device__ __forceinline__
  void sparseReduce(
    T* __restrict__ dst,
    const T* __restrict__ a,
    const T* __restrict__ b,
    size_t nelem, int tid, int nthreads, uint64_t redOpArg
  ) {
    RedOp op(redOpArg);
    for (size_t i = tid; i < nelem; i += nthreads) {
      BytePack<sizeof(T)> pa = toPack<T>(a[i]);
      BytePack<sizeof(T)> pb = toPack<T>(b[i]);
      BytePack<sizeof(T)> result = Apply_Reduce<RedOp, 1>::reduce(op, pa, pb);
      dst[i] = fromPack<T>(result);
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // runRingSparse: header-based ring ReduceScatter
  //
  // Each send is: [SparseChunkHeader][payload]. The receiver reads the header
  // to learn payload_bytes. Currently sends dense data (payload_bytes =
  // nelem*sizeof(T)) to validate the protocol before adding compression.
  //
  // Uses chunkSteps=1, sliceSteps=1 → one FIFO slot per chunk per ring step.
  // The 8-slot FIFO (NCCL_STEPS=8) provides cross-chunk buffering.
  //////////////////////////////////////////////////////////////////////////////
  template<typename T, typename RedOp>
  __device__ void runRingSparse(int tid, int nthreads, struct ncclDevWorkColl* work) {
    ncclRing *ring = &ncclShmem.channel.ring;
    int const *ringRanks = ring->userRanks;
    const int nranks = ncclShmem.comm.nRanks;
    const int rank = ncclShmem.comm.rank;

    // Use ProtoSimple<1,1> for chunk/slice partitioning with chunkSteps=1, sliceSteps=1
    const int protoId = NCCL_PROTO_SIMPLE;
    size_t count;       // elements per rank in the full tensor
    size_t gridOffset;  // element offset where this channel starts
    size_t channelCount;// elements this channel handles
    size_t chunkCount;  // max elements per chunk
    ncclCollCbdPart(work, ncclShmem.channelId, protoId, sizeof(T),
                    &count, &gridOffset, &channelCount, &chunkCount);

    // Access send/recv connections directly (bypass Primitives)
    ncclDevChannelPeer* nextPeer = ncclShmem.channel.peers[ring->next];
    ncclDevChannelPeer* prevPeer = ncclShmem.channel.peers[ring->prev];
    ncclConnInfo* sendConn = &nextPeer->send[0];
    ncclConnInfo* recvConn = &prevPeer->recv[0];

    // Transport buffers and step counters
    char* sendBuf = sendConn->buffs[NCCL_PROTO_SIMPLE];
    char* recvBuf = recvConn->buffs[NCCL_PROTO_SIMPLE];
    int stepSize = sendConn->stepSize;
    ncclConnFifo* sendFifo = sendConn->connFifo; // may be null for P2P

    // Sparse path requires NCCL_BUFFSIZE >= 16 MiB (stepSize >= 2 MiB) so that
    // worst-case compressed formats (COO = 2× dense + header) fit in one slot.
    // Set NCCL_BUFFSIZE=16777216 in the environment. This check runs once per
    // collective (not per chunk), so has zero perf cost.
    if (tid == 0 && stepSize < 2 * 1024 * 1024) {
      printf("[SPARSE-RING] FATAL: stepSize=%d too small for sparse path. "
             "Set NCCL_BUFFSIZE=16777216 (16 MiB). Need stepSize >= 2 MiB.\n",
             stepSize);
      __trap();
    }

    // With NCCL_BUFFSIZE increased (e.g., 16 MiB), ncclCollCbdPart computes a
    // larger chunkCount than dense NCCL would with default 4 MiB buffers. We
    // clamp chunkCount to the dense-equivalent value so the number of chunks
    // (and thus P2P messages) is identical to stock dense NCCL. The enlarged
    // slots provide headroom for sparse format overhead (COO = 2× dense, SPOP
    // ≈ 1.03× dense) + header without needing extra messages.
    const size_t headerSize = sizeof(SparseChunkHeader);

    // Initialize step counters from connection state.
    // With chunkSteps=1, sliceSteps=1, each chunk/step uses exactly 1 FIFO slot.
    uint64_t sendStep = sendConn->step;
    uint64_t recvStep = recvConn->step;

    // Initialize recv credits: tell the sender we're ready for data
    if (tid == 0) {
      st_relaxed_sys_global(recvConn->head, recvStep);
    }

    T* sendbuff = (T*)work->sendbuff;
    T* recvbuff = (T*)work->recvbuff;

    if (tid == 0) {
      printf("[SPARSE-RING] rank=%d nranks=%d channel=%d count=%lu gridOffset=%lu "
             "channelCount=%lu chunkCount=%lu stepSize=%d sendBuf=%p recvBuf=%p "
             "sendFifo=%p\n",
             rank, nranks, ncclShmem.channelId,
             (unsigned long)count, (unsigned long)gridOffset,
             (unsigned long)channelCount, (unsigned long)chunkCount,
             stepSize, sendBuf, recvBuf, sendFifo);
    }

    for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
      uint32_t nelem = min(chunkCount, channelCount - elemOffset);
      size_t dataOffset = gridOffset + elemOffset;
      size_t payloadBytes = nelem * sizeof(T);
      size_t totalBytes = headerSize + payloadBytes;

      /////////////// Ring step 0: SEND ///////////////
      {
        int rankDest = ringRanks[nranks - 1];
        size_t srcOffset = dataOffset + rankDest * count;

        // Wait for send buffer availability
        if (tid == 0) {
          sparseWaitSend(sendConn->head, sendStep);
        }
        __syncthreads();

        // Compute slot pointer
        char* sendSlot = sendBuf + (sendStep % NCCL_STEPS) * stepSize;

        // Thread 0 writes header
        if (tid == 0) {
          SparseChunkHeader* hdr = (SparseChunkHeader*)sendSlot;
          hdr->payload_bytes = payloadBytes;
        }

        // All threads cooperatively copy source data after header
        sparseMemcpy(sendSlot + headerSize,
                     (const char*)(sendbuff + srcOffset),
                     payloadBytes, tid, nthreads);

        __syncthreads();

        // Post send
        if (tid == 0) {
          sparsePostSend(sendConn->tail, sendStep, sendFifo, totalBytes);
        }
        sendStep++;
      }

      /////////////// Ring steps 1..nranks-2: RECV + REDUCE + SEND ///////////////
      for (int j = 2; j < nranks; ++j) {
        int rankDest = ringRanks[nranks - j];
        size_t srcOffset = dataOffset + rankDest * count;

        // Wait for recv data
        if (tid == 0) {
          sparseWaitRecv(recvConn->tail, recvStep);
        }
        __syncthreads();

        char* recvSlot = recvBuf + (recvStep % NCCL_STEPS) * stepSize;

        // Read header (thread 0 reads, broadcast via shared mem not needed —
        // all threads can read from the same location after the sync)
        size_t recvPayload = ((SparseChunkHeader*)recvSlot)->payload_bytes;
        size_t recvNelem = recvPayload / sizeof(T);

        // Wait for send buffer availability
        if (tid == 0) {
          sparseWaitSend(sendConn->head, sendStep);
        }
        __syncthreads();

        char* sendSlot = sendBuf + (sendStep % NCCL_STEPS) * stepSize;

        // Thread 0 writes header (same payload size — dense for now)
        if (tid == 0) {
          SparseChunkHeader* hdr = (SparseChunkHeader*)sendSlot;
          hdr->payload_bytes = payloadBytes;
        }

        // Reduce: local[srcOffset..] + recv payload → send payload
        T* recvData = (T*)(recvSlot + headerSize);
        T* sendData = (T*)(sendSlot + headerSize);
        T* localData = sendbuff + srcOffset;

        sparseReduce<T, RedOp>(sendData, localData, recvData,
                        min((size_t)nelem, recvNelem), tid, nthreads, work->redOpArg);

        __syncthreads();

        // Post recv credits
        if (tid == 0) {
          sparsePostRecv(recvConn->head, recvStep);
        }
        recvStep++;

        // Post send
        if (tid == 0) {
          sparsePostSend(sendConn->tail, sendStep, sendFifo, totalBytes);
        }
        sendStep++;
      }

      /////////////// Ring step nranks-1: RECV + REDUCE + COPY to output ///////////////
      {
        int rankDest = ringRanks[0];
        size_t srcOffset = dataOffset + rankDest * count;

        // Wait for recv data
        if (tid == 0) {
          sparseWaitRecv(recvConn->tail, recvStep);
        }
        __syncthreads();

        char* recvSlot = recvBuf + (recvStep % NCCL_STEPS) * stepSize;
        size_t recvPayload = ((SparseChunkHeader*)recvSlot)->payload_bytes;
        size_t recvNelem = recvPayload / sizeof(T);

        // Reduce: local[srcOffset..] + recv payload → output[dataOffset..]
        T* recvData = (T*)(recvSlot + headerSize);
        T* localData = sendbuff + srcOffset;
        T* outputData = recvbuff + dataOffset;

        sparseReduce<T, RedOp>(outputData, localData, recvData,
                        min((size_t)nelem, recvNelem), tid, nthreads, work->redOpArg);

        __syncthreads();

        // Post recv credits
        if (tid == 0) {
          sparsePostRecv(recvConn->head, recvStep);
        }
        recvStep++;
      }
    }

    // Store final step values back to connection state
    if (tid == 0) {
      sendConn->step = sendStep;
      recvConn->step = recvStep;
    }
  }
}

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    if (work->isSparse) {
      runRingSparse<T, RedOp>(tid, nthreads, work);
      return;
    }
    using Proto = ProtoSimple<REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS>;
    runRing<T, RedOp, Proto>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    runRing<T, RedOp, ProtoLL128>(tid, nthreads, work);
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
#if __CUDA_ARCH__ >= 600
    using Proto = ProtoSimple<1, 1>;
    const int nranks = ncclShmem.comm.nRanks;
    const int rank = ncclShmem.comm.rank;
    size_t count, channelOffset, channelCount, chunkCount;
    ncclCollCbdPart(work, ncclShmem.channelId, Proto::Id, sizeof(T), &count, &channelOffset, &channelCount, &chunkCount);

    static constexpr int nworkers = NCCL_PAT_NWORKERS;
    struct ncclPatShmem* shmem = (struct ncclPatShmem*)ncclScratchForWarp(0);
    uint64_t pollCount = 0;
    __syncthreads(); // Don't start using shared mem until everyone arrives
    for (int i=tid; i<NCCL_SHMEM_PAT_STEPS; i+=nthreads) shmem->patSteps[i].flags = 0;
    if (tid == 0) shmem->localAccSize = 0;
    if (tid == nworkers) shmem->parallelFactor = 0;
    __syncthreads();

    if (tid == nworkers) { // Algo computation thread
      PatRSAlgorithm<T> patAlgo(chunkCount*sizeof(T), NCCL_STEPS, NCCL_PAT_NWORKERS/WARP_SIZE, channelOffset, channelOffset + channelCount, count, chunkCount, rank, nranks);
      int parallelFactor = shmem->parallelFactor = patAlgo.getParallelFactor();
      int step = 0;
      while (1) {
        struct ncclPatStep* ps = shmem->patSteps+(step%NCCL_SHMEM_PAT_STEPS);
        cuda::atomic_ref<int, cuda::thread_scope_block> poll(ps->flags);
        while (poll.load(cuda::memory_order_acquire) != 0) pollCount++; // Wait for workers to be done with step 'step-NCCL_SHMEM_PAT_STEPS'
        patAlgo.getNextOp(ps);
        int last = ps->last;
        step++;
        if (last == 2) break;
      }
    } else if (tid < nworkers) { // Worker threads
      T *inputBuf = (T*)work->sendbuff;
      T *outputBuf = (T*)work->recvbuff;
      int parallelFactor = 0;
      volatile int* pfPtr = &shmem->parallelFactor;
      while (parallelFactor == 0) parallelFactor = *pfPtr;

      int groupSize = nworkers/(WARP_SIZE*parallelFactor) * WARP_SIZE;
      int group = tid / groupSize;
      int nGroups = nworkers / groupSize;
      int tidInGroup = tid - group*groupSize;
      // We don't use recvPeers/sendPeers so let's pass shmem structs instead
      Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0> prims
        (tidInGroup, groupSize, (int*)shmem->recvDims, (int*)shmem->sendDims, inputBuf, outputBuf, work->redOpArg, group, 0, 0, nullptr, nullptr, 0, primsModePatRs);

      int step = group;
      while(1) {
        struct ncclPatStep* ps = shmem->patSteps+(step%NCCL_SHMEM_PAT_STEPS);
        cuda::atomic_ref<int, cuda::thread_scope_block> poll(ps->flags);
        while (poll.load(cuda::memory_order_acquire) == 0) pollCount++; // Wait for compute thread
        int last = ps->last;
        prims.patReduce(ps, shmem);
        if (tidInGroup == 0) poll.store(0, cuda::memory_order_release); // Return element to compute thread
        if (last) break;
        step += nGroups;
      }
    }
#endif
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE> {
  template<bool ReduceSendNotRecv>
  struct Scatterer {
    struct ncclDevWorkColl* work;
    int chunkCount;
    ssize_t railGridOffset;

    template<int SlicePerChunk, int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts, int MultimemSrcs, int MultimemDsts>
    __device__ __forceinline__ void operator()(
        int tid, int tn, int slice, int maxSliceSize,
        int nSrcs, void** srcPtrs, int nDsts, void** dstPtrs, int32_t* dstSizes, uint32_t sendDirectFlag, uint32_t recvDirectFlag
      ) {
      static_assert(SlicePerChunk == 1, "require: SlicePerChunk==1");
      static_assert(MaxDsts <= 1 || MaxSrcs <= 1, "require: MaxDsts<=1 || MaxSrcs<=1");

      struct ncclNvls* nvls = &ncclShmem.channel.nvls;
      int nNodes = ncclShmem.comm.nNodes;
      int nRails = nvls->nHeads;
      int part = ncclShmem.channelId - work->channelLo;
      void* inbuf = (void*)work->sendbuff;
      ssize_t countPerRank = work->collnet.count;

      ssize_t railAllBeg = min(railGridOffset + part * chunkCount, nNodes * countPerRank);
      ssize_t railAllEnd = min(railAllBeg + chunkCount, nNodes * countPerRank);
      int railAllSize = railAllEnd - railAllBeg;
      int rail = nvls->headRank;
      int dst = 0;
      if (ReduceSendNotRecv) {
        if (work->regUsed) return;
        rail = 0;
        nSrcs = 1;
      } else {
        rail = nvls->headRank;
      }
      if (tid < nDsts) dstSizes[tid] = railAllSize;
      do {
        int node = railAllBeg / countPerRank;
        int railAllOffset = 0;
        while (railAllOffset < railAllSize) {
          ssize_t railOneBeg = node * countPerRank;
          ssize_t railOneEnd = railOneBeg + countPerRank;
          ssize_t railOneOffset = (railAllBeg + railAllOffset) - railOneBeg;
          int delta = min(railAllEnd, railOneEnd) - (railAllBeg + railAllOffset);
          int rank = ncclShmem.comm.collNetDenseToUserRank[node * nRails + rail];
          ssize_t userOneBeg = rank * countPerRank + railOneOffset;
          if (nDsts != 0) {
            reduceCopy<ncclCollUnroll(), RedOp, T,
              /*MultimemSrcs=*/MultimemSrcs, 1, 1 + MaxSrcs,
              /*MultimemDsts,MinDsts,MaxDsts=*/MultimemDsts, 1, 1,
              /*PreOpSrcs=*/1>
              (tid, tn, work->redOpArg, &work->redOpArg, false,
                /*nSrcs=*/nSrcs, [=]__device__(int s) {
              return work->regUsed ? (T*)srcPtrs[s] + userOneBeg :
                !ReduceSendNotRecv ? (T*)srcPtrs[s] + railAllOffset:
                (T*)inbuf + userOneBeg;
            },
                /*nDsts=*/1, [=]__device__(int d/*==0*/) {
              return (T*)dstPtrs[dst] + railAllOffset;
            }, delta);
          }
          railAllOffset += delta;
          node += 1;
        }
        dst += 1;
        rail += 1;
      } while (ReduceSendNotRecv && dst < nRails);
    }
  };

  __device__ __forceinline__ void run(int tid, int/*nthreads*/, struct ncclDevWorkColl* work) {
    struct ncclNvls* nvls = &ncclShmem.channel.nvls;
    int nelem;

    /* if we are direct NVLS, we only need to allocate 1 warp to scatter for sync;
     * if not, based on #ranks, we allocate 7 or 5 warps to reduce to saturate bandwidth
     * and the rest are allocated to scatter. */
    const int nThreadsNetRecv = work->oneNode ? 0 : (work->netRegUsed ? WARP_SIZE :  6 * WARP_SIZE);
    const int nThreadsScatter = work->regUsed ? roundUp(nvls->nHeads << 2, WARP_SIZE) : 8 * WARP_SIZE;
    const int nThreadsReduce = NCCL_MAX_NTHREADS - nThreadsNetRecv - nThreadsScatter;
    const int tidEndNetRecv = nThreadsNetRecv;
    const int tidEndScatter = tidEndNetRecv + nThreadsScatter;
    const int tidEndReduce = tidEndScatter + nThreadsReduce;

    if (work->oneNode) {
      const int rank = ncclShmem.comm.rank;
      size_t offset;
      size_t count, gridOffset, channelCount, chunkCount;
      ncclCollCbdPart(work, ncclShmem.channelId, NCCL_PROTO_SIMPLE, sizeof(T), &count, &gridOffset, &channelCount, &chunkCount);
      if (!work->regUsed) {
        if (tid < tidEndScatter) {
          // Scatter
          using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
          Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
            prims(tid, nThreadsScatter, NULL, nvls->up, work->sendbuff, NULL,
              work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
          for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
            offset = gridOffset + elemOffset;
            nelem = min(chunkCount, channelCount - elemOffset);
            prims.scatter(offset, nvls->nHeads * count, nelem, count, -1, 0);
          }
          // coverity[overrun-call] => Coverity think prims.index can be greater than 1
        } else if (tid < tidEndReduce) {
          // Reduce through NVLS
          using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 0>;
          Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
            prims(tid - tidEndScatter, nThreadsReduce, &nvls->down, NULL, NULL, work->recvbuff,
              work->redOpArg, 3 * Proto::MaxGroupWidth, 0, 0);
          for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
            offset = gridOffset + elemOffset;
            nelem = min(chunkCount, channelCount - elemOffset);
            prims.recv(offset, nelem);
          }
        }
      } else {
        if (tid < tidEndScatter) {
          // Scatter
          using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
          Primitives<T, RedOp, FanSymmetric<NCCL_MAX_NVLS_ARITY>, /*Direct=*/0, Proto, 0>
            prims(tid, nThreadsScatter, nvls->up, nvls->up, NULL, NULL,
              work->redOpArg, 0 * Proto::MaxGroupWidth, 1, 1);
          for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
            prims.scatter(0, 0, 0, 0, -1, 0);
          }

          /* gather used as sync */
          prims.gather(0, 0, 0, 0, -1, 0);
        } else if (tid < tidEndReduce) {
          // Reduce through NVLS
          using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 0>;
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
            prims(tid - tidEndScatter, nThreadsReduce, &nvls->down, &nvls->down, NULL, work->recvbuff,
              work->redOpArg, 3 * Proto::MaxGroupWidth, 0, 0, work);
          for (size_t elemOffset = 0; elemOffset < channelCount; elemOffset += chunkCount) {
            size_t outOffset = gridOffset + elemOffset;
            size_t inpOffset = outOffset + rank * count;
            nelem = min(chunkCount, channelCount - elemOffset);
            // Coverity complains about a possible overrun inside the method invoked below, but that's actually
            // a false positive.
            // coverity[overrun-call:FALSE]
            prims.directRecvCopy(inpOffset, outOffset, nelem);
          }

          /* send for sync */
          prims.send(0, 0);
        }
      }
    } else {
      // multi-node
      int nNodes = ncclShmem.comm.nNodes;
      int part = ncclShmem.channelId - work->channelLo;
      ssize_t countPerRank = work->collnet.count;
      const int nChannels = work->channelHi - work->channelLo + 1;
      ssize_t chunkCount = work->collnet.chunkCount;
      if (tid < tidEndNetRecv) {
        using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
        if (work->netRegUsed) {
          if (tid == 0) {
            int steps = (int)divUp(nNodes * countPerRank, nChannels * chunkCount);
            Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>::recvPeerNotify(nvls->out, 0, steps);
          }
          __syncwarp();
        } else {
          Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
            prims(tid, nThreadsNetRecv, &nvls->out, nullptr, nullptr, work->recvbuff,
              work->redOpArg, 0 * Proto::MaxGroupWidth, 0, 0);
          for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkCount) {
            ssize_t railAllBeg = railGridOffset + part * chunkCount;
            ssize_t railAllEnd = min(railAllBeg + chunkCount, nNodes * countPerRank);
            ssize_t railOneBeg = ncclShmem.comm.node * countPerRank;
            ssize_t railOneEnd = railOneBeg + countPerRank;
            ssize_t beg = max(railAllBeg, railOneBeg);
            ssize_t end = min(railAllEnd, railOneEnd);
            prims.recv(beg - railOneBeg, max(ssize_t(0), end - beg), /*postOp=*/true);
          }
        }
      } else {
        if (tid < tidEndScatter) {
          using Proto = ProtoSimple<1, 1, COLL_UNROLL>;
          Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_NVLS_ARITY>, /*Direct=*/1, Proto, 0>
            prims(tid - tidEndNetRecv, nThreadsScatter, nullptr, nvls->up, work->sendbuff, nullptr,
              work->redOpArg, 1 * Proto::MaxGroupWidth, 1, 1, work);
          for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkCount) {
            Scatterer</*ReduceSendNotRecv=*/true> scat;
            scat.work = work;
            scat.chunkCount = chunkCount;
            scat.railGridOffset = railGridOffset;
            prims.template process</*Recv=*/0, /*Send=*/1>(scat);
          }
        } else if (tid < tidEndReduce) {
          using Proto = ProtoSimple<1, 1, COLL_UNROLL, 1, 0>;
          Primitives<T, RedOp, FanSymmetric<1>, /*Direct=*/1, Proto, 0>
            prims(tid - tidEndScatter, nThreadsReduce, &nvls->down, &nvls->out, nullptr, nullptr,
              work->redOpArg, 2 * Proto::MaxGroupWidth, 0, 1, work);
          for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkCount) {
            Scatterer</*ReduceSendNotRecv=*/false> scat;
            scat.work = work;
            scat.chunkCount = chunkCount;
            scat.railGridOffset = railGridOffset;
            prims.template process</*Recv=*/1, /*Send=*/1>(scat);
          }
        }
      }
    }
  }
};

template<typename T, typename RedOp>
struct RunWorkColl<ncclFuncReduceScatter, T, RedOp, NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE> {
  template<bool ReduceSendNotRecv>
  struct Scatterer {
    struct ncclDevWorkColl* work;
    int chunkSize;
    ssize_t railGridOffset;

    template<int SlicePerChunk, int MinSrcs, int MaxSrcs, int MinDsts, int MaxDsts, int MultimemSrcs, int MultimemDsts>
    __device__ __forceinline__ void operator()(
        int tid, int tn, int slice, int maxSliceSize,
        int nSrcs, void** srcPtrs, int nDsts, void** dstPtrs, int32_t* dstSizes, uint32_t sendDirectFlag, uint32_t recvDirectFlag
      ) {
      static_assert(SlicePerChunk==1, "require: SlicePerChunk==1");
      static_assert(MaxDsts<=1 || MaxSrcs<=1, "require: MaxDsts<=1 || MaxSrcs<=1");

      struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
      int nNodes = ncclShmem.comm.nNodes;
      int nRails = direct->nHeads;
      int part = ncclShmem.channelId - work->channelLo;
      void* inbuf = (void*)work->sendbuff;
      ssize_t countPerRank = work->collnet.count;

      ssize_t railAllBeg = min(railGridOffset + part*chunkSize, nNodes*countPerRank);
      ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes*countPerRank);
      int railAllSize = railAllEnd - railAllBeg;
      if (tid < nDsts) dstSizes[tid] = railAllSize;

      int dst = 0;
      int rail;
      if (!ReduceSendNotRecv) {
        rail = direct->headRank;
      } else {
        rail = direct->headRank+1;
        if (rail == nRails) rail = 0;
      }
      do {
        int node = railAllBeg/countPerRank;
        int railAllOffset = 0;
        while (railAllOffset < railAllSize) {
          ssize_t railOneBeg = node*countPerRank;
          ssize_t railOneEnd = railOneBeg + countPerRank;
          ssize_t railOneOffset = (railAllBeg+railAllOffset) - railOneBeg;
          int delta = min(railAllEnd, railOneEnd) - (railAllBeg+railAllOffset);
          int rank = ncclShmem.comm.collNetDenseToUserRank[node*nRails + rail];
          ssize_t userOneBeg = rank*countPerRank + railOneOffset;
          if (nDsts != 0) {
            reduceCopy<ncclCollUnroll(), RedOp, T,
                     /*MultimemSrcs=*/0, 1+MinSrcs, 1+MaxSrcs,
                     /*MultimemDsts,MinDsts,MaxDsts=*/0,1,1,
                     /*PreOpSrcs=*/1>
            (tid, tn, work->redOpArg, &work->redOpArg, false,
             /*nSrcs=*/1+nSrcs, [=]__device__(int s) {
               return s==0 ? (T*)inbuf + userOneBeg
                           : work->regUsed && (recvDirectFlag & NCCL_P2P_READ)
                           ? (T*)srcPtrs[s-1] + userOneBeg
                           : (T*)srcPtrs[s-1] + railAllOffset;
             },
             /*nDsts=*/1, [=]__device__(int d/*==0*/) {
               return (T*)dstPtrs[dst] + railAllOffset;
             },
             delta);
          }
          railAllOffset += delta;
          node += 1;
        }
        dst += 1;
        rail += 1;
        if (rail == nRails) rail = 0;
      } while (ReduceSendNotRecv && dst < nRails-1);
    }
  };

  __device__ __forceinline__ void run(int tid, int nthreads, struct ncclDevWorkColl* work) {
    const int part = ncclShmem.channelId - work->channelLo;
    const int nChannels = work->channelHi - work->channelLo + 1;
    struct ncclDirect* direct = &ncclShmem.channel.collnetDirect;
    int const &nNodes = ncclShmem.comm.nNodes;
    ssize_t chunkSize = int(work->collnet.chunkCount);
    ssize_t countPerRank = work->collnet.count;
    const int hasDn = (direct->down[0] >= 0) ? 1 : 0;

    if (direct->out == -1) __trap();
    bool isMultiRail = (direct->nHeads > 1);
    int nWarps1 = (isMultiRail ? 2 : 0);
    int nWarps2 = (isMultiRail ? 2 : 1);
    int nWarps3 = 1;
    float denom = float(work->nWarps)/float(nWarps1+nWarps2+nWarps3);
    nWarps3 = int(denom*nWarps3);
    nWarps2 = int(denom*nWarps2);
    nWarps1 = work->nWarps - (nWarps2+nWarps3);

    using Proto = ProtoSimple<1, 1>;

    int tn = nWarps1*WARP_SIZE;
    if (tid < tn) {
      // Phase 1: Scatter inputs to peers
      Primitives<T, RedOp, FanAsymmetric<0, NCCL_MAX_DIRECT_ARITY>, /*Direct=*/0, Proto, 0>
        prims(tid, tn, nullptr, direct->heads+1, work->sendbuff, nullptr,
              work->redOpArg, 0*Proto::MaxGroupWidth, 1, 1);
      for (ssize_t railGridOffset=0; railGridOffset < nNodes*countPerRank; railGridOffset += nChannels*chunkSize) {
        Scatterer</*ReduceSendNotRecv=*/true> scat;
        scat.work = work;
        scat.chunkSize = chunkSize;
        scat.railGridOffset = railGridOffset;
        prims.template process</*Recv=*/0, /*Send=*/1>(scat, 0, 0);
      }
      return;
    }
    tid -= tn;

    tn = nWarps2*WARP_SIZE;
    if (tid < tn) {
      if (work->netRegUsed && !hasDn) {
        if (tid == 0) {
          Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 1>, /*Direct=*/0, Proto, 0>::sendPeerNotify(direct->out, 1, 1);
        }
        __syncwarp();
      } else {
        // Phase 2: Reduce from peers + local input -> send to network
        Primitives<T, RedOp, FanAsymmetric<NCCL_MAX_DIRECT_ARITY, 1>, /*Direct=*/0, Proto, 0>
          prims(tid, tn, direct->heads + 1, &direct->out, nullptr, nullptr,
            work->redOpArg, 1 * Proto::MaxGroupWidth, 1, 1);
        for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkSize) {
          Scatterer</*ReduceSendNotRecv=*/false> scat;
          scat.work = work;
          scat.chunkSize = chunkSize;
          scat.railGridOffset = railGridOffset;
          prims.template process</*Recv=*/1, /*Send=*/1>(scat, 0, 0);
        }
      }
      return;
    }
    tid -= tn;

    tn = nWarps3*WARP_SIZE;
    if (tid < tn) {
      if (work->netRegUsed) {
        if (tid == 0) {
          int steps = hasDn ? (int)divUp(nNodes * countPerRank, nChannels * chunkSize) : 1;
          Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>::recvPeerNotify(direct->out, 0, steps);
        }
        __syncwarp();
      } else {
        // Phase 3: recv from network
        Primitives<T, RedOp, FanAsymmetric<1, 0>, /*Direct=*/0, Proto, 0>
          prims(tid, tn, &direct->out, nullptr, nullptr, work->recvbuff,
            work->redOpArg, 2 * Proto::MaxGroupWidth, 0, 0);
        for (ssize_t railGridOffset = 0; railGridOffset < nNodes * countPerRank; railGridOffset += nChannels * chunkSize) {
          ssize_t railAllBeg = railGridOffset + part * chunkSize;
          ssize_t railAllEnd = min(railAllBeg + chunkSize, nNodes * countPerRank);
          ssize_t railOneBeg = ncclShmem.comm.node * countPerRank;
          ssize_t railOneEnd = railOneBeg + countPerRank;
          ssize_t beg = max(railAllBeg, railOneBeg);
          ssize_t end = min(railAllEnd, railOneEnd);
          prims.recv(beg - railOneBeg, max(ssize_t(0), end - beg), /*postOp=*/true);
        }
      }
      return;
    }
  }
};
