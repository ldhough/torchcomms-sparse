/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "network/unpack/unpack.h"
#include "ccd.cuh"
#include <cassert>

enum primsMode {
  primsModeDefault = 0,
  primsModePatRs = 1,
  primsModePatAg = 2
};

template<typename T, typename RedOp, typename Fan, int Direct,
         int SlicePerChunk, int StepPerSlice, int Unroll, int P2p, int MultimemSrcs, int MultimemDsts, bool isNetOffload>
class Primitives<
    T, RedOp, Fan, Direct, ProtoSimple<SlicePerChunk, StepPerSlice, Unroll, MultimemSrcs, MultimemDsts>, P2p, isNetOffload
  > {
  static constexpr int MaxRecv = Fan::MaxRecv, MaxSend = Fan::MaxSend;
  static constexpr int Input=0, Output=1;
  static constexpr int RoleInput = 0x01,
                       RoleOutput = 0x02,
                       RoleWaitRecv = 0x04,
                       RoleWaitSend = 0x08,
                       RolePostSend = 0x10,
                       RolePostRecv = 0x20,
                       Aborted = 0x40,
                       NetRegMode = 0x80,
                       ConnFifoEnabled = 0x100,
                       DirectWrite = 0x200,
                       DirectRead = 0x400,
                       PatMode = 0x800,
                       NvlsMinPolling = 0x1000,
                       NetDeviceUnpack = 0x2000,
                       AnyNetDeviceUnpack = 0x4000;
  const int tid, tidInBlock;
  const int nthreads;
  int nworkers;
  const int stepSize;
  Fan fan;
  int index; // Peer index I'm responsible for
  int flags;
  int group;
  uint64_t step;
  struct ncclConnInfo* conn = NULL;
  struct ncclConnFifo* connFifo = NULL;
  T* connEltsFifo;
  T* directBuff = NULL;
  uint64_t *connStepPtr;
  uint64_t connStepCache; // Cache last seen value of (*connStepPtr)
  int      connStepSize; // Connection step size
  void*    netDeviceHandle;
  uint64_t accSize;
  uint8_t isSparse;
  uint8_t ccdFormatMask;
  float ccdDenseThreshold;
  float ccdTrackedDensity;  // expected post-reduce density of what we send; -1 = unknown
  float ccdBaseSparsity;    // (1 - d₀), per-rank sparsity factor; -1 = unknown

  // Don't use barrier 0 as it's used by the final sync
  __device__ void barrier() {
    if (nthreads == WARP_SIZE) __syncwarp();
    else {
      int bar = 15-group;
      barrier_sync(bar, nthreads);
    }
  }
  __device__ void subBarrier() {
    if (nworkers == WARP_SIZE) __syncwarp();
    else {
      int bar = 15-group - (nworkers!=nthreads ? 1 : 0);
      barrier_sync(bar, nworkers);
    }
  }

  // PAT uses a single barrier across all groups
  __device__ void patBarrier() {
    barrier_sync(15, NCCL_PAT_NWORKERS);
  }

  __device__ bool barrierAny(int vote) {
    if (nthreads == WARP_SIZE) {
      return __any_sync(~0u, vote);
    } else {
      int name = 15-group;
      return barrier_red_or(vote, name, nthreads);
    }
  }
  __device__ bool subBarrierAny(int vote) {
    if (nworkers == WARP_SIZE) {
      return __any_sync(~0u, vote);
    } else {
      int name = 15-group - (nworkers!=nthreads ? 1 : 0);
      return barrier_red_or(vote, name, nworkers);
    }
  }

  inline __device__ uint64_t loadStepValue(uint64_t* ptr) {
    #if __CUDA_ARCH__ >= 900 && CUDART_VERSION >= 12010
    if (flags & NvlsMinPolling) {
      uint64_t ans;
      asm volatile("multimem.ld_reduce.acquire.sys.global.min.u64 %0, [%1];" : "=l"(ans) : "l"(cvta_to_global(ptr)) : "memory");
      return ans;
    }
    #endif
    // volatile is faster than acquire but not as correct. Make sure reduceCopy
    // loads data using volatile so it doesn't see stale data in L1.
    return ld_volatile_global(ptr);
  }

  template <int DirectRecv, int DirectSend, int Recv, int Send, int Src, int Dst>
  __device__ __forceinline__ void waitPeer(intptr_t srcIx, intptr_t dstIx, int offset, int nelts) {
    const bool isSendNotRecv = (Send && Recv) ? (flags & RoleWaitSend) : Send;
    // Yes, for some template arguments this code will be unreachable.  That's fine.
    // coverity[dead_error_line]
    if ((flags & (Recv * RoleWaitRecv)) || (flags & (Send * RoleWaitSend))) {
      int spins = 0;
      while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
        connStepCache = loadStepValue(connStepPtr);
        if (checkAbort(flags, Aborted, spins)) break;
        //if (spins == 0) printf("r=%d b=%d t=%d SPUN OUT got=%d want=%d\n", ncclShmem.comm.rank, blockIdx.x, threadIdx.x, int(connStepCache + (isSendNotRecv ? NCCL_STEPS : 0)), int(step+StepPerSlice));
      }
    }

    if (flags & (Recv*RoleWaitRecv | Send*RoleWaitSend)) {
      if ((flags & ConnFifoEnabled) && (flags & (Send * RoleWaitSend)))
        connFifo[step%NCCL_STEPS].size = isSparse ? sizeof(CcdSparseChunkHeader) + nelts*sizeof(T) : nelts*sizeof(T);

      void **ptrs = isSendNotRecv ? (ncclShmem.groups[group].dsts + Dst)
                                  : (ncclShmem.groups[group].srcs + Src);
      if ((flags & NetRegMode) && ((!isSendNotRecv && DirectRecv) || (isSendNotRecv && DirectSend))) {
        if (P2p) {
          ptrs[index] = NULL;
        } else {
          if (isSendNotRecv) {
            if (!Recv)
              ptrs[index] = NULL;
            else
              ptrs[index] = (T*)ncclShmem.groups[group].userOutput + dstIx + offset;
          } else {
            ptrs[index] = (T*)ncclShmem.groups[group].userOutput + srcIx + offset;
          }
        }
      } else if ((flags & ConnFifoEnabled) && connFifo[step%NCCL_STEPS].mode == NCCL_MODE_OFFSET) {
        ptrs[index] = connEltsFifo + loadInt(&connFifo[step%NCCL_STEPS].offset)/sizeof(T);
      } else if (isSendNotRecv && DirectSend) {
        if (flags & DirectWrite) {
          ptrs[index] = directBuff + dstIx + offset;
        } else if (flags & DirectRead) {  // empty send
          ptrs[index] = nullptr;
        } else {
          ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize
              + (isSparse ? sizeof(CcdSparseChunkHeader)/sizeof(T) : 0);
        }
      } else if (!isSendNotRecv && DirectRecv) {
        if (flags & DirectRead) {
          ptrs[index] = directBuff + srcIx + offset;
        } else if (flags & DirectWrite) {
          ptrs[index] = directBuff + dstIx + offset;  // send to next from my output buffer
        } else {
          ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize
              + (isSparse ? sizeof(CcdSparseChunkHeader)/sizeof(T) : 0);
        }
      }
      else {
        // Yes, for some template arguments this code will be unreachable.  That's fine.
        // coverity[dead_error_line]
        ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize
            + (isSparse ? sizeof(CcdSparseChunkHeader)/sizeof(T) : 0);
      }
      if (flags & NetDeviceUnpack) {
        ncclNetDeviceIncrementHead(group, index);
      }
      step += StepPerSlice;
    }
  }

  template<int Recv, int Send>
  inline __device__ void postPeer(bool dataStored) {
    if (flags & (Recv*RolePostRecv | Send*RolePostSend)) {
      step += StepPerSlice;
      if (Send && (flags & RolePostSend) && (dataStored||(flags&ConnFifoEnabled))) {
        fence_acq_rel_sys();
      }
      st_relaxed_sys_global(connStepPtr, step);
    }
  }

  template <int DirectRecv1, int DirectSend1, int Recv, int Send, int SrcBuf, int DstBuf>
  __device__ __forceinline__ void genericOp(
      intptr_t srcIx, intptr_t dstIx, int nelem, bool postOp
    ) {
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    constexpr int Src = SrcBuf != -1;
    constexpr int Dst = DstBuf != -1;

    nelem = nelem < 0 ? 0 : nelem;
    int sliceSize = stepSize*StepPerSlice;
    sliceSize = max(divUp(nelem, 16*SlicePerChunk)*16, sliceSize/32);
    int slice = 0;
    int offset = 0;

    // CCD sparse: per-side transport detection (NET vs P2P)
    // ConnFifoEnabled is set only for NET (proxy) connections — P2P has null connFifo.
    __shared__ bool ccd_send_is_net, ccd_recv_is_net;
    if (isSparse) {
      if (tid == 0) {
        ccd_send_is_net = false;
        ccd_recv_is_net = false;
      }
      // RoleWaitSend (tid=1) and RoleWaitRecv (tid=0) set their transport flags
      if (flags & RoleWaitSend) ccd_send_is_net = (flags & ConnFifoEnabled) != 0;
      if (flags & RoleWaitRecv) ccd_recv_is_net = (flags & ConnFifoEnabled) != 0;
      barrier();
    }

    if (tid < nworkers && offset < nelem && !isNetOffload) {
      // Worker-only loop for non-empty slices. Non-workers and empty slices are
      // processed in the loop following this if block. The benefit of splitting
      // the loop like this is we pull two branches out of the critical path.
      // Using "number of branch insns (taken or not) encountered dynamically"
      // as the performance metric, then:
      //   perf_orig = 2*numslices
      //   perf_new = 2+numslices
      // So the new code and old code behave the same for numslices=2, and for
      // numslices>2 the new code is superior. And note that in the case
      // numslices=1, the loop is trivially unrollable (single iteration) so we
      // don't incur that that tail branch and we still have perf_new=2.
      //
      // ORIGINAL CODE:
      //   unrolled for(slices) {
      //     if(worker) { // This branch removed
      //       wait();
      //       subBarrier();
      //       if(slice not empty) // This branch removed
      //         ReduceCopyMulti();
      //     }
      //     barrier();
      //     post();
      //   } // Since we no longer unroll, new branch added here
      bool ccdDensityUpdatedThisStep = false;

      #if __CUDA_ARCH__ < 700
        // Above doesn't matter on older hardware.
        #pragma unroll SlicePerChunk
      #else
        #pragma unroll 1
      #endif
      do {
        sliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;
        if (tid == 0) {
          T* userInput = (T*)ncclShmem.groups[group].userInput;
          T* userOutput = (T*)ncclShmem.groups[group].userOutput;
          if (Src) ncclShmem.groups[group].srcs[0] = (SrcBuf==Input ? userInput : userOutput) + srcIx + offset;
          if (Dst) ncclShmem.groups[group].dsts[0] = (DstBuf==Input ? userInput : userOutput) + dstIx + offset;
        }
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(srcIx, dstIx, offset, sliceSize);
        subBarrier();
        /* if user abort the kernel, we don't need to actually perform copy/reduce; just set size
         * to 0 to avoid unnecessary workload. */
        int workSize = ncclShmem.aborted ? 0 : sliceSize;

        if (isSparse && workSize > 0 && ((Recv && ccd_recv_is_net) || (Send && ccd_send_is_net))) {
          // ============================================================
          // CCD Sparse path with per-side transport awareness.
          // Recv and send transport decisions are independent:
          //   P2P recv → data is raw dense in FIFO (no header)
          //   NET recv → data has CcdSparseChunkHeader (may be sparse or dense)
          //   P2P send → always dense, no header/connFifo write
          //   NET send → adaptive compressed, header + connFifo write
          // Dense accumulation uses NCCL's reduceCopy (int4-vectorized).
          // ============================================================

          // sendbuff slice pointer (srcs[0] set by tid==0 above)
          T* ccd_dense_buf = (T*)ncclShmem.groups[group].srcs[0];

          // Virtual 2D shape for SPOP 64×64 tiling (column of tiles).
          const size_t ccd_M = 64;
          const size_t ccd_N = (size_t)workSize / 64;
          const size_t ccd_total_tiles = ceil_div(ccd_N, (size_t)64);

          int ccd_bar = 15 - group - (nworkers != nthreads ? 1 : 0);

          // ---- Phase A: Recv classification ----
          bool recv_data_is_dense = true;  // default for no-recv or P2P recv
          if (Recv) {
            if (!ccd_recv_is_net) {
              // P2P recv: data is raw dense in FIFO, use reduceCopy to accumulate
              recv_data_is_dense = true;
            } else {
              // NET recv: read header to determine format
              char* ccd_recv_after_hdr = (char*)ncclShmem.groups[group].srcs[Src];
              CcdSparseChunkHeader* recv_hdr = (CcdSparseChunkHeader*)(ccd_recv_after_hdr - sizeof(CcdSparseChunkHeader));
              CcdCompressionProtocol recv_format = (CcdCompressionProtocol)recv_hdr->format;

              if (recv_format == CcdCompressionProtocol::DENSE) {
                recv_data_is_dense = true;
              } else {
                recv_data_is_dense = false;
                // Sparse scatter-add into sendbuff
                if (recv_format == CcdCompressionProtocol::COO1D) {
                  const T* recv_vals = (const T*)ccd_recv_after_hdr;
                  const unsigned* recv_keys = (const unsigned*)(ccd_recv_after_hdr + recv_hdr->nnz * sizeof(T));
                  ccd_coo1d_scatter_into<T, unsigned>(
                      ccd_dense_buf, recv_vals, recv_keys, recv_hdr->nnz,
                      0, nworkers / warpSize - 1
                  );
                } else {
                  // SPOP scatter-add
                  const uint64_t* recv_bv  = (const uint64_t*)(ccd_recv_after_hdr + recv_hdr->bv_offset);
                  const unsigned* recv_inds = (const unsigned*)(ccd_recv_after_hdr + recv_hdr->inds_offset);
                  const T* recv_vals        = (const T*)(ccd_recv_after_hdr + recv_hdr->vals_offset);
                  ccd_decompress_or_scatter_into<T, unsigned>(
                      ccd_dense_buf, recv_vals,
                      ccd_N, ccd_M,
                      recv_bv, recv_inds,
                      0, nworkers / warpSize - 1,
                      /*scatter_into=*/true
                  );
                }
              }

            }
          }

          // ---- Phase B: Send format selection ----
          // ccdTrackedDensity tracks the expected post-reduce density of what
          // we send. Updated once per genericOp call (not per slice).
          // Uses asymmetric formula: d = 1 - (1-d_recv) * (1-d₀)
          // because accumulated data (d_recv) is denser than local data (d₀).
          // ccdBaseSparsity = (1-d₀), captured from step 0 compression or
          // first NET recv (which always has j=1, so d_recv = d₀).
          if (!ccdDensityUpdatedThisStep) {
            ccdDensityUpdatedThisStep = true;
            if (Recv && ccd_recv_is_net) {
              // NET recv: observe actual density from header (global mem, committed by waitPeer)
              CcdSparseChunkHeader* density_hdr = (CcdSparseChunkHeader*)(
                  (char*)ncclShmem.groups[group].srcs[Src] - sizeof(CcdSparseChunkHeader));
              float d_recv = (float)density_hdr->nnz / (float)workSize;
              if (ccdBaseSparsity < 0.0f) {
                // First NET recv is always step 1 (1 rank's contribution): d_recv = d₀
                ccdBaseSparsity = 1.0f - d_recv;
              }
              // Post-reduce: 1 - (1 - d_recv) * (1 - d₀)
              ccdTrackedDensity = 1.0f - (1.0f - d_recv) * ccdBaseSparsity;
            } else if (Recv && ccdTrackedDensity >= 0.0f && ccdBaseSparsity >= 0.0f) {
              // P2P recv (not step 0): extrapolate one reduce step forward.
              // Guard on Recv: step 0 (!Recv) sends our own local data, not an
              // extrapolation of accumulated density. Its actual d₀ is captured
              // after compression in Phase C. Without this guard, chunk 2+'s
              // step 0 would extrapolate chunk 1's final density toward 1.0.
              ccdTrackedDensity = 1.0f - (1.0f - ccdTrackedDensity) * ccdBaseSparsity;
            }
            // else: unknown or step 0 → no update (Phase C captures d₀ on step 0)
          }

          bool send_is_compressed = false;
          CcdCompressionProtocol ccd_protocol = CcdCompressionProtocol::DENSE;
          if (Send) {
            if (!ccd_send_is_net) {
              // P2P send: always dense, no compression needed
              send_is_compressed = false;
            } else if (!Recv || ccdTrackedDensity < 0.0f) {
              // Step 0 (!Recv) or no density info: force SPOP.
              // Step 0 must always compress to capture actual d₀ for
              // subsequent extrapolation. This also handles chunk 2+
              // where ccdTrackedDensity is stale from the previous chunk.
              ccd_protocol = (ccdFormatMask & 4) ? CcdCompressionProtocol::SPOP : CcdCompressionProtocol::DENSE;
              send_is_compressed = (ccd_protocol != CcdCompressionProtocol::DENSE);
            } else {
              // ccdTrackedDensity already represents expected post-reduce density
              size_t expected_nnz = (size_t)(ccdTrackedDensity * (float)workSize);
              ccd_protocol = select_ccd_compression_protocol<T, unsigned>(expected_nnz, (size_t)workSize, ccdFormatMask, ccdDenseThreshold);
              send_is_compressed = (ccd_protocol != CcdCompressionProtocol::DENSE);
            }
          }

          // ---- Phase C: Combined execution ----
          if (Recv && recv_data_is_dense && !send_is_compressed) {
            // Dense recv + dense send (P2P or NET DENSE):
            // reduceCopy accumulates recv into destination.
            // Barrier needed after sparse scatter-add? No — recv is dense, no scatter-add done.
            if (Send) {
              // Reduce sendbuff + recv → send FIFO (or sendbuff for P2P)
              void* rc_srcs[2] = { (void*)ccd_dense_buf, ncclShmem.groups[group].srcs[Src] };
              void* rc_dsts[1] = { ncclShmem.groups[group].dsts[Dst] };
              reduceCopy<Unroll, RedOp, T, 0,2,2, 0,1,1, /*PreOpSrcs=*/1>(
                  tid, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, /*postOp=*/false,
                  2, rc_srcs, 1, rc_dsts, workSize);

              // Write header + connFifo for NET dense sends
              if (ccd_send_is_net) {
                barrier_sync(ccd_bar, nworkers);
                size_t ccd_payload = (size_t)workSize * sizeof(T);
                if (tid == 0) {
                  char* ccd_send_after_hdr = (char*)ncclShmem.groups[group].dsts[Dst];
                  CcdSparseChunkHeader* hdr = (CcdSparseChunkHeader*)(ccd_send_after_hdr - sizeof(CcdSparseChunkHeader));
                  hdr->payload_bytes = ccd_payload;
                  hdr->nnz = (size_t)workSize;
                  hdr->bv_offset = 0;
                  hdr->inds_offset = 0;
                  hdr->vals_offset = 0;
                  hdr->format = (size_t)CcdCompressionProtocol::DENSE;
                }
                if ((flags & RoleWaitSend) && (flags & ConnFifoEnabled)) {
                  int ccd_send_size = sizeof(CcdSparseChunkHeader) + ccd_payload;
                  connFifo[(step - StepPerSlice) % NCCL_STEPS].size =
                      max(ccd_send_size, CCD_MIN_SEND_BYTES);
                }
              }
            } else if (Dst) {
              // Final step (recvReduceCopy): reduce sendbuff + recv → output with postOp
              void* rc_srcs[2] = { (void*)ccd_dense_buf, ncclShmem.groups[group].srcs[Src] };
              void* rc_dsts[1] = { ncclShmem.groups[group].dsts[0] };
              reduceCopy<Unroll, RedOp, T, 0,2,2, 0,1,1, /*PreOpSrcs=*/1>(
                  tid, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, /*postOp=*/true,
                  2, rc_srcs, 1, rc_dsts, workSize);
            }
          } else if (Recv && recv_data_is_dense && send_is_compressed) {
            // Dense recv + compressed send:
            // First accumulate recv into sendbuff in-place, then compress.
            // In-place reduceCopy: srcs = {sendbuff, recv}, dst = sendbuff
            {
              void* rc_srcs[2] = { (void*)ccd_dense_buf, ncclShmem.groups[group].srcs[Src] };
              void* rc_dsts[1] = { (void*)ccd_dense_buf };
              reduceCopy<Unroll, RedOp, T, 0,2,2, 0,1,1, /*PreOpSrcs=*/1>(
                  tid, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, /*postOp=*/false,
                  2, rc_srcs, 1, rc_dsts, workSize);
            }
            // Barrier: all workers must finish accumulation before compress reads sendbuff
            barrier_sync(ccd_bar, nworkers);

            // Compress sendbuff → send FIFO
            char* ccd_send_after_hdr = (char*)ncclShmem.groups[group].dsts[Dst];
            if (ccd_protocol == CcdCompressionProtocol::COO1D) {
              const size_t ccd_num_inds = ccd_total_tiles + 1;
              const size_t ccd_bv_bytes = ccd_total_tiles * 64 * sizeof(uint64_t);
              const size_t ccd_inds_bytes = ccd_num_inds * sizeof(unsigned);
              size_t slot_bytes = (size_t)stepSize * StepPerSlice * sizeof(T);
              char* slot_base = ccd_send_after_hdr - sizeof(CcdSparseChunkHeader);
              unsigned* ccd_inds = (unsigned*)(slot_base + slot_bytes - ccd_pad16(ccd_inds_bytes));
              uint64_t* ccd_bv = (uint64_t*)((char*)ccd_inds - ccd_pad16(ccd_bv_bytes));
              T* ccd_vals = (T*)ccd_send_after_hdr;
              ccd_fused_single_block_spop_compress<T, unsigned>(
                  ccd_dense_buf, ccd_vals, ccd_N, ccd_M,
                  ccd_bv, ccd_inds, 0, nworkers / warpSize - 1,
                  ccd_bar, nworkers, CcdCompressionProtocol::COO1D, 0);
              size_t ccd_nnz = ccd_inds[ccd_total_tiles];
              ccdTrackedDensity = (float)ccd_nnz / (float)workSize;  // calibrate from actual
              size_t ccd_payload = ccd_nnz * (sizeof(T) + sizeof(unsigned));
              if (tid == 0) {
                CcdSparseChunkHeader* hdr = (CcdSparseChunkHeader*)(ccd_send_after_hdr - sizeof(CcdSparseChunkHeader));
                hdr->payload_bytes = ccd_payload;
                hdr->nnz = ccd_nnz;
                hdr->bv_offset = 0; hdr->inds_offset = 0; hdr->vals_offset = 0;
                hdr->format = (size_t)CcdCompressionProtocol::COO1D;
              }
              if ((flags & RoleWaitSend) && (flags & ConnFifoEnabled)) {
                connFifo[(step - StepPerSlice) % NCCL_STEPS].size =
                    max((int)(sizeof(CcdSparseChunkHeader) + ccd_payload), CCD_MIN_SEND_BYTES);
              }
            } else {
              // SPOP
              const size_t ccd_num_inds = ccd_total_tiles + 1;
              const size_t ccd_bv_bytes    = ccd_total_tiles * 64 * sizeof(uint64_t);
              const size_t ccd_bv_padded   = ccd_pad16(ccd_bv_bytes);
              const size_t ccd_inds_bytes  = ccd_num_inds * sizeof(unsigned);
              const size_t ccd_inds_padded = ccd_pad16(ccd_inds_bytes);
              const size_t ccd_vals_offset = ccd_bv_padded + ccd_inds_padded;
              uint64_t* ccd_bv   = (uint64_t*)ccd_send_after_hdr;
              unsigned* ccd_inds = (unsigned*)(ccd_send_after_hdr + ccd_bv_padded);
              T* ccd_vals        = (T*)(ccd_send_after_hdr + ccd_vals_offset);
              ccd_fused_single_block_spop_compress<T, unsigned>(
                  ccd_dense_buf, ccd_vals, ccd_N, ccd_M,
                  ccd_bv, ccd_inds, 0, nworkers / warpSize - 1,
                  ccd_bar, nworkers);
              size_t ccd_nnz = ccd_inds[ccd_total_tiles];
              ccdTrackedDensity = (float)ccd_nnz / (float)workSize;  // calibrate from actual
              size_t ccd_payload = ccd_vals_offset + ccd_nnz * sizeof(T);
              if (tid == 0) {
                CcdSparseChunkHeader* hdr = (CcdSparseChunkHeader*)(ccd_send_after_hdr - sizeof(CcdSparseChunkHeader));
                hdr->payload_bytes = ccd_payload;
                hdr->nnz = ccd_nnz;
                hdr->bv_offset = 0;
                hdr->inds_offset = ccd_bv_padded;
                hdr->vals_offset = ccd_vals_offset;
                hdr->format = (size_t)CcdCompressionProtocol::SPOP;
              }
              if ((flags & RoleWaitSend) && (flags & ConnFifoEnabled)) {
                connFifo[(step - StepPerSlice) % NCCL_STEPS].size =
                    max((int)(sizeof(CcdSparseChunkHeader) + ccd_payload), CCD_MIN_SEND_BYTES);
              }
            }
          } else if (Recv && !recv_data_is_dense) {
            // Sparse recv (scatter-add already done in Phase A).
            // Barrier: all workers must finish scatter-add before send reads sendbuff.
            barrier_sync(ccd_bar, nworkers);

            if (Send && send_is_compressed) {
              // Sparse recv + compressed send: compress sendbuff → send FIFO
              char* ccd_send_after_hdr = (char*)ncclShmem.groups[group].dsts[Dst];
              if (ccd_protocol == CcdCompressionProtocol::COO1D) {
                const size_t ccd_num_inds = ccd_total_tiles + 1;
                const size_t ccd_bv_bytes = ccd_total_tiles * 64 * sizeof(uint64_t);
                const size_t ccd_inds_bytes = ccd_num_inds * sizeof(unsigned);
                size_t slot_bytes = (size_t)stepSize * StepPerSlice * sizeof(T);
                char* slot_base = ccd_send_after_hdr - sizeof(CcdSparseChunkHeader);
                unsigned* ccd_inds = (unsigned*)(slot_base + slot_bytes - ccd_pad16(ccd_inds_bytes));
                uint64_t* ccd_bv = (uint64_t*)((char*)ccd_inds - ccd_pad16(ccd_bv_bytes));
                T* ccd_vals = (T*)ccd_send_after_hdr;
                ccd_fused_single_block_spop_compress<T, unsigned>(
                    ccd_dense_buf, ccd_vals, ccd_N, ccd_M,
                    ccd_bv, ccd_inds, 0, nworkers / warpSize - 1,
                    ccd_bar, nworkers, CcdCompressionProtocol::COO1D, 0);
                size_t ccd_nnz = ccd_inds[ccd_total_tiles];
                ccdTrackedDensity = (float)ccd_nnz / (float)workSize;  // calibrate from actual
                size_t ccd_payload = ccd_nnz * (sizeof(T) + sizeof(unsigned));
                if (tid == 0) {
                  CcdSparseChunkHeader* hdr = (CcdSparseChunkHeader*)(ccd_send_after_hdr - sizeof(CcdSparseChunkHeader));
                  hdr->payload_bytes = ccd_payload;
                  hdr->nnz = ccd_nnz;
                  hdr->bv_offset = 0; hdr->inds_offset = 0; hdr->vals_offset = 0;
                  hdr->format = (size_t)CcdCompressionProtocol::COO1D;
                }
                if ((flags & RoleWaitSend) && (flags & ConnFifoEnabled)) {
                  connFifo[(step - StepPerSlice) % NCCL_STEPS].size =
                      max((int)(sizeof(CcdSparseChunkHeader) + ccd_payload), CCD_MIN_SEND_BYTES);
                }
              } else {
                // SPOP
                const size_t ccd_num_inds = ccd_total_tiles + 1;
                const size_t ccd_bv_bytes    = ccd_total_tiles * 64 * sizeof(uint64_t);
                const size_t ccd_bv_padded   = ccd_pad16(ccd_bv_bytes);
                const size_t ccd_inds_bytes  = ccd_num_inds * sizeof(unsigned);
                const size_t ccd_inds_padded = ccd_pad16(ccd_inds_bytes);
                const size_t ccd_vals_offset = ccd_bv_padded + ccd_inds_padded;
                uint64_t* ccd_bv   = (uint64_t*)ccd_send_after_hdr;
                unsigned* ccd_inds = (unsigned*)(ccd_send_after_hdr + ccd_bv_padded);
                T* ccd_vals        = (T*)(ccd_send_after_hdr + ccd_vals_offset);
                ccd_fused_single_block_spop_compress<T, unsigned>(
                    ccd_dense_buf, ccd_vals, ccd_N, ccd_M,
                    ccd_bv, ccd_inds, 0, nworkers / warpSize - 1,
                    ccd_bar, nworkers);
                size_t ccd_nnz = ccd_inds[ccd_total_tiles];
                ccdTrackedDensity = (float)ccd_nnz / (float)workSize;  // calibrate from actual
                size_t ccd_payload = ccd_vals_offset + ccd_nnz * sizeof(T);
                if (tid == 0) {
                  CcdSparseChunkHeader* hdr = (CcdSparseChunkHeader*)(ccd_send_after_hdr - sizeof(CcdSparseChunkHeader));
                  hdr->payload_bytes = ccd_payload;
                  hdr->nnz = ccd_nnz;
                  hdr->bv_offset = 0;
                  hdr->inds_offset = ccd_bv_padded;
                  hdr->vals_offset = ccd_vals_offset;
                  hdr->format = (size_t)CcdCompressionProtocol::SPOP;
                }
                if ((flags & RoleWaitSend) && (flags & ConnFifoEnabled)) {
                  connFifo[(step - StepPerSlice) % NCCL_STEPS].size =
                      max((int)(sizeof(CcdSparseChunkHeader) + ccd_payload), CCD_MIN_SEND_BYTES);
                }
              }
            } else if (Send && !send_is_compressed) {
              // Sparse recv + dense send: copy sendbuff → send FIFO via reduceCopy
              void* rc_srcs[1] = { (void*)ccd_dense_buf };
              void* rc_dsts[1] = { ncclShmem.groups[group].dsts[Dst] };
              reduceCopy<Unroll, RedOp, T, 0,1,1, 0,1,1, /*PreOpSrcs=*/0>(
                  tid, nworkers, 0, nullptr, /*postOp=*/false,
                  1, rc_srcs, 1, rc_dsts, workSize);

              // Write header + connFifo for NET dense sends
              if (ccd_send_is_net) {
                barrier_sync(ccd_bar, nworkers);
                size_t ccd_payload = (size_t)workSize * sizeof(T);
                if (tid == 0) {
                  char* ccd_send_after_hdr = (char*)ncclShmem.groups[group].dsts[Dst];
                  CcdSparseChunkHeader* hdr = (CcdSparseChunkHeader*)(ccd_send_after_hdr - sizeof(CcdSparseChunkHeader));
                  hdr->payload_bytes = ccd_payload;
                  hdr->nnz = (size_t)workSize;
                  hdr->bv_offset = 0;
                  hdr->inds_offset = 0;
                  hdr->vals_offset = 0;
                  hdr->format = (size_t)CcdCompressionProtocol::DENSE;
                }
                if ((flags & RoleWaitSend) && (flags & ConnFifoEnabled)) {
                  int ccd_send_size = sizeof(CcdSparseChunkHeader) + ccd_payload;
                  connFifo[(step - StepPerSlice) % NCCL_STEPS].size =
                      max(ccd_send_size, CCD_MIN_SEND_BYTES);
                }
              }
            } else if (!Send && Dst) {
              // Sparse recv + no send (final step): copy sendbuff → output with postOp
              void* rc_srcs[1] = { (void*)ccd_dense_buf };
              void* rc_dsts[1] = { ncclShmem.groups[group].dsts[0] };
              reduceCopy<Unroll, RedOp, T, 0,1,1, 0,1,1, /*PreOpSrcs=*/0>(
                  tid, nworkers, 0, nullptr, /*postOp=*/true,
                  1, rc_srcs, 1, rc_dsts, workSize);
            }
          } else if (!Recv && Send) {
            // No recv (step 0) + NET send: compress or dense-copy sendbuff → send FIFO
            char* ccd_send_after_hdr = (char*)ncclShmem.groups[group].dsts[Dst];
            size_t ccd_step0_nnz = 0;  // capture actual nnz for d₀ tracking

            if (ccd_protocol == CcdCompressionProtocol::DENSE) {
              // Dense step 0 send
              for (int i = tid; i < workSize; i += nworkers) {
                ((T*)ccd_send_after_hdr)[i] = ccd_dense_buf[i];
              }
              barrier_sync(ccd_bar, nworkers);
              ccd_step0_nnz = (size_t)workSize;
              size_t ccd_payload = (size_t)workSize * sizeof(T);
              if (tid == 0) {
                CcdSparseChunkHeader* hdr = (CcdSparseChunkHeader*)(ccd_send_after_hdr - sizeof(CcdSparseChunkHeader));
                hdr->payload_bytes = ccd_payload;
                hdr->nnz = (size_t)workSize;
                hdr->bv_offset = 0; hdr->inds_offset = 0; hdr->vals_offset = 0;
                hdr->format = (size_t)CcdCompressionProtocol::DENSE;
              }
              if ((flags & RoleWaitSend) && (flags & ConnFifoEnabled)) {
                connFifo[(step - StepPerSlice) % NCCL_STEPS].size =
                    max((int)(sizeof(CcdSparseChunkHeader) + ccd_payload), CCD_MIN_SEND_BYTES);
              }
            } else if (ccd_protocol == CcdCompressionProtocol::COO1D) {
              const size_t ccd_num_inds = ccd_total_tiles + 1;
              const size_t ccd_bv_bytes = ccd_total_tiles * 64 * sizeof(uint64_t);
              const size_t ccd_inds_bytes = ccd_num_inds * sizeof(unsigned);
              size_t slot_bytes = (size_t)stepSize * StepPerSlice * sizeof(T);
              char* slot_base = ccd_send_after_hdr - sizeof(CcdSparseChunkHeader);
              unsigned* ccd_inds = (unsigned*)(slot_base + slot_bytes - ccd_pad16(ccd_inds_bytes));
              uint64_t* ccd_bv = (uint64_t*)((char*)ccd_inds - ccd_pad16(ccd_bv_bytes));
              T* ccd_vals = (T*)ccd_send_after_hdr;
              ccd_fused_single_block_spop_compress<T, unsigned>(
                  ccd_dense_buf, ccd_vals, ccd_N, ccd_M,
                  ccd_bv, ccd_inds, 0, nworkers / warpSize - 1,
                  ccd_bar, nworkers, CcdCompressionProtocol::COO1D, 0);
              size_t ccd_nnz = ccd_inds[ccd_total_tiles];
              ccd_step0_nnz = ccd_nnz;
              size_t ccd_payload = ccd_nnz * (sizeof(T) + sizeof(unsigned));
              if (tid == 0) {
                CcdSparseChunkHeader* hdr = (CcdSparseChunkHeader*)(ccd_send_after_hdr - sizeof(CcdSparseChunkHeader));
                hdr->payload_bytes = ccd_payload;
                hdr->nnz = ccd_nnz;
                hdr->bv_offset = 0; hdr->inds_offset = 0; hdr->vals_offset = 0;
                hdr->format = (size_t)CcdCompressionProtocol::COO1D;
              }
              if ((flags & RoleWaitSend) && (flags & ConnFifoEnabled)) {
                connFifo[(step - StepPerSlice) % NCCL_STEPS].size =
                    max((int)(sizeof(CcdSparseChunkHeader) + ccd_payload), CCD_MIN_SEND_BYTES);
              }
            } else {
              // SPOP step 0
              const size_t ccd_num_inds = ccd_total_tiles + 1;
              const size_t ccd_bv_bytes    = ccd_total_tiles * 64 * sizeof(uint64_t);
              const size_t ccd_bv_padded   = ccd_pad16(ccd_bv_bytes);
              const size_t ccd_inds_bytes  = ccd_num_inds * sizeof(unsigned);
              const size_t ccd_inds_padded = ccd_pad16(ccd_inds_bytes);
              const size_t ccd_vals_offset = ccd_bv_padded + ccd_inds_padded;
              uint64_t* ccd_bv   = (uint64_t*)ccd_send_after_hdr;
              unsigned* ccd_inds = (unsigned*)(ccd_send_after_hdr + ccd_bv_padded);
              T* ccd_vals        = (T*)(ccd_send_after_hdr + ccd_vals_offset);
              ccd_fused_single_block_spop_compress<T, unsigned>(
                  ccd_dense_buf, ccd_vals, ccd_N, ccd_M,
                  ccd_bv, ccd_inds, 0, nworkers / warpSize - 1,
                  ccd_bar, nworkers);
              size_t ccd_nnz = ccd_inds[ccd_total_tiles];
              ccd_step0_nnz = ccd_nnz;
              size_t ccd_payload = ccd_vals_offset + ccd_nnz * sizeof(T);
              if (tid == 0) {
                CcdSparseChunkHeader* hdr = (CcdSparseChunkHeader*)(ccd_send_after_hdr - sizeof(CcdSparseChunkHeader));
                hdr->payload_bytes = ccd_payload;
                hdr->nnz = ccd_nnz;
                hdr->bv_offset = 0;
                hdr->inds_offset = ccd_bv_padded;
                hdr->vals_offset = ccd_vals_offset;
                hdr->format = (size_t)CcdCompressionProtocol::SPOP;
              }
              if ((flags & RoleWaitSend) && (flags & ConnFifoEnabled)) {
                connFifo[(step - StepPerSlice) % NCCL_STEPS].size =
                    max((int)(sizeof(CcdSparseChunkHeader) + ccd_payload), CCD_MIN_SEND_BYTES);
              }
            }
            // Capture d₀ from step 0 compression for asymmetric density extrapolation.
            // Step 0 sends our own local data, so nnz/workSize = d₀.
            float ccd_d0 = (float)ccd_step0_nnz / (float)workSize;
            ccdTrackedDensity = ccd_d0;
            if (ccdBaseSparsity < 0.0f) {
              ccdBaseSparsity = 1.0f - ccd_d0;
            }
          }

        } else {
          // ============================================================
          // Dense path (or workSize==0)
          // ============================================================
          if (flags & AnyNetDeviceUnpack) {
            ncclNetDeviceUnpack<Recv>(tid, tidInBlock, nworkers, group, ncclShmem.groups[group].devicePlugin.unpack.unpackNetDeviceIndexMask, Src, workSize);
            // Sync here to make sure all workers are reading from the updated srcs)
            subBarrier();
          }

          if (DirectRecv && ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[0]
              /* NVLS can have srcs[0] == dsts[0], but we cannot enter this "if branch",
               * so we need to check whether MultimemSrcs and MultimemDsts are 0. */
              && MultimemSrcs == 0 && MultimemDsts == 0 && !Src) {
            // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
            if (Send && Dst && ncclShmem.groups[group].srcs[0] != ncclShmem.groups[group].dsts[1]) {
              reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, MaxSend, /*PreOpSrcs*/0>
                (tid, nworkers, /*redArg*/0, /*preOpArgs*/nullptr, /*postOp*/false,
                 1, ncclShmem.groups[group].srcs,
                 fan.nsend(), ncclShmem.groups[group].dsts+1,
                 workSize);
            }
          } else if (DirectSend && !DirectRecv && SrcBuf != Input && ncclShmem.groups[group].dsts[Dst] == nullptr) {
            // For broadcast in CollNet to do empty send
            reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, 1, /*PreOpSrcs*/0>
              (tid, nworkers, ncclShmem.redOpArgs[0],  nullptr, postOp,
               Recv, ncclShmem.groups[group].srcs,
               Dst, ncclShmem.groups[group].dsts,
               workSize);
          } else if (ncclShmem.groups[group].srcs[0] && ncclShmem.groups[group].dsts[0]) {
            constexpr int PreOpSrcs = SrcBuf != Input ? 0 :
                                      DirectRecv*MaxRecv == NCCL_MAX_DIRECT_ARITY ? (1+NCCL_MAX_DIRECT_ARITY) : 1;
            if (Send && Dst && ncclShmem.groups[group].dsts[1] == nullptr) {
              // this case should only be directCopySend() with registered buffers and send to net peer
              reduceCopy<Unroll, RedOp, T,
                0, Recv + Src, Recv * MaxRecv + Src,
                0, 1, 1, PreOpSrcs>
                (tid, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, postOp,
                  Recv * fan.nrecv() + Src, ncclShmem.groups[group].srcs,
                  1, ncclShmem.groups[group].dsts,
                  workSize);
            } else {
              reduceCopy<Unroll, RedOp, T,
                MultimemSrcs, Recv + Src, Recv * MaxRecv + Src,
                MultimemDsts, Send + Dst, Send * MaxSend + Dst, PreOpSrcs>
                (tid, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, postOp,
                  Recv * fan.nrecv() + Src, ncclShmem.groups[group].srcs,
                  Send * fan.nsend() + Dst, ncclShmem.groups[group].dsts,
                  workSize);
            }
          } else {
            // we will come here when calling prims.directSend with net peer,
            // in this case, ncclShmem.groups[group].dsts[0] == NULL, so we
            // skip data flush.
            workSize = 0;
          }
        }
        barrier(); // This barrier has a counterpart in following loop
        postPeer<Recv, Send>(0 < workSize);
        offset += sliceSize;
        slice += 1;
        // Yes, for some template arguments this code will be unreachable.  That's fine.
        // coverity[dead_error_line]
      } while (slice < SlicePerChunk && offset < nelem);
    }

    // Non-workers come straight here. Workers too but only once the remaining
    // slices are all empty. Since empty slices are the uncommon case, and
    // worker perf is the limiter, perf-wise this loop is effectively unentered,
    // hence just a single branch insn.
    #pragma unroll 1
    while (slice < SlicePerChunk) {
      sliceSize = sliceSize < nelem-offset ? sliceSize : nelem-offset;
      { // Only workers could have Wait roles so we know the slice must be empty
        // since we've exited the loop above.
        waitPeer<DirectRecv, DirectSend, Recv, Send, Src, Dst>(0, 0, 0, sliceSize);
      }
      // Sparse header protocol: write zero-payload header for non-worker slices.
      // Use RoleWaitSend (not tid==0) because there's no subBarrier here,
      // and only the WaitSend thread is guaranteed to see dsts[Dst] it set in waitPeer.
      if (Send && isSparse && ccd_send_is_net && (flags & RoleWaitSend)) {
        CcdSparseChunkHeader* hdr = (CcdSparseChunkHeader*)((char*)ncclShmem.groups[group].dsts[Dst] - sizeof(CcdSparseChunkHeader));
        hdr->payload_bytes = 0;
        hdr->nnz = 0;
        hdr->bv_offset = 0;
        hdr->inds_offset = 0;
        hdr->vals_offset = 0;
        hdr->format = (size_t)CcdCompressionProtocol::DENSE;
      }
      barrier(); // Has couterpart in preceding worker-only loop.
      int workSize = ncclShmem.aborted ? 0 : sliceSize;
      postPeer<Recv, Send>(0 < workSize);
      offset += sliceSize;
      slice += 1;
    }
  }

public:
  static inline __device__ void sendPeerNotify(int peer, int connIndex, int steps) {
    ncclDevChannelPeer* peerPtr = ncclShmem.channel.peers[peer];
    peerPtr->send[connIndex].step += steps;
    st_relaxed_sys_global(peerPtr->send[connIndex].tail, peerPtr->send[connIndex].step);
  }

  static inline __device__ void recvPeerNotify(int peer, int connIndex, int steps) {
    int spins = 0;
    ncclDevChannelPeer* peerPtr = ncclShmem.channel.peers[peer];
    peerPtr->recv[connIndex].step += steps;
    st_relaxed_sys_global(peerPtr->recv[connIndex].head, peerPtr->recv[connIndex].step);
    while (ld_volatile_global(peerPtr->recv[connIndex].tail) < peerPtr->recv[connIndex].step) {
      int abort = 0;
      if (checkAbort(abort, 1, spins)) break;
    }
  }

  template<int Recv, int Send, typename Fn>
  __device__ __forceinline__ void process(Fn &&fn, uint32_t sendDirectFlag = 0, uint32_t recvDirectFlag = 0) {
    #pragma unroll 1
    for (int slice=0; slice < SlicePerChunk; slice++) {
      if (tid < nworkers) {
        int nsend, nrecv;
        if (flags & (Recv*RoleWaitRecv | Send*RoleWaitSend)) {
          const bool isSendNotRecv = (Send && Recv) ? (flags & RoleWaitSend) : Send;
          int spins = 0;
          while (connStepCache + (isSendNotRecv ? NCCL_STEPS : 0) < step + StepPerSlice) {
            connStepCache = loadStepValue(connStepPtr);
            if (checkAbort(flags, Aborted, spins)) break;
          }
          void **ptrs = isSendNotRecv ? ncclShmem.groups[group].dsts
                                      : ncclShmem.groups[group].srcs;
          if ((flags & ConnFifoEnabled) && connFifo[step%NCCL_STEPS].mode == NCCL_MODE_OFFSET) {
            int offset = loadInt(&connFifo[step%NCCL_STEPS].offset);
            ptrs[index] = connEltsFifo + offset/sizeof(T);
          } else if (Direct && fn.work->regUsed) {
            if (isSendNotRecv) {
              if (flags & DirectWrite) {
                ptrs[index] = directBuff;
              } else if (flags & DirectRead) {  // empty send
                ptrs[index] = nullptr;
              } else {
                ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize;
              }
            } else {
              if (flags & DirectRead) {
                ptrs[index] = directBuff;
              } else if (flags & DirectWrite) {
                if (Send)
                  ptrs[index] = directBuff;  // send to next from my output buffer
                else
                  ptrs[index] = nullptr;
              } else {
                ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize;
              }
            }
          } else {
            ptrs[index] = connEltsFifo + (step%NCCL_STEPS)*connStepSize;
          }
        }
        subBarrier();
        if (Recv == 0 || ncclShmem.groups[group].srcs[0] == nullptr) {
          nrecv = 0;
        } else {
          nrecv = fan.nrecv();
        }

        if (Send == 0 || ncclShmem.groups[group].dsts[0] == nullptr) {
          nsend = 0;
        } else {
          nsend = fan.nsend();
        }
        fn.template operator()<SlicePerChunk, 0, Recv*MaxRecv, 0, Send*MaxSend, MultimemSrcs, MultimemDsts>
          (tid, nworkers, slice, stepSize * StepPerSlice,
            nrecv, ncclShmem.groups[group].srcs,
            nsend, ncclShmem.groups[group].dsts, ncclShmem.groups[group].dstSizes, sendDirectFlag, recvDirectFlag);
      }
      barrier();
      int32_t dstSize = 0;
      if (flags & Send*RolePostSend) {
        // Yes, for some template arguments this code will be unreachable.  That's fine.
        // coverity[dead_error_begin]
        dstSize = ncclShmem.groups[group].dstSizes[index];
        ncclShmem.groups[group].dstSizes[index] = 0;
        if (flags & ConnFifoEnabled) connFifo[step%NCCL_STEPS].size = dstSize*sizeof(T);
      }
      barrier();
      if (flags & (Recv*(RoleWaitRecv|RolePostRecv) | Send*(RoleWaitSend|RolePostSend))) {
        step += StepPerSlice;
      }
      if (flags & (Recv*RolePostRecv | Send*RolePostSend)) {
        if (Send && (!Recv || (flags & RolePostSend)) && (dstSize!=0 || (flags&ConnFifoEnabled))) {
          fence_acq_rel_sys();
        }
        st_relaxed_sys_global(connStepPtr, step);
      }
    }
  }

private:
  // Scatter/Gather generic op
  // skip: my own rank order in the buffer chunks
  // shift: peer offset to avoid all ranks sending to or receiving from same peer
  template <int DirectRecv1, int DirectSend1, int Recv, int Send>
  __device__ __forceinline__ void
  ScatterGatherOp(intptr_t inpIx, intptr_t outIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip, int shift, bool postOp) {
    constexpr int DirectRecv = 1 && Direct && DirectRecv1;
    constexpr int DirectSend = 1 && Direct && DirectSend1;
    int offset = 0; // slice offset
    int sliceSize = stepSize*StepPerSlice;
    int dataSize = max(DIVUP(peerElem, 16*SlicePerChunk)*16, sliceSize/32);  // per-peer slice size

    #pragma unroll
    for (int slice=0; slice<SlicePerChunk; ++slice) {
      ssize_t realSize = max(0, min(dataSize, peerElem-offset));
      bool fenceNeeded = false;
      if (tid < nworkers) {
        if (Send) {
          // Scatter pre-scales data of input buffer only in non-Direct case
          constexpr int PreOpSrcs = DirectSend ? 0 : 1;
          if (tid==0) ncclShmem.groups[group].srcs[0] = (T*)ncclShmem.groups[group].userInput + inpIx + offset;
          // realSize is not accurate here; but intra-node does not rely on sizes FIFO
          waitPeer<0, DirectSend, 0, 1, 1, 0>(0, inpIx, offset, realSize);
          subBarrier();
          #pragma unroll
          // Loop over peers
          for (int j=0; j<fan.nsend(); j++) {
            int i = (j+shift)%fan.nsend();
            ssize_t pOffset = i*peerOffset;
            // Skip the data I am responsible of reducing myself
            if (skip >= 0 && i >= skip) pOffset += peerOffset;
            void* src0 = (T*)ncclShmem.groups[group].srcs[0] + pOffset;
            ssize_t realPeerSize = min(realSize, totalElem-pOffset);
            if (realPeerSize > 0 && ncclShmem.groups[group].dsts[i] != nullptr) {
              reduceCopy<Unroll, RedOp, T, 0,1,1, 0,1,1, PreOpSrcs>(tid, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, false, 1, &src0, 1, ncclShmem.groups[group].dsts+i, realPeerSize);
              // Mark for threadfence at the end
              fenceNeeded |= true;
            }
          }
        } else if (Recv) {
          if (tid==0) ncclShmem.groups[group].dsts[0] = (T*)ncclShmem.groups[group].userOutput + outIx + offset;
          ssize_t pOffset = index*peerOffset;
          if (skip >= 0 && index >= skip) pOffset += peerOffset;
          // Adjust remote index with peer offset in case we are directly pulling from peer's output buffer
          waitPeer<DirectRecv, 0, 1, 0, 0, 1>(outIx+pOffset, outIx+pOffset, offset, realSize);
          subBarrier();
          #pragma unroll
          for (int j=0; j<fan.nrecv(); j++) {
            int i = (j+shift)%fan.nrecv();
            pOffset = i*peerOffset;
            if (skip >= 0 && i >= skip) pOffset += peerOffset;
            void* dst0 = (T*)ncclShmem.groups[group].dsts[0] + pOffset;
            ssize_t realPeerSize = min(realSize, totalElem-pOffset);
            if (DirectRecv && ncclShmem.groups[group].srcs[i] == dst0) realPeerSize = 0;
            if (realPeerSize > 0) reduceCopy<Unroll, RedOp, T, 0,1,1, 0,1,1, /*PreOpSrcs=*/0>(tid, nworkers, ncclShmem.redOpArgs[0], ncclShmem.redOpArgs, postOp, 1, ncclShmem.groups[group].srcs+i, 1, &dst0, realPeerSize);
          }
        }
      }
      fenceNeeded = barrierAny(fenceNeeded);
      postPeer<Recv, Send>(fenceNeeded);
      offset += realSize;
    }
  }

  __device__ __forceinline__ void loadRecvConn(ncclDevChannelPeer *peer, int connIndex, uint32_t direct, int ipcRegFlag, int netRegFlag) {
    conn = &peer->recv[connIndex];
    if (conn->netDeviceHandle.netDeviceType == NCCL_NET_DEVICE_UNPACK) {
      // handle must be a device ptr
      netDeviceHandle = conn->netDeviceHandle.handle;
      // Cache the handle
      ncclNetDeviceUnpackSetup(netDeviceHandle, group, index);
      flags |= NetDeviceUnpack;
    }
    step = conn->step;
    step = roundUp(step, SlicePerChunk*StepPerSlice);
    if (flags & RolePostRecv) {
      connStepPtr = conn->head;
      *connStepPtr = step; // Return credits in case we rounded up.
    }
    if (flags & RoleWaitRecv) {
      if ((flags & PatMode) == 0) ncclShmem.groups[group].recvConns[index] = conn; // WaitRecv role saves since that's who needs it in setDataPtrs()
      flags |= (conn->flags & NCCL_NVLS_MIN_POLL) ? NvlsMinPolling : 0;
      connStepPtr = conn->tail;
      connStepCache = loadStepValue(connStepPtr);
      connStepSize = conn->stepSize/sizeof(T);
      connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
      if (conn->connFifo != nullptr) {
        flags |= ConnFifoEnabled;
        connFifo = conn->connFifo;
      }
      if (Direct) {
        if (ipcRegFlag) {
          // User buffers have been registered
          if (conn->flags & (NCCL_P2P_READ | NCCL_P2P_WRITE)) {
            if (P2p) {
              flags |= conn->flags & NCCL_P2P_WRITE ? DirectWrite : DirectRead;
            } else if (connIndex == 1 && direct) {
              flags |= DirectRead;
            } else {
              flags |= direct & NCCL_P2P_READ ? DirectRead : DirectWrite;
            }
          } else if ((conn->flags & NCCL_NVLS_MIN_POLL)) {
            /* NVLS direct */
            flags |= DirectRead;
          }
        }
        if (netRegFlag) {
          if (conn->flags & NCCL_DIRECT_NIC) {
            flags |= NetRegMode;
            connFifo[step % NCCL_STEPS].size = 0;
          }
        }
      }
    }
  }

  __device__ __forceinline__ void loadSendConn(ncclDevChannelPeer *peer, int connIndex, uint32_t direct, int ipcRegFlag, int netRegFlag) {
    conn = &peer->send[connIndex];
    step = conn->step;
    step = roundUp(step, SlicePerChunk*StepPerSlice);

    connFifo = conn->connFifo;
    if (connFifo != nullptr) flags |= ConnFifoEnabled;

    if (flags & RolePostSend) {
      connStepPtr = conn->tail;
      connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
    }
    if (flags & RoleWaitSend) {
      if ((flags & PatMode) == 0) ncclShmem.groups[group].sendConns[index] = conn; // WaitSend role saves since that's who needs it in setDataPtrs()
      flags |= (conn->flags & NCCL_NVLS_MIN_POLL) ? NvlsMinPolling : 0;
      connStepPtr = conn->head;
      connStepCache = loadStepValue(connStepPtr);
      connStepSize = conn->stepSize/sizeof(T);
      connEltsFifo = (T*)conn->buffs[NCCL_PROTO_SIMPLE];
      if (Direct) {
        if (ipcRegFlag) {
          // User buffers have been registered
          if (conn->flags & (NCCL_P2P_WRITE | NCCL_P2P_READ)) {
            if (P2p) {
              flags |= conn->flags & NCCL_P2P_WRITE ? DirectWrite : DirectRead;
            } else if (connIndex == 1 && direct) {
              flags |= DirectRead;  // scatter-reduce use direct pull
            } else {
              flags |= direct & NCCL_P2P_READ ? DirectRead : DirectWrite;
            }
          } else if ((conn->flags & NCCL_NVLS_MIN_POLL)) {
            /* NVLS direct */
            flags |= DirectWrite;
          }
        }
        if (netRegFlag) {
          if (conn->flags & NCCL_DIRECT_NIC) {
            flags |= NetRegMode;
          }
        }
      }
    }
  }

 public:
  __device__ Primitives(
      int tid, int nthreads, int const *recvPeers, int const *sendPeers,
      void const *inputBuf, void *outputBuf, uint64_t redOpArg, uint8_t group=0,
      uint8_t connIndexRecv = 0, uint8_t connIndexSend = 0, struct ncclDevWorkColl* collWork = nullptr,
      struct ncclDevWorkP2p* p2pWork = nullptr, int stepSize_ = 0, int mode = primsModeDefault
    ):
    tid(tid), nthreads(nthreads), tidInBlock(threadIdx.x), group(group),
    stepSize(stepSize_ == 0 ? ncclShmem.comm.buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS/sizeof(T) : stepSize_) {

    int peer = -1;
    flags = 0;
    index = -1;
    isSparse = collWork ? collWork->isSparse : 0;
    ccdFormatMask = collWork ? collWork->ccdFormatMask : CcdMaskAll;
    ccdDenseThreshold = collWork ? collWork->ccdDenseThreshold : 0.3f;
    ccdTrackedDensity = -1.0f;
    ccdBaseSparsity = -1.0f;
    if (mode == primsModeDefault) { // Connect to ranks in sendPeers/recvPeers
      // For send operations, we need an extra warp to overlap the threadfence and the copy
      this->nworkers = nthreads - (MaxSend > 0 && nthreads >= NCCL_SIMPLE_EXTRA_GROUP_IF_NTHREADS_GE ? WARP_SIZE : 0);

      int nrecv=0, nsend=0;
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_line]
      while (nrecv < MaxRecv && recvPeers[nrecv] != -1) nrecv++;
      // coverity[dead_error_line]
      while (nsend < MaxSend && sendPeers[nsend] != -1) nsend++;
      this->fan = Fan(nrecv, nsend);

      constexpr int ThreadPerSync =
        MaxSend >= 16 || MaxRecv >= 16 ? 32 : // NVLS may have an arity > 8. In that case increase the size of the groups
        MaxSend >= 8 || MaxRecv >= 8 ? 16 :
        8; // Allows for all roles (WaitRecv/WaitSend/PostRecv/PostSend) within a single warp
      static_assert(MaxSend <= ThreadPerSync && MaxRecv <= ThreadPerSync, "Not enough threads to cover all peers");

      assert(2*(nrecv+nsend) <= nthreads); // Ensure no thread is assigned more than one role.
      // Coverity assumes that index will equal tid based on the line below, but it doesn't consider the setting
      // of flags.  This results in multiple false positive overruns being reported here and in all_reduce.h.
      // Unfortunately, we've been unsuccessful in trying to silence them with a single directive here so
      // instead it's being done at the callers.
      // coverity[assignment:FALSE]
      if      (tid < nrecv)                 { flags |= RoleWaitRecv; index = tid; }
      // Yes, for some template arguments this code will be unreachable.  That's fine.
      // coverity[dead_error_begin]
      else if (tid < nrecv+nsend)           { flags |= RoleWaitSend; index = tid-nrecv; }
      else if (nthreads-nsend <= tid)       { flags |= RolePostSend; index = tid-(nthreads-nsend); }
      else if (nthreads-nrecv-nsend <= tid) { flags |= RolePostRecv; index = tid-(nthreads-nrecv-nsend); }

      if (flags & (RoleWaitRecv|RolePostRecv)) peer = recvPeers[index];
      if (flags & (RoleWaitSend|RolePostSend)) peer = sendPeers[index];

      // Coverity thinks that index could be -1 here but that's not actually the case.
      // coverity[negative_returns:FALSE]
      int sendIpcReg;
      int recvIpcReg;
      int sendNetReg;
      int recvNetReg;
      if (P2p) {
        sendIpcReg = p2pWork ? p2pWork->sendIpcReg : 0;
        recvIpcReg = p2pWork ? p2pWork->recvIpcReg : 0;
        sendNetReg = p2pWork ? p2pWork->sendNetReg : 0;
        recvNetReg = p2pWork ? p2pWork->recvNetReg : 0;
      } else {
        recvIpcReg = sendIpcReg = collWork ? collWork->regUsed : 0;
        recvNetReg = sendNetReg = collWork ? collWork->netRegUsed : 0;
      }

      // coverity[overrun-call] => Coverity think prims.index can be greater than 1
      if (flags & (RoleWaitRecv|RolePostRecv)) loadRecvConn(ncclShmem.channel.peers[peer], connIndexRecv, collWork ? collWork->direct : 0, recvIpcReg, recvNetReg);
      // coverity[overrun-call] => Coverity think prims.index can be greater than 1
      if (flags & (RoleWaitSend|RolePostSend)) loadSendConn(ncclShmem.channel.peers[peer], connIndexSend, collWork ? collWork->direct : 0, sendIpcReg, sendNetReg);

      if (barrierAny(flags & NetDeviceUnpack)) {
        flags |= AnyNetDeviceUnpack;
        // RoleWaitRecv starts at tid=0, so this creates the bitmask of which recv peers
        // have NetDeviceUnpack.
        uint32_t mask = __ballot_sync(~0u, ((flags & RoleWaitRecv) && (flags & NetDeviceUnpack)) ? 1 : 0);
        if (tid == 0) {
          ncclShmem.groups[this->group].devicePlugin.unpack.unpackNetDeviceIndexMask = mask;
        }
      }

      // coverity[negative_returns:FALSE] => coverity thinks that index could be -1 but that's not actually the case
      // coverity[var_deref_model] => coverity thinks work can dereferenced if NULL but this is not the case
      setDataPtrs(inputBuf, outputBuf, redOpArg, (struct ncclDevWorkCollReg*)collWork, sendIpcReg || recvIpcReg, peer);
      // coverity[uninit_member] => coverity thinks fan.n is not initialized
    } else if (mode == primsModePatRs || mode == primsModePatAg) { // Connect to all ranks +/- 2^n
      flags |= PatMode;
      const int roles[5] = { RoleWaitRecv, RolePostRecv, RoleWaitSend, RolePostSend, RoleInput | RoleOutput };
      if (tid < 5) flags |= roles[tid];

      int nranks = ncclShmem.comm.nRanks;
      if (tid < 32 && ((1UL<<tid) < nranks)) {
        int rank = ncclShmem.comm.rank;
        uint32_t delta = 1 << tid;
        // Load recv peer
        int recvPeer = mode == primsModePatRs ? (rank - delta + nranks) % nranks : (rank + delta) % nranks;
        struct ncclPatPeer* peer = ((struct ncclPatPeer*)recvPeers)+tid;
        struct ncclConnInfo* conn = peer->conn = ncclShmem.channel.peers[recvPeer]->recv+connIndexRecv;
        peer->step = conn->step;
        peer->buff = conn->buffs[NCCL_PROTO_SIMPLE];
        peer->stepCache = loadStepValue(peer->tailPtr = conn->tail);
        peer->headPtr = conn->head;
        peer->accSize = 0;
        peer->connStepSize = conn->stepSize/sizeof(T);
        // Load send peer
        int sendPeer = mode == primsModePatAg ? (rank - delta + nranks) % nranks : (rank + delta) % nranks;
        peer = ((struct ncclPatPeer*)sendPeers)+tid;
        conn = peer->conn = ncclShmem.channel.peers[sendPeer]->send+connIndexSend;
        peer->step = conn->step;
        peer->connFifo = conn->connFifo;
        peer->buff = conn->buffs[NCCL_PROTO_SIMPLE];
        peer->stepCache = loadStepValue(peer->headPtr = conn->head);
        peer->tailPtr = conn->tail;
        peer->accSize = 0;
        peer->connStepSize = conn->stepSize/sizeof(T);
      }
      if (tid==0) {
        ncclShmem.groups[group].userInput = (void*)inputBuf;
        ncclShmem.groups[group].userOutput = (void*)outputBuf;
        ncclShmem.redOpArgs[0] = redOpArg;  // scaler for local input
      }
      patBarrier();
    }
  }

  __device__ ~Primitives() {
    if (flags&PatMode) return;
    // Save steps for the next operation
    if (flags & (RolePostSend|RolePostRecv)) conn->step = step;
    if ((flags & NetRegMode) && (flags & RoleWaitSend)) {
      // Make sure we wait until the proxy has sent data before we return.
      // We don't want the next CUDA kernel to overwrite the send buffer which
      // was accessed directly.
      uint64_t prevStep = step - StepPerSlice;
      volatile ssize_t* ptr = &(connFifo[prevStep%NCCL_STEPS].size);
      int spins = 0;
      while (*ptr != -1) if (checkAbort(flags, Aborted, spins)) break;
    }

    if (flags & NetDeviceUnpack) {
      ncclNetDeviceSaveHead(netDeviceHandle, group, index);
    }

    // Make sure all threads are done writing back conn->step and done using
    // ncclShmem.groups[group]
    barrier();

    if ((flags & DirectRead) && (flags & RoleWaitSend) && P2p) {
      // For sendrecv DirectRead, sender needs to wait for receiver reading data from src.
      // This has to be done after barrier() since post thread might have contention with
      // this check.
      int spins = 0;
      volatile uint64_t* tail = conn->tail;
      volatile uint64_t* head = conn->head;
      while (*tail > *head) if (checkAbort(flags, Aborted, spins)) break;
    }
  }

  __device__ void setDataPtrs(void const *inputBuf, void *outputBuf, uint64_t redOpArg, struct ncclDevWorkCollReg* work, uint8_t ipcReg, int peer) {
    if (tid==0) {
      ncclShmem.groups[group].userInput = (void*)inputBuf;
      ncclShmem.groups[group].userOutput = (void*)outputBuf;
      ncclShmem.redOpArgs[0] = redOpArg;  // scaler for local input
    }

    if (Direct && ipcReg) {
      bool recvProvider = (flags & RoleWaitRecv) && (flags & DirectWrite);
      bool sendAcceptor = (flags & RoleWaitSend) && (flags & DirectWrite);
      bool sendProvider = (flags & RoleWaitSend) && (flags & DirectRead); // sender provides direct buffer (to be fetched)
      bool recvAcceptor = (flags & RoleWaitRecv) && (flags & DirectRead); // receiver accepts direct buffer
      if (recvProvider) {
        int spins = 0;
        void* volatile* slot = ncclShmem.groups[group].recvConns[index]->ptrExchange;
        // Wait for consumer to consume previous value before trampling it.
        if (slot) {
          T* exchgPtr;
          directBuff = (T*)outputBuf;
          while (*slot != nullptr && !checkAbort(flags, Aborted, spins));
          if (P2p) {
            exchgPtr = (T*)outputBuf;
          } else {
            int localPeer = ncclShmem.comm.rankToLocalRank[peer];
            // coverity[deref_parm:FALSE] => work cannot be NULL if ipcReg != NULL
            exchgPtr = (T*)(work->coll.recvbuffOffset + work->coll.recvbuffRmtAddrs[localPeer]);
          }
          *slot = reinterpret_cast<void*>(exchgPtr);
        }
      }
      if (sendAcceptor) {
        int spins = 0;
        void* volatile* slot = ncclShmem.groups[group].sendConns[index]->ptrExchange;
        void* ptr;
        while (slot) {
          ptr = *slot;
          if (ptr != nullptr || checkAbort(flags, Aborted, spins)) break;
        }

        if (slot) {
          directBuff = reinterpret_cast<T*>(ptr);
          *slot = nullptr;
        } else {
          // coverity[var_deref_op]
          directBuff = (T*)work->dnOutputs[index];
        }
      }
      if (sendProvider) {
        int spins = 0;
        void* volatile* slot = ncclShmem.groups[group].sendConns[index]->ptrExchange;
        volatile uint64_t* argSlot0 = ncclShmem.groups[group].sendConns[index]->redOpArgExchange;
        volatile uint64_t* argSlot1 = ncclShmem.groups[group].sendConns[index]->redOpArgExchange + 1;
        // Wait for consumer to consume previous value before trampling it.
        if (slot && argSlot0 && argSlot1) {
          T* exchgPtr;
          while ((*slot != nullptr || *argSlot0 != 0 || *argSlot1 != 0) && !checkAbort(flags, Aborted, spins));
          // If there is no recv, then we are directly pulling from input buffer (e.g. directScatter)
          // Otherwise, we are pulling from output buffer (e.g. recvCopyDirectSend)
          directBuff = MaxRecv == 0 ? (T*)inputBuf : (T*)outputBuf;
          if (P2p) {
            exchgPtr = MaxRecv == 0 ? (T*)inputBuf : (T*)outputBuf;
          } else {
            int localPeer = ncclShmem.comm.rankToLocalRank[peer];
            if (MaxRecv == 0)
              // coverity[var_deref_op]
              exchgPtr = (T*)(work->coll.sendbuffOffset + work->coll.sendbuffRmtAddrs[localPeer]);
            else
              // coverity[var_deref_op]
              exchgPtr = (T*)(work->coll.recvbuffOffset + work->coll.recvbuffRmtAddrs[localPeer]);
          }

          // Exchange pre-scalers for use in direct pull
          *argSlot0 = (uint64_t(1) << 32) | (uint32_t)redOpArg;
          *argSlot1 = (uint64_t(1) << 32) | (uint32_t)(redOpArg >> 32);
          *slot = reinterpret_cast<T*>(exchgPtr);
        }
      }
      if (recvAcceptor) {
        int spins = 0;
        void* volatile* slot = ncclShmem.groups[group].recvConns[index]->ptrExchange;
        volatile uint64_t* argSlot0 = ncclShmem.groups[group].recvConns[index]->redOpArgExchange;
        volatile uint64_t* argSlot1 = ncclShmem.groups[group].recvConns[index]->redOpArgExchange + 1;
        void* ptr;
        while (slot) {
          ptr = *slot;
          if (ptr != nullptr || checkAbort(flags, Aborted, spins)) break;
        }

        if (slot && argSlot0 && argSlot1) {
          directBuff = reinterpret_cast<T*>(ptr);
          if (MaxSend != 0) { // reduce group rather than gather group
            // Store scalers for remote inputs
            uint64_t arg0, arg1;
            while (true) {
              arg0 = *argSlot0;
              arg1 = *argSlot1;
              if ((arg0 != 0 && arg1 != 0) || checkAbort(flags, Aborted, spins)) break;
            }
            ncclShmem.redOpArgs[1 + index] = ((arg1 & 0xffffffff) << 32) | (arg0 & 0xffffffff);
          }
          *argSlot0 = 0; *argSlot1 = 0;
          *slot = nullptr;
        } else {
          // Coverity complains about work being possibly NULL below.  However, slot
          // being NULL means that the NVLS buffer is registered (regUsed == 1)
          // so work can't be NULL in this code path.
          // coverity[var_deref_op]
          directBuff = (T*)work->dnInputs[index];
        }
      }
    }
  }

  __device__ void moveDataPtrs(intptr_t delta) {
    if (tid==0) {
      ncclShmem.groups[group].userInput = (T*)ncclShmem.groups[group].userInput + delta;
      ncclShmem.groups[group].userOutput = (T*)ncclShmem.groups[group].userOutput + delta;
    }
  }

  __device__ __forceinline__ void send(intptr_t inpIx, int eltN) {
    genericOp<0, 0, 0, 1, Input, -1>(inpIx, -1, eltN, false);
  }
  __device__ __forceinline__ void sendFromOutput(intptr_t outIx, int eltN) {
    genericOp<0, 0, 0, 1, Output, -1>(outIx, -1, eltN, false);
  }
  __device__ __forceinline__ void directSend(intptr_t inpIx, intptr_t outIx, int eltN) {
    genericOp<0, 1, 0, 1, Input, -1>(inpIx, outIx, eltN, false);
  }
  __device__ __forceinline__ void directSendFromOutput(intptr_t outIx, int eltN) {
    genericOp<0, 1, 0, 1, Output, -1>(outIx, outIx, eltN, false);
  }

  __device__ __forceinline__ void recv(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 0, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecv(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 0, -1, Output>(outIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvCopy(intptr_t inpIx, intptr_t outIx, int eltN) {
    genericOp<1, 0, 1, 0, -1, Output>(inpIx, outIx, eltN, /*postOp=*/false);
  }

  __device__ __forceinline__ void copySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 0, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvSend(int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, -1, -1>(-1, -1, eltN, postOp);
  }
  __device__ __forceinline__ void recvCopySend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, -1, Output>(-1, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 1, 1, 1, -1, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 1, 1, 1, -1, -1>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void recvDirectSend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 1, 1, -1, -1>(-1, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvSend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 1, -1, -1>(outIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void recvCopyDirectSend(intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 1, 1, -1, Output>(-1, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceCopy(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 0, Input, Output>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceSend(intptr_t inpIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, -1>(inpIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceSend(intptr_t inpIx, int eltN, bool postOp=false) {
    genericOp<1, 0, 1, 1, Input, -1>(inpIx, -1, eltN, postOp);
  }
  __device__ __forceinline__ void recvReduceDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 1, 1, 1, Input, -1>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN, bool postOp=false) {
    genericOp<1, 1, 1, 1, Input, -1>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void recvReduceCopySend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    genericOp<0, 0, 1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void recvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, int eltN, bool postOp=false) {
    // Direct is only for the send part
    genericOp<0, 1, 1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }
  __device__ __forceinline__ void directRecvReduceCopyDirectSend(intptr_t inpIx, intptr_t outIx, ssize_t eltN, bool postOp=false) {
    genericOp<1, 1, 1, 1, Input, Output>(inpIx, outIx, eltN, postOp);
  }

  __device__ __forceinline__ void
  scatter(intptr_t inpIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip, int shift) {
    ScatterGatherOp<0, 0, 0, 1>(inpIx, -1, totalElem, peerElem, peerOffset, skip, shift, /*postOp=*/false);
  }
  __device__ __forceinline__ void
  directScatter(intptr_t inpIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip, int shift) {
    ScatterGatherOp<0, 1, 0, 1>(inpIx, -1, totalElem, peerElem, peerOffset, skip, shift, /*postOp=*/false);
  }

  __device__ __forceinline__ void
  gather(intptr_t outIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip, int shift, bool postOp=false) {
    ScatterGatherOp<0, 0, 1, 0>(-1, outIx, totalElem, peerElem, peerOffset, skip, shift, postOp);
  }
  __device__ __forceinline__ void
  directGather(intptr_t outIx, ssize_t totalElem, int peerElem, ssize_t peerOffset, int skip, int shift) {
    ScatterGatherOp<1, 0, 1, 0>(-1, outIx, totalElem, peerElem, peerOffset, skip, shift, /*postOp=*/false);
  }

  __device__ __forceinline__ void patReduce(struct ncclPatStep* ps, struct ncclPatShmem* shmem) {
    if (ps->flags & PatSkipped) { patBarrier(); patBarrier(); return; } // Skipped
    int nelem = ps->nelem < 0 ? 0 : ps->nelem;
    T* userInput = (T*)ncclShmem.groups[group].userInput;
    T* userOutput = (T*)ncclShmem.groups[group].userOutput;

    bool recv = ps->recvDim >= 0 && (flags & (RolePostRecv|RoleWaitRecv));
    bool send = ps->sendDim >= 0 && (flags & (RolePostSend|RoleWaitSend));
    bool postRecv = ps->postRecv && recv;
    bool postSend = ps->postSend && send;
    struct ncclPatPeer* peer = NULL;
    if (recv) {
      peer = shmem->recvDims+ps->recvDim;
      step = peer->step;
    }
    if (send) {
      peer = shmem->sendDims+ps->sendDim;
      step = peer->step;
    }

    if (recv && (flags & RoleWaitRecv)) {
      ncclShmem.groups[group].srcs[0] = ((T*)peer->buff) + (step%NCCL_STEPS)*peer->connStepSize + ps->recvOffset;
      int spins = 0;
      while (peer->stepCache < step + StepPerSlice) {
        peer->stepCache = loadStepValue(peer->tailPtr);
        if (checkAbort(flags, Aborted, spins)) break;
      }
    }
    if (send && (flags & RoleWaitSend)) {
      int spins = 0;
      while (peer->stepCache + NCCL_STEPS < step + ps->stepOffset + StepPerSlice) {
        peer->stepCache = loadStepValue(peer->headPtr);
        if (checkAbort(flags, Aborted, spins)) break;
      }
      ncclShmem.groups[group].dsts[0] = ((T*)peer->buff) + ((step+ps->stepOffset)%NCCL_STEPS)*peer->connStepSize + ps->sendOffset;
      if (peer->accSize < ps->sendOffset + nelem + (step+ps->stepOffset)*peer->connStepSize) {
        // New data, add our own data to it.
        ncclShmem.groups[group].srcs[1] = userInput + ps->inpIx;
      } else {
        // There is already data in there, accumulate instead of writing to it.
        ncclShmem.groups[group].srcs[1] = ncclShmem.groups[group].dsts[0];
      }
    }
    long long int localAccSize = shmem->localAccSize;
    if (ps->sendDim < 0 && (flags & RoleOutput)) { // Destination is our own local buffer
      ncclShmem.groups[group].dsts[0] = userOutput + ps->outIx;
      if (localAccSize < ps->outIx + nelem) {
        // New data, add our own data to it.
        ncclShmem.groups[group].srcs[1] = userInput + ps->inpIx;
        localAccSize = ps->outIx + nelem;
      } else {
        // There is already data in there, accumulate instead of writing to it.
        ncclShmem.groups[group].srcs[1] = ncclShmem.groups[group].dsts[0];
      }
    }
    patBarrier();
    int nSrcs = 2;
    void** srcs = ncclShmem.groups[group].srcs;
    if (ps->recvDim < 0) { srcs++; nSrcs--; } // No peer to receive from, remove one source

    int workSize = ncclShmem.aborted ? 0 : nelem;

    // [META:PAT_AVG] Apply postOp (division for AVG) on final write to local output
    // isFinalWrite is set only in Phase 4, which is the final write for each chunk
    // For non-PatSumPostDiv ops, Apply_PostOp is identity so this is safe
    const int applyPostOp = ps->isFinalWrite ? 1 : 0;

    reduceCopy<Unroll, RedOp, T, 0, 1, 2, 0, 1, 1, /*PreOpSrcs*/0>
      (tid, nthreads, ncclShmem.redOpArgs[0],  nullptr, /*postOp=*/applyPostOp,
       nSrcs, srcs, 1, ncclShmem.groups[group].dsts, workSize);

    // Store conn step here inside the two barriers to make sure next reload will see the update.
    if (postSend && (flags & RolePostSend)) {
      if (peer->connFifo) {
        peer->connFifo[step%NCCL_STEPS].size = (ps->sendOffset + nelem)*sizeof(T);
      }
      peer->step = step += StepPerSlice;
      st_relaxed_sys_global(&peer->conn->step, step);
    }
    if (postRecv && (flags & RolePostRecv)) {
      peer->step = step += StepPerSlice;
      st_relaxed_sys_global(&peer->conn->step, step); // Also save in global mem for next op
    }

    // Update accSize
    if (ps->sendDim < 0 && (flags & RoleOutput)) atomicMax(&shmem->localAccSize, localAccSize);
    if (ps->sendDim >= 0 && (flags & RoleWaitSend)) atomicMax(&peer->accSize, ps->sendOffset + nelem + (step+ps->stepOffset)*peer->connStepSize);

    patBarrier();

    if (postSend && (flags & RolePostSend)) {
      if (nelem > 0 || peer->connFifo) fence_acq_rel_sys();
      st_relaxed_sys_global(peer->tailPtr, step);
    }
    if (postRecv && (flags & RolePostRecv)) {
      st_relaxed_sys_global(peer->headPtr, step);
    }
  }

  __device__ __forceinline__ void patCopy(struct ncclPatStep* ps, struct ncclPatShmem* shmem) {
    if (ps->flags & PatSkipped) { patBarrier(); patBarrier(); return; } // Skipped
    int nelem = ps->nelem < 0 ? 0 : ps->nelem;
    T* userInput = (T*)ncclShmem.groups[group].userInput;
    T* userOutput = (T*)ncclShmem.groups[group].userOutput;

    bool recv = ps->recvDim >= 0 && (flags & (RolePostRecv|RoleWaitRecv));
    bool send = ps->sendDim >= 0 && (flags & (RolePostSend|RoleWaitSend));
    bool postRecv = ps->postRecv && recv;
    bool postSend = ps->postSend && send;
    struct ncclPatPeer* peer = NULL;
    if (recv) {
      peer = shmem->recvDims+ps->recvDim;
      step = peer->step;
    }
    if (send) {
      peer = shmem->sendDims+ps->sendDim;
      step = peer->step;
    }

    if (recv && (flags & RoleWaitRecv)) {
      ncclShmem.groups[group].srcs[0] = ((T*)peer->buff) + ((step+ps->stepOffset)%NCCL_STEPS)*peer->connStepSize + ps->recvOffset;
      int spins = 0;
      while (peer->stepCache < step + ps->stepOffset + StepPerSlice) {
        peer->stepCache = loadStepValue(peer->tailPtr);
        if (checkAbort(flags, Aborted, spins)) break;
      }
      if (peer->accSize < ps->recvOffset + nelem + (step+ps->stepOffset)*peer->connStepSize) {
        // New data, copy to our output buffer.
        ncclShmem.groups[group].dsts[1] = userOutput + ps->outIx;
      } else {
        ncclShmem.groups[group].dsts[1] = ncclShmem.groups[group].srcs[0]; // Already done
      }
    }
    if (send && (flags & RoleWaitSend)) {
      int spins = 0;
      while (peer->stepCache + NCCL_STEPS < step + StepPerSlice) {
        peer->stepCache = loadStepValue(peer->headPtr);
        if (checkAbort(flags, Aborted, spins)) break;
      }
      ncclShmem.groups[group].dsts[0] = ((T*)peer->buff) + (step%NCCL_STEPS)*peer->connStepSize + ps->sendOffset;
    }
    long long int localAccSize = shmem->localAccSize;
    if (ps->recvDim < 0 && (flags & RoleInput)) { // Source is our own local buffer
      ncclShmem.groups[group].srcs[0] = userInput + ps->inpIx;
      if (localAccSize < ps->inpIx + nelem) {
        // New data, copy to our output buffer.
        ncclShmem.groups[group].dsts[1] = userOutput + ps->outIx;
        localAccSize = ps->inpIx + nelem;
      } else {
        // Already done
        ncclShmem.groups[group].dsts[1] = ncclShmem.groups[group].srcs[0];
      }
    }
    patBarrier();
    int nDsts = 2;
    void** dsts = ncclShmem.groups[group].dsts;
    if (ps->sendDim < 0) { dsts++; nDsts--; } // No peer to send to, remove one dest
    if (ncclShmem.groups[group].srcs[0] == ncclShmem.groups[group].dsts[1]) nDsts--; // In-place or already done.

    int workSize = ncclShmem.aborted ? 0 : nelem;

    reduceCopy<Unroll, RedOp, T, 0, 1, 1, 0, 1, 2, /*PreOpSrcs*/0>
      (tid, nthreads, ncclShmem.redOpArgs[0],  nullptr, /*postOp=*/false,
       1, ncclShmem.groups[group].srcs, nDsts, dsts, workSize);

    // Store conn step here inside the two barriers to make sure next reload will see the update.
    if (postSend && (flags & RolePostSend)) {
      if (peer->connFifo) {
        peer->connFifo[step%NCCL_STEPS].size = (ps->sendOffset + nelem)*sizeof(T);
      }
      peer->step = step += StepPerSlice;
      st_relaxed_sys_global(&peer->conn->step, step);
    }
    if (postRecv && (flags & RolePostRecv)) {
      peer->step = step += StepPerSlice;
      st_relaxed_sys_global(&peer->conn->step, step); // Also save in global mem for next op
    }

    // Update accSize
    if (ps->recvDim < 0 && (flags & RoleInput)) atomicMax(&shmem->localAccSize, localAccSize);
    if (ps->recvDim >= 0 && (flags & RoleWaitRecv)) atomicMax(&peer->accSize, ps->recvOffset + nelem + (step+ps->stepOffset)*peer->connStepSize);

    patBarrier();

    if (postSend && (flags & RolePostSend)) {
      if (nelem > 0 || peer->connFifo) fence_acq_rel_sys();
      st_relaxed_sys_global(peer->tailPtr, step);
    }
    if (postRecv && (flags & RolePostRecv)) {
      st_relaxed_sys_global(peer->headPtr, step);
    }
  }

};
