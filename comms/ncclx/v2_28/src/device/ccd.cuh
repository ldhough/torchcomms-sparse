/*************************************************************************
 * CCD Sparse Communication Support
 * Header-based protocol for variable-size sparse transfers in ring
 * collectives. Used by genericOp in prims_simple.h.
 ************************************************************************/

#ifndef NCCL_DEVICE_CCD_CUH
#define NCCL_DEVICE_CCD_CUH

// Header prepended to each sparse chunk in a FIFO slot.
// Layout per slot: [CcdSparseChunkHeader][bv][inds][values]
// Each section padded to 16 bytes for int4 alignment.
// Byte offsets stored in header so receiver can locate each section.

__device__ __forceinline__
size_t ccd_pad16(size_t x) { return (x + 15) & ~(size_t)15; }

struct __align__(16) CcdSparseChunkHeader {
  size_t payload_bytes;  // total bytes after header (bv + padding + inds + padding + values)
  size_t nnz;            // number of nonzero values
  size_t bv_offset;      // byte offset from after header to bitvector (always 0)
  size_t inds_offset;    // byte offset from after header to tile prefix sums
  size_t vals_offset;    // byte offset from after header to packed values
};

enum CcdCompressionProtocol {
    DENSE,
    COO1D,
    SPOP,
    ADAPTIVE
};

__device__ __forceinline__
size_t ceil_div(size_t a, size_t b) {
    return (a + b - 1) / b;
}

// intended to be used within a single block
template<typename ValType = float, typename IndType = unsigned>
__device__
__forceinline__
void ccd_fused_single_block_spop_compress(
    const ValType * __restrict__ dense,
    ValType * __restrict__ compressed,
    const size_t N, // rows
    const size_t M, // cols
    uint64_t * __restrict__ bv,
    IndType * __restrict__ inds,
    const unsigned warp_start_idx,
    const unsigned warp_end_idx,
    const int bar,
    const int nworkers_count
) {
    // thread/warp indexing
    // const unsigned index      = blockIdx.x * blockDim.x + threadIdx.x;
    // const unsigned stride     = blockDim.x * gridDim.x;
    const unsigned num_warps  = warp_end_idx + 1 - warp_start_idx;          // stride / warpSize;
    const unsigned warp_idx   = warp_start_idx + (threadIdx.x / warpSize);  // index / warpSize;
    const unsigned lane       = threadIdx.x % warpSize;                     // index % warpSize;
    // these include incomplete tiles
    const size_t tiles_in_N   = ceil_div(N, 64);
    const size_t tiles_in_M   = ceil_div(M, 64);
    const size_t total_tiles  = tiles_in_M * tiles_in_N;
    constexpr ValType comp_zero(0);

    if (warp_idx == warp_start_idx && lane == 0) {
        inds[total_tiles] = 0;
    }

    for (
        size_t tile_index = warp_idx;
        tile_index < total_tiles;
        tile_index += num_warps
    ) {
        const size_t idx_N             = tile_index / tiles_in_M;
        const size_t idx_M             = tile_index % tiles_in_M;
        // number of prior rows * elements in row
        const size_t prior_rowc        = 64 * idx_N ;
        // prior rows * elements in row
        const size_t prior_row_elec    = prior_rowc * M;
        // number of elements in this row before tile start
        const size_t prior_colc        = 64 * idx_M;
        size_t base_idx                = prior_row_elec + prior_colc + lane;
        // determine how many rows and cols to process (normally 64, 64)
        const unsigned cols_after      = M - prior_colc;
        const unsigned cols_to_process = (cols_after < 64) ? cols_after : 64;
        const unsigned rows_after      = N - prior_rowc;
        const unsigned rows_to_process = (rows_after < 64) ? rows_after : 64;
        uint64_t col_0_bv   = 0;
        unsigned nz_count_0 = 0;
        // process first col in tile for this thread
        if (lane < cols_to_process) { // if there are < 32 cols, mask some threads
            for (size_t j = 0; j < rows_to_process; ++j) {
                size_t full_idx = base_idx + (j * (size_t) M);
                ValType val = dense[full_idx];
                uint64_t is_nz = val != comp_zero;
                col_0_bv |= (is_nz << j);
                nz_count_0 += (unsigned) is_nz;
            }
        }
        // advance to next col handled by this thread
        base_idx += warpSize;
        uint64_t col_1_bv   = 0;
        unsigned nz_count_1 = 0;
        // process second col in tile for this thread
        if (lane + warpSize < cols_to_process) { // mask if beyond what needs to be processed
            for (size_t j = 0; j < rows_to_process; ++j) {
                size_t full_idx = base_idx + (j * (size_t) M);
                ValType val = dense[full_idx];
                uint64_t is_nz = val != comp_zero;
                col_1_bv |= (is_nz << j);
                nz_count_1 += (unsigned) is_nz;
            }
        }
        size_t bv_base_idx = tile_index * 64 + lane;
        bv[bv_base_idx] = col_0_bv;
        bv_base_idx += warpSize;
        bv[bv_base_idx] = col_1_bv;
        // find popc of whole tile
        unsigned total_popc = nz_count_0 + nz_count_1;
        for (unsigned off = 16; off > 0; off >>= 1) {
            total_popc += __shfl_down_sync(
                0xFFFFFFFF,
                total_popc,
                off
            );
        }
        // write tile popc to indices
        if (lane == 0) {
            inds[tile_index] = total_popc;
        }
    }
    barrier_sync(bar, nworkers_count);
    if (threadIdx.x == 0) {
        IndType running = 0;
        for (unsigned tile_idx = 0; tile_idx < total_tiles; ++tile_idx) {
            IndType count = inds[tile_idx];
            inds[tile_idx] = running;
            running += count;
        }
    }
    barrier_sync(bar, nworkers_count);
    const unsigned dlane = lane * 2;
    for (
        size_t tile_index = warp_idx;
        tile_index < total_tiles;
        tile_index += num_warps
    ) {
        const size_t idx_N             = tile_index / tiles_in_M;
        const size_t idx_M             = tile_index % tiles_in_M;
        // number of prior rows * elements in row
        const size_t prior_rowc        = 64 * idx_N;
        // prior rows * elements in row
        const size_t prior_row_elec    = prior_rowc * M;
        // number of elements in this row before tile start
        const size_t prior_colc        = 64 * idx_M;
        const size_t base_idx          = prior_row_elec + prior_colc + (size_t) dlane;
        // determine how many rows and cols to process (normally 64, 64)
        const unsigned cols_after      = M - prior_colc;
        const unsigned cols_to_process = (cols_after < 64) ? cols_after : 64;
        const IndType nz_before_tile   = inds[tile_index];
        // popc two cols, adjacent
        const size_t bv_word_idx = tile_index * 64 + dlane;
        uint64_t col_0_bv = bv[bv_word_idx];
        uint64_t col_1_bv = bv[bv_word_idx + 1];
        // bv includes some ghost elements which are still zeroed
        const unsigned col_0_popc = __popcll(col_0_bv);
        const unsigned col_1_popc = __popcll(col_1_bv);
        const unsigned dcol_popc  = col_0_popc + col_1_popc;
        unsigned sum = dcol_popc;
        // prefix sum the cols
        for (unsigned offset = 1; offset < 32; offset <<= 1) {
            const unsigned rcv = __shfl_up_sync(
                0xFFFFFFFF, sum, offset
            );
            if (lane >= offset) {
                sum += rcv;
            }
        }
        sum -= dcol_popc;
        size_t nz_count_0 = sum;
        size_t nz_count_1 = sum + col_0_popc;
        nz_count_0 += nz_before_tile;
        nz_count_1 += nz_before_tile;
        // scan the appropriate words/cols in bv
        if (dlane < cols_to_process) {
            for (size_t idx = 0; idx < col_0_popc; ++idx) {
                const unsigned set_lsb_idx = __ffsll(col_0_bv) - 1;
                col_0_bv &= (col_0_bv - 1);
                const size_t full_dense_index = base_idx + (set_lsb_idx * M);
                const size_t full_compressed_index = nz_count_0 + idx;
                compressed[full_compressed_index] = dense[full_dense_index];
            }
        }
        if (dlane + 1 < cols_to_process) {
            for (size_t idx = 0; idx < col_1_popc; ++idx) {
                const unsigned set_lsb_idx = __ffsll(col_1_bv) - 1;
                col_1_bv &= (col_1_bv - 1);
                const size_t full_dense_index = base_idx + 1 + (set_lsb_idx * M);
                const size_t full_compressed_index = nz_count_1 + idx;
                compressed[full_compressed_index] = dense[full_dense_index];
            }
        }
    }
}



#endif /* NCCL_DEVICE_CCD_CUH */
