/* 
CheFSI.cl — OpenCL kernels for Chebyshev Filtering Subspace Iteration (CheFSI).
Implements SpMM (ELLPACK format), Gram matrix, and subspace orthogonalization
for GPU-accelerated frontier orbital computation in large sparse systems.
*/

// Macros for static sizing (can be overridden by compiler flags)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

// -------------------------------------------------------------------------
// Kernel 1: Sparse Matrix - Dense Block Multiplication (SpMM)
// Format: ELLPACK (Fixed neighbors per row, padded with -1)
// 
// This kernel applies H to a block of vectors X. 
// Can optionally perform the Chebyshev update step immediately:
// Out = 2 * ((H-c)/r) * X - Prev
// -------------------------------------------------------------------------
__kernel void spmm_ellpack_chebyshev(
    const int n_atoms,
    const int n_max_neighs,
    const int n_vecs,
    // Arrays
    __global const int* restrict indices,   // [n_atoms, n_max_neighs]
    __global const float* restrict values,  // [n_atoms, n_max_neighs]
    __global const float* restrict X,       // [n_atoms, n_vecs]
    __global const float* restrict X_prev,  // [n_atoms, n_vecs] (Optional, for Cheb)
    __global float* restrict Y,             // [n_atoms, n_vecs] Output
    // Scalar parameters for Shift/Scale
    const float shift, // center of spectrum
    const float scale, // radius of spectrum
    const int is_chebyshev_step // 0 = standard SpMM, 1 = Chebyshev Recurrence
) {
    int row = get_global_id(0);
    
    if (row >= n_atoms) return;

    // Prefetch vector row for X_prev if needed
    // (Optimization: In a real "local cluster" sort, neighbors are close, 
    //  so L1 cache hits on X will be high).

    // We process all vectors for this atom at once to reuse the matrix row
    // Local buffer for accumulators to keep in registers
    // Assumes n_vecs is small (e.g. 16 or 32). 
    // If n_vecs is large, this loop should be tiled.
    float sums[32]; 
    for(int v=0; v<n_vecs; v++) sums[v] = 0.0f;

    // Iterate over neighbors (ELLPACK row)
    for (int n = 0; n < n_max_neighs; n++) {
        int col_idx = indices[row * n_max_neighs + n];
        float val   = values[row * n_max_neighs + n];

        // ELLPACK padding check
        if (col_idx != -1) {
            // Unrolled loop over vectors
            for (int v = 0; v < n_vecs; v++) {
                sums[v] += val * X[col_idx * n_vecs + v];
            }
        }
    }

    // Write result
    float inv_scale = (scale > 1e-8f) ? (1.0f / scale) : 1.0f;

    for (int v = 0; v < n_vecs; v++) {
        if (is_chebyshev_step) {
            // Apply Chebyshev recurrence: T_n+1 = 2 * H_scaled * T_n - T_n-1
            // H_scaled = (H - shift) / scale
            float h_scaled_x = (sums[v] - shift * X[row * n_vecs + v]) * inv_scale;
            Y[row * n_vecs + v] = 2.0f * h_scaled_x - X_prev[row * n_vecs + v];
        } else {
            // Standard H * X
            Y[row * n_vecs + v] = sums[v];
        }
    }
}

// -------------------------------------------------------------------------
// Configuration Macros (Can be defined at compile time)
// -------------------------------------------------------------------------
#ifndef M
#define M 8         // Number of vectors (e.g., 8 or 16)
#endif

#ifndef K
#define K 32        // Tile size (atoms per block), e.g., 32 or 64
#endif

// Derived constants
#define WG_SIZE (M * M) // One thread per output element

// -------------------------------------------------------------------------
// Kernel 1: Tall-Skinny Matrix Multiply (Partial Reduction)
// Computes: Partial_O = A^T * B
// 
// Grid: 1 Workgroup per chunk of 'K' atoms
// Local Size: (M, M) -> one thread computes one element C[i,j]
// -------------------------------------------------------------------------
__kernel void tall_skinny_gram(
    const int n_atoms,
    __global const float* restrict A, // [n_atoms, M]
    __global const float* restrict B, // [n_atoms, M]
    __global float* restrict out_partial // [n_groups, M, M]
) {
    // 1. Thread & Group IDs
    int tx = get_local_id(0); // Row in Output (0..M-1)
    int ty = get_local_id(1); // Col in Output (0..M-1)
    int group_id = get_group_id(0);
    int flat_tid = ty * M + tx; // Flat ID 0..63 (if M=8)

    // Base atom index for this workgroup
    int base_atom = group_id * K;

    // 2. Local Memory for Tile
    // Size: 2 matrices * K rows * M cols
    __local float tile_A[K][M];
    __local float tile_B[K][M];

    // Accumulator for C[tx, ty]
    float sum = 0.0f;

    // 3. Coalesced Loading from Global to Local
    // We need to load K*M floats for A and K*M floats for B.
    // Total floats to load per matrix: K*M
    // Total threads: M*M
    // Elements per thread: (K*M) / (M*M) = K/M
    // Example: K=32, M=8 -> 32/8 = 4 floats per thread per matrix.
    
    int elements_per_thread = K / M; 

    // Loop to load data efficiently
    // We treat the K*M block as a linear array for loading purposes
    for (int i = 0; i < elements_per_thread; i++) {
        // Calculate which element (row_k, col_m) this thread loads
        int linear_idx = flat_tid + i * WG_SIZE;
        
        // Map linear_idx back to [row_k][col_m]
        int row_k = linear_idx / M;
        int col_m = linear_idx % M;

        int global_row = base_atom + row_k;

        // Boundary Check & Load A
        if (global_row < n_atoms) {
            tile_A[row_k][col_m] = A[global_row * M + col_m];
            tile_B[row_k][col_m] = B[global_row * M + col_m];
        } else {
            tile_A[row_k][col_m] = 0.0f;
            tile_B[row_k][col_m] = 0.0f;
        }
    }

    // Wait for all threads to finish loading
    barrier(CLK_LOCAL_MEM_FENCE);

    // 4. Compute Inner Product
    // Each thread computes one element of the M x M output matrix
    // Summing over the K dimension (the tile height)
    for (int k = 0; k < K; k++) {
        sum += tile_A[k][tx] * tile_B[k][ty];
    }

    // 5. Write to Partial Output
    // Output is flattened: [group_id * M*M + flat_tid]
    out_partial[group_id * WG_SIZE + flat_tid] = sum;
}

// -------------------------------------------------------------------------
// Kernel 2: Parallel Global Reduction
// 
// Strategy: 
// - Grid Size: (M * M) Workgroups.
// - Each Workgroup is responsible for summing ONE element (i,j) of the Gram matrix.
// - Inside Workgroup: Threads load chunks of partial sums, reduce in LDS.
// -------------------------------------------------------------------------

#ifndef WG_RED_SIZE
#define WG_RED_SIZE 256
#endif

__kernel void sum_partial_blocks_parallel(
    const int n_partial_blocks,                  // Number of blocks from previous step
    __global const float* restrict partial_sums, // Input: [n_partial_blocks, M, M] flattened
    __global float* restrict final_gram          // Output: [M, M]
) {
    // 1. Identify which Matrix Element (row, col) this Workgroup handles
    // There are M*M groups. Group ID maps linearly to the matrix element.
    int group_id = get_group_id(0); 
    int lid      = get_local_id(0);

    // Map Group ID -> (row, col)
    // We assume the total number of groups launched is exactly M*M
    int row = group_id / M;
    int col = group_id % M;
    
    // Calculate the offset of this element within a flattened MxM block
    // Stride between blocks is M*M
    int element_offset_in_block = row * M + col;
    int stride_block = M * M;

    // 2. Phase 1: Grid-Stride Loop (Load & Accumulate)
    // The threads in this group iterate over the 'n_partial_blocks' dimension.
    // We want to sum: partial_sums[b][row][col] for b in 0..n_partial_blocks
    
    float thread_sum = 0.0f;

    // Stride by the Workgroup Size to cover the array
    for (int b = lid; b < n_partial_blocks; b += WG_RED_SIZE) {
        // Index calculation:
        // Block 'b' starts at index: b * (M*M)
        // We want the specific element inside that block.
        int global_index = b * stride_block + element_offset_in_block;
        thread_sum += partial_sums[global_index];
    }

    // 3. Phase 2: Binary Reduction in Local Memory
    __local float lds[WG_RED_SIZE];
    
    // Load accumulator into LDS
    lds[lid] = thread_sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Standard Binary Tree Reduction
    // Unrolled for common sizes or looped
    for (int offset = WG_RED_SIZE / 2; offset > 0; offset >>= 1) {
        if (lid < offset) {
            lds[lid] += lds[lid + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // 4. Write Final Result
    // Only thread 0 of the workgroup writes the result for this matrix element
    if (lid == 0) {
        final_gram[element_offset_in_block] = lds[0];
    }
}

// -------------------------------------------------------------------------
// Kernel 3: Subspace Rotation / Linear Combination
// Z = X * Matrix (small dense matrix)
// -------------------------------------------------------------------------
__kernel void subspace_rotate(
    const int n_atoms,
    const int n_vecs,
    __global const float* restrict X,
    // NOTE: macro M is used for vector count; rename matrix arg to avoid macro collision when -DM is passed
    __global const float* restrict Mat, // [n_vecs, n_vecs] dense
    __global float* restrict Z        // Output
) {
    int gid = get_global_id(0);
    if (gid >= n_atoms) return;

    // Load Transformation Matrix M into registers or local memory
    // Since M is tiny (16x16), registers are best.
    // All threads read same M. Constant cache handles this well.
    
    float res[32]; // Output buffer
    for(int i=0; i<n_vecs; i++) res[i] = 0.0f;

    for (int i = 0; i < n_vecs; i++) {
        float x_val = X[gid * n_vecs + i];
        for (int j = 0; j < n_vecs; j++) {
            // Z[row, j] += X[row, i] * M[i, j]
            res[j] += x_val * Mat[i * n_vecs + j];
        }
    }

    for(int j=0; j<n_vecs; j++) {
        Z[gid * n_vecs + j] = res[j];
    }
}

// -------------------------------------------------------------------------
// Kernel 4: Dense Linear Combination (AXPY)
// Z = alpha * X + beta * Y
// -------------------------------------------------------------------------
__kernel void axpy_block(
    const int n_size, // total float elements
    const float alpha,
    __global const float* restrict X,
    const float beta,
    __global const float* restrict Y,
    __global float* restrict Z
) {
    int gid = get_global_id(0);
    if (gid >= n_size) return;
    Z[gid] = alpha * X[gid] + beta * Y[gid];
}