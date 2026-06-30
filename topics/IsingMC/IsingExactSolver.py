# run as module:  python -u -m IsingExactSolver

import os
import numpy as np
import pyopencl as cl
from OpenCLBase import OpenCLBase

MAX_SITES = 16
MAX_NEIGH = 8
WG_SIZE   = 16   # one thread per site

# ---- all 16 possible binary logic functions of 2 inputs ----
# Bit encoding: code = out(A=0,B=0) | out(A=0,B=1)<<1 | out(A=1,B=0)<<2 | out(A=1,B=1)<<3
# combos in order: (0,0),(0,1),(1,0),(1,1)
# Truth tables:
#   FALSE  = [0,0,0,0] = 0
#   AND    = [0,0,0,1] = 8
#   A_GT_B = [0,0,1,0] = 4     A and not B
#   A      = [0,0,1,1] = 12
#   B_GT_A = [0,1,0,0] = 2     B and not A
#   B      = [0,1,0,1] = 10
#   XOR    = [0,1,1,0] = 6
#   OR     = [0,1,1,1] = 14
#   NOR    = [1,0,0,0] = 1
#   XNOR   = [1,0,0,1] = 9
#   NOT_B  = [1,0,1,0] = 5
#   A_GE_B = [1,0,1,1] = 13    A or not B
#   NOT_A  = [1,1,0,0] = 3
#   B_GE_A = [1,1,0,1] = 11    B or not A
#   NAND   = [1,1,1,0] = 7
#   TRUE   = [1,1,1,1] = 15
LOGIC_NAMES = {
     0: 'FALSE',
     8: 'AND',
     4: 'A_GT_B',
    12: 'A',
     2: 'B_GT_A',
    10: 'B',
     6: 'XOR',
    14: 'OR',
     1: 'NOR',
     9: 'XNOR',
     5: 'NOT_B',
    13: 'A_GE_B',
     3: 'NOT_A',
    11: 'B_GE_A',
     7: 'NAND',
    15: 'TRUE',
}

USEFUL_LOGIC = {'AND', 'OR', 'XOR', 'NAND', 'NOR', 'XNOR'}

# =====================================================================
#  Square-lattice helper:  build sparse W matrix from (W1, W2)
# =====================================================================

def sq_lattice_sparse(positions, W1, W2, nSite=None):
    """
    Build sparse coupling arrays for a square-lattice cluster.

    positions : (nSite, 2) integer grid coordinates
    W1 : coupling along Cartesian directions (distance=1)
    W2 : coupling along diagonal directions  (distance=sqrt(2))

    Returns (W_val, W_idx, nNeigh) all shape [nSite, MAX_NEIGH]
    """
    if nSite is None: nSite = len(positions)
    pos = np.array(positions, dtype=int)
    W_val  = np.zeros((nSite, MAX_NEIGH), dtype=np.float32)
    W_idx  = np.zeros((nSite, MAX_NEIGH), dtype=np.int32)
    nNeigh = np.zeros(nSite,              dtype=np.int32)
    for i in range(nSite):
        nn = 0
        for j in range(nSite):
            if i == j: continue
            dx = abs(pos[i,0] - pos[j,0])
            dy = abs(pos[i,1] - pos[j,1])
            if dx <= 1 and dy <= 1:
                if dx + dy == 1:
                    w = W1
                else:                  # dx==dy==1  diagonal
                    w = W2
                if nn < MAX_NEIGH:
                    W_val[i, nn] = w
                    W_idx[i, nn] = j
                    nn += 1
        nNeigh[i] = nn
    return W_val, W_idx, nNeigh


def compute_input_bias(positions, input_positions, input_neighbors, input_vals, W1, W2, shift=0.0):
    """
    Compute bias on active sites from fixed input pads, using proper W1/W2 coupling.

    positions       : (nSite, 2) active site grid coordinates
    input_positions : (2,) list of (x,y) coordinates for input pads A and B
    input_neighbors : list of active-site indices adjacent to each input pad
                      e.g. [[0], [2]] means input A neighbors site 0, input B neighbors site 2
    input_vals      : (2,) int 0 or 1 for each input
    W1, W2          : Cartesian and diagonal coupling strengths
    shift           : float, shift applied to input values (default 0.0)
                      shift=0.0: 0→0, 1→1 (no shift)
                      shift=-0.5: 0→-0.5, 1→+0.5 (centered)
                      shift=-1.0: 0→-1, 1→0 (spin mapping)

    Returns bias array (nSite,) to add to Esite_base.
    """
    bias = np.zeros(len(positions), dtype=np.float32)
    positions = np.array(positions, dtype=int)
    input_positions = np.array(input_positions, dtype=int)

    for inp_idx, sites in enumerate(input_neighbors):
        input_val = input_vals[inp_idx]
        # Apply shift: input_val + shift
        spin_val = float(input_val) + shift
        inp_pos = input_positions[inp_idx]
        for s in sites:
            site_pos = positions[s]
            dx = abs(inp_pos[0] - site_pos[0])
            dy = abs(inp_pos[1] - site_pos[1])
            # Determine coupling based on geometry
            if dx + dy == 1:
                w = W1  # Cartesian neighbor
            elif dx == 1 and dy == 1:
                w = W2  # Diagonal neighbor
            else:
                w = 0.0  # Should not happen for properly placed inputs
            bias[s] += w * spin_val
    return bias


def apply_input_bias(Esite_base, positions, input_positions, input_neighbors, input_vals, W1, W2, shift=0.0):
    """
    Return a copy of Esite_base with input bias applied using proper W coupling.
    """
    Esite = Esite_base.copy()
    bias = compute_input_bias(positions, input_positions, input_neighbors, input_vals, W1, W2, shift)
    Esite += bias
    return Esite


def identify_logic(outputs_4):
    """
    outputs_4 : sequence of 4 ints [n(0,0), n(0,1), n(1,0), n(1,1)]
    Returns (code, name) where code is 0-15.
    """
    code = (int(outputs_4[0]) & 1)       \
         | ((int(outputs_4[1]) & 1) << 1) \
         | ((int(outputs_4[2]) & 1) << 2) \
         | ((int(outputs_4[3]) & 1) << 3)
    return code, LOGIC_NAMES.get(code, '?')


def occ_mask_to_array(occ_mask, nSite):
    """Convert integer bitmask to (nSite,) int array."""
    return np.array([(occ_mask >> i) & 1 for i in range(nSite)], dtype=np.int32)


# =====================================================================
#  IsingExactSolver  –  thin OpenCL wrapper
# =====================================================================

class IsingExactSolver(OpenCLBase):
    """
    Brute-force Gray-code Ising ground-state solver (exact, <=16 sites).

    Usage
    -----
    solver = IsingExactSolver()
    E_min, occ_min = solver.solve(Esite_batch, W_val, W_idx, nNeigh, nSite)
    """

    def __init__(self, device_index=0, preferred_vendor='nvidia', bPrint=True):
        super().__init__(nloc=WG_SIZE, device_index=device_index,
                         preferred_vendor=preferred_vendor, bPrint=bPrint)
        base_path = os.path.dirname(os.path.abspath(__file__))
        if not self.load_program(rel_path='cl/ising_exact.cl', base_path=base_path):
            raise RuntimeError('[IsingExactSolver] Failed to compile cl/ising_exact.cl')
        self._nInst_alloc = 0
        self._nSite_alloc = 0
        self._k_gs   = cl.Kernel(self.prg, 'ising_groundstate')
        self._k_gs_W = cl.Kernel(self.prg, 'ising_groundstate_batch_W')
        self._k_gs_top8 = None  # Loaded on demand

    # ------------------------------------------------------------------
    def _realloc(self, nInst, nSite):
        if nInst == self._nInst_alloc and nSite == self._nSite_alloc:
            return
        sz_f = np.dtype(np.float32).itemsize
        sz_i = np.dtype(np.int32  ).itemsize
        buffs = {
            'Esite'  : sz_f * nInst * nSite,
            'W_val'  : sz_f * nSite * MAX_NEIGH,
            'W_idx'  : sz_i * nSite * MAX_NEIGH,
            'nNeigh' : sz_i * nSite,
            'E_out'  : sz_f * nInst,
            'occ_out': sz_i * nInst,
        }
        self.try_make_buffers(buffs)
        self._nInst_alloc = nInst
        self._nSite_alloc = nSite

    def _realloc_batch_W(self, nInst, nSite):
        sz_f = np.dtype(np.float32).itemsize
        sz_i = np.dtype(np.int32  ).itemsize
        buffs = {
            'bw_Esite'  : sz_f * nInst * nSite,
            'bw_W_val'  : sz_f * nInst * nSite * MAX_NEIGH,
            'bw_W_idx'  : sz_i * nInst * nSite * MAX_NEIGH,
            'bw_nNeigh' : sz_i * nInst * nSite,
            'bw_E_out'  : sz_f * nInst,
            'bw_occ_out': sz_i * nInst,
        }
        self.try_make_buffers(buffs)

    # ------------------------------------------------------------------
    def solve(self, Esite_batch, W_val, W_idx, nNeigh, nSite):
        """
        Solve ground state for nInstances independent systems
        sharing the same W geometry.

        Esite_batch : (nInst, nSite) float32 – on-site energies per instance
        W_val       : (nSite, MAX_NEIGH) float32
        W_idx       : (nSite, MAX_NEIGH) int32
        nNeigh      : (nSite,) int32
        nSite       : int

        Returns
        -------
        E_min  : (nInst,) float32
        occ_min: (nInst,) int32  – 16-bit bitmask of ground-state occupancy
        """
        Esite_batch = np.ascontiguousarray(Esite_batch, dtype=np.float32)
        W_val  = np.ascontiguousarray(W_val,  dtype=np.float32)
        W_idx  = np.ascontiguousarray(W_idx,  dtype=np.int32)
        nNeigh = np.ascontiguousarray(nNeigh, dtype=np.int32)
        if Esite_batch.ndim == 1:
            Esite_batch = Esite_batch[np.newaxis, :]
        nInst = Esite_batch.shape[0]

        self._realloc(nInst, nSite)
        self.toGPU_( self.Esite_buff,   Esite_batch.ravel())
        self.toGPU_( self.W_val_buff,   W_val.ravel())
        self.toGPU_( self.W_idx_buff,   W_idx.ravel())
        self.toGPU_( self.nNeigh_buff,  nNeigh.ravel())

        kernel = self._k_gs
        kernel.set_args(
            np.int32(nSite),
            np.int32(nInst),
            self.Esite_buff,
            self.W_val_buff,
            self.W_idx_buff,
            self.nNeigh_buff,
            self.E_out_buff,
            self.occ_out_buff,
        )
        global_size = (nInst * WG_SIZE,)
        local_size  = (WG_SIZE,)
        cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size)
        self.queue.finish()

        E_min   = np.empty(nInst, dtype=np.float32)
        occ_min = np.empty(nInst, dtype=np.int32)
        self.fromGPU_(self.E_out_buff,   E_min)
        self.fromGPU_(self.occ_out_buff, occ_min)
        return E_min, occ_min

    # ------------------------------------------------------------------
    def solve_batch_W(self, Esite_batch, W_val_batch, W_idx_batch, nNeigh_batch, nSite):
        """
        Variant where each instance has its own W matrix.

        Esite_batch  : (nInst, nSite)
        W_val_batch  : (nInst, nSite, MAX_NEIGH)
        W_idx_batch  : (nInst, nSite, MAX_NEIGH)
        nNeigh_batch : (nInst, nSite)
        """
        Esite_batch  = np.ascontiguousarray(Esite_batch,  dtype=np.float32)
        W_val_batch  = np.ascontiguousarray(W_val_batch,  dtype=np.float32)
        W_idx_batch  = np.ascontiguousarray(W_idx_batch,  dtype=np.int32)
        nNeigh_batch = np.ascontiguousarray(nNeigh_batch, dtype=np.int32)
        if Esite_batch.ndim == 1:
            Esite_batch = Esite_batch[np.newaxis, :]
        nInst = Esite_batch.shape[0]

        self._realloc_batch_W(nInst, nSite)
        self.toGPU_(self.bw_Esite_buff,   Esite_batch.ravel())
        self.toGPU_(self.bw_W_val_buff,   W_val_batch.ravel())
        self.toGPU_(self.bw_W_idx_buff,   W_idx_batch.ravel())
        self.toGPU_(self.bw_nNeigh_buff,  nNeigh_batch.ravel())

        kernel = self._k_gs_W
        kernel.set_args(
            np.int32(nSite),
            np.int32(nInst),
            self.bw_Esite_buff,
            self.bw_W_val_buff,
            self.bw_W_idx_buff,
            self.bw_nNeigh_buff,
            self.bw_E_out_buff,
            self.bw_occ_out_buff,
        )
        global_size = (nInst * WG_SIZE,)
        local_size  = (WG_SIZE,)
        cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size)
        self.queue.finish()

        E_min   = np.empty(nInst, dtype=np.float32)
        occ_min = np.empty(nInst, dtype=np.int32)
        self.fromGPU_(self.bw_E_out_buff,   E_min)
        self.fromGPU_(self.bw_occ_out_buff, occ_min)
        return E_min, occ_min

    # ------------------------------------------------------------------
    def _ensure_top8_kernel(self):
        """Load top8 kernel on demand (separate program to avoid conflicts)."""
        if self._k_gs_top8 is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            kernel_src = open(os.path.join(base_path, 'cl', 'ising_exact_top8.cl')).read()
            self._prg_top8 = cl.Program(self.ctx, kernel_src).build()
            self._k_gs_top8 = cl.Kernel(self._prg_top8, 'ising_groundstate_top8')

    def solve_batch_W_top8(self, Esite_batch, W_val_batch, W_idx_batch, nNeigh_batch, nSite):
        """
        Variant tracking top-8 lowest energy states (for degeneracy analysis).

        Returns
        -------
        E_top8   : (nInst, 8) float32 - energies of 8 lowest states (sorted)
        occ_top8 : (nInst, 8) int32   - occupancy masks of 8 lowest states
        """
        self._ensure_top8_kernel()

        Esite_batch  = np.ascontiguousarray(Esite_batch,  dtype=np.float32)
        W_val_batch  = np.ascontiguousarray(W_val_batch,  dtype=np.float32)
        W_idx_batch  = np.ascontiguousarray(W_idx_batch,  dtype=np.int32)
        nNeigh_batch = np.ascontiguousarray(nNeigh_batch, dtype=np.int32)
        if Esite_batch.ndim == 1:
            Esite_batch = Esite_batch[np.newaxis, :]
        nInst = Esite_batch.shape[0]

        self._realloc_batch_W(nInst, nSite)
        # Allocate top8 output buffers
        sz_f = np.dtype(np.float32).itemsize
        sz_i = np.dtype(np.int32).itemsize
        if not hasattr(self, '_top8_buff_alloc') or self._top8_buff_alloc < nInst:
            self.bw_E_top8_buff   = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, sz_f * nInst * 8)
            self.bw_occ_top8_buff = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, sz_i * nInst * 8)
            self._top8_buff_alloc = nInst

        self.toGPU_(self.bw_Esite_buff,   Esite_batch.ravel())
        self.toGPU_(self.bw_W_val_buff,   W_val_batch.ravel())
        self.toGPU_(self.bw_W_idx_buff,   W_idx_batch.ravel())
        self.toGPU_(self.bw_nNeigh_buff,  nNeigh_batch.ravel())

        kernel = self._k_gs_top8
        kernel.set_args(
            np.int32(nSite),
            np.int32(nInst),
            self.bw_Esite_buff,
            self.bw_W_val_buff,
            self.bw_W_idx_buff,
            self.bw_nNeigh_buff,
            self.bw_E_top8_buff,
            self.bw_occ_top8_buff,
        )
        global_size = (nInst * WG_SIZE,)
        local_size  = (WG_SIZE,)
        cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, local_size)
        self.queue.finish()

        E_top8   = np.empty((nInst, 8), dtype=np.float32)
        occ_top8 = np.empty((nInst, 8), dtype=np.int32)
        self.fromGPU_(self.bw_E_top8_buff,   E_top8.ravel())
        self.fromGPU_(self.bw_occ_top8_buff, occ_top8.ravel())
        return E_top8, occ_top8


# =====================================================================
#  High-level: logic-gate evaluation helpers
# =====================================================================

INPUT_COMBOS = [(0,0), (0,1), (1,0), (1,1)]

def eval_logic_table(solver, Esite_base, W_val, W_idx, nNeigh, nSite,
                     positions, input_positions, input_neighbors, output_site, W1, W2, shift=0.0):
    """
    Evaluate logic truth-table for one cluster at one (W1,W2) point.

    positions       : (nSite, 2) for computing input coupling geometry
    input_positions : (2,) list of (x,y) for input pads A and B
    shift           : float, shift applied to input values (default 0.0)

    Returns
    -------
    outputs_4 : (4,) int   output bit for each input combination
    occ_4     : (4, nSite) occupancy arrays for each input combination
    E_4       : (4,) float ground-state energies
    Esite_4   : (4, nSite) on-site energies (for debugging)
    logic_code, logic_name
    """
    Esite_batch = np.zeros((4, nSite), dtype=np.float32)
    for k, (A, B) in enumerate(INPUT_COMBOS):
        Esite_batch[k] = apply_input_bias(Esite_base, positions, input_positions, input_neighbors, [A, B], W1, W2, shift)

    E_4, occ_raw = solver.solve(Esite_batch, W_val, W_idx, nNeigh, nSite)
    outputs_4 = np.array([(occ_raw[k] >> output_site) & 1 for k in range(4)], dtype=np.int32)
    occ_4     = np.array([occ_mask_to_array(occ_raw[k], nSite) for k in range(4)])
    code, name = identify_logic(outputs_4)
    return outputs_4, occ_4, E_4, Esite_batch, code, name


def scan_W1_W2(solver, positions, input_positions, Esite_base, input_neighbors, output_site,
               W1_vals, W2_vals, nSite=None, shift=0.0):
    """
    Scan (W1, W2) parameter space for a fixed cluster geometry.

    shift : float, shift applied to input values (default 0.0)

    Returns
    -------
    logic_map : (nW2, nW1) int   0-15 logic code at each (W1,W2)
    occ_map   : (nW2, nW1, 4, nSite) occupancy for each combo
    """
    if nSite is None: nSite = len(positions)
    nW1, nW2 = len(W1_vals), len(W2_vals)
    nInst = nW1 * nW2 * 4

    # Build Esite batch: shape (nW2, nW1, 4, nSite) → flat (nInst, nSite)
    Esite_batch = np.zeros((nW2, nW1, 4, nSite), dtype=np.float32)
    for iy, W2v in enumerate(W2_vals):
        for ix, W1v in enumerate(W1_vals):
            for k, (A, B) in enumerate(INPUT_COMBOS):
                Esite_batch[iy, ix, k] = apply_input_bias(Esite_base, positions, input_positions, input_neighbors, [A, B], W1v, W2v, shift)

    # W coupling is same per (W1,W2) → replicate into full flat batch
    # Flatten Esite: (nW2*nW1*4, nSite)
    Esite_flat = Esite_batch.reshape(nInst, nSite)

    # Build W batches: (nInst, nSite, MAX_NEIGH)
    W_val_batch  = np.zeros((nInst, nSite, MAX_NEIGH), dtype=np.float32)
    W_idx_batch  = np.zeros((nInst, nSite, MAX_NEIGH), dtype=np.int32)
    nNeigh_batch = np.zeros((nInst, nSite),             dtype=np.int32)
    inst = 0
    for iy, W2 in enumerate(W2_vals):
        for ix, W1 in enumerate(W1_vals):
            Wv, Wi, Wn = sq_lattice_sparse(positions, W1, W2, nSite)
            for k in range(4):
                W_val_batch [inst] = Wv
                W_idx_batch [inst] = Wi
                nNeigh_batch[inst] = Wn
                inst += 1

    E_flat, occ_flat = solver.solve_batch_W(Esite_flat, W_val_batch, W_idx_batch, nNeigh_batch, nSite)

    # Reshape outputs
    E_4d   = E_flat  .reshape(nW2, nW1, 4)
    occ_4d = occ_flat.reshape(nW2, nW1, 4)

    logic_map = np.zeros((nW2, nW1), dtype=np.int32)
    occ_map   = np.zeros((nW2, nW1, 4, nSite), dtype=np.int32)
    for iy in range(nW2):
        for ix in range(nW1):
            outs  = [(occ_4d[iy,ix,k] >> output_site) & 1 for k in range(4)]
            code, _ = identify_logic(outs)
            logic_map[iy, ix] = code
            for k in range(4):
                occ_map[iy, ix, k] = occ_mask_to_array(occ_4d[iy,ix,k], nSite)

    return logic_map, occ_map, E_4d


def scan_W1_W2_top8(solver, positions, input_positions, Esite_base, input_neighbors, output_site,
                   W1_vals, W2_vals, nSite=None, degeneracy_threshold=0.01, shift=0.0):
    """
    Scan (W1, W2) parameter space and check ground state uniqueness using top8 kernel.

    shift : float, shift applied to input values (default 0.0)

    Returns
    -------
    logic_map : (nW2, nW1) int   0-15 logic code at each (W1,W2)
    degenerate_mask : (nW2, nW1) bool  True if ground state is degenerate
    """
    if nSite is None: nSite = len(positions)
    nW1, nW2 = len(W1_vals), len(W2_vals)
    nInst = nW1 * nW2 * 4

    # Build Esite batch: shape (nW2, nW1, 4, nSite) → flat (nInst, nSite)
    Esite_batch = np.zeros((nW2, nW1, 4, nSite), dtype=np.float32)
    for iy, W2v in enumerate(W2_vals):
        for ix, W1v in enumerate(W1_vals):
            for k, (A, B) in enumerate(INPUT_COMBOS):
                Esite_batch[iy, ix, k] = apply_input_bias(Esite_base, positions, input_positions, input_neighbors, [A, B], W1v, W2v, shift)

    # Flatten Esite: (nW2*nW1*4, nSite)
    Esite_flat = Esite_batch.reshape(nInst, nSite)

    # Build W batches: (nInst, nSite, MAX_NEIGH)
    W_val_batch  = np.zeros((nInst, nSite, MAX_NEIGH), dtype=np.float32)
    W_idx_batch  = np.zeros((nInst, nSite, MAX_NEIGH), dtype=np.int32)
    nNeigh_batch = np.zeros((nInst, nSite),             dtype=np.int32)
    inst = 0
    for iy, W2 in enumerate(W2_vals):
        for ix, W1 in enumerate(W1_vals):
            Wv, Wi, Wn = sq_lattice_sparse(positions, W1, W2, nSite)
            for k in range(4):
                W_val_batch [inst] = Wv
                W_idx_batch [inst] = Wi
                nNeigh_batch[inst] = Wn
                inst += 1

    # Use top8 kernel to get ground state uniqueness info
    E_top8, occ_top8 = solver.solve_batch_W_top8(Esite_flat, W_val_batch, W_idx_batch, nNeigh_batch, nSite)

    # Check ground state uniqueness (inlined to avoid circular import with Ising_utils)
    ground_gaps = E_top8[:, 1] - E_top8[:, 0]
    unique = ground_gaps >= degeneracy_threshold

    # Reshape to (nW2, nW1, 4)
    unique_4d = unique.reshape(nW2, nW1, 4)
    occ_4d = occ_top8[:, 0].reshape(nW2, nW1, 4)  # Use ground state occupancy

    # Logic map (use ground state)
    logic_map = np.zeros((nW2, nW1), dtype=np.int32)
    degenerate_mask = np.zeros((nW2, nW1), dtype=bool)

    for iy in range(nW2):
        for ix in range(nW1):
            # Check if any input combo has degenerate ground state
            if not np.all(unique_4d[iy, ix]):
                degenerate_mask[iy, ix] = True
                # Still compute logic from ground state
            outs = [(occ_4d[iy, ix, k] >> output_site) & 1 for k in range(4)]
            code, _ = identify_logic(outs)
            logic_map[iy, ix] = code

    return logic_map, degenerate_mask


# NOTE: Plotting functions (plot_cluster, plot_ground_states, plot_logic_map,
# plot_logic_fraction_map, LOGIC_COLORS) have been moved to IsingPlotting.py
# check_ground_state_uniqueness has been moved to Ising_utils.py
