/*
Dirac4.cl — OpenCL kernels for the 4-component Dirac-Kekule solver.

Solves:  i dPsi/dt = [ -i vF (alpha_x d_x + alpha_y d_y)
                         + Delta_R M1 + Delta_I M2 + V*I4 ] Psi

Basis: (psi_AK, psi_BK, psi_AK', psi_BK')

Matrices (verified correct per spec):
  alpha_x = tau_z x sigma_x  (real)
  alpha_y = tau_0 x sigma_y  (pure imaginary)
  M1      = tau_x x sigma_x  (real)
  M2      = tau_y x sigma_x  (pure imaginary)

Layout: psi[((iy*Nx + ix)*4 + comp)*2 + ri]
  comp = 0..3, ri = 0 (real) or 1 (imag)
  Total size = Nx * Ny * 8 floats.

Three kernels:
  dirac_rhs       — compute dPsi/dt from a given Psi buffer
  dirac_axpy      — dst = src + alpha * k  (elementwise, for RK4 intermediates)
  dirac_rk4_final — psi += dt*(k1 + 2*k2 + 2*k3 + k4)/6
*/

#ifndef D4_NX
#define D4_NX 128
#endif

#ifndef D4_NY
#define D4_NY 128
#endif

// ---- complex helpers (float2 = (real, imag)) ----

float2 i_mul(float2 a) {       // i * a
    return (float2)(-a.y, a.x);
}

float2 neg_i_mul(float2 a) {   // -i * a
    return (float2)(a.y, -a.x);
}

float2 c_mul(float2 a, float2 b) {
    return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// ---- index helpers ----

int idx(int ix, int iy) {
    // periodic wrapping
    ix = (ix + D4_NX) % D4_NX;
    iy = (iy + D4_NY) % D4_NY;
    return (iy * D4_NX + ix) * 8;
}

// Read component comp at (ix, iy) as float2
float2 read_psi(__global const float *psi, int ix, int iy, int comp) {
    int base = idx(ix, iy);
    return (float2)(psi[base + comp * 2], psi[base + comp * 2 + 1]);
}

void write_psi(__global float *psi, int ix, int iy, int comp, float2 val) {
    int base = idx(ix, iy);
    psi[base + comp * 2]     = val.x;
    psi[base + comp * 2 + 1] = val.y;
}

// ---- Dirac RHS kernel ----
// Each work item computes one grid point's 4-component RHS.
// Grid: global_size = (D4_NX, D4_NY)

__kernel void dirac_rhs(
    __global const float *psi,     // input spinor buffer
    __global float       *dpsi,    // output RHS buffer
    __global const float *delta,   // complex Kekule mass: 2 floats per grid point (re, im)
    __global const float *pot,     // scalar potential V per grid point
    __global const float *gamma,   // absorbing damping per grid point (optional, 0 = off)
    const float vF,
    const float dx,
    const float dy
) {
    int ix = get_global_id(0);
    int iy = get_global_id(1);
    if (ix >= D4_NX || iy >= D4_NY) return;

    // Central differences (periodic)
    float2 dx_p0 = (read_psi(psi, ix+1, iy, 0) - read_psi(psi, ix-1, iy, 0)) / (2.0f * dx);
    float2 dx_p1 = (read_psi(psi, ix+1, iy, 1) - read_psi(psi, ix-1, iy, 1)) / (2.0f * dx);
    float2 dx_p2 = (read_psi(psi, ix+1, iy, 2) - read_psi(psi, ix-1, iy, 2)) / (2.0f * dx);
    float2 dx_p3 = (read_psi(psi, ix+1, iy, 3) - read_psi(psi, ix-1, iy, 3)) / (2.0f * dx);

    float2 dy_p0 = (read_psi(psi, ix, iy+1, 0) - read_psi(psi, ix, iy-1, 0)) / (2.0f * dy);
    float2 dy_p1 = (read_psi(psi, ix, iy+1, 1) - read_psi(psi, ix, iy-1, 1)) / (2.0f * dy);
    float2 dy_p2 = (read_psi(psi, ix, iy+1, 2) - read_psi(psi, ix, iy-1, 2)) / (2.0f * dy);
    float2 dy_p3 = (read_psi(psi, ix, iy+1, 3) - read_psi(psi, ix, iy-1, 3)) / (2.0f * dy);

    // Read local spinor
    float2 p0 = read_psi(psi, ix, iy, 0);
    float2 p1 = read_psi(psi, ix, iy, 1);
    float2 p2 = read_psi(psi, ix, iy, 2);
    float2 p3 = read_psi(psi, ix, iy, 3);

    // Read mass and potential
    int gbase = (iy * D4_NX + ix);
    float2 D = (float2)(delta[gbase * 2], delta[gbase * 2 + 1]);  // Delta = Delta_R + i*Delta_I
    float VV = pot[gbase];
    float GG = gamma[gbase];

    // Kinetic term: -vF * (alpha_x * d_x + alpha_y * d_y) Psi
    // Derived component-wise:
    //   rhs_0_kin = -vF*dx_p1 + i*vF*dy_p1
    //   rhs_1_kin = -vF*dx_p0 - i*vF*dy_p0
    //   rhs_2_kin =  vF*dx_p3 + i*vF*dy_p3
    //   rhs_3_kin =  vF*dx_p2 - i*vF*dy_p2
    float2 kin0 = -vF * dx_p1 + i_mul(vF * dy_p1);
    float2 kin1 = -vF * dx_p0 - i_mul(vF * dy_p0);
    float2 kin2 =  vF * dx_p3 + i_mul(vF * dy_p3);
    float2 kin3 =  vF * dx_p2 - i_mul(vF * dy_p2);

    // Mass + potential term: -i * (Delta_R*M1 + Delta_I*M2 + V*I4) Psi
    // Simplified (see derivation):
    //   rhs_0_mass = -i*Delta*p3 - i*V*p0
    //   rhs_1_mass = -i*Delta*p2 - i*V*p1
    //   rhs_2_mass = -i*Delta*p1 - i*V*p2
    //   rhs_3_mass = -i*Delta*p0 - i*V*p3
    float2 iD = i_mul(D);  // i*Delta
    float2 mass0 = -c_mul(iD, p3) - i_mul(VV * p0);
    float2 mass1 = -c_mul(iD, p2) - i_mul(VV * p1);
    float2 mass2 = -c_mul(iD, p1) - i_mul(VV * p2);
    float2 mass3 = -c_mul(iD, p0) - i_mul(VV * p3);

    // Damping (absorbing boundary)
    float2 damp0 = -GG * p0;
    float2 damp1 = -GG * p1;
    float2 damp2 = -GG * p2;
    float2 damp3 = -GG * p3;

    // Total RHS
    write_psi(dpsi, ix, iy, 0, kin0 + mass0 + damp0);
    write_psi(dpsi, ix, iy, 1, kin1 + mass1 + damp1);
    write_psi(dpsi, ix, iy, 2, kin2 + mass2 + damp2);
    write_psi(dpsi, ix, iy, 3, kin3 + mass3 + damp3);
}

// ---- AXPY: dst = src + alpha * k (elementwise) ----
__kernel void dirac_axpy(
    __global float       *dst,
    __global const float *src,
    __global const float *k,
    const float alpha
) {
    int i = get_global_id(0);
    int n = D4_NX * D4_NY * 8;
    if (i >= n) return;
    dst[i] = src[i] + alpha * k[i];
}

// ---- RK4 final combine: psi += dt*(k1 + 2*k2 + 2*k3 + k4)/6 ----
__kernel void dirac_rk4_final(
    __global float       *psi,
    __global const float *k1,
    __global const float *k2,
    __global const float *k3,
    __global const float *k4,
    const float dt
) {
    int i = get_global_id(0);
    int n = D4_NX * D4_NY * 8;
    if (i >= n) return;
    float c = dt / 6.0f;
    psi[i] += c * (k1[i] + 2.0f * k2[i] + 2.0f * k3[i] + k4[i]);
}

// ---- Normalize: divide by sqrt(sum |psi|^2 * dx*dy) ----
__kernel void dirac_norm_sq(
    __global const float *psi,
    __global float       *partial,   // work-group partial sums
    const int n_per_group            // elements per work group
) {
    int i = get_global_id(0);
    int n = D4_NX * D4_NY * 8;
    if (i >= n) return;
    // We need sum of |psi_comp|^2 = sum of (re^2 + im^2) over all components and grid points
    // Each float element is either a real or imag part
    // We need to pair them: element 2k and 2k+1 form a complex number
    // |psi|^2 = sum (psi[2k]^2 + psi[2k+1]^2)
    // Since we're iterating over all elements, we can just sum psi[i]^2
    // But we need to be careful: the sum of squares of all elements = sum of |comp|^2
    // because each complex |c|^2 = re^2 + im^2 and we iterate over both re and im.
    float v = psi[i];
    // Use work-group reduction
    __local float sdata[256];
    int lid = get_local_id(0);
    sdata[lid] = v * v;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (lid < s) {
            sdata[lid] += sdata[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        partial[get_group_id(0)] = sdata[0];
    }
}
