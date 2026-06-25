/*
KekuleQM.cl — OpenCL kernels for the local valence-bond / QEq / SSH π-electron model.

Physical picture
----------------
The persistent electronic degree of freedom is y[i,s,a].  For each atom i and spin s:

    a = 0      localized/non-bonding p_z electron density on atom i
    a = 1..3   half-bond electron donation from atom i to one graph neighbor

The local spin population is

    rho[i,s] = sum_a y[i,s,a]

and a single p_z orbital gives the Pauli/valence capacity 0 <= rho[i,s] <= 1.  The simplest fast mode keeps
rho[i,s] fixed by projecting the four local channels onto a small simplex.  That is the local analogue of a
classical force-field valence constraint: one atom cannot reuse the same π electron in several bonds.

A covalent π bond needs electron amplitude from both atoms.  If y_i and y_j are classical positive populations,
the tight-binding-like amplitude c_i c_j is represented by sqrt(y_i y_j), giving

    E_pi(ij,s) = -2 eps_ij(r) sqrt( y[i,s,i->j] y[j,s,j->i] )

This is why the half-bond gradient contains sqrt(y_opposite/(y_own+small)).  It pulls up the weaker side of an
asymmetric bond, while the simplex constraint makes different bonds compete for the same local electron budget.

The Peierls/SSH coupling enters through eps_ij(r).  A shorter bond has larger eps, so electrons prefer it; the same
term pulls atoms together.  The sigma skeleton resists this by K_r (r-r0)^2/2.  These two local terms are analogous
to bonding terms in a classical force field.

Electrostatics is intentionally separate.  The charge Q_i = Zpi_i - rho_i produces a nonlocal Coulomb potential
phi_i = sum_j J_ij Q_j.  Direct evaluation is O(N^2); FFT/FMM/treecode can replace only that kernel without changing
the local bonding/update kernels.

Performance notes
-----------------
The local kernels are gather-only: one work item owns one atom, reads neighbors and opposite half-bonds, and writes
only that atom's y/R/F buffers.  This avoids atomics.  Degree is fixed to <=3, so all local loops are tiny, bounded,
and branch divergence comes mostly from edge atoms with missing neighbors.  Workgroup size is not important for the
local kernels; occupancy is limited by global-memory latency and exp/sqrt throughput.

The Coulomb kernel is the bottleneck.  It tiles atom positions and charges through local memory so each global load is
reused by all work items in the group.  It has synchronization at tile boundaries.  Use local size KQ_COUL_WG.  For
large systems replace KekuleQM_gatherCoulombDirect() by screened cutoff, FFT, FMM, or treecode.
*/

#ifndef KQ_MAX_NEIGH
#define KQ_MAX_NEIGH 3
#endif

#ifndef KQ_NSPIN
#define KQ_NSPIN 2
#endif

#ifndef KQ_NCHAN
#define KQ_NCHAN 4
#endif

#ifndef KQ_COUL_WG
#define KQ_COUL_WG 128
#endif

#ifndef KQ_SMALL_Y
#define KQ_SMALL_Y 1.0e-12f
#endif

#ifndef KQ_SMALL_R2
#define KQ_SMALL_R2 1.0e-20f
#endif

#define KQ_FLAG_LOCAL_SIMPLEX        1
#define KQ_FLAG_UPDATE_R             2
#define KQ_FLAG_USE_COULOMB_ELECTRON 4
#define KQ_FLAG_USE_COULOMB_FORCE    8

#define KQ_NEIGH(i,k)      neigh[(i)*KQ_MAX_NEIGH + (k)]
#define KQ_REV(i,k)        rev[(i)*KQ_MAX_NEIGH + (k)]
#define KQ_SLOT(i,k)       ((i)*KQ_MAX_NEIGH + (k))
#define KQ_Y(i,s,a)        y[(((i)*KQ_NSPIN + (s))*KQ_NCHAN + (a))]
#define KQ_GY(i,s,a)       gyLocal[(((i)*KQ_NSPIN + (s))*KQ_NCHAN + (a))]

inline float3 kq_xyz(float4 v){ return (float3)(v.x,v.y,v.z); }

inline float kq_eps_r(const float eps0, const float beta, const float r, const float r0){
    return eps0 * exp(-beta*(r-r0));
}

inline void kq_project_simplex4(float* y0, float* y1, float* y2, float* y3, const float target, const int nchan){
    float a0 = fmax(*y0,0.0f);
    float a1 = (nchan > 1) ? fmax(*y1,0.0f) : 0.0f;
    float a2 = (nchan > 2) ? fmax(*y2,0.0f) : 0.0f;
    float a3 = (nchan > 3) ? fmax(*y3,0.0f) : 0.0f;
    float sum = a0+a1+a2+a3;
    if(sum > KQ_SMALL_Y){
        float s = target/sum;
        *y0 = a0*s; *y1 = a1*s; *y2 = a2*s; *y3 = a3*s;
    }else{
        float v = target/(float)nchan;
        *y0 = v;
        *y1 = (nchan > 1) ? v : 0.0f;
        *y2 = (nchan > 2) ? v : 0.0f;
        *y3 = (nchan > 3) ? v : 0.0f;
    }
}

__kernel void KekuleQM_gatherLocalBonding(
    const int nAtoms,
    __global const float*  restrict y,
    __global const float4* restrict R,
    __global const int*    restrict neigh,
    __global const int*    restrict rev,
    __global const float*  restrict eps0,
    __global const float*  restrict beta,
    __global const float*  restrict r0,
    __global const float*  restrict Kr,
    __global const float4* restrict atomParams,  // (chi, eta, U, Zpi)
    __global float4*       restrict rhoQ,        // (rho_up, rho_dn, Q, rho_total)
    __global float*        restrict gyLocal,     // local dE/dy excluding nonlocal -phi
    __global float4*       restrict FLocal,      // sigma + pi Peierls force on atom i
    __global float4*       restrict bondData     // per directed slot: (eps, sum_s sqrt(y_i y_j), r, 2*sum_s sqrt)
){
    int i = get_global_id(0);
    if(i >= nAtoms) return;

    float yu0 = KQ_Y(i,0,0), yu1 = KQ_Y(i,0,1), yu2 = KQ_Y(i,0,2), yu3 = KQ_Y(i,0,3);
    float yd0 = KQ_Y(i,1,0), yd1 = KQ_Y(i,1,1), yd2 = KQ_Y(i,1,2), yd3 = KQ_Y(i,1,3);
    float ru = yu0+yu1+yu2+yu3;
    float rd = yd0+yd1+yd2+yd3;
    float rt = ru+rd;

    float4 par = atomParams[i];
    float chi = par.x, eta = par.y, U = par.z, Zpi = par.w;
    float Q = Zpi - rt;
    rhoQ[i] = (float4)(ru,rd,Q,rt);

    float base = chi + eta*(rt-Zpi);
    float mu_u = base + U*rd;
    float mu_d = base + U*ru;

    KQ_GY(i,0,0) = mu_u;
    KQ_GY(i,1,0) = mu_d;

    float3 Ri = kq_xyz(R[i]);
    float3 Fi = (float3)(0.0f,0.0f,0.0f);

    for(int k=0; k<KQ_MAX_NEIGH; k++){
        int j = KQ_NEIGH(i,k);
        int slot = KQ_SLOT(i,k);
        if(j < 0){
            KQ_GY(i,0,1+k) = 0.0f;
            KQ_GY(i,1,1+k) = 0.0f;
            bondData[slot] = (float4)(0.0f,0.0f,0.0f,0.0f);
            continue;
        }

        int rk = KQ_REV(i,k);
        float3 d = kq_xyz(R[j]) - Ri;
        float r = sqrt(dot(d,d) + KQ_SMALL_R2);
        float3 h = d/r;
        float eij = kq_eps_r(eps0[slot], beta[slot], r, r0[slot]);
        float deps = -beta[slot]*eij;

        float yiu = KQ_Y(i,0,1+k);
        float yid = KQ_Y(i,1,1+k);
        float yju = KQ_Y(j,0,1+rk);
        float yjd = KQ_Y(j,1,1+rk);

        float su = sqrt(fmax(yiu*yju,0.0f));
        float sd = sqrt(fmax(yid*yjd,0.0f));
        float sij = su + sd;

        KQ_GY(i,0,1+k) = mu_u - eij*sqrt((yju+KQ_SMALL_Y)/(yiu+KQ_SMALL_Y));
        KQ_GY(i,1,1+k) = mu_d - eij*sqrt((yjd+KQ_SMALL_Y)/(yid+KQ_SMALL_Y));

        float fpair = Kr[slot]*(r-r0[slot]) - 2.0f*deps*sij;
        Fi += fpair*h;
        bondData[slot] = (float4)(eij,sij,r,2.0f*sij);
    }

    FLocal[i] = (float4)(Fi.x,Fi.y,Fi.z,0.0f);
}

__kernel void KekuleQM_gatherCoulombDirect(
    const int nAtoms,
    __global const float4* restrict R,
    __global const float4* restrict rhoQ,
    __global float*        restrict phi,
    __global float4*       restrict FCoul,
    const float kCoul,       // eV Angstrom / e^2, already divided by dielectric if desired
    const float softening2,  // Ohno-like finite size a^2; keeps J finite near r=0
    const float cutoff2      // <=0 means no cutoff; cutoff is a physics approximation, not just an optimization
){
    int i = get_global_id(0);
    int lid = get_local_id(0);
    int lsize = get_local_size(0);
    if(i >= nAtoms) return;

    __local float4 lR[KQ_COUL_WG];
    __local float  lQ[KQ_COUL_WG];

    float3 Ri = kq_xyz(R[i]);
    float Qi = rhoQ[i].z;
    float phii = 0.0f;
    float3 Fi = (float3)(0.0f,0.0f,0.0f);

    for(int base=0; base<nAtoms; base+=KQ_COUL_WG){
        int ntile = min(KQ_COUL_WG, nAtoms-base);
        for(int t=lid; t<ntile; t+=lsize){
            int j = base+t;
            lR[t] = R[j];
            lQ[t] = rhoQ[j].z;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for(int t=0; t<ntile; t++){
            int j = base+t;
            if(j == i) continue;
            float3 d = kq_xyz(lR[t]) - Ri;
            float r2 = dot(d,d) + softening2;
            if((cutoff2 > 0.0f) && (r2 > cutoff2)) continue;
            float invr = rsqrt(r2);
            float J = kCoul*invr;
            float Qj = lQ[t];
            phii += J*Qj;
            Fi += -(Qi*Qj*kCoul*invr*invr*invr)*d;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    phi[i] = phii;
    FCoul[i] = (float4)(Fi.x,Fi.y,Fi.z,0.0f);
}

__kernel void KekuleQM_updateDOFs(
    const int nAtoms,
    __global float*        restrict y,
    __global float4*       restrict R,
    __global float4*       restrict vel,       // persistent velocity buffer for heavy-ball momentum
    __global const int*    restrict neigh,
    __global const float*  restrict gyLocal,
    __global const float4* restrict FLocal,
    __global const float4* restrict FCoul,
    __global const float*  restrict phi,
    __global const float2* restrict rhoTarget,
    const float dtY,
    const float dtR,
    const float momentum,   // heavy-ball damping coefficient (0 = plain Euler, ~0.9 = damped)
    const float lambdaGlobal,
    const int flags
){
    int i = get_global_id(0);
    if(i >= nAtoms) return;

    int active[KQ_NCHAN];
    active[0] = 1;
    int nchan = 1;
    for(int k=0; k<KQ_MAX_NEIGH; k++){
        int ok = (KQ_NEIGH(i,k) >= 0);
        active[1+k] = ok;
        nchan += ok;
    }

    float phii = ((flags & KQ_FLAG_USE_COULOMB_ELECTRON) != 0) ? phi[i] : 0.0f;

    for(int s=0; s<KQ_NSPIN; s++){
        float g0 = KQ_GY(i,s,0) - phii;
        float g1 = active[1] ? (KQ_GY(i,s,1) - phii) : 0.0f;
        float g2 = active[2] ? (KQ_GY(i,s,2) - phii) : 0.0f;
        float g3 = active[3] ? (KQ_GY(i,s,3) - phii) : 0.0f;

        float lam = lambdaGlobal;
        if((flags & KQ_FLAG_LOCAL_SIMPLEX) != 0){
            lam = (g0 + g1 + g2 + g3)/(float)nchan;
        }

        float y0 = KQ_Y(i,s,0) - dtY*(g0-lam);
        float y1 = active[1] ? (KQ_Y(i,s,1) - dtY*(g1-lam)) : 0.0f;
        float y2 = active[2] ? (KQ_Y(i,s,2) - dtY*(g2-lam)) : 0.0f;
        float y3 = active[3] ? (KQ_Y(i,s,3) - dtY*(g3-lam)) : 0.0f;

        if((flags & KQ_FLAG_LOCAL_SIMPLEX) != 0){
            float target = (s == 0) ? rhoTarget[i].x : rhoTarget[i].y;
            target = clamp(target,0.0f,1.0f);
            kq_project_simplex4(&y0,&y1,&y2,&y3,target,nchan);
        }else{
            y0 = fmax(y0,0.0f);
            y1 = active[1] ? fmax(y1,0.0f) : 0.0f;
            y2 = active[2] ? fmax(y2,0.0f) : 0.0f;
            y3 = active[3] ? fmax(y3,0.0f) : 0.0f;
        }

        KQ_Y(i,s,0) = y0;
        KQ_Y(i,s,1) = y1;
        KQ_Y(i,s,2) = y2;
        KQ_Y(i,s,3) = y3;
    }

    if((flags & KQ_FLAG_UPDATE_R) != 0){
        float3 F = kq_xyz(FLocal[i]);
        if((flags & KQ_FLAG_USE_COULOMB_FORCE) != 0) F += kq_xyz(FCoul[i]);
        // Heavy-ball (a.k.a. accelerated gradient) update:
        //   v_{t+1} = momentum * v_t + dtR * F
        //   R_{t+1} = R_t + v_{t+1}
        // The momentum term provides inertial smoothing that stabilizes
        // the Jacobi-type electronic solver when coupled to geometry.
        // With momentum=0 this reduces to plain Euler: R += dtR * F.
        float3 v_old = kq_xyz(vel[i]);
        float3 v_new = momentum * v_old + dtR * F;
        vel[i] = (float4)(v_new.x, v_new.y, v_new.z, 0.0f);
        float3 Ri = kq_xyz(R[i]) + v_new;
        R[i] = (float4)(Ri.x,Ri.y,Ri.z,0.0f);
    }
}
