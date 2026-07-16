// BRBFFF.cl - GPU evaluator for two blended rigid-body molecular frames.
//
// The two kernels are deliberately split at the atom-force buffer boundary:
// 1. brbfff_reconstruct_positions creates skinned Cartesian atom positions.
// 2. brbfff_project_atomic_forces applies the exact Jacobian-transpose rule
//    to any forces written to atom_force by a compatible OpenCL interaction.
//
// This makes the evaluator composable with Coulomb, Morse, grid, or UFF kernels
// without hard-coding one external potential here.  Frame positions and
// quaternions are independent coordinates: a small left quaternion increment
// changes x_a=R_a r+p_a by omega cross (R_a r), while a position increment changes
// it by dp_a.  The projected torque is consequently about each frame origin,
// tau_a=sum_i w_ia (R_a r_i) cross f_i.  This is the coordinate pair used by the
// overdamped GPU relaxer below.
//
// One workgroup reduces one molecule/system.  All work-items reach the barriers;
// the Python wrapper enforces BRBFFF_WG as its local workgroup size.

#ifndef BRBFFF_WG
#define BRBFFF_WG 32
#endif

inline float3 brbfff_rotate(const float4 q, const float3 v) {
    const float3 qv = q.xyz;
    const float3 t = 2.0f * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}

inline float4 brbfff_quat_mult(const float4 a, const float4 b) {
    return (float4)(
        a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz),
        a.w * b.w - dot(a.xyz, b.xyz)
    );
}

inline float4 brbfff_quat_exp(const float3 theta) {
    const float theta2 = dot(theta, theta);
    if (theta2 < 1.0e-10f) {
        return (float4)(theta * (0.5f - theta2 / 48.0f), 1.0f - theta2 / 8.0f);
    }
    const float theta_len = sqrt(theta2);
    const float half = 0.5f * theta_len;
    return (float4)(theta * (sin(half) / theta_len), cos(half));
}

inline float3 brbfff_clamp_norm(const float3 v, const float max_norm) {
    const float norm2 = dot(v, v);
    const float max2 = max_norm * max_norm;
    return (norm2 > max2) ? v * (max_norm / sqrt(norm2)) : v;
}

__kernel void brbfff_reconstruct_positions(
    __global const float4* ref_pos,
    __global const float2* weights,
    __global const float4* frame_pos,
    __global const float4* frame_quat,
    __global float4* atom_pos,
    const int natoms,
    const int nsystems
) {
    const int index = get_global_id(0);
    const int total = natoms * nsystems;
    if (index >= total) return;
    const int system = index / natoms;
    const int atom = index - system * natoms;
    const int frame0 = 2 * system;
    const float3 ref = ref_pos[atom].xyz;
    const float2 w = weights[atom];
    const float3 x0 = brbfff_rotate(frame_quat[frame0], ref) + frame_pos[frame0].xyz;
    const float3 x1 = brbfff_rotate(frame_quat[frame0 + 1], ref) + frame_pos[frame0 + 1].xyz;
    atom_pos[index] = (float4)(w.x * x0 + w.y * x1, 0.0f);
}

__kernel void brbfff_project_atomic_forces(
    __global const float4* ref_pos,
    __global const float2* weights,
    __global const float4* frame_quat,
    __global const float4* atom_force,
    __global float4* frame_force,
    __global float4* frame_torque,
    const int natoms,
    const int nsystems
) {
    const int system = get_group_id(0);
    const int lid = get_local_id(0);
    const int lsize = get_local_size(0);
    if (system >= nsystems) return;

    __local float4 local_quat[2];
    __local float4 sum_f0[BRBFFF_WG];
    __local float4 sum_f1[BRBFFF_WG];
    __local float4 sum_t0[BRBFFF_WG];
    __local float4 sum_t1[BRBFFF_WG];
    const int frame0 = 2 * system;
    if (lid < 2) {
        local_quat[lid] = frame_quat[frame0 + lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float3 f0 = (float3)(0.0f);
    float3 f1 = (float3)(0.0f);
    float3 t0 = (float3)(0.0f);
    float3 t1 = (float3)(0.0f);
    const int atom0 = system * natoms;
    for (int atom = lid; atom < natoms; atom += lsize) {
        const float3 ref = ref_pos[atom].xyz;
        const float2 w = weights[atom];
        const float3 force = atom_force[atom0 + atom].xyz;
        const float3 r0 = brbfff_rotate(local_quat[0], ref);
        const float3 r1 = brbfff_rotate(local_quat[1], ref);
        f0 += w.x * force;
        f1 += w.y * force;
        t0 += w.x * cross(r0, force);
        t1 += w.y * cross(r1, force);
    }
    sum_f0[lid] = (float4)(f0, 0.0f);
    sum_f1[lid] = (float4)(f1, 0.0f);
    sum_t0[lid] = (float4)(t0, 0.0f);
    sum_t1[lid] = (float4)(t1, 0.0f);
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = lsize >> 1; stride > 0; stride >>= 1) {
        if (lid < stride) {
            sum_f0[lid] += sum_f0[lid + stride];
            sum_f1[lid] += sum_f1[lid + stride];
            sum_t0[lid] += sum_t0[lid + stride];
            sum_t1[lid] += sum_t1[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        frame_force[frame0] = sum_f0[0];
        frame_force[frame0 + 1] = sum_f1[0];
        frame_torque[frame0] = sum_t0[0];
        frame_torque[frame0 + 1] = sum_t1[0];
    }
}

// One capped, overdamped reduced-coordinate step.  The persistent step buffers
// are optional momentum-like smoothing, not physical velocities or masses.
// Damping=0 produces ordinary capped gradient descent.  Positive forces move
// along -dE/dq because brbfff_project_atomic_forces evaluates J^T F.
// This kernel intentionally adds no molecular internal potential; callers must
// include it in atom_force or provide a separate projected frame contribution.
__kernel void brbfff_relax_step(
    __global float4* frame_pos,
    __global float4* frame_quat,
    __global const float4* frame_force,
    __global const float4* frame_torque,
    __global float4* linear_step_state,
    __global float4* angular_step_state,
    const float linear_step,
    const float angular_step,
    const float damping,
    const float max_translation,
    const float max_rotation,
    const int nframes
) {
    const int frame = get_global_id(0);
    if (frame >= nframes) return;

    float3 dpos = damping * linear_step_state[frame].xyz + linear_step * frame_force[frame].xyz;
    float3 dtheta = damping * angular_step_state[frame].xyz + angular_step * frame_torque[frame].xyz;
    dpos = brbfff_clamp_norm(dpos, max_translation);
    dtheta = brbfff_clamp_norm(dtheta, max_rotation);

    frame_pos[frame] += (float4)(dpos, 0.0f);
    frame_quat[frame] = normalize(brbfff_quat_mult(brbfff_quat_exp(dtheta), frame_quat[frame]));
    linear_step_state[frame] = (float4)(dpos, 0.0f);
    angular_step_state[frame] = (float4)(dtheta, 0.0f);
}
