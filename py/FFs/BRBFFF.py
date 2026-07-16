"""BRBFFF.py — persistent GPU evaluator for two-frame blended molecular mechanics.

This is the OpenCL bridge between atom-based external force fields and the reduced
two-frame model.  It keeps reference geometry, skinning weights, two frame poses,
skinned atom positions, atom forces, and projected frame wrenches resident on the
GPU.  Any compatible OpenCL force evaluator can consume ``atom_pos`` and write
``atom_force``; this module then applies the work-preserving reduction ``J.T @ F``.

Design:
- Atom positions use weighted absolute quaternion transforms, so equal frame poses
  reproduce an exact global rigid transform.
- The force-reduction kernel uses one workgroup per system and local-memory tree
  reduction; it never uses atomics and supports many replicas of one topology.
- ``relax_step`` is a capped, overdamped GPU propagator.  It needs no masses:
  generalized forces already point downhill in the frame coordinates.

Open issues:
- Only two frames and float32 GPU arithmetic are supported.
- The external force kernel must write forces for the reconstructed atom ordering.
- The current GPU path does not evaluate the fitted relative-frame potential.  A
  distorted molecule is restored only when ``atom_force`` includes an intrinsic
  molecular force or after that potential is ported to the GPU.
- A generic external force evaluator must supply an energy separately if strict
  energy-acceptance/backtracking is required; the GPU step itself only sees forces.
"""

from __future__ import annotations

import os

import numpy as np
import pyopencl as cl

from ..OpenCLBase import OpenCLBase


DEFAULT_WORKGROUP_SIZE = 32


def _float4(values, w=0.0):
    """Pack Cartesian vectors into the aligned float4 layout used by the kernels."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] not in (3, 4):
        raise ValueError(f"expected (n,3) or (n,4) array, got {values.shape}")
    if values.shape[1] == 4:
        return np.ascontiguousarray(values, dtype=np.float32)
    packed = np.empty((len(values), 4), dtype=np.float32)
    packed[:, :3] = values
    packed[:, 3] = np.float32(w)
    return packed


def _quaternions(quaternions, nsystems):
    """Validate normalized ``(n_systems, 2, 4)`` quaternion frame state."""
    quaternions = np.asarray(quaternions, dtype=np.float32)
    if quaternions.shape != (nsystems, 2, 4):
        raise ValueError(f"expected quaternions shape ({nsystems},2,4), got {quaternions.shape}")
    norm = np.linalg.norm(quaternions, axis=2)
    if np.any(norm < 1e-7) or not np.all(np.isfinite(quaternions)):
        raise ValueError("frame quaternions must be finite and nonzero")
    return np.ascontiguousarray(quaternions / norm[:, :, None], dtype=np.float32)


class BRBFFF(OpenCLBase):
    """GPU two-frame skinning, force projection, and capped pose relaxation."""

    def __init__(self, nloc=DEFAULT_WORKGROUP_SIZE, debug=False, ctx=None, queue=None):
        if nloc != DEFAULT_WORKGROUP_SIZE:
            raise ValueError(f"BRBFFF.cl is compiled for workgroup size {DEFAULT_WORKGROUP_SIZE}, got {nloc}")
        super().__init__(nloc=nloc, ctx=ctx, queue=queue)
        kernel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../kernels/BRBFFF.cl')
        build_options = [f'-DBRBFFF_WG={nloc}', f'-DBRBFFF_DEBUG={1 if debug else 0}']
        if not self.load_program(kernel_path, build_options=build_options, bMakeHeaders=False):
            raise RuntimeError(f"failed to compile BRBFFF kernel {kernel_path}")
        self.debug = bool(debug)
        self.natoms = 0
        self.nsystems = 0
        self._initialized = False

    def initialize(self, reference_positions, weights, nsystems=1, frame_positions=None, frame_quaternions=None):
        """Allocate persistent buffers and upload one shared topology plus frame state."""
        reference_positions = _float4(reference_positions)
        natoms = len(reference_positions)
        weights = np.asarray(weights, dtype=np.float32)
        if weights.shape != (natoms, 2) or not np.all(np.isfinite(weights)):
            raise ValueError(f"weights must have shape ({natoms},2) and be finite")
        if not np.allclose(np.sum(weights, axis=1), 1.0, atol=1e-6):
            raise ValueError("BRBFFF weights must satisfy partition of unity")
        nsystems = int(nsystems)
        if nsystems < 1:
            raise ValueError("nsystems must be positive")
        if frame_positions is None:
            frame_positions = np.zeros((nsystems, 2, 3), dtype=np.float32)
        frame_positions = np.asarray(frame_positions, dtype=np.float32)
        if frame_positions.shape != (nsystems, 2, 3):
            raise ValueError(f"expected frame_positions shape ({nsystems},2,3), got {frame_positions.shape}")
        if frame_quaternions is None:
            frame_quaternions = np.zeros((nsystems, 2, 4), dtype=np.float32)
            frame_quaternions[:, :, 3] = 1.0
        frame_quaternions = _quaternions(frame_quaternions, nsystems)

        self.natoms = natoms
        self.nsystems = nsystems
        nframes = 2 * nsystems
        total_atoms = natoms * nsystems
        f4 = np.dtype(np.float32).itemsize * 4
        f2 = np.dtype(np.float32).itemsize * 2
        mf = cl.mem_flags
        self.check_buf('ref_pos', natoms * f4, mf.READ_ONLY)
        self.check_buf('weights', natoms * f2, mf.READ_ONLY)
        self.check_buf('frame_pos', nframes * f4, mf.READ_WRITE)
        self.check_buf('frame_quat', nframes * f4, mf.READ_WRITE)
        self.check_buf('atom_pos', total_atoms * f4, mf.READ_WRITE)
        self.check_buf('atom_force', total_atoms * f4, mf.READ_WRITE)
        self.check_buf('frame_force', nframes * f4, mf.READ_WRITE)
        self.check_buf('frame_torque', nframes * f4, mf.READ_WRITE)
        self.check_buf('linear_step_state', nframes * f4, mf.READ_WRITE)
        self.check_buf('angular_step_state', nframes * f4, mf.READ_WRITE)

        self.toGPU('ref_pos', reference_positions)
        self.toGPU('weights', np.ascontiguousarray(weights, dtype=np.float32))
        self.set_frame_state(frame_positions, frame_quaternions)
        zeros_atoms = np.zeros((total_atoms, 4), dtype=np.float32)
        zeros_frames = np.zeros((nframes, 4), dtype=np.float32)
        self.toGPU('atom_force', zeros_atoms)
        self.toGPU('atom_pos', zeros_atoms)
        self.toGPU('frame_force', zeros_frames)
        self.toGPU('frame_torque', zeros_frames)
        self.toGPU('linear_step_state', zeros_frames)
        self.toGPU('angular_step_state', zeros_frames)
        self.queue.finish()
        self._initialized = True

    def _require_initialized(self):
        if not self._initialized:
            raise RuntimeError("call initialize(...) before launching BRBFFF kernels")

    def set_frame_state(self, frame_positions, frame_quaternions):
        """Upload the two frame positions and unit quaternions for every system."""
        if self.nsystems == 0:
            raise RuntimeError("initialize(...) must set system count before setting frame state")
        positions = np.asarray(frame_positions, dtype=np.float32)
        if positions.shape != (self.nsystems, 2, 3) or not np.all(np.isfinite(positions)):
            raise ValueError(f"expected finite frame_positions shape ({self.nsystems},2,3)")
        quaternions = _quaternions(frame_quaternions, self.nsystems)
        self.toGPU('frame_pos', _float4(positions.reshape(-1, 3)))
        self.toGPU('frame_quat', quaternions.reshape(-1, 4))
        if self._initialized:
            self.reset_relaxation()

    def reset_relaxation(self):
        """Discard damped-step history after a pose teleport or changed constraints."""
        self._require_initialized()
        zeros = np.zeros((2 * self.nsystems, 4), dtype=np.float32)
        self.toGPU('linear_step_state', zeros)
        self.toGPU('angular_step_state', zeros)

    def get_frame_state(self):
        """Download current frame positions and quaternions."""
        self._require_initialized()
        positions = np.empty((2 * self.nsystems, 4), dtype=np.float32)
        quaternions = np.empty((2 * self.nsystems, 4), dtype=np.float32)
        self.fromGPU('frame_pos', positions)
        self.fromGPU('frame_quat', quaternions)
        self.queue.finish()
        return positions[:, :3].reshape(self.nsystems, 2, 3), quaternions.reshape(self.nsystems, 2, 4)

    def upload_atom_forces(self, atom_forces):
        """Upload atom forces or replace this transfer with a compatible GPU force kernel."""
        self._require_initialized()
        forces = np.asarray(atom_forces, dtype=np.float32)
        if forces.shape == (self.natoms, 3):
            forces = np.broadcast_to(forces[None, :, :], (self.nsystems, self.natoms, 3)).copy()
        if forces.shape != (self.nsystems, self.natoms, 3) or not np.all(np.isfinite(forces)):
            raise ValueError(
                f"atom_forces must have shape ({self.nsystems},{self.natoms},3) or ({self.natoms},3)"
            )
        self.toGPU('atom_force', _float4(forces.reshape(-1, 3)))

    def reconstruct_positions(self):
        """Run skinning and leave positions resident in ``atom_pos``."""
        self._require_initialized()
        total_atoms = self.nsystems * self.natoms
        self.prg.brbfff_reconstruct_positions(
            self.queue, (self.roundUpGlobalSize(total_atoms),), (self.nloc,),
            self.buffer_dict['ref_pos'], self.buffer_dict['weights'],
            self.buffer_dict['frame_pos'], self.buffer_dict['frame_quat'],
            self.buffer_dict['atom_pos'], np.int32(self.natoms), np.int32(self.nsystems),
        )

    def project_atomic_forces(self):
        """Reduce atom forces to frame-origin force/torque pairs in direct pose coordinates."""
        self._require_initialized()
        self.prg.brbfff_project_atomic_forces(
            self.queue, (self.nsystems * self.nloc,), (self.nloc,),
            self.buffer_dict['ref_pos'], self.buffer_dict['weights'],
            self.buffer_dict['frame_quat'],
            self.buffer_dict['atom_force'], self.buffer_dict['frame_force'],
            self.buffer_dict['frame_torque'], np.int32(self.natoms), np.int32(self.nsystems),
        )

    def evaluate(self, atom_forces=None, download_positions=False):
        """Reconstruct geometry and project supplied/resident atomic forces to frame wrenches."""
        self._require_initialized()
        if atom_forces is not None:
            self.upload_atom_forces(atom_forces)
        self.reconstruct_positions()
        self.project_atomic_forces()
        self.queue.finish()
        result = self.download_frame_wrenches()
        if download_positions:
            result['positions'] = self.download_positions()
        return result

    def relax_step(
        self, atom_forces=None, linear_step=1.0e-3, angular_step=1.0e-4,
        damping=0.0, max_translation=0.02, max_rotation=0.02,
        reconstruct=True,
    ):
        """Apply one capped GPU gradient-descent or damped-relaxation pose update.

        ``linear_step`` has units Angstrom/eV times force units and
        ``angular_step`` has units rad/eV times torque units.  With ``damping=0``
        this is ordinary capped gradient descent; nonzero damping smooths successive
        updates but can require smaller step sizes.  The returned wrench belongs to
        the geometry *before* the update.  A generic force-only kernel cannot prove
        energy decrease, so callers needing that guarantee must evaluate energy and
        do host-side backtracking.  This kernel adds no fitted internal restoring
        force; ``atom_force`` must contain every force that should drive relaxation.
        """
        self._require_initialized()
        values = (linear_step, angular_step, damping, max_translation, max_rotation)
        if not np.all(np.isfinite(values)):
            raise ValueError("relaxation parameters must be finite")
        if linear_step < 0.0 or angular_step < 0.0:
            raise ValueError("relaxation step scales must be non-negative")
        if not 0.0 <= damping < 1.0:
            raise ValueError("damping must satisfy 0 <= damping < 1")
        if max_translation <= 0.0 or max_rotation <= 0.0:
            raise ValueError("maximum translation and rotation must be positive")
        if atom_forces is not None:
            self.upload_atom_forces(atom_forces)
        self.project_atomic_forces()
        self.prg.brbfff_relax_step(
            self.queue, (self.roundUpGlobalSize(2 * self.nsystems),), (self.nloc,),
            self.buffer_dict['frame_pos'], self.buffer_dict['frame_quat'],
            self.buffer_dict['frame_force'], self.buffer_dict['frame_torque'],
            self.buffer_dict['linear_step_state'], self.buffer_dict['angular_step_state'],
            np.float32(linear_step), np.float32(angular_step), np.float32(damping),
            np.float32(max_translation), np.float32(max_rotation), np.int32(2 * self.nsystems),
        )
        if reconstruct:
            self.reconstruct_positions()
        self.queue.finish()
        return self.download_frame_wrenches()

    def download_positions(self):
        """Download skinned positions; use only for diagnostics or CPU-side force evaluators."""
        self._require_initialized()
        positions = np.empty((self.nsystems * self.natoms, 4), dtype=np.float32)
        self.fromGPU('atom_pos', positions)
        self.queue.finish()
        if not np.all(np.isfinite(positions[:, :3])):
            raise FloatingPointError("non-finite GPU skinned atom position")
        return positions[:, :3].reshape(self.nsystems, self.natoms, 3)

    def download_frame_wrenches(self):
        """Download the small projected force/torque state for host-side reduced integration."""
        self._require_initialized()
        forces = np.empty((2 * self.nsystems, 4), dtype=np.float32)
        torques = np.empty((2 * self.nsystems, 4), dtype=np.float32)
        self.fromGPU('frame_force', forces)
        self.fromGPU('frame_torque', torques)
        self.queue.finish()
        if not (np.all(np.isfinite(forces[:, :3])) and np.all(np.isfinite(torques[:, :3]))):
            raise FloatingPointError("non-finite GPU projected frame wrench")
        return {
            'force': forces[:, :3].reshape(self.nsystems, 2, 3),
            'torque': torques[:, :3].reshape(self.nsystems, 2, 3),
        }

    def device_buffers(self):
        """Expose persistent atom buffers for a compatible external OpenCL force kernel."""
        self._require_initialized()
        return {
            'atom_pos': self.buffer_dict['atom_pos'],
            'atom_force': self.buffer_dict['atom_force'],
            'frame_pos': self.buffer_dict['frame_pos'],
            'frame_quat': self.buffer_dict['frame_quat'],
        }
