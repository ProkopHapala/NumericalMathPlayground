#!/usr/bin/env python3

import math
import sys
from pathlib import Path

import numpy as np


MAX_PARTICLES = 64
TETRA_DIRS = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ],
    dtype=np.float32,
) / math.sqrt(3.0)


KERNEL_SRC = r"""
#define MAX_PARTICLES 64

typedef struct {
    float3 f;
    float3 tau;
} PairEval;

inline float4 qnorm(float4 q) {
    float s = rsqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w + 1.0e-20f);
    return q * s;
}

inline float3 rot_fwd(float3 v, float4 q) {
    float3 u = (float3)(q.y, q.z, q.w);
    return v + 2.0f * cross(u, cross(u, v) + q.x*v);
}

inline float3 rot_inv(float3 v, float4 q) {
    float3 u = (float3)(-q.y, -q.z, -q.w);
    return v + 2.0f * cross(u, cross(u, v) + q.x*v);
}

inline float4 qstep_lab(float4 q, float3 w, float dt) {
    float4 dq;
    dq.x = -0.5f * dt * (w.x*q.y + w.y*q.z + w.z*q.w);
    dq.y =  0.5f * dt * (w.x*q.x + w.y*q.w - w.z*q.z);
    dq.z =  0.5f * dt * (-w.x*q.w + w.y*q.x + w.z*q.y);
    dq.w =  0.5f * dt * (w.x*q.z - w.y*q.y + w.z*q.x);
    return qnorm(q + dq);
}

inline PairEval eval_source(float3 ps, float4 qs, float3 pt, float rc, float cang, float rmin) {
    PairEval out;
    out.f = (float3)(0.0f, 0.0f, 0.0f);
    out.tau = (float3)(0.0f, 0.0f, 0.0f);
    float3 rlab = pt - ps;
    float r2 = dot(rlab, rlab);
    float rc2 = rc*rc;
    if (r2 >= rc2) return out;
    float3 rloc = rot_inv(rlab, qs);
    float r2s = fmax(r2, rmin*rmin);
    float inv2 = 1.0f / r2s;
    float inv4 = inv2 * inv2;
    float q = r2 / rc2;
    float oneq = 1.0f - q;
    float fcut = oneq * oneq;
    float x = rloc.x / rc;
    float y = rloc.y / rc;
    float z = rloc.z / rc;
    float fang = x * y * z;
    float base = inv2 + cang * fang;
    float3 g_inv = -2.0f * rloc * inv4;
    float3 g_ang = (float3)(y*z/rc, x*z/rc, x*y/rc);
    float3 g_cut = -4.0f * oneq * rloc / rc2;
    float3 grad_loc = (g_inv + cang*g_ang) * fcut + base * g_cut;
    float3 grad_lab = rot_fwd(grad_loc, qs);
    out.f = -grad_lab;
    out.tau = cross(out.f, rlab);
    return out;
}

__kernel void step_particles(
    __global float4* pos,
    __global float4* vel,
    __global float4* quat,
    __global float4* omega,
    const int n,
    const int nsub,
    const float dt,
    const float damp,
    const float rdamp,
    const float rc,
    const float cang,
    const float rmin,
    const float box,
    const float wallk,
    const float4 gravity,
    const int drag_index,
    const float4 drag_pos,
    const float dragk
) {
    const int lid = get_local_id(0);
    __local float3 lpos[MAX_PARTICLES];
    __local float3 lvel[MAX_PARTICLES];
    __local float4 lquat[MAX_PARTICLES];
    __local float3 low[MAX_PARTICLES];
    if (lid < n) {
        float4 p = pos[lid];
        float4 v = vel[lid];
        float4 w = omega[lid];
        lpos[lid] = (float3)(p.x, p.y, p.z);
        lvel[lid] = (float3)(v.x, v.y, v.z);
        lquat[lid] = qnorm(quat[lid]);
        low[lid] = (float3)(w.x, w.y, w.z);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int isub = 0; isub < nsub; isub++) {
        if (lid < n) {
            float3 p = lpos[lid];
            float4 q = lquat[lid];
            float3 f = (float3)(0.0f, 0.0f, 0.0f);
            float3 tau = (float3)(0.0f, 0.0f, 0.0f);
            for (int j = 0; j < n; j++) {
                if (j == lid) continue;
                PairEval own = eval_source(p, q, lpos[j], rc, cang, rmin);
                PairEval other = eval_source(lpos[j], lquat[j], p, rc, cang, rmin);
                f -= own.f;
                f += other.f;
                tau += own.tau;
            }
            if (p.x > box) f.x -= wallk * (p.x - box);
            if (p.x < -box) f.x -= wallk * (p.x + box);
            if (p.y > box) f.y -= wallk * (p.y - box);
            if (p.y < -box) f.y -= wallk * (p.y + box);
            if (p.z > box) f.z -= wallk * (p.z - box);
            if (p.z < -box) f.z -= wallk * (p.z + box);
            f += gravity.xyz;
            if (lid == drag_index) {
                float3 dp = drag_pos.xyz - p;
                f += dragk * dp;
            }
            float3 v = (lvel[lid] + dt * f) * damp;
            p += dt * v;
            float3 w = (low[lid] + dt * tau) * rdamp;
            q = qstep_lab(q, w, dt);
            lpos[lid] = p;
            lvel[lid] = v;
            lquat[lid] = q;
            low[lid] = w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid < n) {
        pos[lid] = (float4)(lpos[lid].x, lpos[lid].y, lpos[lid].z, 0.0f);
        vel[lid] = (float4)(lvel[lid].x, lvel[lid].y, lvel[lid].z, 0.0f);
        quat[lid] = lquat[lid];
        omega[lid] = (float4)(low[lid].x, low[lid].y, low[lid].z, 0.0f);
    }
}
"""


def import_qt_and_vispy():
    from vispy import app as vispy_app

    errors = []
    for api, module in (("pyqt6", "PyQt6"), ("pyside6", "PySide6"), ("pyqt5", "PyQt5"), ("pyside2", "PySide2")):
        try:
            vispy_app.use_app(api)
            qt_mod = __import__(module, fromlist=["QtCore", "QtWidgets"])
            return qt_mod.QtCore, qt_mod.QtWidgets
        except Exception as exc:
            errors.append(f"{module}: {exc}")
    raise RuntimeError("No Qt binding usable by VisPy. Tried:\n" + "\n".join(errors))


def c_for_xyz_minimum(rc, r0, lobe_value=1.0/(3.0*math.sqrt(3.0))):
    q = (r0 / rc) ** 2
    return 2.0 * rc**3 * (1.0 + q) / (lobe_value * r0**5 * (3.0 - 7.0 * q))


def select_opencl_context():
    import pyopencl as cl

    platforms = cl.get_platforms()
    devices = []
    for platform in platforms:
        for device in platform.get_devices():
            score = 0 if device.type & cl.device_type.GPU else 1
            devices.append((score, platform, device))
    if not devices:
        raise RuntimeError("No OpenCL device found")
    devices.sort(key=lambda item: item[0])
    device = devices[0][2]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)
    return cl, ctx, queue, device


def random_quaternions(rng, n):
    u1 = rng.random(n)
    u2 = rng.random(n)
    u3 = rng.random(n)
    q = np.zeros((n, 4), dtype=np.float32)
    q[:, 0] = np.sqrt(1.0 - u1) * np.sin(2.0 * math.pi * u2)
    q[:, 1] = np.sqrt(1.0 - u1) * np.cos(2.0 * math.pi * u2)
    q[:, 2] = np.sqrt(u1) * np.sin(2.0 * math.pi * u3)
    q[:, 3] = np.sqrt(u1) * np.cos(2.0 * math.pi * u3)
    q[:, [0, 1, 2, 3]] = q[:, [3, 0, 1, 2]]
    return q


def quat_rotate(v, q):
    u = q[..., 1:4]
    w = q[..., 0:1]
    return v + 2.0 * np.cross(u, np.cross(u, v) + w*v)


class ReactiveFF3DSolver:
    def __init__(self, n=32):
        self.cl, self.ctx, self.queue, self.device = select_opencl_context()
        self.program = self.cl.Program(self.ctx, KERNEL_SRC).build()
        self.kernel = self.program.step_particles
        self.n = int(n)
        mf = self.cl.mem_flags
        self.pos = np.zeros((MAX_PARTICLES, 4), dtype=np.float32)
        self.vel = np.zeros((MAX_PARTICLES, 4), dtype=np.float32)
        self.quat = np.zeros((MAX_PARTICLES, 4), dtype=np.float32)
        self.omega = np.zeros((MAX_PARTICLES, 4), dtype=np.float32)
        self.pos_buf = self.cl.Buffer(self.ctx, mf.READ_WRITE, self.pos.nbytes)
        self.vel_buf = self.cl.Buffer(self.ctx, mf.READ_WRITE, self.vel.nbytes)
        self.quat_buf = self.cl.Buffer(self.ctx, mf.READ_WRITE, self.quat.nbytes)
        self.omega_buf = self.cl.Buffer(self.ctx, mf.READ_WRITE, self.omega.nbytes)
        self.reset(self.n)

    def reset(self, n=None, seed=1234):
        if n is not None:
            self.n = max(1, min(MAX_PARTICLES, int(n)))
        rng = np.random.default_rng(seed)
        self.pos.fill(0.0)
        self.vel.fill(0.0)
        self.quat.fill(0.0)
        self.omega.fill(0.0)
        m = int(math.ceil(self.n ** (1.0 / 3.0)))
        spacing = 0.85
        k = 0
        for iz in range(m):
            for iy in range(m):
                for ix in range(m):
                    if k >= self.n:
                        break
                    xyz = np.array(
                        [
                            (ix - 0.5*(m - 1))*spacing,
                            (iy - 0.5*(m - 1))*spacing,
                            (iz - 0.5*(m - 1))*spacing,
                        ],
                        dtype=np.float32,
                    )
                    xyz += rng.normal(0.0, 0.08, size=3).astype(np.float32)
                    self.pos[k, :3] = xyz
                    self.vel[k, :3] = rng.normal(0.0, 0.02, size=3).astype(np.float32)
                    self.omega[k, :3] = rng.normal(0.0, 0.05, size=3).astype(np.float32)
                    k += 1
        self.quat[:self.n, :] = random_quaternions(rng, self.n)
        self.upload()

    def upload(self):
        self.cl.enqueue_copy(self.queue, self.pos_buf, self.pos)
        self.cl.enqueue_copy(self.queue, self.vel_buf, self.vel)
        self.cl.enqueue_copy(self.queue, self.quat_buf, self.quat)
        self.cl.enqueue_copy(self.queue, self.omega_buf, self.omega)
        self.queue.finish()

    def step(self, nsub, dt, damp, rdamp, rc, r0, rmin, box, wallk,
             gravity=(0, 0, 0), drag_index=-1, drag_pos=(0, 0, 0), dragk=0.0):
        cang = c_for_xyz_minimum(rc, r0)
        self.kernel(
            self.queue,
            (MAX_PARTICLES,),
            (MAX_PARTICLES,),
            self.pos_buf,
            self.vel_buf,
            self.quat_buf,
            self.omega_buf,
            np.int32(self.n),
            np.int32(nsub),
            np.float32(dt),
            np.float32(damp),
            np.float32(rdamp),
            np.float32(rc),
            np.float32(cang),
            np.float32(rmin),
            np.float32(box),
            np.float32(wallk),
            np.array([gravity[0], gravity[1], gravity[2], 0.0], dtype=np.float32),
            np.int32(drag_index),
            np.array([drag_pos[0], drag_pos[1], drag_pos[2], 0.0], dtype=np.float32),
            np.float32(dragk),
        )

    def download(self):
        self.cl.enqueue_copy(self.queue, self.pos, self.pos_buf)
        self.cl.enqueue_copy(self.queue, self.vel, self.vel_buf)
        self.cl.enqueue_copy(self.queue, self.quat, self.quat_buf)
        self.cl.enqueue_copy(self.queue, self.omega, self.omega_buf)
        self.queue.finish()
        return self.pos[:self.n].copy(), self.quat[:self.n].copy()


class ReactiveFF3DWindow:
    def __init__(self, QtCore, QtWidgets):
        from vispy import app as vispy_app
        from vispy import scene

        self.QtCore = QtCore
        self.QtWidgets = QtWidgets
        self.scene = scene
        self.vispy_app = vispy_app
        self.solver = ReactiveFF3DSolver(n=32)
        self.paused = False
        self.frame = 0
        self.drag_index = -1
        self.drag_pos = np.zeros(3, dtype=np.float32)
        self.drag_mouse_pos = None
        self.last_mouse_pos = None
        self.last_pos = None
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("ReactiveFF OpenCL sp3 particles")
        self.root = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout(self.root)
        self.canvas = scene.SceneCanvas(keys="interactive", bgcolor="white", size=(1000, 800), show=False)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(fov=45.0, distance=8.0, azimuth=35.0, elevation=25.0, center=(0.0, 0.0, 0.0))
        self.view.camera.interactive = False
        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self.arms = scene.visuals.Line(parent=self.view.scene, color=(0.03, 0.03, 0.03, 0.82), width=1.1, connect="segments")
        self.box_lines = scene.visuals.Line(parent=self.view.scene, color=(0.15, 0.15, 0.15, 0.35), width=1.0, connect="segments")
        self.drag_line = scene.visuals.Line(parent=self.view.scene, color=(0.95, 0.55, 0.02, 0.90), width=1.6, connect="segments")
        self.ray_line = scene.visuals.Line(parent=self.view.scene, color=(0.0, 0.8, 0.0, 0.7), width=1.0, connect="segments")
        self.layout.addWidget(self.canvas.native, 1)
        self.controls = QtWidgets.QWidget()
        self.controls_layout = QtWidgets.QFormLayout(self.controls)
        self.n_spin = self.make_spin(1, MAX_PARTICLES, 32, 1)
        self.sub_spin = self.make_spin(1, 500, 30, 1)
        self.dt_spin = self.make_double(0.0, 1.0, 0.02, 0.00005, 8)
        self.rc_spin = self.make_double(0.3, 3.0, 1.0, 0.05, 3)
        self.r0_spin = self.make_double(0.67, 0.95, 0.75, 0.01, 3)
        self.damp_spin = self.make_double(0.900, 1.000, 0.996, 0.001, 4)
        self.rdamp_spin = self.make_double(0.900, 1.000, 0.995, 0.001, 4)
        self.wall_spin = self.make_double(0.0, 50.0, 6.0, 0.25, 3)
        self.box_spin = self.make_double(1.0, 10.0, 3.0, 0.1, 3)
        self.gravity_spin = self.make_double(0.0, 20.0, 0.0, 0.1, 3)
        self.dragk_spin = self.make_double(0.0, 200.0, 50.0, 1.0, 2)
        self.controls_layout.addRow("particles", self.n_spin)
        self.controls_layout.addRow("substeps/kernel/frame", self.sub_spin)
        self.controls_layout.addRow("dt / substep", self.dt_spin)
        self.controls_layout.addRow("rc", self.rc_spin)
        self.controls_layout.addRow("r0/rc", self.r0_spin)
        self.controls_layout.addRow("velocity damp", self.damp_spin)
        self.controls_layout.addRow("angular damp", self.rdamp_spin)
        self.controls_layout.addRow("box half-size", self.box_spin)
        self.controls_layout.addRow("wall K", self.wall_spin)
        self.controls_layout.addRow("gravity (z-down)", self.gravity_spin)
        self.controls_layout.addRow("drag spring K", self.dragk_spin)
        self.pause_button = QtWidgets.QPushButton("pause")
        self.reset_button = QtWidgets.QPushButton("reset")
        self.info_label = QtWidgets.QLabel("")
        self.controls_layout.addRow(self.pause_button)
        self.controls_layout.addRow(self.reset_button)
        self.controls_layout.addRow(self.info_label)
        self.layout.addWidget(self.controls)
        self.window.setCentralWidget(self.root)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.reset_button.clicked.connect(self.reset)
        self.n_spin.valueChanged.connect(self.reset)
        self.box_spin.valueChanged.connect(self.update_visuals)
        self.canvas.events.mouse_press.connect(self.on_mouse_press)
        self.canvas.events.mouse_move.connect(self.on_mouse_move)
        self.canvas.events.mouse_release.connect(self.on_mouse_release)
        self.canvas.events.key_press.connect(self.on_key_press)
        self.canvas.events.mouse_wheel.connect(self.on_mouse_wheel)
        self.timer = vispy_app.Timer(interval=0.0, connect=self.on_timer, start=True)
        self.update_visuals()

    def make_spin(self, mn, mx, val, step):
        w = self.QtWidgets.QSpinBox()
        w.setRange(mn, mx)
        w.setSingleStep(step)
        w.setValue(val)
        return w

    def make_double(self, mn, mx, val, step, decimals):
        w = self.QtWidgets.QDoubleSpinBox()
        w.setRange(mn, mx)
        w.setSingleStep(step)
        w.setDecimals(decimals)
        w.setValue(val)
        return w

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.setText("resume" if self.paused else "pause")

    def reset(self):
        self.solver.reset(self.n_spin.value(), seed=1234 + self.frame)
        self.update_visuals()

    def screen_to_ray(self, mouse_pos):
        cam = self.view.camera
        w, h = self.canvas.size
        x, y = float(mouse_pos[0]), float(mouse_pos[1])
        view_mat = cam.transform.matrix
        proj_mat = cam._projection.matrix
        inv_view = np.linalg.inv(view_mat)
        VP = inv_view @ proj_mat
        inv_VP = np.linalg.inv(VP)
        ndc_near = np.array([2*x/w - 1, 1 - 2*y/h, -1.0, 1.0])
        ndc_far  = np.array([2*x/w - 1, 1 - 2*y/h,  1.0, 1.0])
        w_near = ndc_near @ inv_VP
        w_far  = ndc_far  @ inv_VP
        w_near = w_near[:3] / w_near[3]
        w_far  = w_far[:3]  / w_far[3]
        origin = w_near.astype(np.float32)
        direction = (w_far - w_near).astype(np.float32)
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
        return origin, direction

    def pick_particle(self, mouse_pos):
        if self.last_pos is None:
            return -1
        eye, ray_dir = self.screen_to_ray(mouse_pos)
        xyz = self.last_pos[:, :3]
        best_i = -1
        best_dist = 1e10
        for i in range(self.solver.n):
            d = xyz[i] - eye
            t = float(np.dot(d, ray_dir))
            if t < 0:
                continue
            closest = eye + t * ray_dir
            dist = float(np.linalg.norm(xyz[i] - closest))
            if dist < best_dist:
                best_dist = dist
                best_i = i
        if best_dist < 0.3:
            return best_i
        return -1

    def update_drag_pos(self):
        if self.drag_index < 0 or self.drag_mouse_pos is None or self.last_pos is None:
            return
        eye, ray_dir = self.screen_to_ray(self.drag_mouse_pos)
        p = self.last_pos[self.drag_index, :3]
        t = float(np.dot(p - eye, ray_dir))
        self.drag_pos = (eye + t * ray_dir).astype(np.float32)

    def on_mouse_press(self, event):
        idx = self.pick_particle(event.pos)
        if idx >= 0:
            self.drag_index = idx
            self.drag_mouse_pos = event.pos
            self.update_drag_pos()
        else:
            self.drag_index = -1

    def on_mouse_move(self, event):
        self.last_mouse_pos = event.pos
        if self.drag_index >= 0:
            self.drag_mouse_pos = event.pos
            self.update_drag_pos()
        self.update_ray_visual()

    def on_mouse_release(self, event):
        self.drag_index = -1

    def on_key_press(self, event):
        cam = self.view.camera
        step = 5.0
        if event.key == 'Left':
            cam.azimuth -= step
        elif event.key == 'Right':
            cam.azimuth += step
        elif event.key == 'Up':
            cam.elevation = min(90.0, cam.elevation + step)
        elif event.key == 'Down':
            cam.elevation = max(-90.0, cam.elevation - step)
        elif event.key in ('+', '='):
            cam.distance = max(1.0, cam.distance * 0.9)
        elif event.key in ('-', '_'):
            cam.distance = min(50.0, cam.distance * 1.1)
        cam.view_changed()
        self.update_ray_visual()

    def on_mouse_wheel(self, event):
        cam = self.view.camera
        delta = float(event.delta[1])
        factor = 0.9 if delta > 0 else 1.1
        cam.distance = max(1.0, min(50.0, cam.distance * factor))
        cam.view_changed()
        self.update_ray_visual()

    def update_ray_visual(self):
        if self.last_mouse_pos is not None:
            origin, direction = self.screen_to_ray(self.last_mouse_pos)
            length = 10.0
            seg = np.array([origin, origin + length * direction], dtype=np.float32)
            self.ray_line.set_data(seg)
        else:
            self.ray_line.set_data(np.zeros((0, 3), dtype=np.float32))
        self.canvas.update()

    def on_timer(self, event):
        if not self.paused:
            self.update_drag_pos()
            rc = self.rc_spin.value()
            r0 = self.r0_spin.value() * rc
            gravity = np.array([0.0, 0.0, -self.gravity_spin.value()], dtype=np.float32)
            self.solver.step(
                self.sub_spin.value(),
                self.dt_spin.value(),
                self.damp_spin.value(),
                self.rdamp_spin.value(),
                rc,
                r0,
                0.08 * rc,
                self.box_spin.value(),
                self.wall_spin.value(),
                gravity,
                self.drag_index,
                self.drag_pos,
                self.dragk_spin.value(),
            )
            self.frame += 1
            self.update_visuals()

    def box_segments(self, box):
        corners = np.array(
            [
                [-box, -box, -box],
                [ box, -box, -box],
                [ box,  box, -box],
                [-box,  box, -box],
                [-box, -box,  box],
                [ box, -box,  box],
                [ box,  box,  box],
                [-box,  box,  box],
            ],
            dtype=np.float32,
        )
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        segs = np.zeros((len(edges)*2, 3), dtype=np.float32)
        for i, (a, b) in enumerate(edges):
            segs[2*i] = corners[a]
            segs[2*i + 1] = corners[b]
        return segs

    def update_visuals(self, pos=None, quat=None):
        if pos is None or quat is None:
            pos, quat = self.solver.download()
        self.last_pos = pos
        xyz = pos[:, :3]
        rc = self.rc_spin.value()
        r0 = self.r0_spin.value() * rc
        self.markers.set_data(xyz, face_color=(1.0, 1.0, 1.0, 1.0), edge_color=(0.02, 0.02, 0.02, 0.95), size=8.0)
        segs = np.zeros((self.solver.n * 8, 3), dtype=np.float32)
        arm = 0.5 * r0
        for i in range(self.solver.n):
            q = quat[i]
            dirs = quat_rotate(TETRA_DIRS, q[None, :])
            for k in range(4):
                idx = 8*i + 2*k
                segs[idx] = xyz[i]
                segs[idx + 1] = xyz[i] + arm*dirs[k]
        self.arms.set_data(segs)
        self.box_lines.set_data(self.box_segments(self.box_spin.value()))
        if self.drag_index >= 0 and self.drag_index < self.solver.n:
            self.drag_line.set_data(np.array([xyz[self.drag_index], self.drag_pos], dtype=np.float32))
        else:
            self.drag_line.set_data(np.zeros((0, 3), dtype=np.float32))
        cang = c_for_xyz_minimum(rc, r0)
        self.info_label.setText(f"device: {self.solver.device.name}\nc={cang:.4f}\nframe={self.frame}")
        self.canvas.update()

    def run(self):
        self.window.resize(1200, 820)
        self.window.show()
        self.vispy_app.run()


def main():
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    QtCore, QtWidgets = import_qt_and_vispy()
    window = ReactiveFF3DWindow(QtCore, QtWidgets)
    window.run()


if __name__ == "__main__":
    main()
