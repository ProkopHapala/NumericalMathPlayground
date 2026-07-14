#!/usr/bin/env python3

import math
import sys
from pathlib import Path

import numpy as np


MAX_PARTICLES = 64


KERNEL_SRC = r"""
#define MAX_PARTICLES 64
#define M_PI_F 3.14159265358979323846f

typedef struct {
    float2 f;
    float tau;
} PairEval;

inline float2 rot_fwd(float2 v, float4 q) {
    float cs = q.x*q.x - q.w*q.w;
    float sn = 2.0f*q.x*q.w;
    return (float2)(cs*v.x - sn*v.y, sn*v.x + cs*v.y);
}

inline float2 rot_inv(float2 v, float4 q) {
    float cs = q.x*q.x - q.w*q.w;
    float sn = 2.0f*q.x*q.w;
    return (float2)(cs*v.x + sn*v.y, -sn*v.x + cs*v.y);
}

inline PairEval eval_source(float2 ps, float4 qs, float2 pt, float rc, float cang, float rmin) {
    PairEval out;
    out.f = (float2)(0.0f, 0.0f);
    out.tau = 0.0f;
    float2 rlab = pt - ps;
    float r2 = dot(rlab, rlab);
    float rc2 = rc*rc;
    if (r2 >= rc2) return out;
    float2 rloc = rot_inv(rlab, qs);
    float r2s = fmax(r2, rmin*rmin);
    float inv2 = 1.0f / r2s;
    float inv4 = inv2 * inv2;
    float q = r2 / rc2;
    float oneq = 1.0f - q;
    float fcut = oneq * oneq;
    float x = rloc.x / rc;
    float y = rloc.y / rc;
    float fang = x * (x*x - 3.0f*y*y);
    float base = inv2 + cang * fang;
    float2 g_inv = -2.0f * rloc * inv4;
    float2 g_ang = (float2)(3.0f*(x*x - y*y)/rc, -6.0f*x*y/rc);
    float2 g_cut = -4.0f * oneq * rloc / rc2;
    float2 grad_loc = (g_inv + cang*g_ang) * fcut + base * g_cut;
    float2 grad_lab = rot_fwd(grad_loc, qs);
    out.f = -grad_lab;
    out.tau = out.f.x*rlab.y - out.f.y*rlab.x;
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
    const int drag_index,
    const float2 drag_pos,
    const float dragk
) {
    const int lid = get_local_id(0);
    __local float2 lpos[MAX_PARTICLES];
    __local float2 lvel[MAX_PARTICLES];
    __local float4 lquat[MAX_PARTICLES];
    __local float low[MAX_PARTICLES];
    if (lid < n) {
        float4 p = pos[lid];
        float4 v = vel[lid];
        lpos[lid] = (float2)(p.x, p.y);
        lvel[lid] = (float2)(v.x, v.y);
        lquat[lid] = quat[lid];
        low[lid] = omega[lid].x;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int isub = 0; isub < nsub; isub++) {
        if (lid < n) {
            float2 p = lpos[lid];
            float4 q = lquat[lid];
            float2 f = (float2)(0.0f, 0.0f);
            float tau = 0.0f;
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
            if (lid == drag_index) f += dragk * (drag_pos - p);
            float2 v = (lvel[lid] + dt * f) * damp;
            p += dt * v;
            float om = (low[lid] + dt * tau) * rdamp;
            float qw = q.x;
            float qz = q.w;
            q.x = qw - 0.5f * dt * om * qz;
            q.y = 0.0f;
            q.z = 0.0f;
            q.w = qz + 0.5f * dt * om * qw;
            float iq = rsqrt(q.x*q.x + q.w*q.w + 1.0e-20f);
            q.x *= iq;
            q.w *= iq;
            lpos[lid] = p;
            lvel[lid] = v;
            lquat[lid] = q;
            low[lid] = om;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid < n) {
        pos[lid] = (float4)(lpos[lid].x, lpos[lid].y, 0.0f, 0.0f);
        vel[lid] = (float4)(lvel[lid].x, lvel[lid].y, 0.0f, 0.0f);
        quat[lid] = lquat[lid];
        omega[lid] = (float4)(low[lid], 0.0f, 0.0f, 0.0f);
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


def c_for_angular_minimum(rc, r0, fang_sign=-1.0):
    q = (r0 / rc) ** 2
    return 2.0 * rc**3 * (1.0 + q) / (fang_sign * r0**5 * (3.0 - 7.0 * q))


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


class ReactiveFFSolver:
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

    def reset(self, n=None, seed=1234, box=3.0):
        if n is not None:
            self.n = max(1, min(MAX_PARTICLES, int(n)))
        rng = np.random.default_rng(seed)
        self.pos.fill(0.0)
        self.vel.fill(0.0)
        self.quat.fill(0.0)
        self.omega.fill(0.0)
        m = int(math.ceil(math.sqrt(self.n)))
        spacing = 0.72
        k = 0
        for iy in range(m):
            for ix in range(m):
                if k >= self.n:
                    break
                xy = np.array([(ix - 0.5*(m - 1))*spacing, (iy - 0.5*(m - 1))*spacing], dtype=np.float32)
                xy += rng.normal(0.0, 0.08, size=2).astype(np.float32)
                self.pos[k, :2] = xy
                self.vel[k, :2] = rng.normal(0.0, 0.03, size=2).astype(np.float32)
                angle = rng.uniform(0.0, 2.0*math.pi)
                self.quat[k, 0] = math.cos(0.5*angle)
                self.quat[k, 3] = math.sin(0.5*angle)
                self.omega[k, 0] = rng.normal(0.0, 0.1)
                k += 1
        self.upload()

    def upload(self):
        self.cl.enqueue_copy(self.queue, self.pos_buf, self.pos)
        self.cl.enqueue_copy(self.queue, self.vel_buf, self.vel)
        self.cl.enqueue_copy(self.queue, self.quat_buf, self.quat)
        self.cl.enqueue_copy(self.queue, self.omega_buf, self.omega)
        self.queue.finish()

    def step(self, nsub, dt, damp, rdamp, rc, r0, rmin, box, wallk, drag_index=-1, drag_pos=(0.0, 0.0), dragk=0.0):
        cang = c_for_angular_minimum(rc, r0)
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
            np.int32(drag_index),
            np.array(drag_pos, dtype=np.float32),
            np.float32(dragk),
        )

    def download(self):
        self.cl.enqueue_copy(self.queue, self.pos, self.pos_buf)
        self.cl.enqueue_copy(self.queue, self.vel, self.vel_buf)
        self.cl.enqueue_copy(self.queue, self.quat, self.quat_buf)
        self.cl.enqueue_copy(self.queue, self.omega, self.omega_buf)
        self.queue.finish()
        return self.pos[:self.n].copy(), self.quat[:self.n].copy()


class ReactiveFFWindow:
    def __init__(self, QtCore, QtWidgets):
        from vispy import app as vispy_app
        from vispy import scene

        self.QtCore = QtCore
        self.QtWidgets = QtWidgets
        self.scene = scene
        self.vispy_app = vispy_app
        self.solver = ReactiveFFSolver(n=32)
        self.paused = False
        self.frame = 0
        self.drag_index = -1
        self.drag_pos = np.zeros(2, dtype=np.float32)
        self.last_pos = None
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        self.window = QtWidgets.QMainWindow()
        self.window.setWindowTitle("ReactiveFF OpenCL particles")
        self.root = QtWidgets.QWidget()
        self.layout = QtWidgets.QHBoxLayout(self.root)
        self.canvas = scene.SceneCanvas(keys="interactive", bgcolor="white", size=(1000, 800), show=False)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.PanZoomCamera(rect=(-3.5, -3.5, 7.0, 7.0), aspect=1.0)
        self.view.camera.interactive = False
        self.rc_circles = scene.visuals.Line(parent=self.view.scene, color=(0.15, 0.35, 0.80, 0.16), width=0.75, connect="segments")
        self.r0_circles = scene.visuals.Line(parent=self.view.scene, color=(0.95, 0.45, 0.05, 0.30), width=0.9, connect="segments")
        self.markers = scene.visuals.Markers(parent=self.view.scene)
        self.arms = scene.visuals.Line(parent=self.view.scene, color=(0.05, 0.05, 0.05, 0.78), width=1.1, connect="segments")
        self.drag_line = scene.visuals.Line(parent=self.view.scene, color=(0.95, 0.55, 0.02, 0.90), width=1.6, connect="segments")
        self.box = scene.visuals.Rectangle(center=(0.0, 0.0), width=6.0, height=6.0, border_color=(0.25, 0.25, 0.25, 0.45), color=(0, 0, 0, 0), parent=self.view.scene)
        self.layout.addWidget(self.canvas.native, 1)
        self.controls = QtWidgets.QWidget()
        self.controls_layout = QtWidgets.QFormLayout(self.controls)
        self.n_spin = self.make_spin(1, MAX_PARTICLES, 32, 1)
        self.sub_spin = self.make_spin(1, 500, 30, 1)
        self.dt_spin = self.make_double(0.0, 1.0, 0.0005, 0.0001, 8)
        self.rc_spin = self.make_double(0.3, 3.0, 1.0, 0.05, 3)
        self.r0_spin = self.make_double(0.67, 0.95, 0.75, 0.01, 3)
        self.show_circles_check = QtWidgets.QCheckBox()
        self.show_circles_check.setChecked(True)
        self.damp_spin = self.make_double(0.900, 1.000, 0.996, 0.001, 4)
        self.rdamp_spin = self.make_double(0.900, 1.000, 0.995, 0.001, 4)
        self.wall_spin = self.make_double(0.0, 20.0, 4.0, 0.25, 3)
        self.controls_layout.addRow("particles", self.n_spin)
        self.controls_layout.addRow("substeps/kernel/frame", self.sub_spin)
        self.controls_layout.addRow("dt / substep", self.dt_spin)
        self.controls_layout.addRow("rc", self.rc_spin)
        self.controls_layout.addRow("r0/rc", self.r0_spin)
        self.controls_layout.addRow("show r0/rc circles", self.show_circles_check)
        self.controls_layout.addRow("velocity damp", self.damp_spin)
        self.controls_layout.addRow("angular damp", self.rdamp_spin)
        self.controls_layout.addRow("wall K", self.wall_spin)
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
        self.show_circles_check.stateChanged.connect(self.update_visuals)
        self.canvas.events.mouse_press.connect(self.on_mouse_press)
        self.canvas.events.mouse_move.connect(self.on_mouse_move)
        self.canvas.events.mouse_release.connect(self.on_mouse_release)
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
        self.drag_index = -1
        self.solver.reset(self.n_spin.value(), seed=1234 + self.frame)
        self.update_visuals()

    def on_timer(self, event):
        if not self.paused:
            rc = self.rc_spin.value()
            r0 = self.r0_spin.value() * rc
            self.solver.step(
                self.sub_spin.value(),
                self.dt_spin.value(),
                self.damp_spin.value(),
                self.rdamp_spin.value(),
                rc,
                r0,
                0.08 * rc,
                3.0,
                self.wall_spin.value(),
                self.drag_index,
                self.drag_pos,
                35.0,
            )
            self.frame += 1
            self.update_visuals()

    def screen_to_world(self, pos):
        tr = self.canvas.scene.node_transform(self.view.scene)
        mapped = tr.map(pos)
        return np.array(mapped[:2], dtype=np.float32)

    def on_mouse_press(self, event):
        if event.button != 1:
            return
        world = self.screen_to_world(event.pos)
        pos, quat = self.solver.download()
        xy = pos[:, :2]
        d2 = np.sum((xy - world[None, :])**2, axis=1)
        idx = int(np.argmin(d2))
        if d2[idx] < (0.35 * self.rc_spin.value())**2:
            self.drag_index = idx
            self.drag_pos[:] = world
            event.handled = True
            self.update_visuals(pos, quat)

    def on_mouse_move(self, event):
        if self.drag_index < 0:
            return
        self.drag_pos[:] = self.screen_to_world(event.pos)
        event.handled = True
        self.update_visuals()

    def on_mouse_release(self, event):
        if event.button == 1:
            self.drag_index = -1
            event.handled = True
            self.update_visuals()

    def circle_segments(self, xy, radius, nseg=72):
        angles = np.linspace(0.0, 2.0*math.pi, nseg + 1, dtype=np.float32)
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        segs = np.zeros((xy.shape[0] * nseg * 2, 2), dtype=np.float32)
        for i, p in enumerate(xy):
            base = i * nseg * 2
            pts = p[None, :] + radius * dirs
            segs[base:base + 2*nseg:2] = pts[:-1]
            segs[base + 1:base + 2*nseg:2] = pts[1:]
        return segs

    def update_visuals(self, pos=None, quat=None):
        if pos is None or quat is None:
            pos, quat = self.solver.download()
        self.last_pos = pos
        xy = pos[:, :2]
        rc = self.rc_spin.value()
        r0 = self.r0_spin.value() * rc
        if self.show_circles_check.isChecked():
            self.rc_circles.set_data(self.circle_segments(xy, rc))
            self.r0_circles.set_data(self.circle_segments(xy, r0))
        else:
            self.rc_circles.set_data(np.zeros((0, 2), dtype=np.float32))
            self.r0_circles.set_data(np.zeros((0, 2), dtype=np.float32))
        angles = 2.0 * np.arctan2(quat[:, 3], quat[:, 0])
        self.markers.set_data(xy, face_color=(1.0, 1.0, 1.0, 1.0), edge_color=(0.02, 0.02, 0.02, 0.95), size=9.0)
        segs = np.zeros((self.solver.n * 6, 2), dtype=np.float32)
        arm = 0.5 * r0
        for i in range(self.solver.n):
            a0 = angles[i] + math.pi / 3.0
            for k in range(3):
                a = a0 + 2.0 * math.pi * k / 3.0
                idx = 6*i + 2*k
                segs[idx] = xy[i]
                segs[idx + 1] = xy[i] + arm*np.array([math.cos(a), math.sin(a)], dtype=np.float32)
        self.arms.set_data(segs)
        if self.drag_index >= 0 and self.drag_index < self.solver.n:
            self.drag_line.set_data(np.array([xy[self.drag_index], self.drag_pos], dtype=np.float32))
        else:
            self.drag_line.set_data(np.zeros((0, 2), dtype=np.float32))
        cang = c_for_angular_minimum(rc, r0)
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
    window = ReactiveFFWindow(QtCore, QtWidgets)
    window.run()


if __name__ == "__main__":
    main()
