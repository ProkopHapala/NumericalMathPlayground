#!/usr/bin/env python3
"""Interactive radial function explorer with matplotlib blitting."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

# ---------------------------------------------------------------------------
# Function families (all C2 at cutoff, blunt maximum at r=0)
# x = (r/Rc)^2, so all f(x) are evaluated in terms of r^2
# ---------------------------------------------------------------------------

def f_quintic(x, alpha):
    """f(x) = (1-x)^3 * (1 + alpha*x)"""
    return (1 - x)**3 * (1 + alpha * x)

def df_quintic(x, alpha):
    """df/dx for quintic"""
    return (1 - x)**2 * (alpha - 3 - 4 * alpha * x)

def d2f_quintic(x, alpha):
    """d2f/dx2 for quintic"""
    return (1 - x) * (6 - 6 * alpha + 12 * alpha * x)


def f_bump(x, n):
    """f(x) = (1 - x^n)^3,  n >= 1"""
    xn = x**n
    return (1 - xn)**3

def df_bump(x, n):
    """df/dx for generalized bump"""
    xn = x**n
    return -3 * n * x**(n - 1) * (1 - xn)**2

def d2f_bump(x, n):
    """d2f/dx2 for generalized bump"""
    x = np.where(x == 0, 1e-12, x)
    xn = x**n
    return -3 * n * (1 - xn) * x**(n - 2) * (n - 1 - (3 * n - 1) * xn)


def f_shoulder(x, beta):
    """f(x) = (1-x)^3 * (1 - beta*x*(1-x))"""
    return (1 - x)**3 * (1 - beta * x * (1 - x))

def df_shoulder(x, beta):
    """df/dx for shoulder-tuned"""
    return (1 - x)**2 * (-3 - beta + 6 * beta * x - 5 * beta * x**2)

def d2f_shoulder(x, beta):
    """d2f/dx2 for shoulder-tuned"""
    return (1 - x) * (6 + 8 * beta - 28 * beta * x + 20 * beta * x**2)


def f_polyA(x, a):
    """f(x) = (1-x)^3 * (1 + 3x + a*x^2)"""
    return (1 - x)**3 * (1 + 3 * x + a * x**2)

def df_polyA(x, a):
    """df/dx for polynomial family A"""
    return (1 - x)**2 * ((2 * a - 12) * x - 5 * a * x**2)

def d2f_polyA(x, a):
    """d2f/dx2 for polynomial family A"""
    return (1 - x) * ((2 * a - 12) + (-16 * a + 36) * x + 20 * a * x**2)


def f_lorentz(r, Rc, W):
    """Lorentzian with C2 smoothstep envelope: f(r) = (1-(r/Rc)^2)^3 / (1 + (r/W)^2)"""
    W = max(W, 1e-6)
    x = (r / Rc)**2
    envelope = (1 - x)**3
    lorentz = 1.0 / (1.0 + (r / W)**2)
    return np.where(r <= Rc, envelope * lorentz, 0.0)

def df_lorentz(r, Rc, W):
    """df/dr for Lorentzian envelope"""
    W = max(W, 1e-6)
    x = (r / Rc)**2
    envelope = (1 - x)**3
    de = -6 * r / Rc**2 * (1 - x)**2
    lorentz = 1.0 / (1.0 + (r / W)**2)
    dl = -2 * r / W**2 / (1.0 + (r / W)**2)**2
    return np.where(r <= Rc, de * lorentz + envelope * dl, 0.0)

def d2f_lorentz(r, Rc, W):
    """d2f/dr2 for Lorentzian envelope"""
    W = max(W, 1e-6)
    x = (r / Rc)**2
    e = (1 - x)**3
    de = -6 * r / Rc**2 * (1 - x)**2
    d2e = (1 - x) * (-6 / Rc**2 + 30 * r**2 / Rc**4)
    L = 1.0 / (1.0 + (r / W)**2)
    dL = -2 * r / W**2 * L**2
    d2L = 8 * r**2 / W**4 * L**3 - 2 / W**2 * L**2
    return np.where(r <= Rc, d2e * L + 2 * de * dL + e * d2L, 0.0)


def f_lorentz_pure(r, W):
    """Pure Lorentzian L(r) = 1/(1+(r/W)^2), no cutoff envelope."""
    W = max(W, 1e-6)
    return 1.0 / (1.0 + (r / W)**2)

def df_lorentz_pure(r, W):
    W = max(W, 1e-6)
    L = 1.0 / (1.0 + (r / W)**2)
    return -2 * r / W**2 * L**2

def d2f_lorentz_pure(r, W):
    W = max(W, 1e-6)
    L = 1.0 / (1.0 + (r / W)**2)
    return 8 * r**2 / W**4 * L**3 - 2 / W**2 * L**2


def f_r2smooth(x):
    """C1 r2smooth envelope R(r) = (1-x)^2,  x=(r/Rc)^2"""
    return (1 - x)**2

def df_r2smooth(x):
    return -2 * (1 - x)

def d2f_r2smooth(x):
    return 2.0 * np.ones_like(x)


def f_combo(x, r, Rc, W):
    """L(r)*R(r) = (1-x)^2 / (1+(r/W)^2)"""
    W = max(W, 1e-6)
    return (1 - x)**2 / (1.0 + (r / W)**2)

def df_combo(x, r, Rc, W):
    W = max(W, 1e-6)
    c = (Rc / W)**2
    d = 1 + c * x
    return -2 * (1 - x) / d - c * (1 - x)**2 / d**2

def d2f_combo(x, r, Rc, W):
    W = max(W, 1e-6)
    c = (Rc / W)**2
    d = 1 + c * x
    return 2 / d + 4 * c * (1 - x) / d**2 + 2 * c**2 * (1 - x)**2 / d**3


def f_r2smooth_tuned(x, r, Rc, a):
    """R(r)*(1+a*r^2) = (1-x)^2 * (1 + a*Rc^2*x)"""
    return (1 - x)**2 * (1 + a * Rc**2 * x)

def df_r2smooth_tuned(x, r, Rc, a):
    c = a * Rc**2
    return (1 - x) * (c - 2 - 3 * c * x)

def d2f_r2smooth_tuned(x, r, Rc, a):
    c = a * Rc**2
    return 2 - 4 * c + 6 * c * x


# --- node families (Morse-like zero-crossing) ---

def f_linear_node(r, Rc, rho):
    """(1 - r/(rho*Rc)) * (1-(r/Rc)^2)^3,  node at r = rho*Rc  (needs sqrt)"""
    r_node = max(rho * Rc, 1e-12)
    E = (1 - (r / Rc)**2)**3
    return np.where(r <= Rc, (1 - r / r_node) * E, 0.0)

def df_linear_node(r, Rc, rho):
    r_node = max(rho * Rc, 1e-12)
    E = (1 - (r / Rc)**2)**3
    dE = -6 * r / Rc**2 * (1 - (r / Rc)**2)**2
    g = 1 - r / r_node
    dg = -1.0 / r_node
    return np.where(r <= Rc, dg * E + g * dE, 0.0)

def d2f_linear_node(r, Rc, rho):
    r_node = max(rho * Rc, 1e-12)
    E = (1 - (r / Rc)**2)**3
    dE = -6 * r / Rc**2 * (1 - (r / Rc)**2)**2
    d2E = (1 - (r / Rc)**2) * (-6 / Rc**2 + 30 * r**2 / Rc**4)
    g = 1 - r / r_node
    dg = -1.0 / r_node
    return np.where(r <= Rc, 2 * dg * dE + g * d2E, 0.0)


def f_quadratic_node(x, rho):
    """(1 - x/rho^2) * (1-x)^3,  node at x=rho^2  (pure r^2, no sqrt)"""
    return (1 - x / rho**2) * (1 - x)**3

def df_quadratic_node(x, rho):
    k = 1.0 / rho**2
    return -(1 - x)**2 * (3 + k - 4 * k * x)

def d2f_quadratic_node(x, rho):
    k = 1.0 / rho**2
    return 6 * (1 - x) * (1 + k - 2 * k * x)


# ---------------------------------------------------------------------------
# r-derivative helpers (chain rule: x = r^2/Rc^2)
#   f'(r) = f'(x) * 2r/Rc^2
#   f''(r) = f''(x) * 4r^2/Rc^4 + f'(x) * 2/Rc^2
# ---------------------------------------------------------------------------

def d_r(fx, dfx, r, Rc):
    """First derivative w.r.t. r."""
    return dfx * (2 * r / Rc**2)

def d2_r(fx, dfx, d2fx, r, Rc):
    """Second derivative w.r.t. r."""
    return d2fx * (4 * r**2 / Rc**4) + dfx * (2 / Rc**2)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class RadialExplorer:
    def __init__(self):
        self.fig, self.axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        self.ax_f, self.ax_df, self.ax_d2f = self.axes
        self.ax_f.set_ylabel("f(r)")
        self.ax_df.set_ylabel("f'(r)")
        self.ax_d2f.set_ylabel("f''(r)")
        self.ax_d2f.set_xlabel("r")

        for ax in self.axes:
            ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
            ax.grid(True, alpha=0.3)

        # Fixed r-grid (sufficient resolution, independent of Rc)
        self.r = np.linspace(0, 3.0, 2000)

        # Default parameters
        self.Rc = 1.5
        self.param1 = 1.0   # primary tunable (alpha / n / beta / a / W)
        self.param2 = 0.0   # secondary (unused for most)
        self.family = "quintic"

        # Placeholder lines
        (self.ln_f,) = self.ax_f.plot([], [], lw=1.5)
        (self.ln_df,) = self.ax_df.plot([], [], lw=1.5)
        (self.ln_d2f,) = self.ax_d2f.plot([], [], lw=1.5)

        # Vertical cutoff line
        self.ln_cut = self.ax_f.axvline(x=self.Rc, color="gray", ls="--", lw=1)

        # Formula annotation (updated dynamically)
        self.txt_formula = self.ax_f.text(
            0.02, 0.95, "", transform=self.ax_f.transAxes,
            fontsize=10, family="monospace", va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
        )

        self.ax_f.set_xlim(0, 3.0)
        self.ax_f.set_ylim(-0.2, 1.2)
        self.ax_df.set_ylim(-8, 4)
        self.ax_d2f.set_ylim(-20, 20)

        # ---- widgets ----
        plt.subplots_adjust(left=0.25, bottom=0.25)

        # Family selector
        self.ax_family = plt.axes([0.02, 0.45, 0.15, 0.25])
        self.radio = RadioButtons(
            self.ax_family,
            ("quintic", "bump", "shoulder", "polyA",
             "lorentz_c2", "lorentz", "r2smooth", "combo", "r2smooth_tuned",
             "linear_node", "quadratic_node"),
            active=0,
        )
        self.radio.on_clicked(self._on_family)

        # Parameter 1 slider
        self.ax_p1 = plt.axes([0.30, 0.12, 0.55, 0.03])
        self.slider_p1 = Slider(self.ax_p1, r"$\alpha$", 0.0, 6.0, valinit=self.param1)
        self.slider_p1.on_changed(self._on_param)

        # Parameter 2 slider (mostly unused)
        self.ax_p2 = plt.axes([0.30, 0.06, 0.55, 0.03])
        self.slider_p2 = Slider(self.ax_p2, r"$p_2$", 0.0, 2.0, valinit=self.param2)
        self.slider_p2.on_changed(self._on_param)

        # Rc slider
        self.ax_Rc = plt.axes([0.30, 0.18, 0.55, 0.03])
        self.slider_Rc = Slider(self.ax_Rc, r"$R_c$", 0.3, 3.0, valinit=self.Rc)
        self.slider_Rc.on_changed(self._on_param)

        self._family_configs = {
            "quintic":         {"p1": (r"$\alpha$", 0.0, 6.0, 1.0),  "p2": None, "ylim_f": (-0.2, 1.2), "ylim_df": (-8, 4),  "ylim_d2f": (-20, 20),  "tag": "[r²]",  "c2": True},
            "bump":            {"p1": ("n", 1, 5, 2),                "p2": None, "ylim_f": (-0.2, 1.2), "ylim_df": (-8, 4),  "ylim_d2f": (-20, 20),  "tag": "[r²]",  "c2": True},
            "shoulder":        {"p1": (r"$\beta$", -5.0, 5.0, 0.0), "p2": None, "ylim_f": (-0.2, 1.2), "ylim_df": (-8, 4),  "ylim_d2f": (-20, 20),  "tag": "[r²]",  "c2": True},
            "polyA":           {"p1": ("a", -6.0, 6.0, 0.0),         "p2": None, "ylim_f": (-0.2, 1.2), "ylim_df": (-8, 4),  "ylim_d2f": (-20, 20),  "tag": "[r²]",  "c2": True},
            "lorentz_c2":      {"p1": ("W", 0.1, 2.0, 0.5),          "p2": None, "ylim_f": (-0.2, 1.2), "ylim_df": (-8, 4),  "ylim_d2f": (-30, 30),  "tag": "[r²]",  "c2": True},
            "lorentz":         {"p1": ("W", 0.1, 2.0, 0.5),          "p2": None, "ylim_f": (-0.2, 1.2), "ylim_df": (-8, 4),  "ylim_d2f": (-30, 30),  "tag": "[r²]",  "c2": False},
            "r2smooth":        {"p1": None,                           "p2": None, "ylim_f": (-0.2, 1.2), "ylim_df": (-8, 4),  "ylim_d2f": (-20, 20),  "tag": "[r²]",  "c2": False},
            "combo":           {"p1": ("W", 0.1, 2.0, 0.5),          "p2": None, "ylim_f": (-0.2, 1.2), "ylim_df": (-8, 4),  "ylim_d2f": (-30, 30),  "tag": "[r²]",  "c2": False},
            "r2smooth_tuned":  {"p1": ("a", -2.0, 4.0, 0.0),         "p2": None, "ylim_f": (-0.2, 1.2), "ylim_df": (-8, 4),  "ylim_d2f": (-20, 20),  "tag": "[r²]",  "c2": False},
            "linear_node":     {"p1": (r"$\rho$", 0.1, 0.99, 0.5),  "p2": None, "ylim_f": (-1.2, 1.2), "ylim_df": (-12, 8),  "ylim_d2f": (-30, 30), "tag": "[sqrt]", "c2": True},
            "quadratic_node":  {"p1": (r"$\rho$", 0.1, 0.99, 0.5),  "p2": None, "ylim_f": (-1.2, 1.2), "ylim_df": (-12, 8),  "ylim_d2f": (-30, 30), "tag": "[r²]",  "c2": True},
        }

        # Blitting setup
        self.fig.canvas.draw()
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self._artists = [self.ln_f, self.ln_df, self.ln_d2f, self.ln_cut, self.txt_formula]
        self._blit_artists = [self.ln_f, self.ln_df, self.ln_d2f, self.ln_cut,
                              self.ax_f.yaxis, self.ax_df.yaxis, self.ax_d2f.yaxis,
                              self.ax_d2f.xaxis]

        # Initial draw
        self._update(True)
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)

    # --- callbacks ---

    def _on_draw(self, event):
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self._update(True)

    def _on_family(self, label):
        self.family = label
        cfg = self._family_configs[label]
        # reconfigure slider 1
        if cfg["p1"]:
            name, vmin, vmax, v0 = cfg["p1"]
            self.ax_p1.set_visible(True)
            self.ax_p1.clear()
            self.slider_p1 = Slider(self.ax_p1, name, vmin, vmax, valinit=v0)
            self.slider_p1.on_changed(self._on_param)
        else:
            self.ax_p1.set_visible(False)
            self.slider_p1 = type("obj", (object,), {"val": 0.0})()  # dummy
        # reconfigure slider 2 (hidden if None)
        if cfg["p2"]:
            n2, vmin2, vmax2, v02 = cfg["p2"]
            self.ax_p2.set_visible(True)
            self.ax_p2.clear()
            self.slider_p2 = Slider(self.ax_p2, n2, vmin2, vmax2, valinit=v02)
            self.slider_p2.on_changed(self._on_param)
        else:
            self.ax_p2.set_visible(False)
        # y-limits
        self.ax_f.set_ylim(*cfg["ylim_f"])
        self.ax_df.set_ylim(*cfg["ylim_df"])
        self.ax_d2f.set_ylim(*cfg["ylim_d2f"])
        self.fig.canvas.draw_idle()
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self._update(True)

    def _formula_str(self, fam, p1, Rc):
        tag = self._family_configs[fam].get("tag", "")
        c2  = "C²" if self._family_configs[fam].get("c2", False) else "C¹"
        hdr = f"{tag}  {c2}  "
        if fam == "quintic":
            return rf"{hdr}$f(r)=(1-(r/R_c)^2)^3(1+\alpha(r/R_c)^2)$    $\alpha$={p1:.3f},  $R_c$={Rc:.3f}"
        elif fam == "bump":
            n = int(round(p1))
            return rf"{hdr}$f(r)=(1-(r/R_c)^{{2n}})^3$    $n$={n},  $R_c$={Rc:.3f}"
        elif fam == "shoulder":
            return rf"{hdr}$f(r)=(1-(r/R_c)^2)^3[1-\beta(r/R_c)^2(1-(r/R_c)^2)]$    $\beta$={p1:.3f},  $R_c$={Rc:.3f}"
        elif fam == "polyA":
            return rf"{hdr}$f(r)=(1-(r/R_c)^2)^3(1+3(r/R_c)^2+a(r/R_c)^4)$    $a$={p1:.3f},  $R_c$={Rc:.3f}"
        elif fam == "lorentz_c2":
            return rf"{hdr}$f(r)=(1-(r/R_c)^2)^3/(1+(r/W)^2)$    $W$={p1:.3f},  $R_c$={Rc:.3f}"
        elif fam == "lorentz":
            return rf"{hdr}$f(r)=1/(1+(r/W)^2)$    $W$={p1:.3f},  $R_c$={Rc:.3f}"
        elif fam == "r2smooth":
            return rf"{hdr}$f(r)=(1-(r/R_c)^2)^2$    $R_c$={Rc:.3f}"
        elif fam == "combo":
            return rf"{hdr}$f(r)=(1-(r/R_c)^2)^2/(1+(r/W)^2)$    $W$={p1:.3f},  $R_c$={Rc:.3f}"
        elif fam == "r2smooth_tuned":
            return rf"{hdr}$f(r)=(1-(r/R_c)^2)^2(1+a r^2)$    $a$={p1:.3f},  $R_c$={Rc:.3f}"
        elif fam == "linear_node":
            return rf"{hdr}$f(r)=(1-r/(\rho R_c))(1-(r/R_c)^2)^3$    $\rho$={p1:.3f},  $R_c$={Rc:.3f}"
        elif fam == "quadratic_node":
            return rf"{hdr}$f(r)=(1-(r/R_c)^2/\rho^2)(1-(r/R_c)^2)^3$    $\rho$={p1:.3f},  $R_c$={Rc:.3f}"
        return ""

    def _on_param(self, val=None):
        self._update()

    def _update(self, full=False):
        Rc = self.slider_Rc.val
        p1 = self.slider_p1.val
        fam = self.family

        x = (self.r / Rc)**2
        x = np.clip(x, 0, 1)  # enforce support

        if fam == "quintic":
            fx = f_quintic(x, p1)
            dfx = df_quintic(x, p1)
            d2fx = d2f_quintic(x, p1)
            f = fx
            df = d_r(fx, dfx, self.r, Rc)
            d2f = d2_r(fx, dfx, d2fx, self.r, Rc)
        elif fam == "bump":
            n = int(round(p1))
            n = max(1, n)
            fx = f_bump(x, n)
            dfx = df_bump(x, n)
            d2fx = d2f_bump(x, n)
            f = fx
            df = d_r(fx, dfx, self.r, Rc)
            d2f = d2_r(fx, dfx, d2fx, self.r, Rc)
        elif fam == "shoulder":
            fx = f_shoulder(x, p1)
            dfx = df_shoulder(x, p1)
            d2fx = d2f_shoulder(x, p1)
            f = fx
            df = d_r(fx, dfx, self.r, Rc)
            d2f = d2_r(fx, dfx, d2fx, self.r, Rc)
        elif fam == "polyA":
            fx = f_polyA(x, p1)
            dfx = df_polyA(x, p1)
            d2fx = d2f_polyA(x, p1)
            f = fx
            df = d_r(fx, dfx, self.r, Rc)
            d2f = d2_r(fx, dfx, d2fx, self.r, Rc)
        elif fam == "lorentz_c2":
            W = p1
            f = f_lorentz(self.r, Rc, W)
            df = df_lorentz(self.r, Rc, W)
            d2f = d2f_lorentz(self.r, Rc, W)
        elif fam == "lorentz":
            W = p1
            f = f_lorentz_pure(self.r, W)
            df = df_lorentz_pure(self.r, W)
            d2f = d2f_lorentz_pure(self.r, W)
        elif fam == "r2smooth":
            fx = f_r2smooth(x)
            dfx = df_r2smooth(x)
            d2fx = d2f_r2smooth(x)
            f = fx
            df = d_r(fx, dfx, self.r, Rc)
            d2f = d2_r(fx, dfx, d2fx, self.r, Rc)
        elif fam == "combo":
            W = p1
            fx = f_combo(x, self.r, Rc, W)
            dfx = df_combo(x, self.r, Rc, W)
            d2fx = d2f_combo(x, self.r, Rc, W)
            f = fx
            df = d_r(fx, dfx, self.r, Rc)
            d2f = d2_r(fx, dfx, d2fx, self.r, Rc)
        elif fam == "r2smooth_tuned":
            a = p1
            fx = f_r2smooth_tuned(x, self.r, Rc, a)
            dfx = df_r2smooth_tuned(x, self.r, Rc, a)
            d2fx = d2f_r2smooth_tuned(x, self.r, Rc, a)
            f = fx
            df = d_r(fx, dfx, self.r, Rc)
            d2f = d2_r(fx, dfx, d2fx, self.r, Rc)
        elif fam == "linear_node":
            rho = p1
            f = f_linear_node(self.r, Rc, rho)
            df = df_linear_node(self.r, Rc, rho)
            d2f = d2f_linear_node(self.r, Rc, rho)
        elif fam == "quadratic_node":
            rho = p1
            fx = f_quadratic_node(x, rho)
            dfx = df_quadratic_node(x, rho)
            d2fx = d2f_quadratic_node(x, rho)
            f = fx
            df = d_r(fx, dfx, self.r, Rc)
            d2f = d2_r(fx, dfx, d2fx, self.r, Rc)
        else:
            raise ValueError(f"unknown family: {fam}")

        # zero outside cutoff (skip for pure Lorentzian which has no cutoff)
        if fam not in ("lorentz",):
            mask = self.r > Rc
            f[mask] = 0.0
            df[mask] = 0.0
            d2f[mask] = 0.0

        self.ln_f.set_data(self.r, f)
        self.ln_df.set_data(self.r, df)
        self.ln_d2f.set_data(self.r, d2f)
        self.ln_cut.set_xdata([Rc, Rc])
        self.txt_formula.set_text(self._formula_str(fam, p1, Rc))

        if full:
            self.fig.canvas.draw_idle()
        else:
            self.fig.canvas.restore_region(self.bg)
            for art in self._artists:
                self.ax_f.draw_artist(art) if art.axes == self.ax_f else None
                self.ax_df.draw_artist(art) if art.axes == self.ax_df else None
                self.ax_d2f.draw_artist(art) if art.axes == self.ax_d2f else None
            # redraw axes that might change
            for ax in self.axes:
                ax.draw_artist(ax.xaxis)
                ax.draw_artist(ax.yaxis)
            self.fig.canvas.blit(self.fig.bbox)
            self.fig.canvas.flush_events()


if __name__ == "__main__":
    app = RadialExplorer()
    plt.show()
