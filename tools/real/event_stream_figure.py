"""Space-time (x, y, t) rendering of a real DAVIS346 event stream, for the paper's event-input figure.

A small software renderer (CPU only): every event is an anti-aliased disc in a perspective view of
the (x, y, t) volume, composited back to front. A few time slices are drawn as translucent panes
carrying the events of the last `slice_ms` before them; thin rails join the pane corners and a
dashed arrow marks the time axis.

    python tools/real/event_stream_figure.py --layout events      # the paper figure -> docs/assets/event_stream_input_20261113/
    python tools/real/event_stream_figure.py --layout diagonal --set width=2000 slices='[35.05, 35.44]'

Layout `events` is the paper figure: the stream is one continuous cloud of events (no intermediate
slices, which read as sampled frames) and a single projection of the newest events closes the
space-time box, seen nearly face-on so that the scene can be recognised. Layout `lens` is its earlier
version: a more oblique view plus a magnifier that draws the events of one small patch one by one, each
on its pixel at its own time.

Events are raw sensor events of one recording (no undistortion). For display only, events with
fewer than `support_min` other events on the 8 neighbouring pixels within about +-5 ms are not drawn
and weakly supported ones are drawn lighter; nothing else is changed. Red = ON, blue = OFF.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

W, H = 346, 260
ROOT = Path("/data1/lyq/data/hand_data_real/data_collection")
OUT = Path(__file__).resolve().parents[2] / "docs" / "assets" / "event_stream_input_20261113"
FONT = "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Italic.otf"

DEFAULT = dict(
    # data: session, window [t_start, t_end) in seconds from the first event, pane times, pane accumulation
    session="ch_global", stream="event", t_start=34.25, t_end=35.42, slices=[34.72, 35.05, 35.42], slice_ms=35,
    # camera: time axis `length` in sensor pixels; up = "y" time recedes into depth, "z" time is vertical
    length=900, up="y", az=-33, el=22, dist=3600, roll=0, width=3200, margin=60, ss=2,
    # display filter
    support_dt_us=5000, support_min=3, support_lo=3, support_hi=9, support_floor=0.25, pane_support_min=2,
    # event cloud
    red=(222, 45, 38), blue=(33, 88, 214), dot=1.2, alpha=1.0, fade=0.8, fade_gamma=1.0, tint=0.08, tail=0.18,
    sublayers=24,
    # panes
    tex_scale=8, pane_fill=(0.955, 0.962, 0.98), pane_alpha=0.55, pane_alpha_last=0.86, pane_dot=1.2,
    border_px=1.0, border_col=(64, 68, 82), border_alpha=0.95,
    # rails between pane corners (sizes below are for a 1400 px wide figure and scale with `width`)
    rails=True, rails_tail=170, rails_lw=1.0, rails_col=(64, 68, 82), rails_alpha=0.38,
    # time axis
    axis=True, axis_xy=(-W / 2 - 16, -H / 2 - 16), axis_z0=120, axis_z1=70, axis_pad=40, axis_dash=(16, 11),
    axis_lw=2.4, axis_head=7.5, axis_col=(64, 68, 82), axis_alpha=0.95, axis_label="t", axis_font=46,
    crop_margin=28, crop_alpha=0.06, pad=(0, 0, 0, 0),              # extra canvas left, top, right, bottom
    # magnifier: the events of pixels x0..x1, y0..y1 during the last `ms` before the last pane, one bead each
    lens=None, lens_d=0.27, lens_pos=(-0.50, -0.88), lens_fill=(0.972, 0.977, 0.99), lens_bead=0.78,
    lens_fit=0.43, lens_shift=(0.0, -0.045), lens_lw=1.7,
    # text of the `events` layout (em sizes for a 1400 px wide figure)
    text_col=(64, 68, 82), text_em=44, formula=r"$e_i=(x_i,\,y_i,\,t_i,\,p_i)$", formula_em=41, event_label=r"$e_i$",
    event_em=36, event_at=(0.1, 0.62), xy_labels=False, legend=False, legend_em=36,
)
LAYOUTS = {
    # time flows from the far upper left to the near lower right; the newest pane is in front
    "diagonal": dict(),
    # time is vertical and the panes are stacked like floors, the newest on top
    "stack": dict(up="z", az=-28, el=48, length=500, axis_xy=(-W / 2 - 22, H / 2 + 22), axis_z0=60, axis_z1=60,
                  rails_tail=110),
    # one projection of the newest events closes the box, seen nearly face-on so that the scene reads at a glance
    "events": dict(slices=[35.42], length=880, az=-20, el=15, rails_tail=770, pane_alpha_last=0.92, axis_label="",
                   xy_labels=True, legend=True, pad=(0, 0, 60, 0)),      # room on the right for the letter y
    # the earlier version of `events`: a more oblique view and a magnifier that shows single events
    "lens": dict(slices=[35.42], length=720, az=-30, rails_tail=610, axis_label="", xy_labels=True, legend=True,
                 lens=dict(x0=211, x1=218, y0=120, y1=126, ms=12), pad=(0, 330, 110, 0)),
}


# ----------------------------------------------------------------------------- data
def load_session(session, stream="event", cache=None):
    if cache and Path(cache).exists():
        z = np.load(cache)
        return z["t"], z["x"].astype(np.int32), z["y"].astype(np.int32), z["p"].astype(np.uint8)
    ts, xs, ys, ps = [], [], [], []
    with open(ROOT / session / stream / "index.jsonl") as f:
        for rec in map(json.loads, f):
            z = np.load(ROOT / session / rec["file"])
            ts.append(z["t_us"]); xs.append(z["x"]); ys.append(z["y"]); ps.append(z["p"])
    return np.concatenate(ts), np.concatenate(xs), np.concatenate(ys), np.concatenate(ps).astype(np.uint8)


def support(t, x, y, dt_us):
    """Number of other events on the 8 neighbouring pixels in the same and the two adjacent time bins."""
    k = ((t - t[0]) // dt_us).astype(np.int64)
    vol = np.zeros((H, W, int(k.max()) + 1), np.float32)
    np.add.at(vol, (y, x, k), 1)
    box = ndimage.uniform_filter(vol, size=(3, 3, 3), mode="constant") * 27
    own = ndimage.uniform_filter(vol, size=(1, 1, 3), mode="constant") * 3
    return np.rint((box - own)[y, x, k])


def smooth(q):
    q = np.clip(q, 0, 1)
    return q * q * (3 - 2 * q)


# ----------------------------------------------------------------------------- camera
class Camera:
    def __init__(self, az, el, dist, roll=0.0, up="y"):
        az, el, roll = np.deg2rad([az, el, roll])
        ex, ey, ez = np.eye(3)
        e1, e2, e3 = (ex, ez, ey) if up == "y" else (ex, -ey, ez)
        d = np.sin(az) * np.cos(el) * e1 + np.cos(az) * np.cos(el) * e2 + np.sin(el) * e3
        self.C = d * dist
        f = -d
        r = np.cross(f, e3); r /= np.linalg.norm(r)
        u = np.cross(r, f)
        c, s = np.cos(roll), np.sin(roll)
        self.r, self.u, self.f = c * r + s * u, -s * r + c * u, f
        self.scale, self.off = 1.0, np.zeros(2)

    def raw(self, P):
        Q = np.asarray(P, np.float64) - self.C
        z = Q @ self.f
        return (Q @ self.r) / z, -(Q @ self.u) / z, z

    def fit(self, pts, width, margin):
        a, b, _ = self.raw(pts)
        self.scale = (width - 2 * margin) / (a.max() - a.min())
        self.off = np.array([margin - a.min() * self.scale, margin - b.min() * self.scale])
        return int(round(width)), int(round((b.max() - b.min()) * self.scale + 2 * margin))

    def project(self, P):
        a, b, z = self.raw(P)
        return a * self.scale + self.off[0], b * self.scale + self.off[1], z


# ----------------------------------------------------------------------------- rasteriser
def over(dst, src):
    """Premultiplied (4, H, W) arrays: src over dst, in place."""
    dst *= (1 - src[3])[None]
    dst += src


class Layer:
    """Order-independent accumulation of soft discs inside a window: alpha = 1 - prod(1 - w)."""

    def __init__(self, hw, origin=(0, 0)):
        self.hw, self.origin = hw, origin
        n = hw[0] * hw[1]
        self.loga, self.sw, self.sc = np.zeros(n), np.zeros(n), np.zeros((3, n))

    def discs(self, u, v, rad, col, alpha, batch=3_000_000):
        Hc, Wc = self.hw
        u, v = u - self.origin[1], v - self.origin[0]
        order = np.argsort(rad)
        u, v, rad, col, alpha = u[order], v[order], rad[order], col[order], alpha[order]
        i, n = 0, self.loga.size
        while i < len(u):
            R = int(np.ceil(rad[i] + 0.5))
            j = min(int(np.searchsorted(rad, R + 0.5 - 1e-9, side="right")), i + max(1, batch // (2 * R + 1) ** 2))
            j = max(j, i + 1)
            R = int(np.ceil(rad[j - 1] + 0.5))
            o = np.arange(-R, R + 1)
            oy, ox = (g.ravel() for g in np.meshgrid(o, o, indexing="ij"))
            px = np.floor(u[i:j]).astype(np.int64)[:, None] + ox[None]
            py = np.floor(v[i:j]).astype(np.int64)[:, None] + oy[None]
            dist = np.hypot(px + 0.5 - u[i:j, None], py + 0.5 - v[i:j, None])
            w = np.clip(rad[i:j, None] + 0.5 - dist, 0, 1) * alpha[i:j, None]
            ok = (w > 0) & (px >= 0) & (px < Wc) & (py >= 0) & (py < Hc)
            idx = (py * Wc + px)[ok]
            w = np.minimum(w[ok], 0.999)
            self.loga += np.bincount(idx, np.log1p(-w), minlength=n)
            self.sw += np.bincount(idx, w, minlength=n)
            cc = np.broadcast_to(col[i:j, None, :], ok.shape + (3,))[ok]
            for c in range(3):
                self.sc[c] += np.bincount(idx, w * cc[:, c], minlength=n)
            i = j

    def rgba(self):
        a = 1 - np.exp(self.loga)
        rgb = self.sc / np.maximum(self.sw, 1e-9)
        return np.concatenate([rgb * a, a[None]], 0).reshape(4, *self.hw).astype(np.float32)


def draw_discs(canvas, u, v, rad, col, alpha):
    """Composite one layer of discs over the canvas, touching only its bounding box."""
    if len(u) == 0:
        return
    Hc, Wc = canvas.shape[1:]
    m = int(np.ceil(rad.max())) + 2
    x0, x1 = max(int(u.min()) - m, 0), min(int(u.max()) + m + 1, Wc)
    y0, y1 = max(int(v.min()) - m, 0), min(int(v.max()) + m + 1, Hc)
    if x1 <= x0 or y1 <= y0:
        return
    lay = Layer((y1 - y0, x1 - x0), (y0, x0))
    lay.discs(u, v, rad, col, alpha)
    over(canvas[:, y0:y1, x0:x1], lay.rgba())


def line(mask, a, b, lw, value=1.0):
    cv2.line(mask, (int(round(a[0] * 16)), int(round(a[1] * 16))), (int(round(b[0] * 16)), int(round(b[1] * 16))),
             float(value), max(1, int(round(lw))), cv2.LINE_AA, shift=4)


def tinted(mask, col, alpha):
    col = np.array(col, np.float32) / 255
    mask = np.clip(mask, 0, 1) * alpha
    return np.concatenate([col[:, None, None] * mask[None], mask[None]], 0)


# ----------------------------------------------------------------------------- panes, rails, axis
def pane_texture(x, y, p, cfg, alpha, pad):
    """Event frame on a translucent pane: premultiplied RGBA texture with `pad` texels of margin."""
    ts = cfg["tex_scale"]
    h, w = H * ts + 2 * pad, W * ts + 2 * pad
    tex = np.zeros((4, h, w), np.float32)
    tex[:, pad:pad + H * ts, pad:pad + W * ts] = np.array(list(cfg["pane_fill"]) + [1.0], np.float32)[:, None, None] * alpha
    red, blue = (np.array(cfg[k], np.float32) / 255 for k in ("red", "blue"))
    col = np.where(p[:, None] == 1, red[None], blue[None]).astype(np.float32)
    key = (y.astype(np.int64) * W + x) * 2 + p              # one disc per (pixel, polarity)
    _, first, cnt = np.unique(key, return_index=True, return_counts=True)
    draw_discs(tex, (x[first] + 0.5) * ts + pad, (y[first] + 0.5) * ts + pad,
               np.full(len(first), cfg["pane_dot"] * ts / 2), col[first], 0.45 + 0.55 * (1 - np.exp(-cnt / 1.0)))
    bw = cfg["border_px"] * ts
    if bw > 0:
        b = np.zeros((h, w), np.float32)
        r = bw / 2
        cv2.rectangle(b, (int(round(pad - r)), int(round(pad - r))),
                      (int(round(pad + W * ts + r)) - 1, int(round(pad + H * ts + r)) - 1), 1.0, int(round(bw)), cv2.LINE_AA)
        over(tex, tinted(b, cfg["border_col"], cfg["border_alpha"]))
    return tex


def warp_pane(tex, quad, hw, ts, pad):
    src = np.float32([[pad, pad], [pad + W * ts, pad], [pad + W * ts, pad + H * ts], [pad, pad + H * ts]])
    M = cv2.getPerspectiveTransform(src, np.float32(quad))
    out = cv2.warpPerspective(np.moveaxis(tex, 0, -1), M, (hw[1], hw[0]), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.moveaxis(out, -1, 0)


def rails_layer(cam, cfg, hw, corners, za, zb, fade_in):
    """Lines along time through the given pane corners, from za to zb; `fade_in` ramps them up from nothing."""
    k = cfg["ss"] * cfg["width"] / 1400.0
    mask = np.zeros(hw, np.float32)
    n = 24 if fade_in else 1
    for cx, cy in corners:
        zs = np.linspace(za, zb, n + 1)
        uu, vv, _ = cam.project(np.stack([np.full(n + 1, cx), np.full(n + 1, cy), zs], -1))
        for i in range(n):
            piece = np.zeros(hw, np.float32)
            line(piece, (uu[i], vv[i]), (uu[i + 1], vv[i + 1]), cfg["rails_lw"] * k, smooth((i + 1) / n) if fade_in else 1.0)
            mask = np.maximum(mask, piece)
    return tinted(mask, cfg["rails_col"], cfg["rails_alpha"])


def axis_layer(cam, cfg, hw, L):
    """Dashed time axis with an arrow head and a label."""
    k = cfg["ss"] * cfg["width"] / 1400.0
    mask = np.zeros(hw, np.float32)
    ax, ay = cfg["axis_xy"]
    z0, z1 = -L / 2 + cfg["axis_z0"], L / 2 + cfg["axis_z1"]
    on, off = cfg["axis_dash"]
    z = z0
    while z < z1 - 1e-6:
        e = min(z + on, z1)
        uu, vv, _ = cam.project(np.array([[ax, ay, z], [ax, ay, e]]))
        line(mask, (uu[0], vv[0]), (uu[1], vv[1]), cfg["axis_lw"] * k)
        z = e + off
    uu, vv, _ = cam.project(np.array([[ax, ay, z1 - 1], [ax, ay, z1]]))
    d = np.array([uu[1] - uu[0], vv[1] - vv[0]]); d /= np.linalg.norm(d)
    nrm = np.array([-d[1], d[0]])
    head = cfg["axis_head"] * k
    tip = np.array([uu[1], vv[1]]) + d * head * 0.9
    base = tip - d * head * 2.6
    cv2.fillPoly(mask, [np.round(np.array([tip, base + nrm * head, base - nrm * head]) * 16).astype(np.int32)], 1.0,
                 cv2.LINE_AA, shift=4)
    size = cfg["axis_font"] * k
    pos = tip + d * size * 0.6
    if cfg["axis_label"]:
        img = Image.new("L", (hw[1], hw[0]), 0)
        ImageDraw.Draw(img).text((float(pos[0]), float(pos[1])), cfg["axis_label"], fill=255,
                                 font=ImageFont.truetype(FONT, int(size)), anchor="mm")
        mask = np.maximum(mask, np.asarray(img, np.float32) / 255)
    return tinted(mask, cfg["axis_col"], cfg["axis_alpha"]), pos


# ----------------------------------------------------------------------------- magnifier and text
def math_mask(text, em):
    """Alpha mask of a mathtext string set at `em` pixels (STIX, close to Times)."""
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    with matplotlib.rc_context({"mathtext.fontset": "stix"}):
        fig = Figure(figsize=(max(len(text), 6) * em * 0.8 / 100, em * 3.4 / 100), dpi=100)
        fig.patch.set_alpha(0)
        fig.text(0.5, 0.5, text, fontsize=em * 0.72, color="black", ha="center", va="center")
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        a = np.asarray(canvas.buffer_rgba())[..., 3].astype(np.float32) / 255
    ys, xs = np.nonzero(a > 0)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def paste(canvas, patch, x0, y0):
    """Premultiplied patch over the canvas with its top-left corner at (x0, y0), clipped."""
    Hc, Wc = canvas.shape[1:]
    x0, y0 = int(round(x0)), int(round(y0))
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + patch.shape[2], Wc), min(y0 + patch.shape[1], Hc)
    if xb > xa and yb > ya:
        over(canvas[:, ya:yb, xa:xb], patch[:, ya - y0:yb - y0, xa - x0:xb - x0])


def put_text(canvas, text, em, cx, cy, col, anchor="c"):
    m = math_mask(text, em)
    x0 = cx - m.shape[1] / 2 if anchor == "c" else cx if anchor == "l" else cx - m.shape[1]
    paste(canvas, tinted(m, col, 1.0), x0, cy - m.shape[0] / 2)


def bead_sprite(R, base):
    """A small shaded sphere, premultiplied, with its half size."""
    n = int(np.ceil(R)) + 2
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1].astype(np.float32)
    d = np.hypot(xx, yy)
    a = np.clip(R + 0.5 - d, 0, 1)
    nx, ny, nz = xx / R, yy / R, np.sqrt(np.clip(1 - (d / R) ** 2, 0, 1))
    l = np.array([-0.45, -0.60, 0.66]); l /= np.linalg.norm(l)
    hv = l + np.array([0, 0, 1.0]); hv /= np.linalg.norm(hv)
    diff = np.clip(nx * l[0] + ny * l[1] + nz * l[2], 0, 1)
    spec = np.clip(nx * hv[0] + ny * hv[1] + nz * hv[2], 0, 1) ** 40
    rgb = np.asarray(base, np.float32)[:, None, None] * (0.62 + 0.50 * diff)[None] + 0.55 * spec[None]
    rgb = rgb * (1 - 0.18 * np.clip((d - (R - 1.6)) / 1.6, 0, 1))[None]
    return np.concatenate([np.clip(rgb, 0, 1) * a[None], a[None]], 0), n


def stroke(canvas, a, b, lw, col, alpha=1.0):
    """One anti-aliased line segment composited over the canvas inside its bounding box."""
    pad = int(lw) + 3
    x0, y0 = int(np.floor(min(a[0], b[0]))) - pad, int(np.floor(min(a[1], b[1]))) - pad
    w, h = int(np.ceil(max(a[0], b[0]))) + pad - x0 + 1, int(np.ceil(max(a[1], b[1]))) + pad - y0 + 1
    m = np.zeros((h, w), np.float32)
    line(m, (a[0] - x0, a[1] - y0), (b[0] - x0, b[1] - y0), lw)
    paste(canvas, np.concatenate([np.asarray(col, np.float32)[:, None, None] * m[None], m[None]], 0) * alpha, x0, y0)


def ring_mask(hw, c, r, lw):
    yy, xx = np.mgrid[0:hw[0], 0:hw[1]].astype(np.float32)
    return np.clip(lw / 2 + 0.5 - np.abs(np.hypot(xx + 0.5 - c[0], yy + 0.5 - c[1]) - r), 0, 1)


def lens_disc(cfg, cam, zt, ev, box, D):
    """Magnified view of one patch: the pixel grid at the oldest time and one bead per event on a stem along time.

    ev = (x, y, sec, p) of the events of the patch, box = (x0, x1, y0, y1, ta, tb).
    Returns the premultiplied disc (4, D, D) and, in disc pixels, the centre and radius of the marked event.
    """
    x, y, s, p = ev
    x0, x1, y0, y1, ta, tb = box
    world = lambda xx, yy, zz: np.stack(np.broadcast_arrays(np.asarray(xx, float) - W / 2, H / 2 - np.asarray(yy, float),
                                                            np.asarray(zz, float)), -1)
    corners = world([x0, x1 + 1, x0, x1 + 1] * 2, [y0, y0, y1 + 1, y1 + 1] * 2, [zt(ta)] * 4 + [zt(tb)] * 4)
    c0 = corners.mean(0)
    proj = lambda P: ((P - c0) @ cam.r, -((P - c0) @ cam.u), (P - c0) @ cam.f)      # same orientation as the main view
    a, b, _ = proj(corners)
    ctr = np.array([(a.max() + a.min()) / 2, (b.max() + b.min()) / 2])
    S = cfg["lens_fit"] * D / np.hypot(a - ctr[0], b - ctr[1]).max()
    org = np.array([D / 2 + cfg["lens_shift"][0] * D, D / 2 + cfg["lens_shift"][1] * D])

    def to_px(P):
        a, b, d = proj(P)
        return org[0] + (a - ctr[0]) * S, org[1] + (b - ctr[1]) * S, d

    slate = np.array(cfg["axis_col"], np.float32) / 255
    can = np.zeros((4, D, D), np.float32)
    can[:3] = np.asarray(cfg["lens_fill"], np.float32)[:, None, None]
    can[3] = 1.0
    zp = zt(ta)
    quad = [to_px(world(cx, cy, zp)) for cx, cy in ((x0, y0), (x1 + 1, y0), (x1 + 1, y1 + 1), (x0, y1 + 1))]
    m = np.zeros((D, D), np.float32)
    cv2.fillPoly(m, [np.round(np.array([[q[0], q[1]] for q in quad]) * 16).astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
    over(can, np.concatenate([np.array([0.925, 0.937, 0.965], np.float32)[:, None, None] * m[None], m[None]], 0))
    for g in range(x0, x1 + 2):
        stroke(can, to_px(world(g, y0, zp)), to_px(world(g, y1 + 1, zp)), 0.0026 * D, slate * 0.38 + 0.62)
    for g in range(y0, y1 + 2):
        stroke(can, to_px(world(x0, g, zp)), to_px(world(x1 + 1, g, zp)), 0.0026 * D, slate * 0.38 + 0.62)
    red, blue = (np.array(cfg[k], np.float32) / 255 for k in ("red", "blue"))
    R = cfg["lens_bead"] / 2 * S
    prims, centres = [], []
    for i in range(len(x)):
        base = red if p[i] == 1 else blue
        n = max(1, int(np.ceil((zt(s[i]) - zp) / 0.35)))
        px, py, dd = to_px(world(x[i] + 0.5, y[i] + 0.5, np.linspace(zp, zt(s[i]), n + 1)))
        for j in range(n):                                  # stem pieces, so that nearer beads hide them correctly
            prims.append(((dd[j] + dd[j + 1]) / 2, (px[j], py[j]), (px[j + 1], py[j + 1]), base))
        prims.append((dd[-1] - 0.45 * cfg["lens_bead"], (px[-1], py[-1]), None, base))
        centres.append((px[-1], py[-1]))
    sprites = {1: bead_sprite(R, red), 0: bead_sprite(R, blue)}
    for d, pa, pb, base in sorted(prims, key=lambda q: -q[0]):
        if pb is not None:
            stroke(can, pa, pb, 0.0042 * D, base * 0.42 + 0.58)
        else:
            sp, n = sprites[1 if base is red else 0]
            paste(can, sp, round(pa[0]) - n, round(pa[1]) - n)
    centres = np.array(centres)
    mark = int(np.argmax(centres[:, 0] + 0.35 * centres[:, 1]))        # the bead furthest to the lower right
    k = cfg["ss"] * cfg["width"] / 1400.0
    ax, ay = x0 - 0.9, y1 + 1 + 1.3                                    # time arrow beside the grid, parallel to the stems
    a0, a1 = to_px(world(ax, ay, zp)), to_px(world(ax, ay, zt(tb)))
    dvec = np.array([a1[0] - a0[0], a1[1] - a0[1]]); dvec /= np.linalg.norm(dvec)
    nrm, head = np.array([-dvec[1], dvec[0]]), 4.2 * k
    tip = np.array([a1[0], a1[1]])
    stroke(can, a0, tip - dvec * head * 2.0, cfg["lens_lw"] * k * 0.8, slate)
    tri = np.zeros((D, D), np.float32)
    cv2.fillPoly(tri, [np.round(np.array([tip, tip - dvec * head * 2.6 + nrm * head, tip - dvec * head * 2.6 - nrm * head]) * 16)
                       .astype(np.int32)], 1.0, cv2.LINE_AA, shift=4)
    over(can, tinted(tri, cfg["axis_col"], 1.0))
    t_at = tip + dvec * 0.55 * cfg["event_em"] * k
    over(can, tinted(ring_mask((D, D), centres[mark], 1.75 * R, cfg["lens_lw"] * k), cfg["axis_col"], 1.0))
    over(can, tinted(ring_mask((D, D), (D / 2, D / 2), D / 2 - 1 - cfg["lens_lw"] * k, 2 * cfg["lens_lw"] * k),
                     cfg["axis_col"], 1.0))
    yy, xx = np.mgrid[0:D, 0:D].astype(np.float32)
    can *= np.clip(D / 2 - np.hypot(xx + 0.5 - D / 2, yy + 0.5 - D / 2), 0, 1)[None]
    return can, centres[mark], R, t_at


def tangents(c1, r1, c2, r2):
    """The two outer tangent segments between two circles."""
    d = np.asarray(c2, float) - np.asarray(c1, float)
    th, ph = np.arctan2(d[1], d[0]), np.arccos(np.clip((r1 - r2) / np.linalg.norm(d), -1, 1))
    out = []
    for sgn in (1, -1):
        e = np.array([np.cos(th + sgn * ph), np.sin(th + sgn * ph)])
        out.append((np.asarray(c1) + r1 * e, np.asarray(c2) + r2 * e))
    return out


# ----------------------------------------------------------------------------- render
def frame(cfg):
    """The view of a configuration: camera (fitted, padded), time -> depth map, canvas size (supersampled pixels).

    Returns (cam, zt, L, span, (Wc, Hc), k14); cam.project maps (x - W/2, H/2 - y, zt(seconds)) to canvas pixels."""
    L, span = cfg["length"], cfg["t_end"] - cfg["t_start"]
    zt = lambda s: (s - cfg["t_start"]) / span * L - L / 2
    cam = Camera(cfg["az"], cfg["el"], cfg["dist"], cfg["roll"], cfg["up"])
    fit_pts = np.array([[sx * W / 2, sy * H / 2, sz * L / 2] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    if cfg["axis"]:
        ax, ay = cfg["axis_xy"]
        fit_pts = np.concatenate([fit_pts, [[ax, ay, -L / 2 + cfg["axis_z0"]],
                                            [ax, ay, L / 2 + cfg["axis_z1"] + cfg["axis_pad"]]]])
    ss = cfg["ss"]
    Wc, Hc = cam.fit(fit_pts, cfg["width"] * ss, cfg["margin"] * ss)
    k14 = ss * cfg["width"] / 1400.0                        # annotation sizes are given for a 1400 px wide figure
    pl, pt, pr, pb = (int(round(v * k14)) for v in cfg["pad"])
    cam.off = cam.off + np.array([pl, pt])
    return cam, zt, L, span, (Wc + pl + pr, Hc + pt + pb), k14


def render(cfg, events):
    t, x, y, p = events
    t0 = t[0]
    lo, hi = np.searchsorted(t, [t0 + int((cfg["t_start"] - 0.05) * 1e6), t0 + int((cfg["t_end"] + 0.05) * 1e6)])
    T, X, Y, P = (v[lo:hi] for v in (t, x, y, p))
    sec = (T - t0) / 1e6
    sup = support(T, X, Y, cfg["support_dt_us"])
    strength = cfg["support_floor"] + (1 - cfg["support_floor"]) * smooth(
        (sup - cfg["support_lo"]) / (cfg["support_hi"] - cfg["support_lo"]))
    inwin = (sec >= cfg["t_start"]) & (sec < cfg["t_end"])
    sel = inwin & (sup >= cfg["support_min"])
    stats = dict(events_in_window=int(inwin.sum()), events_drawn=int(sel.sum()), slices={})

    cam, zt, L, span, (Wc, Hc), k14 = frame(cfg)
    ss = cfg["ss"]
    canvas = np.zeros((4, Hc, Wc), np.float32)

    ex, ey, es, ep = X[sel].astype(np.float64), Y[sel].astype(np.float64), sec[sel], P[sel]
    u, v, z = cam.project(np.stack([ex + 0.5 - W / 2, H / 2 - (ey + 0.5), zt(es)], -1))
    near = (es - cfg["t_start"]) / span                       # 0 oldest .. 1 newest
    rad = cfg["dot"] / 2 * cam.scale / z
    alpha = cfg["alpha"] * (cfg["fade"] + (1 - cfg["fade"]) * near ** cfg["fade_gamma"]) * strength[sel]
    alpha = alpha * smooth(near / max(cfg["tail"], 1e-6))     # the stream emerges from nothing
    red, blue = (np.array(cfg[k], np.float32) / 255 for k in ("red", "blue"))
    col = np.where(ep[:, None] == 1, red[None], blue[None]).astype(np.float32)
    tint = (cfg["tint"] * (1 - near))[:, None]
    col = col * (1 - tint) + tint

    slices = sorted(cfg["slices"])
    ts, pad = cfg["tex_scale"], 8
    corners = [(sx * W / 2, sy * H / 2) for sx in (-1, 1) for sy in (-1, 1)]
    depth = lambda c: cam.project(np.array([[c[0], c[1], 0.0]]))[2][0]
    mid = depth((0, 0))
    far, nearc = [c for c in corners if depth(c) > mid], [c for c in corners if depth(c) <= mid]
    seg = np.searchsorted(np.array(slices), es, side="right")
    for k in range(len(slices) + 1):
        draw_rails = cfg["rails"] and k < len(slices) and (k > 0 or cfg["rails_tail"] > 0)
        if draw_rails:
            za = zt(slices[k - 1]) if k > 0 else zt(slices[0]) - cfg["rails_tail"]
            over(canvas, rails_layer(cam, cfg, (Hc, Wc), far, za, zt(slices[k]), k == 0))
        idx = np.flatnonzero(seg == k)
        if len(idx):
            idx = idx[np.argsort(-z[idx])]
            for part in np.array_split(idx, cfg["sublayers"]):
                draw_discs(canvas, u[part], v[part], rad[part], col[part], alpha[part])
        if draw_rails:
            over(canvas, rails_layer(cam, cfg, (Hc, Wc), nearc, za, zt(slices[k]), k == 0))
        if k < len(slices):
            s = slices[k]
            w0 = (sec >= s - cfg["slice_ms"] / 1000) & (sec < s) & (sup >= cfg["pane_support_min"])
            last = k == len(slices) - 1
            tex = pane_texture(X[w0], Y[w0], P[w0], cfg, cfg["pane_alpha_last"] if last else cfg["pane_alpha"], pad)
            cz = zt(s)
            qu, qv, _ = cam.project(np.array([[-W / 2, H / 2, cz], [W / 2, H / 2, cz], [W / 2, -H / 2, cz], [-W / 2, -H / 2, cz]]))
            over(canvas, warp_pane(tex, np.stack([qu, qv], -1), (Hc, Wc), ts, pad))
            stats["slices"]["%.3f" % s] = int(w0.sum())
    overlays, t_pos = {}, None
    if cfg["axis"]:
        overlays["axis"], t_pos = axis_layer(cam, cfg, (Hc, Wc), L)
    text = np.zeros((4, Hc, Wc), np.float32)
    col, zf = cfg["text_col"], zt(slices[-1]) if slices else L / 2
    if cfg["lens"]:
        ln = cfg["lens"]
        tb = slices[-1]
        ta = tb - ln["ms"] / 1000
        m = (sec >= ta) & (sec < tb) & (X >= ln["x0"]) & (X <= ln["x1"]) & (Y >= ln["y0"]) & (Y <= ln["y1"])
        D = int(round(cfg["lens_d"] * cfg["width"] * ss))
        disc, mark, R, t_at = lens_disc(cfg, cam, zt, (X[m], Y[m], sec[m], P[m]),
                                  (ln["x0"], ln["x1"], ln["y0"], ln["y1"], ta, tb), D)
        tr = cam.project(np.array([[W / 2, H / 2, zf]]))
        centre = np.array([tr[0][0] + cfg["lens_pos"][0] * D, tr[1][0] + cfg["lens_pos"][1] * D])
        # the patch on the pane, the magnifier and the two lines that join them
        pu, pv, _ = cam.project(np.array([[cx - W / 2, H / 2 - cy, zf] for cx in (ln["x0"], ln["x1"] + 1)
                                          for cy in (ln["y0"], ln["y1"] + 1)]))
        src = np.array([pu.mean(), pv.mean()])
        r_src = 1.7 * np.hypot(pu - src[0], pv - src[1]).max()
        lw = cfg["lens_lw"] * k14
        mask = ring_mask((Hc, Wc), src, r_src, lw)
        for a, b in tangents(src, r_src, centre, D / 2 - lw):
            line(mask, a, b, lw)
        lens = tinted(mask, cfg["axis_col"], 0.95)
        paste(lens, disc, centre[0] - D / 2, centre[1] - D / 2)
        overlays["lens"] = lens
        stats["lens"] = dict(events=int(m.sum()), on=int((P[m] == 1).sum()), t=[round(ta, 4), round(tb, 4)],
                             pixels=[ln["x0"], ln["x1"], ln["y0"], ln["y1"]])
        origin = centre - D / 2
        if cfg["event_label"]:
            em = cfg["event_em"] * k14
            put_text(text, cfg["event_label"], em, origin[0] + mark[0] + cfg["event_at"][0] * em,
                     origin[1] + mark[1] + 1.75 * R * np.sign(cfg["event_at"][1]) + cfg["event_at"][1] * em, col)
            put_text(text, "$t$", em, origin[0] + t_at[0], origin[1] + t_at[1], col)
        if cfg["formula"]:
            em = cfg["formula_em"] * k14
            put_text(text, cfg["formula"], em, centre[0], centre[1] - D / 2 - 0.95 * em, col)
    if cfg["xy_labels"]:
        em = cfg["text_em"] * k14
        for label, pt3, shift in (("$x$", (0, -H / 2, zf), (0.25, 0.85)), ("$y$", (W / 2, 0, zf), (0.75, 0.1))):
            uu, vv, _ = cam.project(np.array([pt3]))
            put_text(text, label, em, uu[0] + shift[0] * em, vv[0] + shift[1] * em, col)
        if t_pos is not None and not cfg["axis_label"]:
            put_text(text, "$t$", em, t_pos[0], t_pos[1], col)
    if cfg["legend"] and cfg["axis"]:
        em = cfg["legend_em"] * k14
        ax, ay = cfg["axis_xy"]
        uu, vv, _ = cam.project(np.array([[ax, ay, -L / 2 + cfg["axis_z0"]]]))
        x0, y0 = uu[0], (t_pos[1] if t_pos is not None else vv[0] + 12 * em)
        for i, (base, label) in enumerate(((red, "$p=+1$"), (blue, "$p=-1$"))):
            sp, n = bead_sprite(0.36 * em, base)
            cx = x0 + 0.4 * em + i * 5.6 * em
            paste(text, sp, cx - n, y0 - n)
            put_text(text, label, em, cx + 0.85 * em, y0, col, anchor="l")
    if text[3].any():
        overlays["text"] = text
    return canvas, overlays, stats


def downsample(img, ss):
    c, h, w = img.shape
    return img[:, :h // ss * ss, :w // ss * ss].reshape(c, h // ss, ss, w // ss, ss).mean((2, 4))


def crop_box(alpha, margin, thr):
    ys, xs = np.nonzero(alpha > thr)
    h, w = alpha.shape
    return max(ys.min() - margin, 0), min(ys.max() + 1 + margin, h), max(xs.min() - margin, 0), min(xs.max() + 1 + margin, w)


def to_png(premult, path, background=None):
    a = premult[3:4]
    if background is None:
        out = np.concatenate([np.where(a > 1e-6, premult[:3] / np.maximum(a, 1e-6), 1.0), a], 0)
    else:
        out = premult[:3] + (1 - a) * np.asarray(background, np.float32)[:, None, None]
    out = (np.clip(np.moveaxis(out, 0, -1), 0, 1) * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA if out.shape[-1] == 4 else cv2.COLOR_RGB2BGR))


def make(cfg, events, out, preview=False):
    canvas, overlays, stats = render(cfg, events)
    ss = cfg["ss"]
    base = downsample(canvas, ss)
    full = base.copy()
    for name in ("axis", "lens", "text"):
        if name in overlays:
            over(full, downsample(overlays[name], ss))
    # second image: without the letters when the layout has any, else without the time axis
    if "text" in overlays:
        second, tag = base.copy(), "_notext"
        for name in ("axis", "lens"):
            if name in overlays:
                over(second, downsample(overlays[name], ss))
    else:
        second, tag = base, "_noaxis"
    y0, y1, x0, x1 = crop_box(full[3], int(cfg["crop_margin"] * cfg["width"] / 1400.0), cfg["crop_alpha"])
    full, second = full[:, y0:y1, x0:x1], second[:, y0:y1, x0:x1]
    out.parent.mkdir(parents=True, exist_ok=True)
    to_png(full, out.with_name(out.name + "_white.png"), background=(1, 1, 1))
    if not preview:
        to_png(full, out.with_name(out.name + ".png"))
        to_png(second, out.with_name(out.name + tag + ".png"))
        stats["size"] = [int(x1 - x0), int(y1 - y0)]
        with open(out.with_name(out.name + ".json"), "w") as f:
            json.dump(dict(config=cfg, stats=stats), f, indent=1)
    print(out.name, "%dx%d" % (x1 - x0, y1 - y0), stats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", choices=list(LAYOUTS) + ["all"], default="events")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--name", default="event_stream")
    ap.add_argument("--set", nargs="*", default=[], help="key=json overrides of DEFAULT / the layout preset")
    ap.add_argument("--cache", help="npz with t, x, y, p of the session (skips reading the recording)")
    ap.add_argument("--preview", action="store_true", help="write only the white-background image")
    args = ap.parse_args()
    over_cfg = {}
    for kv in args.set:
        key, val = kv.split("=", 1)
        if key not in DEFAULT:
            ap.error("unknown key %s" % key)
        over_cfg[key] = json.loads(val)
    base = {**DEFAULT, **over_cfg}
    events = load_session(base["session"], base["stream"], args.cache)
    for layout in (list(LAYOUTS) if args.layout == "all" else [args.layout]):
        cfg = {**DEFAULT, **LAYOUTS[layout], **over_cfg}
        make(cfg, events, Path(args.out_dir) / ("%s_%s" % (args.name, layout)), args.preview)
