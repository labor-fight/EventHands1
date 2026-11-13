"""ESMNet's joint assignment of one real DAVIS346 event packet, for the paper figure (user 2026-11-13:
"画一个ESMNet处理后的event信息图，就是知道我event归属哪个关节……
最好让人能看出来是属于哪个关节").

ESMNet is the paper's name for EvHandTrack's router; on the real data it is r_s2dx2 (outputs/real/20261110/routers/
chosen.json, docs/EXPERIMENT_RECORDS.md#rfsu-20261113). For every event of a 50 ms packet it gives an on-hand
probability, one of the 16 MANO parts and a point on the rest template. A part is the region whose skinning one MANO
joint dominates, so the part names the joint the event belongs to: the wrist (palm), MCP / PIP / DIP of the four
fingers, CMC / MCP / IP of the thumb; each joint owns the segment distal to it.

Layout `full` (the figure):
  left   the packet on the network's own canvas (n346: undistorted 346 x 260), every event on its pixel, colored by
         its joint: hue = finger, shade = proximal (dark) -> distal (light), palm tan; events whose on-hand
         probability is below tau (0.5, the pipeline's) are light gray. A pixel takes the majority joint of its
         events (in the default packet the majority holds 99.8 % of a pixel's events on average).
  right  the key: the MANO rest template seen from the side the camera sees (palmar for the default packet; chosen
         from the faces ESMNet maps the events to), its 16 parts in the same colors (the template is split twice at
         edge midpoints and each small face takes the joint whose interpolated skinning dominates it, after three
         rounds of averaging the weights with the neighbours', so part borders are smooth; display only, ESMNet's own
         parts are the 1538 faces' majority parts), joints and bones on top, finger names in their finger's color, and
         MCP / PIP / DIP named on one finger.
Layouts `events` and `key` are the two panels alone (for assembling the method figure). Every layout is written with
a transparent background, on white (`_white`), and without any lettering (`_notext`, same size and position as the
lettered image), with the parameters, the router's identity and the counts in a json of the same name.

The default packet ends where the input figure's projection pane ends (ch_global, 35.42 s after the first raw event,
tools/real/event_stream_figure.py), so both figures show the same events. ESMNet reads the whole 50 ms packet exactly
as the route-first pipeline does (`semkine.s45sm_fast.router_forward` on every event, CPU); the panels draw its newest
`show_ms` (35 ms, the span of the input figure's pane). ch is a training subject, so the assignment is in-sample;
for a held-out packet set session / end_ms (the labels then come from the held-out XAG sources).

    /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python tools/real/esmnet_joint_figure.py --layout all
    ... --layout full --preview --set width=1400 show_ms=50
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA = Path("/data1/lyq/code/mesh/EventHands/data/hand_data_real_n346")
CHOSEN = ROOT / "outputs/real/20261110/routers/chosen.json"
LABELS = {"train": ROOT / "outputs/real/20261110/xag_sources",
          "heldout": ROOT / "outputs/real/20261110/analysis/router/xag_heldout"}
MANO = ROOT / "assets/mano_right.npz"
OUT = ROOT / "docs" / "assets" / "esmnet_joint_assignment_20261113"
FONTS = Path("/usr/share/fonts/opentype/urw-base35")

# MANO joint order (assets/mano_right.npz kintree): 0 wrist, 1-3 index, 4-6 middle, 7-9 little, 10-12 ring, 13-15 thumb
FINGERS = {"index": (1, 2, 3), "middle": (4, 5, 6), "little": (7, 8, 9), "ring": (10, 11, 12), "thumb": (13, 14, 15)}
TIPS = {"thumb": 744, "index": 320, "middle": 443, "ring": 554, "little": 671}      # fingertip vertices
JOINT_NAMES = {"index": ("MCP", "PIP", "DIP"), "middle": ("MCP", "PIP", "DIP"), "ring": ("MCP", "PIP", "DIP"),
               "little": ("MCP", "PIP", "DIP"), "thumb": ("CMC", "MCP", "IP")}

DEFAULT = dict(
    # packet: n346 millisecond index of the last millisecond (ms since the sequence's t0), packet and display spans
    session="ch_global", end_ms=35407, packet_ms=50, show_ms=35, tau=0.5, checkpoint=None,
    # crop of the n346 canvas shown on the left: x0, x1, y0, y1 in pixels
    crop=[142, 236, 86, 186],
    # figure (sizes in pixels at `width`; lengths below scale with width / 1400)
    width=3200, margin=40, gap=70, ss=3, events_w=640,     # events panel width (px at a 1400 px figure)
    # colors: the finger base colors of the method figure's part (a) (tools/real/pipeline_part_a.py FINGER_RGB: palm
    # tan, thumb orange, index gold, middle green, ring purple, little pink; no pure red / blue, those mean polarity),
    # shaded per segment by an OKLCH lightness offset (proximal, middle, distal; wider than part (a)'s x0.86 / +32 % so
    # the three joints of a finger tell apart); finger names use the finger's color at lightness <= label_L
    finger_rgb=dict(palm=[204, 172, 128], index=[222, 172, 32], middle=[58, 160, 100], little=[214, 96, 160],
                    ring=[138, 108, 204], thumb=[236, 132, 40]),
    seg_dL=[-0.12, 0.0, 0.10], label_L=0.52, off=[0.85, 0.008, 260],
    pane_fill=[244, 245, 250], border_col=[64, 68, 82], border_lw=1.0,
    # event dots on the left, diameters in canvas pixels
    on_dot=0.80, off_dot=0.54,
    # key: subdivision levels, face tint (share of the joint color, the rest white), Lambert shading, outline, skeleton
    # (weight_smooth: rounds of averaging the skinning weights with the neighbours' before the parts are taken, display
    # only; ESMNet's own parts are the 1538 faces' majority parts)
    subdiv=2, weight_smooth=3, key_tint=0.9, key_shade=0.16, key_light=[-0.3, -1.0, -0.3], outline_lw=1.2,
    outline_col=[150, 154, 165],
    bone_lw=1.1, bone_col=[64, 68, 82], joint_r=4.0, joint_rim=1.2, tip_r=1.8, key_vpad=[70, 14], key_hpad=10,
    view="auto",                                    # "palmar" | "dorsal" | "auto" (the side most mapped faces face)
    # text (em sizes for a 1400 px wide figure)
    text=True, font="NimbusSans-Regular.otf", text_col=[48, 52, 64], finger_em=33, joint_em=26, legend_em=29,
    label_color="finger", halo=0.14,              # finger names in their finger's color; halo around "Palm"
    finger_labels=dict(thumb="Thumb", index="Index", middle="Middle", ring="Ring", little="Little"),
    palm_label="Palm", palm_at=[0.0, 0.42], joint_finger="little", joint_off_m=0.0135, joint_leader=0,
    label_gap=12, legend=True, legend_label="Off-hand",
)
LAYOUTS = {"full": dict(), "events": dict(), "key": dict()}


# ----------------------------------------------------------------------------- colors
def _oklch_lin(L, C, h):
    a, b = C * np.cos(np.deg2rad(h)), C * np.sin(np.deg2rad(h))
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    return np.array([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                     -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                     -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s])


def oklch(L, C, h):
    """sRGB 8-bit of an OKLCH color; chroma is reduced until the color is inside the sRGB gamut."""
    while C > 0 and not ((_oklch_lin(L, C, h) >= -1e-4).all() and (_oklch_lin(L, C, h) <= 1 + 1e-4).all()):
        C -= 0.002
    lin = np.clip(_oklch_lin(L, max(C, 0.0), h), 0, 1)
    c = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return tuple(int(round(v * 255)) for v in c)


def to_oklch(rgb):
    c = np.asarray(rgb, float) / 255.0
    r, g, b = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    l_ = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return float(L), float(np.hypot(a, bb)), float(np.rad2deg(np.arctan2(bb, a)) % 360)


def palette(cfg):
    """Joint colors {joint: rgb}, the off-hand gray, and the finger-name colors {finger: rgb}."""
    col = {0: tuple(int(v) for v in cfg["finger_rgb"]["palm"])}
    names = {}
    for finger, joints in FINGERS.items():
        L, C, h = to_oklch(cfg["finger_rgb"][finger])
        for seg, k in enumerate(joints):
            col[k] = oklch(min(max(L + cfg["seg_dL"][seg], 0.0), 0.97), C, h)
        names[finger] = oklch(min(L + cfg["seg_dL"][0], cfg["label_L"]), C, h)
    return col, oklch(*cfg["off"]), names


# ----------------------------------------------------------------------------- data + ESMNet
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sequence_info(seq):
    conv = json.loads((DATA / "conversion.json").read_text())
    for rec in conv["sequences"]:
        if rec["seq"] == seq:
            return rec
    raise KeyError(seq)


def raw_first_us(info):
    with open(Path(info["source_dir"]) / "event" / "index.jsonl") as f:
        return int(json.loads(f.readline())["first_timestamp_us"])


def infer(cfg):
    """ESMNet on every event of the packet ending at `end_ms`: on-hand probability, joint (part), template point."""
    import torch
    from tools.real import evht as X
    X.enter()                                                   # the router's and the evaluator's frame = n346
    from semkine import eval_track as ET
    from semkine import pfo_router as PR
    from semkine import s45sm_fast as FAST
    from semkine.xag_sources import load_sources

    chosen = json.loads(CHOSEN.read_text())
    ckpt = Path(cfg["checkpoint"] or chosen["checkpoint"])
    net, obj = PR.load(ckpt)
    info = sequence_info(cfg["session"])
    seq, sub = info["seq"], info["legacy_dir"]
    events, offsets, aux, _ = ET.load_sequence(DATA, sub, seq)
    tsub = np.load(DATA / sub / f"{seq}_tsub.npy", mmap_mode="r")
    end, span = int(cfg["end_ms"]), int(cfg["packet_ms"])
    ev = ET._window_events(events, offsets, tsub, end, span)
    betas = torch.tensor(np.asarray(aux["betas"], np.float32)).view(1, -1)
    K = torch.tensor(np.asarray(aux["camera_K"], np.float32)).view(1, 3, 3)
    b = ET.make_eval_packet(ev, torch.zeros(1, 51), betas, K, span, torch.device("cpu"))
    n = int(b.events.shape[0])
    src, mask = torch.arange(n).view(1, -1), torch.ones(1, n, dtype=torch.bool)
    with torch.no_grad():
        o = FAST.router_forward(net, b.events, b.ptr, b.delta_t_s, src, mask)
        prob = torch.sigmoid(o["onhand"].float())[0].numpy()
        face, bary = FAST.route(net, "sr", o, mask)
    face = face[0]
    point = (bary[0].float().unsqueeze(-1) * net.tmpl[net.faces[face]]).sum(-2).numpy()
    part = net.face_part[face].numpy()
    e = b.events.numpy()
    out = dict(x=e[:, 1].astype(np.int64), y=e[:, 2].astype(np.int64), t=e[:, 3].astype(np.float64),
               p=prob, part=part, face=face.numpy(), point=point)

    # agreement with the XAG pseudo-labels (descriptive; the router's training labels for training subjects)
    role = info["role"]
    lab_root = LABELS["train" if role == "train" else "heldout"]
    agree = None
    if (lab_root / seq).exists():
        start = end - span + 1
        off = np.asarray(offsets)
        rows = np.arange(int(off[start]), int(off[end + 1]), dtype=np.int64)
        ms = np.repeat(np.arange(start, end + 1, dtype=np.int64), np.diff(off[start:end + 2]).astype(np.int64))
        lab = load_sources(str(lab_root), seq, rows, ms)
        ok = lab["xag_valid"] & (lab["xag_face"] >= 0) & (lab["xag_face"] < net.faces.shape[0])
        lpart = net.face_part.numpy()[np.clip(lab["xag_face"], 0, net.faces.shape[0] - 1)]
        on = prob >= cfg["tau"]
        both = on & ok
        finger = np.zeros(16, np.int64)
        for i, joints in enumerate(FINGERS.values()):
            finger[list(joints)] = i + 1
        agree = dict(labels=str(lab_root.relative_to(ROOT)), labelled=int(ok.sum()),
                     onhand_and_labelled=int(both.sum()),
                     part_acc=float((lpart[both] == part[both]).mean()),
                     finger_acc=float((finger[lpart[both]] == finger[part[both]]).mean()),
                     onhand_precision=float(ok[on].mean()), onhand_recall=float(on[ok].mean()))
    first = raw_first_us(info)
    t0_us = int(aux["t0_ns"]) / 1e3
    meta = dict(
        router=dict(variant=chosen["variant"], checkpoint=str(ckpt.relative_to(ROOT)), sha256=sha256(ckpt),
                    params=int(sum(q.numel() for q in net.parameters())), design=obj["design"]),
        packet=dict(session=seq, subject=info["subject"], role=role, light=info["light"], canvas="n346 (346x260)",
                    ms=[end - span + 1, end], events=n,
                    raw_seconds=[round((t0_us + (end - span + 1) * 1e3 - first) / 1e6, 4),
                                 round((t0_us + (end + 1) * 1e3 - first) / 1e6, 4)]),
        agreement_with_labels=agree)
    return out, meta


# ----------------------------------------------------------------------------- drawing helpers
def font(cfg, em):
    return ImageFont.truetype(str(FONTS / cfg["font"]), max(int(round(em)), 4))


def disc(draw, x, y, r, fill):
    draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def visible_side(cfg, data, verts, faces):
    if cfg["view"] != "auto":
        return cfg["view"]
    n = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]], verts[faces[:, 2]] - verts[faces[:, 0]])
    ny = n[:, 1] / np.linalg.norm(n, axis=1)
    on = data["p"] >= cfg["tau"]
    side = ny[data["face"][on]]
    return "dorsal" if (side > 0.5).sum() > (side < -0.5).sum() else "palmar"


# ----------------------------------------------------------------------------- left panel: events
def events_panel(cfg, data, colors, off_col, height):
    """RGBA panel (`height` px tall) and per-joint statistics of the shown events."""
    ss = cfg["ss"]
    x0, x1, y0, y1 = cfg["crop"]
    s = height / (y1 - y0)
    W, H = int(round((x1 - x0) * s)), int(round(height))
    img = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lw = max(cfg["border_lw"] * cfg["width"] / 1400 * ss, 1)
    d.rectangle([0, 0, W * ss - 1, H * ss - 1], fill=tuple(cfg["pane_fill"]) + (255,),
                outline=tuple(cfg["border_col"]) + (255,), width=int(round(lw)))
    show = data["t"] >= cfg["packet_ms"] / 1e3 - cfg["show_ms"] / 1e3
    x, y = data["x"], data["y"]
    inside = show & (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    on = data["p"] >= cfg["tau"]
    pix = y * 346 + x
    onpix = np.unique(pix[inside & on])
    offpix = np.setdiff1d(np.unique(pix[inside & ~on]), onpix)
    k = s * ss
    for q in offpix:
        disc(d, (q % 346 - x0 + 0.5) * k, (q // 346 - y0 + 0.5) * k, cfg["off_dot"] / 2 * k, off_col + (255,))
    sel = inside & on
    pp, pa = pix[sel], data["part"][sel]
    order = np.lexsort((pa, pp))
    pp, pa = pp[order], pa[order]
    uniq, first, count = np.unique(pp, return_index=True, return_counts=True)
    shares = []
    for q, f0, c in zip(uniq, first, count):
        votes = np.bincount(pa[f0:f0 + c], minlength=16)
        j = int(votes.argmax())
        shares.append(votes[j] / c)
        disc(d, (q % 346 - x0 + 0.5) * k, (q // 346 - y0 + 0.5) * k, cfg["on_dot"] / 2 * k, colors[j] + (255,))
    img = img.resize((W, H), Image.LANCZOS)
    stats = dict(shown_events=int(inside.sum()), shown_onhand_events=int(sel.sum()), onhand_pixels=int(len(uniq)),
                 offhand_pixels=int(len(offpix)), pixel_majority_share_mean=float(np.mean(shares)) if shares else None,
                 onhand_events_per_joint=np.bincount(pa, minlength=16).tolist())
    return img, stats, sel


# ----------------------------------------------------------------------------- right panel: MANO template key
def subdivided_template(levels, smooth=0):
    """MANO rest template split `levels` times at edge midpoints (shape unchanged); skinning weights and vertex normals
    are interpolated, so each small face takes the joint whose interpolated skinning dominates it (smooth part
    boundaries instead of the 1538 faces' zigzag; the same parts)."""
    z = np.load(MANO)
    V, F = z["v_template"].astype(np.float64), z["f"].astype(np.int64)
    Wt = z["weights"].astype(np.float64)
    if smooth:                                      # display only: average each vertex's weights with its neighbours
        nb = [set() for _ in range(len(V))]
        for a, b, c in F:
            nb[a] |= {b, c}
            nb[b] |= {a, c}
            nb[c] |= {a, b}
        nb = [np.fromiter(q, np.int64) for q in nb]
        for _ in range(int(smooth)):
            Wt = np.stack([0.5 * Wt[i] + 0.5 * Wt[q].mean(0) for i, q in enumerate(nb)])
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    N = np.zeros_like(V)
    for j in range(3):
        np.add.at(N, F[:, j], fn)
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    for _ in range(levels):
        index = {}
        vs, ws, ns = list(V), list(Wt), list(N)

        def mid(a, b):
            key = (a, b) if a < b else (b, a)
            if key not in index:
                index[key] = len(vs)
                vs.append((V[a] + V[b]) / 2)
                ws.append((Wt[a] + Wt[b]) / 2)
                ns.append(N[a] + N[b])
            return index[key]
        out = []
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            out += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        V, Wt, F = np.array(vs), np.array(ws), np.array(out, np.int64)
        N = np.array(ns)
        N /= np.linalg.norm(N, axis=1, keepdims=True)
    return V, F, Wt[F].mean(1).argmax(1), N, z


def key_panel(cfg, data, colors, name_col, height):
    """MANO rest template with parts in the joint colors, joints and bones (RGBA), and the finger / joint names on a
    layer of the same size; both cropped to their joint content."""
    from scipy import ndimage
    ss = cfg["ss"]
    u = cfg["width"] / 1400.0
    V, F, fpart, N, z = subdivided_template(cfg["subdiv"], cfg["weight_smooth"])
    J = z["J"].astype(np.float64)
    parent = z["kintree_table"][0].astype(np.int64)
    F0 = z["f"].astype(np.int64)
    side = visible_side(cfg, data, z["v_template"].astype(np.float64), F0)
    # rest template: fingers along -x, thumb +z, back of the hand +y. Screen: fingers up; the palmar view looks along
    # +y (thumb on the right), the dorsal view along -y (thumb on the left).
    sgn = 1.0 if side == "palmar" else -1.0
    toward = np.array([0.0, -sgn, 0.0])                                    # unit vector toward the viewer
    s = (height - cfg["key_vpad"][0] * u - cfg["key_vpad"][1] * u) / (V[:, 0].max() - V[:, 0].min())
    side_room = 400 * u                                                     # generous; cropped at the end
    W = int(round((V[:, 2].max() - V[:, 2].min()) * s + 2 * side_room))
    H = int(round(height))
    ox = side_room - (sgn * V[:, 2]).min() * s
    oy = cfg["key_vpad"][0] * u - V[:, 0].min() * s

    def to_px(P):
        P = np.atleast_2d(P)
        return (sgn * P[:, 2] * s + ox) * ss, (P[:, 0] * s + oy) * ss, P @ toward

    px, py, depth = to_px(V)
    img = Image.new("RGBA", (W * ss, H * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fnorm = N[F].mean(1)
    fnorm /= np.linalg.norm(fnorm, axis=1, keepdims=True)
    front = fnorm @ toward > -0.2
    light = np.asarray(cfg["key_light"], float) * np.array([1.0, sgn, sgn])
    light /= np.linalg.norm(light)
    # silhouette outline under the mesh
    if cfg["outline_lw"] > 0:
        mask = Image.new("L", img.size, 0)
        md = ImageDraw.Draw(mask)
        for fi in np.flatnonzero(front):
            md.polygon([(px[F[fi, j]], py[F[fi, j]]) for j in range(3)], fill=255)
        m = np.asarray(mask) > 127
        r = int(round(cfg["outline_lw"] * u * ss))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        ring = ndimage.binary_dilation(m, structure=(xx ** 2 + yy ** 2) <= r * r) & ~m
        layer = np.zeros(m.shape + (4,), np.uint8)
        layer[ring] = tuple(cfg["outline_col"]) + (255,)
        img.alpha_composite(Image.fromarray(layer, "RGBA"))
    for fi in np.argsort(depth[F].mean(1)):                                # far to near
        if not front[fi]:
            continue
        c = np.array(colors[int(fpart[fi])], float)
        tint = cfg["key_tint"] * c + (1 - cfg["key_tint"]) * 255.0
        shade = 1.0 - cfg["key_shade"] + cfg["key_shade"] * max(float(fnorm[fi] @ light), 0.0)
        col = tuple(int(round(v)) for v in np.clip(tint * shade, 0, 255)) + (255,)
        d.polygon([(px[F[fi, j]], py[F[fi, j]]) for j in range(3)], fill=col, outline=col)
    jx, jy, _ = to_px(J)
    bone = tuple(cfg["bone_col"]) + (255,)
    lw = max(int(round(cfg["bone_lw"] * u * ss)), 1)
    for k in range(1, 16):
        d.line([(jx[parent[k]], jy[parent[k]]), (jx[k], jy[k])], fill=bone, width=lw)
    tips = {}
    for f, joints in FINGERS.items():
        tx, ty, _ = to_px(z["v_template"][TIPS[f]])
        tips[f] = (float(tx[0]), float(ty[0]))
        d.line([(jx[joints[-1]], jy[joints[-1]]), tips[f]], fill=bone, width=lw)
        disc(d, *tips[f], cfg["tip_r"] * u * ss, bone)
    jr, rim = cfg["joint_r"] * u * ss, cfg["joint_rim"] * u * ss
    for k in range(16):
        disc(d, jx[k], jy[k], jr + rim, (255, 255, 255, 255))
        disc(d, jx[k], jy[k], jr, colors[k] + (255,))
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    tc = tuple(cfg["text_col"]) + (255,)
    ff = font(cfg, cfg["finger_em"] * u * ss)
    gap = cfg["label_gap"] * u * ss
    label_col = {cfg["finger_labels"][f]: (name_col[f] if cfg["label_color"] == "finger" else tuple(cfg["text_col"]))
                 + (255,) for f in FINGERS}
    placed = []
    cands = []
    for f, joints in FINGERS.items():
        tx, ty = tips[f]
        dx, dy = tx - jx[joints[-1]], ty - jy[joints[-1]]
        nrm = np.hypot(dx, dy)
        cands.append((cfg["finger_labels"][f], tx + dx / nrm * gap, ty + dy / nrm * gap))
    for name, ax, ay in sorted(cands, key=lambda r: -r[2]):            # lowest first; later ones move up
        w = d.textlength(name, font=ff)
        h = ff.size * 0.74
        base = ay
        moved = True
        while moved:
            moved = False
            for (bx0, bx1, by0, by1) in placed:
                if ax - w / 2 < bx1 + 0.25 * h and ax + w / 2 > bx0 - 0.25 * h and base > by0 - 0.2 * h \
                        and base - h < by1 + 0.2 * h:
                    base = by0 - 0.2 * h
                    moved = True
        placed.append((ax - w / 2, ax + w / 2, base - h, base))
        d.text((ax, base), name, font=ff, fill=label_col[name], anchor="ms")
    if cfg["palm_label"]:
        cx = float(jx[0] + cfg["palm_at"][0] * (jx[4] - jx[0]))
        cy = float(jy[0] + cfg["palm_at"][1] * (jy[4] - jy[0]))
        # dark text with a halo in the palm's own face color, so the palm bones break around the word
        halo = np.array(colors[0], float) * cfg["key_tint"] + (1 - cfg["key_tint"]) * 255.0
        d.text((cx, cy), cfg["palm_label"], font=ff, anchor="mm", fill=tc,
               stroke_width=int(round(cfg["halo"] * ff.size)),
               stroke_fill=tuple(int(round(v)) for v in halo) + (255,))
    if cfg["joint_finger"]:
        fj = font(cfg, cfg["joint_em"] * u * ss)
        joints = FINGERS[cfg["joint_finger"]]
        # perpendicular to the finger axis, on the side away from the middle finger's MCP (outside the hand)
        a, b_ = jx[joints[-1]] - jx[joints[0]], jy[joints[-1]] - jy[joints[0]]
        nx, ny = b_ / np.hypot(a, b_), -a / np.hypot(a, b_)
        if nx * (jx[joints[0]] - jx[4]) + ny * (jy[joints[0]] - jy[4]) < 0:
            nx, ny = -nx, -ny
        g = cfg["joint_off_m"] * s * ss
        for name, k in zip(JOINT_NAMES[cfg["joint_finger"]], joints):
            lx, ly = jx[k] + nx * g, jy[k] + ny * g
            if cfg["joint_leader"]:
                lead = cfg["joint_leader"] * u * ss
                d.line([(jx[k] + nx * (jr + rim + 0.3 * lead), jy[k] + ny * (jr + rim + 0.3 * lead)),
                        (lx - nx * lead, ly - ny * lead)], fill=tc, width=max(int(round(0.8 * u * ss)), 1))
            d.text((lx, ly), name, font=fj, fill=tc, anchor="rm" if nx < 0 else "lm")
    img, layer = img.resize((W, H), Image.LANCZOS), layer.resize((W, H), Image.LANCZOS)
    a = np.maximum(np.asarray(img)[..., 3], np.asarray(layer)[..., 3])
    xs = np.nonzero(a.max(0) > 3)[0]
    pad = int(round(cfg["key_hpad"] * u))
    box = (max(xs.min() - pad, 0), 0, min(xs.max() + pad + 1, W), H)
    return img.crop(box), layer.crop(box), dict(view=side, subdivided_faces=int(len(F)))


# ----------------------------------------------------------------------------- assembly
def legend_row(cfg, off_col, width):
    u = cfg["width"] / 1400.0
    ss = cfg["ss"]
    em = cfg["legend_em"] * u
    H = int(round(em * 1.7))
    img = Image.new("RGBA", (int(width) * ss, H * ss), (0, 0, 0, 0))
    if not cfg["legend"]:
        return img.resize((int(width), H), Image.LANCZOS)
    d = ImageDraw.Draw(img)
    r = em * 0.28 * ss
    cx, cy = r + 2 * ss, H * ss / 2
    disc(d, cx, cy, r, off_col + (255,))
    d.text((cx + r + em * 0.35 * ss, cy), cfg["legend_label"], font=font(cfg, em * ss),
           fill=tuple(cfg["text_col"]) + (255,), anchor="lm")
    return img.resize((int(width), H), Image.LANCZOS)


def build(cfg, data, layout):
    """The layout twice on canvases of the same size: with the text, and without it (for lettering by hand)."""
    colors, off_col, name_col = palette(cfg)
    u = cfg["width"] / 1400.0
    m, gap = cfg["margin"] * u, cfg["gap"] * u
    x0, x1, y0, y1 = cfg["crop"]
    Hp = cfg["events_w"] * u * (y1 - y0) / (x1 - x0)
    stats = {}
    a = b = bt = leg = None
    if layout in ("full", "events"):
        a, stats["events_panel"], _ = events_panel(cfg, data, colors, off_col, Hp)
        leg = legend_row(cfg, off_col, a.size[0])
    if layout in ("full", "key"):
        b, bt, stats["key_panel"] = key_panel(cfg, data, colors, name_col, Hp)
    W = int(round(2 * m + (a.size[0] if a is not None else 0) + (b.size[0] if b is not None else 0)
                  + (gap if a is not None and b is not None else 0)))
    H = int(round(2 * m + Hp + (leg.size[1] + 0.2 * cfg["legend_em"] * u if leg is not None else 0)))
    bare = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    text = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    xo, yo = int(round(m)), int(round(m))
    if a is not None:
        bare.alpha_composite(a, (xo, yo))
        text.alpha_composite(leg, (xo, int(round(m + Hp + 0.2 * cfg["legend_em"] * u))))
        xo += a.size[0] + int(round(gap))
    if b is not None:
        bare.alpha_composite(b, (xo, yo))
        text.alpha_composite(bt, (xo, yo))
    full = bare.copy()
    full.alpha_composite(text)
    stats["colors"] = {str(k): list(v) for k, v in colors.items()}
    stats["name_colors"] = {k: list(v) for k, v in name_col.items()}
    stats["offhand_color"] = list(off_col)
    return full, bare, stats


def content_box(img, margin):
    a = np.asarray(img)[..., 3]
    ys, xs = np.nonzero(a > 3)
    return (max(xs.min() - margin, 0), max(ys.min() - margin, 0), min(xs.max() + margin + 1, a.shape[1]),
            min(ys.max() + margin + 1, a.shape[0]))


def save(img, path, white=False):
    if white:
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        bg.convert("RGB").save(path)
    else:
        img.save(path)


def make(cfg, data, meta, layout, out_dir, name, preview=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{name}_{layout}"
    margin = int(round(cfg["margin"] * cfg["width"] / 1400.0 * 0.5))
    full, bare, stats = build(cfg, data, layout)
    if not cfg["text"]:
        full = bare
    box = content_box(full, margin)                  # one box for both, so the bare image lines up with the lettered
    full, bare = full.crop(box), bare.crop(box)
    save(full, out_dir / f"{stem}_white.png", white=True)
    if not preview:
        save(full, out_dir / f"{stem}.png")
        save(bare, out_dir / f"{stem}_notext.png")
        with open(out_dir / f"{stem}.json", "w") as f:
            json.dump(dict(config=cfg, layout=layout, meta=meta, stats=dict(stats, size=list(full.size))), f, indent=1)
    print(stem, "%dx%d" % full.size, json.dumps({k: v for k, v in stats.items() if k.endswith("panel")}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layout", choices=list(LAYOUTS) + ["all"], default="full")
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--name", default="esmnet_joints")
    ap.add_argument("--set", nargs="*", default=[], help="key=json overrides of DEFAULT")
    ap.add_argument("--preview", action="store_true", help="write only the white-background image")
    args = ap.parse_args(argv)
    cfg = dict(DEFAULT)
    for kv in args.set:
        key, val = kv.split("=", 1)
        if key not in cfg:
            raise SystemExit("unknown key " + key)
        cfg[key] = json.loads(val)
    data, meta = infer(cfg)
    print(json.dumps(meta))
    for layout in (list(LAYOUTS) if args.layout == "all" else [args.layout]):
        make(dict(cfg, **LAYOUTS[layout]), data, meta, layout, Path(args.out_dir), args.name, args.preview)


if __name__ == "__main__":
    main()
