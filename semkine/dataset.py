#!/usr/bin/env python3
"""S1 dataset: one sampler, two input faces (`legacy_lnes`, `raw_packed`), optional domrand.

The design constraint is that S1 must not move the baseline. `INPUT_MODE: legacy_lnes` with
`AUG.DOMRAND.ENABLED: false` therefore has to be *bitwise* identical to
`model/fastevc.py::HandData51Dataset`, which is checked by `tests/test_s1_dataset.py` rather
than argued. Everything new is reachable only by opting in.

Two structural differences from the legacy loader, both required by later stages:

* Sequences are addressed by `(seq, legacy_dir)` instead of by split, because the S1 protocol
  reform reassigns sequences across splits while their files stay in the directory
  `prepare_hand_data.py` wrote them to.
* Randomness comes from a per-sample seeded generator rather than global `np.random`, so a
  sample's augmentation is a pure function of `(epoch_seed, index)`. Without this, the domrand
  arm could not be replicated, and the `worker_id`-dependent stream would make the "off is
  bitwise identical" check depend on the worker count.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from semkine import domrand as DR
from semkine import events as EV
from semkine.events import EventPacket

H_DEFAULT, W_DEFAULT = 180, 240


@dataclass
class SequenceHandles:
    """Lazily-opened memmaps for one sequence. Workers must not share open memmaps."""

    seq: str
    legacy_dir: str
    base: str
    events: np.ndarray
    offsets: np.ndarray
    tsub: Optional[np.ndarray]
    pos51: np.ndarray
    betas: np.ndarray
    camera_K: np.ndarray
    valid_runs: np.ndarray
    category: str
    j0: np.ndarray


def _read_meta51(path: str) -> np.ndarray:
    import struct

    with open(path, "rb") as f:
        (ncomps,) = struct.unpack("<i", f.read(4))
        assert ncomps == 51, ncomps
        dt = np.dtype([("data", np.float64, ncomps), ("m0", np.uint8), ("m1", np.uint8)])
        raw = np.fromfile(f, dtype=dt)
    assert np.all(raw["m0"] == 4) and np.all(raw["m1"] == 13)
    return raw["data"].astype(np.float32)


def mano_root_joint(betas: np.ndarray, mano_npz: str) -> np.ndarray:
    """`j0 = (J_regressor @ (v_template + shapedirs beta))[0]`, the LBS pivot domrand needs."""
    d = np.load(mano_npz, allow_pickle=False)
    v = d["v_template"].astype(np.float64) + np.einsum(
        "l,mkl->mk", betas.astype(np.float64), d["shapedirs"].astype(np.float64)
    )
    return (d["J_regressor"].astype(np.float64) @ v)[0]


class SemKineDataset(Dataset):
    """Windowed sampler over one or more sequences.

    A sample is a window of `window` milliseconds ending at `end_idx`, entirely inside one
    `valid_run`, exactly as the legacy loader defines it.
    """

    INPUT_MODES = ("legacy_lnes", "raw_packed", "both")

    def __init__(
        self,
        root: str | Path,
        sequences: Sequence[Tuple[str, str]],
        components: np.ndarray,
        mano_npz: str,
        input_mode: str = "legacy_lnes",
        window_min: int = 30,
        window_max: int = 300,
        width: int = W_DEFAULT,
        height: int = H_DEFAULT,
        fixed_window: Optional[int] = None,
        train: bool = True,
        speed_aug: bool = True,
        polarity_flip: bool = True,
        pixel_polarity_swap: bool = True,
        domrand: Optional[DR.DomRandConfig] = None,
        prev_noise: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        prev_noise_mode: str = "gaussian",
        prev_noise_large: Tuple[float, float, float] = (0.05, 0.3, 0.3),
        prev_noise_mix: Tuple[float, float, float] = (0.5, 0.3, 0.2),
        render_scale: float = 0.375,
        event_channels: Sequence[str] = ("last",),
        seed: int = 0,
        unroll_pair: bool = False,
    ):
        if input_mode not in self.INPUT_MODES:
            raise ValueError(f"INPUT_MODE must be one of {self.INPUT_MODES}, got {input_mode!r}")
        self.root = Path(root)
        self.sequences = [tuple(s) for s in sequences]
        self.components = np.asarray(components, dtype=np.float32)
        self.mano_npz = str(mano_npz)
        self.input_mode = input_mode
        self.window_min, self.window_max = int(window_min), int(window_max)
        self.width, self.height = int(width), int(height)
        self.fixed_window = fixed_window
        self.train = bool(train)
        self.speed_aug = bool(speed_aug) and self.train
        self.polarity_flip = bool(polarity_flip) and self.train
        self.pixel_polarity_swap = bool(pixel_polarity_swap)
        self.domrand = domrand or DR.DomRandConfig()
        if self.domrand.enabled and not self.train:
            # Randomising the evaluation input would make the metric a moving target.
            self.domrand = DR.DomRandConfig(enabled=False)
        self.prev_noise = tuple(float(v) for v in prev_noise)
        self.prev_noise_mode = str(prev_noise_mode)
        self.prev_noise_large = tuple(float(v) for v in prev_noise_large)
        mix = tuple(float(v) for v in prev_noise_mix)
        s = sum(mix) or 1.0
        self.prev_noise_mix = tuple(v / s for v in mix)
        self.render_scale = float(render_scale)
        self.seed = int(seed)
        # S22. Return `(leading, main)` instead of one packet, so training can condition a window
        # on the model's own prediction for the window before it. Training only: randomising or
        # re-conditioning the evaluation input would make the metric a moving target.
        self.unroll_pair = bool(unroll_pair) and self.train
        self.event_channels = tuple(event_channels)

        self._handles: Dict[int, SequenceHandles] = {}
        self._pid = os.getpid()

        max_w = int(fixed_window) if fixed_window is not None else self.window_max
        min_w = int(fixed_window) if fixed_window is not None else self.window_min
        self.min_w, self.max_w = min_w, max_w

        # Index construction reads only small files, so it stays in the parent process.
        index: List[Tuple[int, int, int, int]] = []   # (seq_idx, end_idx, run_start, run_end)
        self.seq_meta: List[dict] = []
        for si, (seq, legacy_dir) in enumerate(self.sequences):
            base = str(self.root / legacy_dir / seq)
            offsets = np.load(base + "_offsets.npy", mmap_mode="r")
            aux = np.load(base + "_aux.npz", allow_pickle=True)
            runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
            self.seq_meta.append({
                "seq": seq, "legacy_dir": legacy_dir,
                "category": str(aux["category"]), "n_ms": int(len(offsets) - 1),
                "has_subms": Path(base + "_tsub.npy").exists(),
            })
            for a, b in runs:
                start_end = a + max_w - 1
                if start_end >= b:
                    start_end = a + min_w - 1
                if start_end >= b:
                    continue
                for end in range(int(start_end), int(b)):
                    index.append((si, end, int(a), int(b)))
        self.index = np.asarray(index, dtype=np.int64).reshape(-1, 4)
        self.pinv_C6 = np.linalg.pinv(self.components[:6]).astype(np.float32)

        if self.input_mode in ("raw_packed", "both"):
            missing = [m["seq"] for m in self.seq_meta if not m["has_subms"]]
            if missing:
                raise FileNotFoundError(
                    f"INPUT_MODE={input_mode} needs microsecond timestamps; run "
                    f"tools/extract_subms.py. Missing <seq>_tsub.npy for: {missing[:5]}"
                    + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else "")
                )

    # ---------------------------------------------------------------- handles
    def _handle(self, si: int) -> SequenceHandles:
        if os.getpid() != self._pid:
            self._handles.clear()
            self._pid = os.getpid()
        h = self._handles.get(si)
        if h is not None:
            return h
        seq, legacy_dir = self.sequences[si]
        base = str(self.root / legacy_dir / seq)
        aux = np.load(base + "_aux.npz", allow_pickle=True)
        betas = np.asarray(aux["betas"], dtype=np.float32)
        tsub_path = Path(base + "_tsub.npy")
        h = SequenceHandles(
            seq=seq,
            legacy_dir=legacy_dir,
            base=base,
            events=np.load(base + "_events.npy", mmap_mode="r"),
            offsets=np.load(base + "_offsets.npy"),
            tsub=np.load(str(tsub_path), mmap_mode="r") if tsub_path.exists() else None,
            pos51=_read_meta51(base + ".meta"),
            betas=betas,
            camera_K=np.asarray(aux["camera_K"], dtype=np.float32).reshape(3, 3),
            valid_runs=np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2),
            category=str(aux["category"]),
            j0=mano_root_joint(betas, self.mano_npz),
        )
        self._handles[si] = h
        return h

    def __len__(self) -> int:
        return int(len(self.index))

    # ---------------------------------------------------------------- targets
    def _to_target(self, p51: np.ndarray) -> np.ndarray:
        return np.asarray(p51, dtype=np.float32)

    def _sample_prev_noise(self, rng: np.random.Generator) -> Optional[np.ndarray]:
        sig_t, sig_r, sig_p = self.prev_noise
        if sig_t == 0.0 and sig_r == 0.0 and sig_p == 0.0:
            return None

        def fill(a, b, c):
            n = np.empty(51, np.float32)
            n[0:3] = rng.standard_normal(3) * a
            n[3:6] = rng.standard_normal(3) * b
            n[6:51] = rng.standard_normal(45) * c
            return n

        if self.prev_noise_mode != "mixed":
            return fill(sig_t, sig_r, sig_p)
        u = float(rng.random())
        p_small, p_large, _ = self.prev_noise_mix
        if u < p_small:
            return fill(sig_t, sig_r, sig_p)
        lt, lr, lp = self.prev_noise_large
        if u < p_small + p_large:
            return fill(lt, lr, lp)

        def directed(n, mag):
            v = rng.standard_normal(n).astype(np.float32)
            return v / (np.linalg.norm(v) + 1e-8) * np.float32(rng.uniform(0.0, mag))

        n = np.empty(51, np.float32)
        n[0:3] = directed(3, lt)
        n[3:6] = directed(3, lr)
        n[6:51] = directed(45, lp)
        return n

    # ---------------------------------------------------------------- windows
    def _window(self, idx: int, rng: np.random.Generator) -> Tuple[int, int, int, int]:
        si, end, run_a, run_b = (int(v) for v in self.index[idx])
        if self.fixed_window is not None:
            window = int(self.fixed_window)
        elif self.speed_aug:
            lo, hi = np.log(self.min_w), np.log(self.max_w)
            window = int(round(np.exp(rng.uniform(lo, hi))))
            window = max(self.min_w, min(self.max_w, window))
        else:
            window = self.min_w
        window = max(1, min(window, end - run_a + 1))
        return si, end, window, run_a

    def _raw_events(self, h: SequenceHandles, end: int, window: int):
        start = end - window + 1
        a0, a1 = int(h.offsets[start]), int(h.offsets[end + 1])
        if a1 <= a0:
            return (np.zeros(0, np.int64),) * 4
        ev = np.asarray(h.events[a0:a1])
        counts = np.diff(h.offsets[start : end + 2]).astype(np.int64)
        ms_rel = np.repeat(np.arange(window, dtype=np.int64), counts)
        if h.tsub is not None:
            us = ms_rel * 1000 + np.asarray(h.tsub[a0:a1]).astype(np.int64)
        else:
            us = ms_rel * 1000
        return ev[:, 0].astype(np.int64), ev[:, 1].astype(np.int64), \
            np.clip(ev[:, 2].astype(np.int64), 0, 1), us

    def _domrand_events(self, xs, ys, ps, us, window, dr, camera_K, rng):
        """Geometric map, dropout and hot pixels, in the draw order the S1 arms were trained with.

        Factored out so the S22 leading window can be transformed by the *same* virtual camera as
        the main window: a pair whose two halves saw different cameras would train the model to
        estimate a state in one frame and apply it in another.
        """
        if len(xs):
            keep = DR.transform_events(xs, ys, dr, camera_K, self.render_scale,
                                       self.width, self.height)
            m = DR.keep_mask(len(xs), dr.keep, rng)
            if m is not None:
                keep = keep & m
            xs, ys, ps, us = xs[keep], ys[keep], ps[keep], us[keep]
        hot = DR.sample_hot_pixels(dr.n_hot, window, self.width, self.height, rng)
        if hot is not None:
            # Hot pixels are camera noise: injected after the geometric map, since a hot
            # pixel is a property of the sensor and does not move with the virtual camera.
            xs = np.concatenate([xs, hot[:, 1]])
            ys = np.concatenate([ys, hot[:, 2]])
            ps = np.concatenate([ps, hot[:, 3]])
            us = np.concatenate([us, hot[:, 0] * 1000 + rng.integers(0, 1000, len(hot))])
            order = np.argsort(us, kind="stable")
            xs, ys, ps, us = xs[order], ys[order], ps[order], us[order]
        return xs, ys, ps, us

    @staticmethod
    def _polarity(xs, ys, ps, flip, swap_mask):
        """Polarity augmentation, applied once, to the event polarity itself."""
        if not len(ps):
            return ps
        p_eff = 1 - ps if flip else ps
        if swap_mask is not None:
            sel = swap_mask[ys.astype(np.intp), xs.astype(np.intp)]
            p_eff = np.where(sel, 1 - p_eff, p_eff)
        return p_eff

    def _splat_lnes(self, xs, ys, ps, ms_rel, window, height, width) -> np.ndarray:
        """Delegates to the one shared builder, which `eval_track` also calls.

        The two used to be separate implementations kept in step by inspection. They are now the
        same function, so a representation change cannot reach training without reaching
        evaluation -- with more than one plane there is much more to get out of step.
        """
        return EV.splat_event_image(xs, ys, ps, ms_rel, window, height, width,
                                    self.event_channels)

    # ---------------------------------------------------------------- getitem
    def __getitem__(self, idx: int):
        # Per-sample stream: augmentation is a pure function of (seed, epoch-free index).
        rng = np.random.default_rng((self.seed * 1_000_003 + idx) & 0x7FFFFFFF)
        si, end, window, _ = self._window(idx, rng)
        h = self._handle(si)
        start = end - window + 1

        xs, ys, ps, us = self._raw_events(h, end, window)
        target = self._to_target(h.pos51[end])
        prev = self._to_target(h.pos51[start])
        camera_K = h.camera_K.copy()

        # -------- domain randomisation, mirrored onto labels and intrinsics
        n_slots = self.height * self.width * 2 * window
        dr = DR.sample_params(self.domrand, rng, n_slots)
        if self.domrand.enabled:
            xs, ys, ps, us = self._domrand_events(xs, ys, ps, us, window, dr, camera_K, rng)
            stacked, camera_K = DR.transform_labels(
                np.stack([prev, target]), dr, camera_K, h.j0, self.render_scale
            )
            prev, target = stacked[0], stacked[1]

        # -------- teacher forcing noise on the previous state
        if self.train:
            noise = self._sample_prev_noise(rng)
            if noise is not None:
                prev = prev + noise

        flip = self.polarity_flip and int(rng.integers(0, 2)) == 1
        swap_mask = None
        if self.pixel_polarity_swap and self.train:
            swap_mask = rng.integers(0, 2, size=(self.height, self.width)).astype(bool)
        p_eff = self._polarity(xs, ys, ps, flip, swap_mask)

        packet = self._pack(
            si, h, xs, ys, p_eff, us, window,
            t_start_us=int(start) * 1000, t_end_us=(int(end) + 1) * 1000,
            target=target, prev=prev, camera_K=camera_K,
            is_start=bool(start == int(self.index[idx][2])),
            is_end=bool(end == int(self.index[idx][3]) - 1),
            meta={"window_ms": window, "end_idx": end, "seq": h.seq,
                  "category": h.category, "domrand": dr},
        )
        if self.input_mode == "legacy_lnes":
            # Legacy 5-tuple so the existing training loop and model take this dataset as-is.
            return (
                torch.from_numpy(packet.lnes),
                torch.from_numpy(packet.prev_state),
                torch.from_numpy(packet.target),
                torch.from_numpy(packet.betas),
                torch.from_numpy(packet.camera_K),
            )
        if not self.unroll_pair:
            return packet
        # Drawn after the main window is complete, so every draw the non-pair path makes keeps its
        # position in the stream and `unroll_pair=False` stays bitwise identical to S1.
        lead = self._leading_packet(idx, si, h, start, window, dr, flip, swap_mask, rng)
        return lead, packet

    def _pack(self, si, h, xs, ys, p_eff, us, window, *, t_start_us, t_end_us,
              target, prev, camera_K, is_start, is_end, meta) -> EventPacket:
        out: Dict[str, object] = {}
        if self.input_mode in ("legacy_lnes", "both"):
            out["lnes"] = self._splat_lnes(xs, ys, p_eff, us // 1000, window,
                                           self.height, self.width)
        if self.input_mode in ("raw_packed", "both"):
            ev = np.empty((len(xs), 4), dtype=np.float32)
            if len(xs):
                ev[:, 0] = xs
                ev[:, 1] = ys
                ev[:, 2] = us.astype(np.float32) * 1e-6
                ev[:, 3] = p_eff
            out["events"] = ev
        return EventPacket(
            events=out.get("events", np.zeros((0, 4), np.float32)),
            sequence_id=si,
            t_start_us=int(t_start_us),
            t_end_us=int(t_end_us),
            is_sequence_start=bool(is_start),
            is_sequence_end=bool(is_end),
            target=np.asarray(target, np.float32),
            prev_state=np.asarray(prev, np.float32),
            betas=h.betas.copy(),
            camera_K=np.asarray(camera_K, np.float32),
            lnes=out.get("lnes"),
            meta=meta,
        )

    def _leading_packet(self, idx, si, h, main_start, window, dr, flip, swap_mask,
                        rng) -> EventPacket:
        """The window immediately before the main one, for the S22 unroll pair.

        It ends exactly where the main window starts and its target is the main window's
        *un-noised* conditioning state, so a prediction on it estimates what the main window is
        conditioned on -- which is what turns a pair into one step of the deployed recursion.

        At a run boundary there is nothing before the main window. The pair then degenerates to a
        zero-length interval at that same instant, where the zero-event gate returns the
        conditioning state unchanged and the sample falls back to plain teacher forcing.
        """
        run_a = int(self.index[idx][2])
        end_l = int(main_start) - 1
        w = min(int(window), end_l - run_a + 1)
        camera_K = h.camera_K.copy()
        if w < 1:
            start_l = int(main_start)
            xs = ys = ps = us = np.zeros(0, np.int64)
        else:
            start_l = end_l - w + 1
            xs, ys, ps, us = self._raw_events(h, end_l, w)
        target = self._to_target(h.pos51[int(main_start)])
        prev = self._to_target(h.pos51[start_l])

        if self.domrand.enabled:
            if w >= 1:
                xs, ys, ps, us = self._domrand_events(xs, ys, ps, us, w, dr, camera_K, rng)
            stacked, camera_K = DR.transform_labels(
                np.stack([prev, target]), dr, camera_K, h.j0, self.render_scale
            )
            prev, target = stacked[0], stacked[1]
        if self.train:
            noise = self._sample_prev_noise(rng)
            if noise is not None:
                prev = prev + noise
        p_eff = self._polarity(xs, ys, ps, flip, swap_mask)

        return self._pack(
            si, h, xs, ys, p_eff, us, max(w, 1),
            t_start_us=start_l * 1000, t_end_us=int(main_start) * 1000,
            target=target, prev=prev, camera_K=camera_K,
            is_start=bool(start_l == run_a), is_end=False,
            meta={"window_ms": max(w, 0), "end_idx": end_l, "seq": h.seq,
                  "category": h.category, "domrand": dr, "leading": True},
        )


# -------------------------------------------------------------------- builders
def sequences_for_split(root: Path, split: str,
                        manifest_path: Optional[Path] = None) -> List[Tuple[str, str]]:
    """`[(seq, legacy_dir)]` for a split, from the SemKine manifest or the legacy splits."""
    mp = Path(manifest_path) if manifest_path else root / "splits_semkine.json"
    if mp.exists():
        m = json.loads(mp.read_text())
        if split in m:
            return [(s, m[split]["legacy_dir"][s]) for s in m[split]["trials"]]
    legacy = json.loads((root / "splits.json").read_text())
    key = split if split in legacy else ("val" if split == "test" else split)
    return [(s, key) for s in legacy[key]["trials"]]


def splits_manifest(cfg: dict) -> Optional[Path]:
    """The manifest a config pins itself to, so which subject split a run used is recorded.

    Unset means the loader's default (`splits_semkine.json`). Relative names resolve against the
    data root, which is where both manifests live.
    """
    name = (cfg.get("DATA", {}) or {}).get("SPLITS_MANIFEST")
    if not name:
        return None
    p = Path(name)
    return p if p.is_absolute() else Path(cfg["DATA"]["ROOT"]) / p


def build_dataset(cfg: dict, split: str, components: np.ndarray,
                  train: Optional[bool] = None, input_mode: Optional[str] = None,
                  manifest_path: Optional[Path] = None) -> SemKineDataset:
    root = Path(cfg["DATA"]["ROOT"])
    manifest_path = manifest_path or splits_manifest(cfg)
    if train is None:
        train = split == "train"
    track = cfg.get("TRACK", {}) or {}
    aug = cfg.get("AUG", {}) or {}
    ev = cfg.get("EVAL", {}) or {}
    return SemKineDataset(
        root=root,
        sequences=sequences_for_split(root, split, manifest_path),
        components=components,
        mano_npz=cfg["MANO"]["NPZ"],
        input_mode=input_mode or cfg["DATA"].get("INPUT_MODE", "legacy_lnes"),
        window_min=int(cfg["DATA"]["WINDOW_MIN"]),
        window_max=int(cfg["DATA"]["WINDOW_MAX"]),
        width=int(cfg["DATA"]["WIDTH"]),
        height=int(cfg["DATA"]["HEIGHT"]),
        fixed_window=None if train else int(ev.get("WINDOW_MS", 100)),
        train=train,
        speed_aug=bool(aug.get("SPEED_AUG", True)),
        polarity_flip=bool(aug.get("POLARITY_FLIP", True)),
        pixel_polarity_swap=bool(aug.get("PIXEL_POLARITY_SWAP", True)),
        domrand=DR.DomRandConfig.from_cfg(cfg),
        prev_noise=(float(track.get("PREV_NOISE_T", 0.0)),
                    float(track.get("PREV_NOISE_R", 0.0)),
                    float(track.get("PREV_NOISE_POSE", 0.0))),
        prev_noise_mode=str(track.get("PREV_NOISE_MODE", "gaussian")),
        prev_noise_large=(float(track.get("PREV_NOISE_LARGE_T", 0.05)),
                          float(track.get("PREV_NOISE_LARGE_R", 0.3)),
                          float(track.get("PREV_NOISE_LARGE_POSE", 0.3))),
        prev_noise_mix=(float(track.get("PREV_NOISE_P_SMALL", 0.5)),
                        float(track.get("PREV_NOISE_P_LARGE", 0.3)),
                        float(track.get("PREV_NOISE_P_CORR", 0.2))),
        render_scale=float(cfg.get("MODEL", {}).get("RENDER_SCALE", 0.375)),
        seed=int(cfg.get("SEED", 0)),
        unroll_pair=bool(track.get("UNROLL_PAIR", False)) and bool(train),
        event_channels=EV.event_channels(cfg),
    )
