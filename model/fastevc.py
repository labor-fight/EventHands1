import numpy as np
import torch
from torch.utils.data import Dataset
import os
import struct
from tqdm import tqdm

try:
    import evcreader
except ImportError:
    evcreader = None

class EVCDataset(Dataset):
    def __init__(self):
        pass

    def load(self, name, offset, count):
        evt = open(name+".evc", "rb")
        meta = open(name+".meta", "rb")
        ncomps, = struct.unpack("<i", meta.read(4))

        self.offset = offset
        self.count = count

        self.events = []
        self.pos = []

        # skip first frame
        # gt = struct.unpack("<"+"d"*ncomps, meta.read(8*ncomps))
        # magic = struct.unpack("<BB", meta.read(2))
        # assert(magic == [4, 13])

        # print(gt, magic)

        # ignore first new frame construct
        x, y, p = struct.unpack('<HBB', evt.read(4))
        assert(p == 255)
        # # ignore first frame
        # x, y, p = struct.unpack('<HBB', evt.read(4))
        # print(x, y, p)
        # while True:
        #     x, y, p = struct.unpack('<HBB', evt.read(4))
        #     if p == 255:
        #         break

        # start reading frames
        for idx in tqdm(range(self.offset+self.count)):
            # read events
            ev = []
            while True:
                x, y, p = struct.unpack('<HBB', evt.read(4))
                if p == 255:
                    break
                ev.append((x, y, p))
            ev = np.array(ev, np.int16)
            self.events.append(ev)

            # read metadata
            gt = struct.unpack("<"+"d"*ncomps, meta.read(8*ncomps))
            magic = struct.unpack("<BB", meta.read(2))
            if idx == 0:
                print(gt, magic)
            assert(magic == (4, 13))

            self.pos.append(np.array(gt, np.float32))
        evt.close()
        meta.close()

    def view(self, offset, count):
        assert(self.offset + offset + count <= self.count)
        res = EVCDataset()
        res.offset = self.offset + offset
        res.count = count
        res.events = self.events
        res.pos = self.pos
        return res

    def __len__(self):
        return self.count

    WINDOW = 100
    def __getitem__(self, idx):
        WIDTH, HEIGHT = 240, 180
        WINDOW = 100
        idx += self.offset
        img = np.zeros((HEIGHT, WIDTH, 2), np.float32)
        prevpos = self.pos[idx]
        for i in range(WINDOW):
            for x, y, p in self.events[idx+i]:
                img[y, x, p] = i/WINDOW

            # ignore if it's randomizer transition frame that
            # is every 50 seconds in the dataset
            transition_frame = ((idx+i)%(50*1000) == 0)
            if transition_frame:  # TODO: what if it's the last frame?
                img *= 0
                prevpos = self.pos[idx+i]
            pos = self.pos[idx+i]
        X = torch.tensor(img, dtype=torch.float32)
        prevpos = torch.tensor(prevpos, dtype=torch.float32)
        Y = torch.tensor(pos, dtype=torch.float32)
        return X, prevpos, Y

class EVCDatasetFast(Dataset):
    def __init__(self):
        pass

    def load(self, name, offset, count):
        # evt = open(name+".evc", "rb")
        meta = open(name+".meta", "rb")
        ncomps, = struct.unpack("<i", meta.read(4))

        self.offset = offset
        self.count = count

        # self.events = []
        self.pos = []

        # skip first frame
        # gt = struct.unpack("<"+"d"*ncomps, meta.read(8*ncomps))
        # magic = struct.unpack("<BB", meta.read(2))
        # assert(magic == [4, 13])

        # print(gt, magic)

        # # ignore first frame
        # x, y, p = struct.unpack('<HBB', evt.read(4))
        # print(x, y, p)
        # while True:
        #     x, y, p = struct.unpack('<HBB', evt.read(4))
        #     if p == 255:
        #         break

        # start reading event camera frames
        self.events = evcreader.read(name+".evc", self.offset+self.count)
        # evtdt = np.dtype([('x', np.uint16), ('y', np.uint8), ('p', np.uint8)])
        # BUF_SIZE = 65536*512
        # # BUF_SIZE = 512
        # buf = np.fromfile(evt, dtype=evtdt, count=BUF_SIZE)
        # new_frame_pos = np.where(buf['p']==255)[0]
        # nfp_off = 0
        # # buf = np.hsplit(buf, new_frame_pos)
        # # print(buf)

        # for idx in tqdm(range(self.offset+self.count)):
        #     # read events
        #     if idx+1-nfp_off >= len(new_frame_pos):
        #         tmpbuf = np.fromfile(evt, dtype=evtdt, count=BUF_SIZE)
        #         # read until we find the next frame starting point
        #         while np.all(tmpbuf['p'] != 255):
        #             tmpbuf = np.r_[tmpbuf, np.fromfile(evt, dtype=evtdt, count=BUF_SIZE)]
        #         # concat with old data
        #         buf = np.r_[buf[new_frame_pos[idx-nfp_off]:], tmpbuf]
        #         # reset the frame indexing starting with the current frame
        #         new_frame_pos = np.where(buf['p']==255)[0]
        #         nfp_off = idx
        #     # extract the frame, unite the columns
        #     ev = buf[new_frame_pos[idx-nfp_off]+1:new_frame_pos[idx+1-nfp_off]]
        #     ev = np.c_[ev['x'].astype(np.uint8),
        #                ev['y'].astype(np.uint8),
        #                ev['p'].astype(np.uint8)]
        #     self.events.append(ev)

        # read metadata: ncomps of doubles and the 2-byte magic
        metadt = np.dtype([('data', np.float64, ncomps), ('m0', np.uint8), ('m1', np.uint8)])
        self.pos = np.fromfile(meta, dtype=metadt, count=(self.offset+self.count))
        # check the magic
        assert(np.all(self.pos['m0'] == 4) and np.all(self.pos['m1'] == 13))
        # convert to float
        self.pos = self.pos['data'].astype(np.float32)

        # for idx in tqdm(range(self.offset+self.count)):
        #     # read metadata
        #     gt = struct.unpack("<"+"d"*ncomps, meta.read(8*ncomps))
        #     magic = struct.unpack("<BB", meta.read(2))
        #     if idx == 0:
        #         print(gt, magic)
        #     assert(magic == (4, 13))

        #     self.pos.append(np.array(gt, np.float32))
        # evt.close()
        meta.close()

    def view(self, offset, count):
        assert(self.offset + offset + count <= self.count)
        res = type(self)()
        res.offset = self.offset + offset
        res.count = count
        res.events = self.events
        res.pos = self.pos
        return res

    def __len__(self):
        return self.count

    WINDOW = 100
    def __getitem__(self, idx):
        WIDTH, HEIGHT = 240, 180
        # WINDOW = 100
        def log_uniform_int(a, b):
            l = np.random.uniform(np.log(a), np.log(b))
            return int(round(np.exp(l)))
        WINDOW = log_uniform_int(30,300)  # speed augmentation
        idx += self.offset
        img = np.zeros((HEIGHT, WIDTH, 2), np.float32)
        prevpos = self.pos[idx+0]
        for i in range(WINDOW):
            xs, ys, ps = self.events[idx+i].T
            img[ys, xs, ps] = i/WINDOW
            # this is EOI
            # img[ys, xs, ps] = 1.
            # this is ECI
            # img[ys, xs, ps] += 1./self.WINDOW*2/2
            # if you add the next line to ECI, this is ECI-S
            # img[ys, xs, 1-ps] += 1./self.WINDOW*2/2

            # ignore if it's randomizer transition frame that
            # is every 50 seconds in the dataset
            transition_frame = ((idx+i)%(50*1000) == 0)
            if transition_frame:
                img *= 0
                prevpos = self.pos[idx+i]
            pos = self.pos[idx+i]

        # random polarity augmentation
        if np.random.randint(0, 2) == 1:
            img[:, :, (0, 1)] = img[:, :, (1, 0)]

        # global random polarity augmentation
        whatmask = np.random.randint(0, 2, size=(HEIGHT, WIDTH))
        mask0 = np.zeros((HEIGHT, WIDTH, 2), dtype=np.bool)
        mask0[..., 0][whatmask==1] = True
        mask1 = np.zeros((HEIGHT, WIDTH, 2), dtype=np.bool)
        mask1[..., 1][whatmask==1] = True

        img[mask0], img[mask1] = img[mask1], img[mask0]

        X = torch.tensor(img, dtype=torch.float32)
        prevpos = torch.tensor(prevpos, dtype=torch.float32)

        Y = torch.tensor(pos, dtype=torch.float32)
        return X, prevpos, Y


class HandData51Dataset(Dataset):
    """
    hand_data51 loader: memmap events + 51D .meta, with pose_repr target adapter.

    Returns (X, prevpos, Y, betas) where Y matches MODEL.POSE_REPR layout.
    Sampling is restricted to continuous valid runs with enough LNES history.
    """

    def __init__(
        self,
        root,
        seq,
        split,
        pose_repr="mano_pca6",
        components=None,
        window_min=30,
        window_max=300,
        width=240,
        height=180,
        speed_aug=True,
        polarity_flip=True,
        pixel_polarity_swap=True,
        fixed_window=None,
        train=True,
        prev_noise=(0.0, 0.0, 0.0),
        prev_noise_mode="gaussian",
        prev_noise_large=(0.05, 0.3, 0.3),
        prev_noise_mix=(0.5, 0.3, 0.2),
    ):
        import struct as _struct

        self.root = root
        self.seq = seq
        self.split = split
        self.pose_repr = pose_repr
        self.window_min = int(window_min)
        self.window_max = int(window_max)
        self.width = int(width)
        self.height = int(height)
        self.speed_aug = bool(speed_aug) and train
        self.polarity_flip = bool(polarity_flip) and train
        self.pixel_polarity_swap = bool(pixel_polarity_swap) and train
        self.fixed_window = fixed_window
        self.train = train
        # (sigma_t, sigma_R, sigma_pose): teacher-forcing noise on prevpos to
        # emulate tracking drift / RGB-init error (se(3)-TrackNet style).
        self.prev_noise = tuple(float(s) for s in prev_noise)
        self.prev_noise_mode = str(prev_noise_mode)
        self.prev_noise_large = tuple(float(s) for s in prev_noise_large)
        mix = tuple(float(p) for p in prev_noise_mix)
        s = sum(mix) if mix else 1.0
        self.prev_noise_mix = tuple(p / s for p in mix)

        base = os.path.join(root, split, seq)
        self.events = np.load(base + "_events.npy", mmap_mode="r")
        self.offsets = np.load(base + "_offsets.npy")  # (n_ms+1,)
        aux = np.load(base + "_aux.npz", allow_pickle=True)
        self.betas = np.asarray(aux["betas"], dtype=np.float32)
        self.camera_K = np.asarray(aux["camera_K"], dtype=np.float32).reshape(3, 3)
        self.valid_ms = np.asarray(aux["valid_ms"], dtype=bool)
        self.runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
        self.category = str(aux["category"])

        # Read .meta (ncomps float64 + magic)
        meta_path = base + ".meta"
        with open(meta_path, "rb") as f:
            ncomps, = _struct.unpack("<i", f.read(4))
            assert ncomps == 51, ncomps
            dt = np.dtype([("data", np.float64, ncomps), ("m0", np.uint8), ("m1", np.uint8)])
            raw = np.fromfile(f, dtype=dt)
        assert np.all(raw["m0"] == 4) and np.all(raw["m1"] == 13)
        self.pos51 = raw["data"].astype(np.float32)
        assert len(self.pos51) == len(self.valid_ms)

        if components is None:
            raise ValueError("components (hands_components) required")
        self.components = np.asarray(components, dtype=np.float32)

        # Precompute sample end indices: last ms of LNES window must be valid,
        # and [end-window_max+1, end] must lie inside a valid run.
        # We store end indices (inclusive) for windows ending at that frame.
        max_w = self.window_max if self.fixed_window is None else int(self.fixed_window)
        min_w = self.window_min if self.fixed_window is None else int(self.fixed_window)
        self.min_w = min_w
        self.max_w = max_w
        ends = []
        for a, b in self.runs:
            # need at least max_w frames of history inside the run for speed aug
            start_end = a + max_w - 1
            if start_end >= b:
                # still allow if run long enough for min_w
                start_end = a + min_w - 1
            if start_end >= b:
                continue
            ends.append(np.arange(start_end, b, dtype=np.int64))
        if ends:
            self.sample_ends = np.concatenate(ends)
        else:
            self.sample_ends = np.zeros(0, dtype=np.int64)

        # Precompute PCA projection matrix pinv(C6) for 12D adapter
        C6 = self.components[:6]
        self.pinv_C6 = np.linalg.pinv(C6).astype(np.float32)  # (45, 6)

    def __len__(self):
        return int(len(self.sample_ends))

    def _params_to_target(self, p51):
        t = p51[0:3]
        R = p51[3:6]
        residual = p51[6:51]
        if self.pose_repr == "mano_full_axis_angle":
            return p51
        if self.pose_repr == "mano_pca6":
            alpha = residual @ self.pinv_C6
            return np.concatenate([alpha, t, R]).astype(np.float32)
        raise ValueError(self.pose_repr)

    def _fill_noise(self, sig_t, sig_r, sig_pose):
        if self.pose_repr == "mano_full_axis_angle":
            noise = np.empty(51, np.float32)
            noise[0:3] = np.random.randn(3) * sig_t
            noise[3:6] = np.random.randn(3) * sig_r
            noise[6:51] = np.random.randn(45) * sig_pose
        else:  # mano_pca6: [alpha6, t3, R3]
            noise = np.empty(12, np.float32)
            noise[0:6] = np.random.randn(6) * sig_pose
            noise[6:9] = np.random.randn(3) * sig_t
            noise[9:12] = np.random.randn(3) * sig_r
        return noise

    def _correlated_noise(self, mag_t, mag_r, mag_pose):
        def directed(n, mag):
            v = np.random.randn(n).astype(np.float32)
            nrm = float(np.linalg.norm(v)) + 1e-8
            return v / nrm * np.float32(np.random.uniform(0.0, mag))

        if self.pose_repr == "mano_full_axis_angle":
            noise = np.empty(51, np.float32)
            noise[0:3] = directed(3, mag_t)
            noise[3:6] = directed(3, mag_r)
            noise[6:51] = directed(45, mag_pose)
        else:
            noise = np.empty(12, np.float32)
            noise[0:6] = directed(6, mag_pose)
            noise[6:9] = directed(3, mag_t)
            noise[9:12] = directed(3, mag_r)
        return noise

    def _sample_prev_noise(self):
        """Teacher-forcing noise on prevpos. mixed mode = small / large / correlated."""
        sig_t, sig_r, sig_pose = self.prev_noise
        if sig_t == 0.0 and sig_r == 0.0 and sig_pose == 0.0:
            return None
        if self.prev_noise_mode != "mixed":
            return self._fill_noise(sig_t, sig_r, sig_pose)
        u = float(np.random.rand())
        p_small, p_large, _ = self.prev_noise_mix
        if u < p_small:
            return self._fill_noise(sig_t, sig_r, sig_pose)
        if u < p_small + p_large:
            lt, lr, lp = self.prev_noise_large
            return self._fill_noise(lt, lr, lp)
        lt, lr, lp = self.prev_noise_large
        return self._correlated_noise(lt, lr, lp)

    def _build_lnes(self, end_idx, window):
        start = end_idx - window + 1
        img = np.zeros((self.height, self.width, 2), np.float32)
        # Vectorized: concatenate all events in [start, end_idx], assign i/WINDOW
        # offsets[i] .. offsets[i+1] are events belonging to ms frame i
        a0 = int(self.offsets[start])
        a1 = int(self.offsets[end_idx + 1])
        if a1 <= a0:
            return img
        ev = self.events[a0:a1]  # (N, 3) uint8
        # Build per-event time value = (frame_index - start) / window
        counts = np.diff(self.offsets[start : end_idx + 2]).astype(np.int64)
        if counts.sum() == 0:
            return img
        frame_rel = np.repeat(np.arange(window, dtype=np.float32), counts)
        tval = frame_rel / float(window)
        xs = ev[:, 0].astype(np.intp)
        ys = ev[:, 1].astype(np.intp)
        # Clamp polarity to {0,1}
        ps = np.clip(ev[:, 2].astype(np.intp), 0, 1)
        img[ys, xs, ps] = tval
        return img

    def __getitem__(self, idx):
        end_idx = int(self.sample_ends[idx])
        if self.fixed_window is not None:
            window = int(self.fixed_window)
        elif self.speed_aug:
            lo, hi = np.log(self.min_w), np.log(self.max_w)
            window = int(round(np.exp(np.random.uniform(lo, hi))))
            window = max(self.min_w, min(self.max_w, window))
        else:
            window = self.min_w
        # Ensure window fits in the valid run containing end_idx
        # Find run
        for a, b in self.runs:
            if a <= end_idx < b:
                window = min(window, end_idx - a + 1)
                break
        window = max(1, window)

        img = self._build_lnes(end_idx, window)
        start = end_idx - window + 1
        prevpos = self._params_to_target(self.pos51[start])
        pos = self._params_to_target(self.pos51[end_idx])

        if self.train:
            noise = self._sample_prev_noise()
            if noise is not None:
                prevpos = prevpos + noise

        if self.polarity_flip and np.random.randint(0, 2) == 1:
            img[:, :, (0, 1)] = img[:, :, (1, 0)]

        if self.pixel_polarity_swap:
            whatmask = np.random.randint(0, 2, size=(self.height, self.width))
            mask0 = np.zeros((self.height, self.width, 2), dtype=bool)
            mask0[..., 0][whatmask == 1] = True
            mask1 = np.zeros((self.height, self.width, 2), dtype=bool)
            mask1[..., 1][whatmask == 1] = True
            tmp = img[mask0].copy()
            img[mask0] = img[mask1]
            img[mask1] = tmp

        X = torch.from_numpy(img)
        prevpos_t = torch.from_numpy(prevpos.copy())
        Y = torch.from_numpy(pos.copy())
        betas = torch.from_numpy(self.betas.copy())
        camera_K = torch.from_numpy(self.camera_K.copy())
        return X, prevpos_t, Y, betas, camera_K


def build_hand_data51_datasets(cfg, split, components, train=None):
    """Build list of HandData51Dataset for a split from cfg."""
    import json

    root = cfg["DATA"]["ROOT"]
    with open(os.path.join(root, "splits.json")) as f:
        splits = json.load(f)
    # Accept both 'val' and 'test' keys
    key = split if split in splits else ("val" if split == "test" and "val" in splits else split)
    trials = splits[key]["trials"]
    if train is None:
        train = split == "train"
    pose_repr = cfg["MODEL"]["POSE_REPR"]
    track = cfg.get("TRACK", {}) or {}
    prev_noise = (
        float(track.get("PREV_NOISE_T", 0.0)),
        float(track.get("PREV_NOISE_R", 0.0)),
        float(track.get("PREV_NOISE_POSE", 0.0)),
    )
    prev_noise_large = (
        float(track.get("PREV_NOISE_LARGE_T", 0.05)),
        float(track.get("PREV_NOISE_LARGE_R", 0.3)),
        float(track.get("PREV_NOISE_LARGE_POSE", 0.3)),
    )
    prev_noise_mix = (
        float(track.get("PREV_NOISE_P_SMALL", 0.5)),
        float(track.get("PREV_NOISE_P_LARGE", 0.3)),
        float(track.get("PREV_NOISE_P_CORR", 0.2)),
    )
    prev_noise_mode = str(track.get("PREV_NOISE_MODE", "gaussian"))
    ds_list = []
    for seq in trials:
        ds_list.append(
            HandData51Dataset(
                root=root,
                seq=seq,
                split=key if key in ("train", "val") else split,
                pose_repr=pose_repr,
                components=components,
                window_min=cfg["DATA"]["WINDOW_MIN"],
                window_max=cfg["DATA"]["WINDOW_MAX"],
                width=cfg["DATA"]["WIDTH"],
                height=cfg["DATA"]["HEIGHT"],
                speed_aug=cfg["AUG"]["SPEED_AUG"],
                polarity_flip=cfg["AUG"]["POLARITY_FLIP"],
                pixel_polarity_swap=cfg["AUG"]["PIXEL_POLARITY_SWAP"],
                train=train,
                prev_noise=prev_noise,
                prev_noise_mode=prev_noise_mode,
                prev_noise_large=prev_noise_large,
                prev_noise_mix=prev_noise_mix,
            )
        )
    return ds_list


if __name__ == '__main__':
    # Test whether it's the same as the gold one
    DATASETS = ['new_handfingersvlad_hard_ng', 'new_handfingersvlad_hard_ng2']
    # DATA_DIR = 'data'
    DATA_DIR = '/scratch/inf0/user/vrudnev/data'
    COUNT = 1000*110  # 110 seconds of 1000 FPS
    # COUNT = 1000*1000
    name = DATASETS[0]
    ds1 = EVCDatasetFast()
    print('loading dataset', name)
    ds1.load(os.path.join(DATA_DIR, name), 0, COUNT)
    print('loaded dataset', name)

    ds2 = EVCDataset()
    print('loading dataset', name)
    ds2.load(os.path.join(DATA_DIR, name), 0, COUNT)
    print('loaded dataset', name)

    def cool_assert(a, b):
        if not torch.allclose(a, b):
            idx = (a != b)
            print("IDX:", torch.where(idx))
            print("A:", a[idx])
            print("B:", b[idx])
            assert(False)

    assert(len(ds1.events) == len(ds2.events))
    try:
        for i, (x, y) in enumerate(zip(ds1.events, ds2.events)):
            assert(np.allclose(x, y))
    except:
        print(i, x, y)
        raise

    assert(len(ds1.pos) == len(ds2.pos))
    try:
        for i, (x, y) in enumerate(zip(ds1.pos, ds2.pos)):
            assert(np.allclose(x, y))
    except:
        print(i, x, y)
        raise


    DIM = max(len(ds1[0]), len(ds2[0]))
    assert(DIM == 3 and len(ds1[0]) == DIM and len(ds2[0]) == DIM)

    for d in range(DIM):
        cool_assert(ds1[0*1000][d], ds2[0*1000][d])

    for d in range(DIM):
        cool_assert(ds1[10*1000][d], ds2[10*1000][d])

    for d in range(DIM):
        cool_assert(ds1[100*1000][d], ds2[100*1000][d])

    for i in tqdm(range(0, 1000*100, 123)):
        for d in range(DIM):
            cool_assert(ds1[i][d], ds2[i][d])

    print('Congrats, all is OK; fast data loading works correctly.')
