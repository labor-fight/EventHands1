"""OpenCV display: the LNES image (or the DAVIS346 APS frame) with the tracked / template mesh projected on it."""
from __future__ import annotations

import numpy as np

from .mano import SKELETON


def lnes_image(lnes: np.ndarray, scale: int) -> np.ndarray:
    """(180, 240, 2) LNES -> BGR image, ON events red, OFF events blue, brighter = more recent."""
    img = np.zeros(lnes.shape[:2] + (3,), np.uint8)
    img[..., 2] = (np.clip(lnes[..., 1], 0, 1) ** 0.5 * 255 * (lnes[..., 1] > 0)).astype(np.uint8)
    img[..., 0] = (np.clip(lnes[..., 0], 0, 1) ** 0.5 * 255 * (lnes[..., 0] > 0)).astype(np.uint8)
    both = (lnes[..., 0] > 0) & (lnes[..., 1] > 0)
    img[both, 1] = 160
    return np.repeat(np.repeat(img, scale, 0), scale, 1)


def draw_hand(img: np.ndarray, uv_verts: np.ndarray, uv_joints: np.ndarray, scale: float, color=(0, 255, 0)):
    import cv2
    h, w = img.shape[:2]
    for u, v in (uv_verts[::4] * scale).astype(np.int32):
        if 1 <= u < w - 1 and 1 <= v < h - 1:
            img[v - 1:v + 1, u - 1:u + 1] = color
    j = (uv_joints * scale).astype(np.int32)
    for a, b in SKELETON:
        cv2.line(img, tuple(int(c) for c in j[a]), tuple(int(c) for c in j[b]), color, 1, cv2.LINE_AA)
    cv2.circle(img, tuple(int(c) for c in j[0]), 3, color, -1)
    return img


def put_lines(img: np.ndarray, lines, color=(255, 255, 255)):
    import cv2
    for i, t in enumerate(lines):
        cv2.putText(img, t, (6, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (6, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img
