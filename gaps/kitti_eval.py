"""KITTI 3D object detection evaluation (AP over 40 recall positions), used for every number in the paper.

A port of the reference Python evaluator used by SECOND, PointPillars and OpenPCDet (kitti_object_eval_python), which
itself follows the official devkit (evaluate_object.cpp). Differences from that evaluator, all deliberate:
  * a detection's 2D box (used only for the minimum-height filter) is the projection of its 3D box, clipped to the
    image, as OpenPCDet writes it;
  * when one class is scored, detections of other classes are left out.

Boxes everywhere are (x, y, z_centre, w, l, h, yaw) in the LiDAR frame.
"""
import os

import numpy as np
from numba import njit

from .calib import Calib
from .iou3d import iou3d

CLASSES = ("Car", "Pedestrian", "Cyclist")
CLS2IDX = {"Car": 1, "Pedestrian": 2, "Cyclist": 3}
MIN_HEIGHT = [40, 25, 25]
MAX_OCCLUSION = [0, 1, 2]
MAX_TRUNCATION = [0.15, 0.3, 0.5]
N_SAMPLE_PTS = 41
MIN_OVERLAP = {"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5}
_NEIGHBOUR = {"Car": "van", "Pedestrian": "person_sitting"}


class Obj:
    """one line of a KITTI label file"""
    __slots__ = ("type", "truncated", "occluded", "alpha", "box2d", "h", "w", "l", "loc", "ry", "cls")

    def __init__(self, f):
        self.type = f[0]
        self.truncated = float(f[1])
        self.occluded = int(float(f[2]))
        self.alpha = float(f[3])
        self.box2d = np.array([float(x) for x in f[4:8]], np.float32)
        self.h, self.w, self.l = (float(f[8]), float(f[9]), float(f[10]))
        self.loc = np.array([float(f[11]), float(f[12]), float(f[13])], np.float32)
        self.ry = float(f[14])
        self.cls = CLS2IDX.get(self.type, -1)


class Kitti:
    """reads labels and calibration from a KITTI 'training' folder (label_2/, calib/)"""

    def __init__(self, root):
        self.root = root
        self._cache = {}

    def labels(self, fid):
        out = []
        with open(os.path.join(self.root, "label_2", fid + ".txt")) as fh:
            for line in fh:
                f = line.split()
                if len(f) >= 15:
                    out.append(Obj(f))
        return out

    def calib(self, fid):
        if fid not in self._cache:
            self._cache[fid] = Calib.from_file(os.path.join(self.root, "calib", fid + ".txt"))
        return self._cache[fid]


def objs_to_velo_boxes(objs, calib):
    """KITTI label objects -> (N,7) LiDAR-frame boxes (KITTI stores the bottom centre in the rectified camera frame)"""
    if not objs:
        return np.zeros((0, 7), np.float32)
    loc = np.stack([o.loc for o in objs])
    hh = np.array([o.h for o in objs], np.float32)
    centre_rect = loc.copy()
    centre_rect[:, 1] -= hh / 2.0
    cen = calib.cam2_to_velo(calib.rect_to_cam2(centre_rect))
    box = np.zeros((len(objs), 7), np.float32)
    box[:, 0:3] = cen
    box[:, 3] = [o.w for o in objs]
    box[:, 4] = [o.l for o in objs]
    box[:, 5] = hh
    box[:, 6] = [-o.ry - np.pi / 2 for o in objs]
    return box


def box2d_from_velo(boxes, calib, hw):
    """(N,7) LiDAR boxes -> (N,4) image boxes, clipped to the image"""
    n = boxes.shape[0]
    if n == 0:
        return np.zeros((0, 4), np.float32)
    x, y, z = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    w, l, h, yaw = boxes[:, 3], boxes[:, 4], boxes[:, 5], boxes[:, 6]
    dx = np.stack([l, l, -l, -l, l, l, -l, -l], 1) * 0.5
    dy = np.stack([-w, w, w, -w, -w, w, w, -w], 1) * 0.5
    dz = np.stack([-h, -h, -h, -h, h, h, h, h], 1) * 0.5
    c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    pts = np.stack([x[:, None] + dx * c - dy * s, y[:, None] + dx * s + dy * c, z[:, None] + dz], -1).reshape(-1, 3)
    cam = calib.velo_to_cam2(pts)
    depth = cam[:, 2].reshape(n, 8)
    cam[:, 2] = np.maximum(cam[:, 2], 0.1)
    uv = calib.project_cam2(cam).reshape(n, 8, 2)
    H, W = hw
    out = np.stack([uv[..., 0].min(1), uv[..., 1].min(1), uv[..., 0].max(1), uv[..., 1].max(1)], 1)
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, W - 1)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, H - 1)
    out[(depth <= 0.1).all(1)] = 0.0
    return out.astype(np.float32)


def clean_data(objs, dt_box2d, cls_name, level):
    cur = cls_name.lower()
    ig_gt = np.empty(len(objs), np.int64)
    n_valid = 0
    for i, o in enumerate(objs):
        name = o.type.lower()
        height = o.box2d[3] - o.box2d[1]
        valid = 1 if name == cur else (0 if _NEIGHBOUR.get(cls_name) == name else -1)
        ignore = (o.occluded > MAX_OCCLUSION[level] or o.truncated > MAX_TRUNCATION[level]
                  or height <= MIN_HEIGHT[level])
        if valid == 1 and not ignore:
            ig_gt[i] = 0
            n_valid += 1
        elif valid == 0 or (ignore and valid == 1):
            ig_gt[i] = 1
        else:
            ig_gt[i] = -1
    hgt = np.abs(dt_box2d[:, 3] - dt_box2d[:, 1])
    ig_dt = np.where(hgt < MIN_HEIGHT[level], 1, 0).astype(np.int64)
    return n_valid, ig_gt, ig_dt


@njit(cache=True)
def _statistics(overlaps, scores, ig_gt, ig_dt, min_overlap, thresh, compute_fp):
    det_size, gt_size = overlaps.shape[0], overlaps.shape[1]
    assigned = np.zeros(det_size, np.bool_)
    below = np.zeros(det_size, np.bool_)
    if compute_fp:
        for i in range(det_size):
            if scores[i] < thresh:
                below[i] = True
    NO_DET = -10000000.0
    tp, fp, fn = 0, 0, 0
    ths = np.zeros(gt_size, np.float64)
    nth = 0
    for i in range(gt_size):
        if ig_gt[i] == -1:
            continue
        det_idx = -1
        valid = NO_DET
        max_ov = 0.0
        assigned_ignored = False
        for j in range(det_size):
            if ig_dt[j] == -1 or assigned[j] or below[j]:
                continue
            ov = overlaps[j, i]
            if (not compute_fp) and ov > min_overlap and scores[j] > valid:
                det_idx = j
                valid = scores[j]
            elif compute_fp and ov > min_overlap and (ov > max_ov or assigned_ignored) and ig_dt[j] == 0:
                max_ov = ov
                det_idx = j
                valid = 1.0
                assigned_ignored = False
            elif compute_fp and ov > min_overlap and valid == NO_DET and ig_dt[j] == 1:
                det_idx = j
                valid = 1.0
                assigned_ignored = True
        if valid == NO_DET and ig_gt[i] == 0:
            fn += 1
        elif valid != NO_DET and (ig_gt[i] == 1 or ig_dt[det_idx] == 1):
            assigned[det_idx] = True
        elif valid != NO_DET:
            tp += 1
            ths[nth] = scores[det_idx]
            nth += 1
            assigned[det_idx] = True
    if compute_fp:
        for i in range(det_size):
            if not (assigned[i] or ig_dt[i] == -1 or ig_dt[i] == 1 or below[i]):
                fp += 1
    return tp, fp, fn, ths[:nth]


@njit(cache=True)
def _pr_at_thresholds(overlaps, scores, ig_gt, ig_dt, min_overlap, thresholds, pr):
    for t in range(thresholds.shape[0]):
        tp, fp, fn, _ = _statistics(overlaps, scores, ig_gt, ig_dt, min_overlap, thresholds[t], True)
        pr[t, 0] += tp
        pr[t, 1] += fp
        pr[t, 2] += fn


def _thresholds(scores, num_gt):
    scores = np.sort(scores)[::-1]
    current_recall = 0.0
    out = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        r_recall = (i + 2) / num_gt if i < len(scores) - 1 else l_recall
        if (r_recall - current_recall) < (current_recall - l_recall) and i < len(scores) - 1:
            continue
        out.append(score)
        current_recall += 1 / (N_SAMPLE_PTS - 1.0)
    return np.array(out, np.float64)


def _ap(prepared, cls_name, level):
    mo = MIN_OVERLAP[cls_name]
    per, all_th, n_gt = [], [], 0
    for objs, dt2d, ov, sc in prepared:
        n_valid, ig_gt, ig_dt = clean_data(objs, dt2d, cls_name, level)
        n_gt += n_valid
        _, _, _, th = _statistics(ov, sc, ig_gt, ig_dt, mo, 0.0, False)
        all_th.append(th)
        per.append((ov, sc, ig_gt, ig_dt))
    if n_gt == 0:
        return float("nan")
    th = _thresholds(np.concatenate(all_th) if all_th else np.zeros(0), n_gt)
    if th.size == 0:
        return 0.0
    pr = np.zeros((th.size, 4))
    for ov, sc, ig_gt, ig_dt in per:
        _pr_at_thresholds(ov, sc, ig_gt, ig_dt, mo, th, pr)
    prec = np.zeros(N_SAMPLE_PTS)
    for i in range(th.size):
        prec[i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
    for i in range(th.size):
        prec[i] = prec[i:].max()
    return prec[1:].sum() / 40.0 * 100.0


def evaluate(dets, kitti, image_sizes, level=1, min_range=None):
    """3D AP|R40 for Car, Pedestrian, Cyclist at one difficulty level (0 easy, 1 moderate, 2 hard).

    dets: {frame id: {"boxes": (N,7), "name": (N,), "score": (N,)}} for EVERY evaluated frame (also frames without
    detections, whose ground truth still counts).
    min_range: if set, only objects at least this far away count (ground truth closer than that is ignored, detections
    closer than that are dropped) - the paper's "cars beyond 40 m" column uses min_range=40.
    """
    out = []
    for cname in CLASSES:
        prepared = []
        for fid, d in dets.items():
            raw = np.asarray(d["boxes"]).reshape(-1, 7)
            boxes = raw.astype(np.float64)
            m = np.asarray(d["name"]) == cname
            if min_range is not None:
                m &= np.hypot(raw[:, 0], raw[:, 1]) >= min_range
            b, s = boxes[m], np.asarray(d["score"], np.float64)[m]
            objs, cal = kitti.labels(fid), kitti.calib(fid)
            gt = objs_to_velo_boxes(objs, cal).astype(np.float64)
            if min_range is not None:
                objs = [_far_only(o, min_range) for o in objs]
            ov = np.asarray(iou3d(b, gt), np.float64) if len(b) and len(gt) else np.zeros((len(b), len(gt)))
            prepared.append((objs, box2d_from_velo(b, cal, image_sizes[fid]), ov, s))
        out.append(_ap(prepared, cname, level))
    return out


def _far_only(o, min_range):
    """ground truth closer than min_range is marked as ignored (KITTI-style)"""
    if o.cls > 0 and float(np.hypot(o.loc[0], o.loc[2])) < min_range:
        import copy
        o = copy.copy(o)
        o.occluded = 99
    return o
