"""Rotated 3D IoU for the official 3D AP, JIT compiled.

KITTI reports two localisation metrics: bird's-eye view (the box footprint must
overlap) and 3D (the footprint AND the vertical extent must overlap).  Every
number in this project so far was BEV.  The 3D overlap is the BEV polygon
intersection times the height overlap, over the union of the two volumes -
exactly what the official devkit's d3_box_overlap computes in camera
coordinates, done here in the velodyne frame where height is z.

Boxes are x, y, z_centre, w, l, h, yaw (velodyne frame), the convention of
kitti_data.objs_to_velo_boxes and of every detection this project writes.
"""
import numpy as np

try:
    from numba import njit
except ImportError:                                     # pragma: no cover
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not a else a[0]


@njit(cache=True, fastmath=True, inline="always")
def _corners(box, out):
    c = np.cos(box[6])
    s = np.sin(box[6])
    hl = box[4] * 0.5
    hw = box[3] * 0.5
    dx = (hl, hl, -hl, -hl)
    dy = (-hw, hw, hw, -hw)
    for k in range(4):
        out[k, 0] = box[0] + dx[k] * c - dy[k] * s
        out[k, 1] = box[1] + dx[k] * s + dy[k] * c


@njit(cache=True, fastmath=True)
def bev_intersection(a, b):
    """Area of the intersection of two rotated BEV rectangles
    (Sutherland-Hodgman clip of a against b, then the shoelace formula)."""
    ca = np.empty((4, 2), np.float64)
    cb = np.empty((4, 2), np.float64)
    _corners(a, ca)
    _corners(b, cb)
    poly = np.empty((16, 2), np.float64)
    tmp = np.empty((16, 2), np.float64)
    for k in range(4):
        poly[k, 0] = ca[k, 0]
        poly[k, 1] = ca[k, 1]
    n = 4
    for e in range(4):
        p0x = cb[e, 0]
        p0y = cb[e, 1]
        e2 = e + 1
        if e2 == 4:
            e2 = 0
        nx = -(cb[e2, 1] - p0y)
        ny = cb[e2, 0] - p0x
        m = 0
        for i in range(n):
            j = i + 1
            if j == n:
                j = 0
            cx = poly[i, 0]
            cy = poly[i, 1]
            qx = poly[j, 0]
            qy = poly[j, 1]
            d0 = (cx - p0x) * nx + (cy - p0y) * ny
            d1 = (qx - p0x) * nx + (qy - p0y) * ny
            if d0 >= 0.0:
                tmp[m, 0] = cx
                tmp[m, 1] = cy
                m += 1
            if (d0 >= 0.0) != (d1 >= 0.0):
                den = d0 - d1
                if den > 1e-12 or den < -1e-12:
                    t = d0 / den
                    tmp[m, 0] = cx + (qx - cx) * t
                    tmp[m, 1] = cy + (qy - cy) * t
                    m += 1
        n = m
        if n == 0:
            return 0.0
        for i in range(n):
            poly[i, 0] = tmp[i, 0]
            poly[i, 1] = tmp[i, 1]
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = i + 1
        if j == n:
            j = 0
        area += poly[i, 0] * poly[j, 1] - poly[j, 0] * poly[i, 1]
    area *= 0.5
    return area if area >= 0.0 else -area


@njit(cache=True, fastmath=True)
def iou3d_matrix(a, b):
    """(Na,7) x (Nb,7) velodyne boxes -> (Na,Nb) rotated 3D IoU."""
    na = a.shape[0]
    nb = b.shape[0]
    out = np.zeros((na, nb), np.float32)
    for i in range(na):
        ri = 0.5 * np.sqrt(a[i, 3] ** 2 + a[i, 4] ** 2)
        za0 = a[i, 2] - 0.5 * a[i, 5]
        za1 = a[i, 2] + 0.5 * a[i, 5]
        va = a[i, 3] * a[i, 4] * a[i, 5]
        for j in range(nb):
            dx = a[i, 0] - b[j, 0]
            dy = a[i, 1] - b[j, 1]
            rj = 0.5 * np.sqrt(b[j, 3] ** 2 + b[j, 4] ** 2)
            if dx * dx + dy * dy > (ri + rj) * (ri + rj):
                continue
            zb0 = b[j, 2] - 0.5 * b[j, 5]
            zb1 = b[j, 2] + 0.5 * b[j, 5]
            oh = min(za1, zb1) - max(za0, zb0)
            if oh <= 0.0:
                continue
            inter = bev_intersection(a[i], b[j]) * oh
            if inter <= 0.0:
                continue
            vb = b[j, 3] * b[j, 4] * b[j, 5]
            den = va + vb - inter
            if den > 1e-9:
                out[i, j] = inter / den
    return out


def iou3d(a, b):
    """numpy-friendly wrapper with the same empty-input handling as the BEV one."""
    a = np.ascontiguousarray(a, np.float64).reshape(-1, 7)
    b = np.ascontiguousarray(b, np.float64).reshape(-1, 7)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), np.float32)
    return iou3d_matrix(a, b)
