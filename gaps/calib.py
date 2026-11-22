"""KITTI calibration handling and the camera<->polar geometry that links the
two independent branches together at fusion time."""
import numpy as np


class Calib:
    """Rectified KITTI calibration for the left colour camera (cam2).

    The polar grid is defined in *cam2's own optical frame* rather than the
    rectified reference frame.  That matters: in cam2's frame the pinhole
    model gives ``u = fx * tan(azimuth) + cx`` exactly, with no dependence on
    range, so azimuth bins map to image columns as an exact bijection.
    """

    __slots__ = ("P2", "R0", "V2C", "fx", "fy", "cx", "cy", "_v2c2", "_c2v")

    def __init__(self, P2, R0, V2C):
        self.P2 = P2                      # 3x4, rectified-reference -> image
        self.R0 = R0                      # 3x3 rectifying rotation
        self.V2C = V2C                    # 3x4 velodyne -> reference camera
        self.fx, self.fy = P2[0, 0], P2[1, 1]
        self.cx, self.cy = P2[0, 2], P2[1, 2]
        # Offset from the rectified reference frame to cam2's optical centre.
        off = np.array([P2[0, 3] / self.fx, P2[1, 3] / self.fy, P2[2, 3]])
        RV = R0 @ V2C                     # 3x4 velodyne -> reference camera
        self._v2c2 = np.vstack([RV, [0, 0, 0, 1.0]])
        self._v2c2[:3, 3] += off          # ... -> cam2 optical frame
        self._c2v = np.linalg.inv(self._v2c2)

    # ------------------------------------------------------------------ io --
    @staticmethod
    def from_file(path):
        d = {}
        with open(path) as fh:
            for line in fh:
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                d[k.strip()] = np.array([float(x) for x in v.split()])
        return Calib(d["P2"].reshape(3, 4),
                     d["R0_rect"].reshape(3, 3),
                     d["Tr_velo_to_cam"].reshape(3, 4))

    # ----------------------------------------------------------- transforms --
    def velo_to_cam2(self, pts):
        """(N,3) velodyne xyz -> (N,3) cam2 optical xyz (x right, y down, z fwd)."""
        n = pts.shape[0]
        h = np.concatenate([pts[:, :3], np.ones((n, 1), pts.dtype)], 1)
        return (h @ self._v2c2[:3].T).astype(np.float32)

    def cam2_to_velo(self, pts):
        n = pts.shape[0]
        h = np.concatenate([pts[:, :3], np.ones((n, 1), pts.dtype)], 1)
        return (h @ self._c2v[:3].T).astype(np.float32)

    def rect_to_cam2(self, pts):
        """Rectified-reference camera xyz (what KITTI labels use) -> cam2 xyz."""
        off = np.array([self.P2[0, 3] / self.fx,
                        self.P2[1, 3] / self.fy,
                        self.P2[2, 3]], np.float32)
        return pts[:, :3].astype(np.float32) + off

    def project_cam2(self, pts):
        """(N,3) cam2 xyz -> (N,2) pixel uv in the ORIGINAL image resolution."""
        z = np.clip(pts[:, 2], 1e-3, None)
        return np.stack([self.fx * pts[:, 0] / z + self.cx,
                         self.fy * pts[:, 1] / z + self.cy], 1)

    def project_rect(self, pts):
        """(N,3) rectified-reference xyz -> (N,2) pixel uv (standard KITTI path)."""
        n = pts.shape[0]
        h = np.concatenate([pts[:, :3], np.ones((n, 1), pts.dtype)], 1)
        p = h @ self.P2.T
        z = np.clip(p[:, 2], 1e-3, None)
        return np.stack([p[:, 0] / z, p[:, 1] / z], 1)
