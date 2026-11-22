"""Writing revised box parts back into a 3D box. The distance along the camera's line of sight (depth) always stays
the 3D detector's; vertical position, sideways position and height are set directly, and length and width are scaled
together by the footprint-size factor."""
import numpy as np


def cam_frame(box, cal):
    """box centre in the camera frame (x right, y down, z forward), and w, l, h"""
    c = cal.velo_to_cam2(box[None, :3])[0]
    return c, box[3], box[4], box[5]


def compose(box, vals, cal):
    cen, w, l, h = cam_frame(box, cal)
    nb = box.copy()
    nb[:3] = cal.cam2_to_velo(np.array([[vals["lat"], vals["vert"], cen[2]]], np.float32))[0]
    nb[5] = vals["height"]; nb[3] *= vals["scale"]; nb[4] *= vals["scale"]
    return nb


def load_detections(path):
    """{frame id: {"boxes", "name", "score"}} from a detections .npz (boxes: x, y, z_centre, w, l, h, yaw, LiDAR frame)"""
    d = np.load(path, allow_pickle=False)
    out, k = {}, 0
    for fid, n in zip(d["fid"], d["count"]):
        out[str(fid)] = {"boxes": d["boxes"][k:k + n], "name": d["name"][k:k + n], "score": d["score"][k:k + n]}
        k += n
    return out


def apply(dets, recs, new_s, vals, kitti):
    """GAPS output: dets with revised scores (new_s) and box parts (vals), both keyed by row of recs"""
    out = {fid: {"boxes": d["boxes"].copy(), "name": d["name"], "score": d["score"].astype(np.float64).copy()}
           for fid, d in dets.items()}
    for j, q in new_s.items():
        out[str(recs.fid[j])]["score"][recs.det[j]] = q
    for j in sorted(vals):
        fid, i = str(recs.fid[j]), recs.det[j]
        out[fid]["boxes"][i] = compose(dets[fid]["boxes"][i], vals[j], kitti.calib(fid))
    for d in out.values():
        d["score"] = d["score"].astype(np.float32)
    return out
