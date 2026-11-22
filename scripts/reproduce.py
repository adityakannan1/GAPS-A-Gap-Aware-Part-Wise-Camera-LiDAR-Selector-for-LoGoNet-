"""Reproduce the paper's KITTI validation results (Tables 1 and 2) from the saved detector outputs.

Fits GAPS on the cross-fitted detections of the training images, applies it to LoGoNet's detections on the
validation images, and scores everything with the KITTI 3D AP (40 recall positions, moderate difficulty). Runs on a
CPU in a few minutes. Needs the KITTI object labels and calibration files (see README).

    python scripts/reproduce.py --kitti /path/to/kitti/training
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from gaps import gates, boxes, kitti_eval  # noqa: E402

DATA = os.path.join(HERE, "..", "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti", required=True, help="KITTI 'training' folder containing label_2/ and calib/")
    ap.add_argument("--save", default=None, help="optional: write GAPS's validation detections to this .npz")
    args = ap.parse_args()
    kitti = kitti_eval.Kitti(args.kitti)
    sizes = {k: tuple(v) for k, v in json.load(open(os.path.join(DATA, "val_image_sizes.json"))).items()}
    tr = gates.Records(os.path.join(DATA, "train_records.npz"))
    te = gates.Records(os.path.join(DATA, "val_records.npz"))
    base = boxes.load_detections(os.path.join(DATA, "val_logonet_retrained.npz"))
    published = boxes.load_detections(os.path.join(DATA, "val_logonet_published.npz"))
    expected = json.load(open(os.path.join(DATA, "expected.json")))

    rows = {}

    def score(name, dets):
        a = kitti_eval.evaluate(dets, kitti, sizes)
        far = kitti_eval.evaluate(dets, kitti, sizes, min_range=40)[0]
        rows[name] = (a, far)
        print(f"{name:42s} Car {a[0]:6.2f}  Ped {a[1]:6.2f}  Cyc {a[2]:6.2f}  | cars >= 40 m {far:6.2f}", flush=True)
        return dets

    print("Moderate 3D AP (%) on the KITTI validation set\n")
    score("LoGoNet (published)", published)
    score("LoGoNet (retrained)", base)
    s_conf, _ = gates.run(tr, te, parts=False)
    for grouping, label in (("box", "one weight per box"), ("direction", "one weight per direction"),
                            ("part", "one weight per part")):
        _, vals = gates.run(tr, te, grouping=grouping, confidence=False)
        out = score("GAPS, " + label, boxes.apply(base, te, s_conf, vals, kitti))
        if grouping == "part":
            ours = out
    paper = rows["GAPS, one weight per part"]
    rows["GAPS, one weight per part (paper model)"] = paper

    print("\nCheck against the paper's numbers:")
    ok = True
    for name, ref in expected.items():
        a, far = rows[name]
        diff = max(max(abs(x - y) for x, y in zip(a, ref["3d_moderate"])), abs(far - ref["cars_40m"]))
        ok &= diff < 0.01
        print(f"  {name:42s} {'OK' if diff < 0.01 else 'DIFFERENT'} (largest difference {diff:.4f})")
    if args.save:
        fids = list(ours)
        np.savez_compressed(args.save, fid=np.array(fids), count=np.array([len(ours[f]["boxes"]) for f in fids]),
                            boxes=np.concatenate([ours[f]["boxes"] for f in fids]),
                            name=np.concatenate([ours[f]["name"] for f in fids]),
                            score=np.concatenate([ours[f]["score"] for f in fids]))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
