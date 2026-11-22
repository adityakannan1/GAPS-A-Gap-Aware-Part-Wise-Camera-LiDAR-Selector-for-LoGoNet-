# GAPS: A Gap-Aware Part-Wise Camera–LiDAR Selector

GAPS works after a camera–LiDAR 3D detector (here LoGoNet) and revises the boxes it has already found. For every
detected box it decides, separately for each part of the box (vertical position, height, sideways position and
footprint size), how far to move the 3D detector's estimate toward the estimate of a 2D camera detector. The weight
for each part is set from the spacing of the LiDAR's beams at the object in that part's direction: far away, the
beams are spread out (especially vertically), so the camera is given more weight. GAPS also rescores the confidence
of each detection. It is a small set of single logistic neurons (132 learned numbers in total) and does not change
the 3D detector itself.

This repository contains the code of GAPS, the detector outputs it was trained and tested on, and a script that
reproduces the paper's KITTI validation results on a normal computer (no GPU, no detector retraining).

## Results (KITTI validation set, moderate difficulty, 3D AP over 40 recall positions)

| Method | Car | Pedestrian | Cyclist | Cars beyond 40 m |
|---|---|---|---|---|
| LoGoNet (published weights, our evaluation) | 84.87 | 63.65 | 75.26 | 44.27 |
| LoGoNet (fusion part retrained, the base detector) | 85.43 | 63.50 | 75.49 | 44.58 |
| + GAPS, one camera weight per box | 86.01 | 65.53 | 76.11 | 50.53 |
| + GAPS, one camera weight per direction | 86.03 | 67.73 | 75.91 | 51.92 |
| **+ GAPS, one camera weight per part (paper model)** | **86.05** | **67.89** | **77.69** | **51.99** |

## Reproducing the results

1. Install Python 3.10 or newer and the requirements:

   ```
   pip install -r requirements.txt
   ```

2. Download two small files of the KITTI 3D object detection benchmark from
   https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d (a free account is needed):
   "training labels of object data set" (`data_object_label_2.zip`, 5 MB) and
   "camera calibration matrices of object data set" (`data_object_calib.zip`, 16 MB).
   Unzip both into the same folder, so that it contains `training/label_2/` and `training/calib/`.

3. Run:

   ```
   python scripts/reproduce.py --kitti /path/to/kitti/training
   ```

   The script fits GAPS on the training detections, applies it to the validation detections, prints the table above
   and checks every number against `data/expected.json`. It takes a few minutes on a CPU.
   `--save out.npz` also writes GAPS's final validation detections.

## What is in this repository

```
gaps/gates.py        the gates: fitting (Adam, full batch, fixed number of steps) and applying them
gaps/boxes.py        writing the revised parts back into a 3D box; reading detection files
gaps/kitti_eval.py   KITTI 3D AP (port of the reference evaluator used by SECOND, PointPillars and OpenPCDet)
gaps/iou3d.py        rotated 3D IoU
gaps/calib.py        KITTI calibration
scripts/reproduce.py reproduces the results table
splits/              image lists: standard train/val split (Chen et al., 2015) and the two cross-fitting halves
data/                detector outputs and the per-detection inputs of GAPS (below)
```

### Data

All detections use the LiDAR frame of KITTI, with boxes as (x, y, z of the box center, width, length, height, yaw).

- `val_logonet_published.npz`: detections of the publicly released LoGoNet on the 3,769 validation images.
- `val_logonet_retrained.npz`: detections of the base detector (LoGoNet with only its camera–LiDAR fusion part
  retrained for 10 epochs on the 3,712 training images) on the validation images.
- `gaps_val_detections.npz`: the final detections of GAPS (the paper model) on the validation images.
- `train_records.npz`, `val_records.npz`: one row per 3D detection with everything GAPS uses.
  - `train_records` come from **cross-fitting**: the training images were split into two halves by drive
    (`splits/cfold1.txt`, `splits/cfold2.txt`), LoGoNet and the camera detector were trained on one half and run on
    the other, and vice versa, so every training detection comes from detectors that never saw that image.
  - `val_records` come from the detectors trained on all 3,712 training images, run on the validation images.
  - Columns: `fid` (image), `det` (row of the detection in the detection file), `cls` (1 car, 2 pedestrian,
    3 cyclist), `tp` (1 if the detection matches a labeled object of its class, 3D IoU ≥ 0.7 for cars and ≥ 0.5
    otherwise; used only for training), `lid` / `cam` / `tgt` (the 3D detector's, the camera's and the true value of
    each part: vertical position and sideways position in meters in the camera frame, height in meters, footprint
    size as a scale factor where the 3D detector's box is 1), `has_geo` (a camera box is paired with the detection,
    2D IoU ≥ 0.5, and the box is not cut off at the left or right image border), `has_tgt` (a true box is known for
    training), and `feats`, named in `feat_names`:

| Feature | Meaning |
|---|---|
| `s_logit` | 3D detector's confidence, as a logit |
| `v_space_rel` | log of the vertical spacing between LiDAR beams at the object divided by the object's angular height |
| `gap_top_rel`, `gap_bot_rel` | gap between the object's top (bottom) and the next beam above (below), divided by the object's height |
| `top_uncov` | 1 if no beam passes above the object's top |
| `h_space_rel` | log of the horizontal beam spacing (0.18°) divided by the object's angular width |
| `lz_vert`, `lz_height`, `lz_lat`, `lz_scale` | log(1 + camera–LiDAR disagreement on that part, measured in beam spacings) |
| `trunc_own_v`, `trunc_own_h` | 1 if the projected box touches the top/bottom (v) or left/right (h) image border |
| `cam_match`, `cam_iou`, `yolo_score`, `cls_agree` | camera cues: a camera box was paired, its 2D IoU, its confidence, whether the classes agree |
| `above_corr`, `above_ratio` | image edges continuing above the box (a sign of a pole or tree trunk) |
| `above_pts_ring`, `above_any` | LiDAR returns above the box |
| `window`, `rings_in` | how much of the space above the box the LiDAR can see (height in meters, number of beams) |

The detectors were LoGoNet (https://github.com/PJLab-ADG/LoGoNet), starting from its released KITTI weights, and
YOLO26-m (Ultralytics) as the 2D camera detector, starting from COCO weights and trained on the KITTI 2D boxes of the
same training images. The features were computed from these detections, the KITTI LiDAR scans and the KITTI images.

## Notes

- All results are on the KITTI validation set; GAPS has not been evaluated on the KITTI test server.
- The KITTI data itself is not included and is subject to the KITTI license (CC BY-NC-SA 3.0).

## License

MIT (see `LICENSE`). LoGoNet is released under the Apache 2.0 license and Ultralytics YOLO under AGPL-3.0.
