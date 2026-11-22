"""The GAPS gates: fitting and applying them to per-detection records.

Every gate is a single logistic neuron. For a box part c (vertical position, height, sideways position, footprint size):

    camera weight  w_c = n_c * t_c
    need           n_c = sigmoid(beta_c . o + b_c)                         o = beam spacing relative to object size
    trust          t_c = sigmoid(g0 - softplus(g2) * u_c - softplus(g3) * log(1 + z_c))
                                                                           u_c = truncation flag, z_c = disagreement
    revised part   x   = x_L + w_c * (x_C - x_L)

and the confidence of each detection is rescored as

    y = a + b * s + sum_k sigmoid(O_k . w_k + c_k) * (U_k . v_k),   new confidence = sigmoid(y)

with s the 3D detector's confidence as a logit, U_k the cues of group k and O_k what decides how visible they are.
One set of gates is fitted per class (and per part), with Adam (learning rate 0.05) on all training examples at once
(full batch) for a fixed number of steps, from fixed starting values; there is nothing random in training.
"""
import numpy as np
import torch

PARTS = ("vert", "height", "lat", "scale")
SIDE = {"vert": "v", "height": "v", "lat": "h", "scale": "h"}
# beam-spacing inputs of the need factor
NEED = {"vert": ["v_space_rel", "gap_top_rel", "gap_bot_rel", "top_uncov"],
        "height": ["v_space_rel", "gap_top_rel", "gap_bot_rel", "top_uncov"],
        "lat": ["h_space_rel"], "scale": ["h_space_rel"]}
# confidence cue groups: (cues, visibility inputs of the group's gate)
CONF_GROUPS = [("cam_app", ["cam_match", "cam_iou", "yolo_score", "cls_agree"], ["v_space_rel", "h_space_rel"]),
               ("cam_above", ["above_corr", "above_ratio"], ["v_space_rel"]),
               ("lidar_above", ["above_pts_ring", "above_any"], ["window", "rings_in"])]
GROUPINGS = {"part": [["vert"], ["height"], ["lat"], ["scale"]], "direction": [["vert", "height"], ["lat", "scale"]],
             "box": [list(PARTS)]}
SP = torch.nn.functional.softplus
T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)


def zfit(X):
    """standardize with the training mean and standard deviation"""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return lambda Z: (Z - mu) / sd


class Records:
    """per-detection inputs (see data/README in the repository)"""

    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.fid, self.det, self.cls, self.tp = d["fid"], d["det"], d["cls"], d["tp"]
        self.col = {k: j for j, k in enumerate(d["feat_names"])}
        self.feats, self.has_geo, self.has_tgt = d["feats"], d["has_geo"], d["has_tgt"]
        self.lid, self.cam, self.tgt = d["lid"], d["cam"], d["tgt"]

    def f(self, idx, names):
        return self.feats[np.ix_(idx, [self.col[k] for k in names])]


def fit_part(tr, itr, te, ite, part, steps=800):
    """one weight for one box part; returns (revised values, camera weights) for the te records"""
    p = PARTS.index(part); side = SIDE[part]
    norm = zfit(tr.f(itr, NEED[part]))
    Xt = T(norm(tr.f(itr, NEED[part])))
    Lt, Ct, Yt = T(tr.lid[itr, p]), T(tr.cam[itr, p]), T(tr.tgt[itr, p])
    beta = torch.zeros(Xt.shape[1], requires_grad=True); c = torch.full((1,), -2.0, requires_grad=True)
    g = torch.tensor([4.0, 0.0, 0.0, 0.0], requires_grad=True)   # g[1] belonged to an occlusion input not used here

    def trust(R, idx):
        t = T(R.f(idx, [f"trunc_own_{side}"])[:, 0]); z = T(R.f(idx, [f"lz_{part}"])[:, 0])
        return torch.sigmoid(g[0] - SP(g[1]) * 0.0 - SP(g[2]) * t - SP(g[3]) * z)
    opt = torch.optim.Adam([beta, c, g], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        w = torch.sigmoid(Xt @ beta + c) * trust(tr, itr)
        loss = torch.nn.functional.smooth_l1_loss(Lt + w * (Ct - Lt), Yt, beta=0.05) + 1e-3 * (beta ** 2).sum()
        loss.backward(); opt.step()
    with torch.no_grad():
        w = (torch.sigmoid(T(norm(te.f(ite, NEED[part]))) @ beta + c) * trust(te, ite)).numpy()
    L, C = te.lid[ite, p], te.cam[ite, p]
    return L + w * (C - L), w


def fit_group(tr, itr, te, ite, parts, steps=800):
    """one weight shared by several parts (per direction / per box); returns the camera weight for each te record"""
    need = sorted({q for c in parts for q in NEED[c]}); sides = sorted({SIDE[c] for c in parts})
    norm = zfit(tr.f(itr, need)); Xt = T(norm(tr.f(itr, need)))

    def tparts(R, idx):
        t = np.max(R.f(idx, [f"trunc_own_{s}" for s in sides]), 1); z = np.mean(R.f(idx, [f"lz_{c}" for c in parts]), 1)
        return T(np.zeros(len(idx))), T(t), T(z)
    L = {c: T(tr.lid[itr, PARTS.index(c)]) for c in parts}; C = {c: T(tr.cam[itr, PARTS.index(c)]) for c in parts}
    Y = {c: T(tr.tgt[itr, PARTS.index(c)]) for c in parts}
    scale = {c: float(torch.nn.functional.smooth_l1_loss(L[c], Y[c], beta=0.05)) + 1e-9 for c in parts}
    o, t, z = tparts(tr, itr)
    beta = torch.zeros(Xt.shape[1], requires_grad=True); b = torch.full((1,), -2.0, requires_grad=True)
    g = torch.tensor([4.0, 0.0, 0.0, 0.0], requires_grad=True)
    w_of = lambda Xn, o, t, z: torch.sigmoid(Xn @ beta + b) * torch.sigmoid(g[0] - SP(g[1]) * o - SP(g[2]) * t - SP(g[3]) * z)
    opt = torch.optim.Adam([beta, b, g], lr=0.05)
    for _ in range(steps):
        opt.zero_grad(); w = w_of(Xt, o, t, z)
        loss = sum(torch.nn.functional.smooth_l1_loss(L[c] + w * (C[c] - L[c]), Y[c], beta=0.05) / scale[c] for c in parts) \
            + 1e-3 * (beta ** 2).sum()
        loss.backward(); opt.step()
    with torch.no_grad():
        return w_of(T(norm(te.f(ite, need))), *tparts(te, ite)).numpy()


def fit_confidence(tr, itr, te, ite, steps=500):
    """rescored confidence for the te records"""
    def mats(R, idx):
        return (R.f(idx, ["s_logit"]), [R.f(idx, u) for _, u, _ in CONF_GROUPS], [R.f(idx, o) for _, _, o in CONF_GROUPS])
    s, U, O = mats(tr, itr); y = torch.tensor(tr.tp[itr], dtype=torch.float32)
    nu = [zfit(u) for u in U]; no = [zfit(o) for o in O]
    Ut = [T(n(u)) for n, u in zip(nu, U)]; Ot = [T(n(o)) for n, o in zip(no, O)]; st = T(s)[:, 0]
    a = torch.zeros(1, requires_grad=True); bb = torch.ones(1, requires_grad=True)
    v = [torch.zeros(u.shape[1], requires_grad=True) for u in U]
    w = [torch.zeros(o.shape[1], requires_grad=True) for o in O]; c = [torch.zeros(1, requires_grad=True) for _ in O]
    opt = torch.optim.Adam([a, bb] + v + w + c, lr=0.05)

    def logit(st, Ut, Ot):
        z = a + bb * st
        for k in range(len(Ut)):
            z = z + torch.sigmoid(Ot[k] @ w[k] + c[k]) * (Ut[k] @ v[k])
        return z
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit(st, Ut, Ot), y) \
            + 1e-3 * sum((p ** 2).sum() for p in v + w)
        loss.backward(); opt.step()
    s2, U2, O2 = mats(te, ite)
    with torch.no_grad():
        z = logit(T(s2)[:, 0], [T(n(u)) for n, u in zip(nu, U2)], [T(n(o)) for n, o in zip(no, O2)])
    return torch.sigmoid(z).numpy()


def run(tr, te, grouping="part", parts=True, confidence=True):
    """fit GAPS on tr and apply it to te: returns {te row: new confidence} and {te row: {part: value}}"""
    new_s, vals = {}, {}
    for k in (1, 2, 3):
        itr = np.where(tr.cls == k)[0]; ite = np.where(te.cls == k)[0]
        if confidence and len(itr) >= 50 and len(ite) and 0 < tr.tp[itr].sum() < len(itr):
            for j, q in zip(ite, fit_confidence(tr, itr, te, ite)):
                new_s[j] = float(q)
        gtr = itr[tr.has_geo[itr] & tr.has_tgt[itr]]; gte = ite[te.has_geo[ite]]
        if not parts or len(gtr) < 30 or not len(gte):
            continue
        for grp in GROUPINGS[grouping]:
            if grouping == "part":
                v, _ = fit_part(tr, gtr, te, gte, grp[0])
                for j, q in zip(gte, v):
                    vals.setdefault(j, dict(zip(PARTS, te.lid[j])))[grp[0]] = float(q)
                continue
            wts = fit_group(tr, gtr, te, gte, grp)
            for j, wi in zip(gte, wts):
                d = vals.setdefault(j, dict(zip(PARTS, te.lid[j])))
                for c in grp:
                    p = PARTS.index(c)
                    d[c] = te.lid[j, p] + float(wi) * (te.cam[j, p] - te.lid[j, p])
    return new_s, vals
