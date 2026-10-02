# Relative-pose estimators on the SAME cached k1024 matches, true pairs only (COLMAP: >= 30 shared points), 2026-10-01:
#  F      : F-RANSAC 2 px -> E = K^T F K -> recoverPose          (SpawnScene until 4b5a86b)
#  F+LM   : F + Sampson LM on the F inliers, 5 DoF               (SpawnScene 4b5a86b)
#  E      : 5-point E-RANSAC with K, 2 px (cv2)                  (research harness)
#  F->E   : 5-point E-RANSAC on the F inliers only
#  E+LM   : E then the same Sampson LM
import os, sys
import numpy as np, cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "dj_matchers.py")).read().split("def skew")[0])
n = len(names)
Rgt = [R[nm] for nm in names]
def rot_err(A, B): return np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1)))
truepairs = {(a, b) for a in range(n) for b in range(a + 1, n) if len(P3[names[a]] & P3[names[b]]) >= 30}
z = np.load(os.path.join(HERE, "dj_k1024.npz"), allow_pickle=True)
kp = z["kp"]
f = 1035.5 * 1024 / 1332
Kc = np.array([[f, 0, 666 * 1024 / 1332], [0, f, 438 * 1024 / 1332], [0, 0, 1.0]])
Ki = np.linalg.inv(Kc)

def skew(v): return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
def refine(R0, t0, ra, rb, huber=1.5):
    t0 = t0 / np.linalg.norm(t0)
    a = np.array([1.0, 0, 0]) if abs(t0[0]) < 0.9 else np.array([0, 1.0, 0])
    u1 = np.cross(a, t0); u1 /= np.linalg.norm(u1); u2 = np.cross(t0, u1)
    def pose(p):
        Rr = R0 @ Rot.from_rotvec(p[:3]).as_matrix()
        t = t0 + p[3] * u1 + p[4] * u2
        return Rr, t / np.linalg.norm(t)
    def res(p):
        Rr, t = pose(p); E = skew(t) @ Rr
        Ex = ra @ E.T; Etx = rb @ E
        num = np.sum(rb * Ex, axis=1)
        den = np.sqrt(Ex[:, 0] ** 2 + Ex[:, 1] ** 2 + Etx[:, 0] ** 2 + Etx[:, 1] ** 2)
        return num / den * f
    r = least_squares(res, np.zeros(5), loss="huber", f_scale=huber, method="trf")
    return pose(r.x)

def hom(p): return np.c_[(p - Kc[:2, 2]) / f, np.ones(len(p))]

errs = {k: [] for k in ["F", "F+LM", "E", "F->E", "E+LM"]}
for a, b, ia, ib in zip(z["a"], z["b"], z["ia"], z["ib"]):
    if (a, b) not in truepairs or len(ia) < 15: continue
    pa = kp[a][ia].astype(np.float64); pb = kp[b][ib].astype(np.float64)
    truth = Rgt[b] @ Rgt[a].T
    F, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 2.0, 0.999)
    if F is None or F.shape != (3, 3) or mask.sum() < 15: continue
    inl = mask.ravel().astype(bool)
    E = Kc.T @ F @ Kc
    cnt, Rf, tf, _ = cv2.recoverPose(E, pa[inl], pb[inl], Kc)
    errs["F"].append(rot_err(Rf, truth))
    Rl, tl = refine(Rf, tf.ravel(), hom(pa[inl]), hom(pb[inl]))
    errs["F+LM"].append(rot_err(Rl, truth))
    Ee, me = cv2.findEssentialMat(pa, pb, Kc, cv2.RANSAC, 0.999, 2.0)
    if Ee is not None and Ee.shape == (3, 3):
        _, Re, te, _ = cv2.recoverPose(Ee, pa, pb, Kc, mask=me.copy())
        errs["E"].append(rot_err(Re, truth))
        ein = me.ravel().astype(bool)
        Rel, tel = refine(Re, te.ravel(), hom(pa[ein]), hom(pb[ein]))
        errs["E+LM"].append(rot_err(Rel, truth))
    Ef, mf = cv2.findEssentialMat(pa[inl], pb[inl], Kc, cv2.RANSAC, 0.999, 2.0)
    if Ef is not None and Ef.shape == (3, 3):
        _, Rfe, _, _ = cv2.recoverPose(Ef, pa[inl], pb[inl], Kc, mask=mf.copy())
        errs["F->E"].append(rot_err(Rfe, truth))
for k, v in errs.items():
    v = np.sort(v)
    print(f"{k:5s}: {len(v)} true pairs, rel rot err median {np.median(v):.2f} p75 {v[len(v)*3//4]:.2f} p90 {v[len(v)*9//10]:.2f} deg, "
          f"<5 deg {np.mean(v < 5):.0%}")
