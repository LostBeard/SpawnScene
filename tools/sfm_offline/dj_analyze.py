# SpawnScene's pair filtering replayed on the cached k1024 matches, labelled with COLMAP truth (2026-10-01).
# Emulates: pairs with >= 15 matches -> F-RANSAC 2 px (>= 15 inliers) -> relative pose from E = K^T F K (cheirality) ->
# loop filter (1 triangle, 5 deg) -> rotation averaging -> FilterByRotations (10 deg vs the averaged rotations).
# Truth: a pair is TRUE when its images share >= 30 COLMAP points.
import os, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "dj_matchers.py")).read().split("def skew")[0])
n = len(names)
Rgt = [R[nm] for nm in names]
Cgt = [-R[nm].T @ T[nm] for nm in names]
src = open(os.path.join(HERE, "dj_rotavg.py")).read()
exec(src[src.index("def rot_err"):src.index("def edges_for")] + src[src.index("def average"):src.index("truepairs = set()")])
truepairs = {(a, b) for a in range(n) for b in range(a + 1, n) if len(P3[names[a]] & P3[names[b]]) >= 30}

MODE = sys.argv[1] if len(sys.argv) > 1 else "F"
z = np.load(os.path.join(HERE, "dj_k1024.npz"), allow_pickle=True)
kp = z["kp"]
f = 1035.5 * 1024 / 1332
Kc = np.array([[f, 0, 666 * 1024 / 1332], [0, f, 438 * 1024 / 1332], [0, 0, 1.0]])
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
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

edges = []   # (a, b, R, inliers, true?, relrot err vs GT)
for a, b, ia, ib in zip(z["a"], z["b"], z["ia"], z["ib"]):
    if len(ia) < 15: continue
    pa = kp[a][ia].astype(np.float64); pb = kp[b][ib].astype(np.float64)
    F, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 2.0, 0.999)
    if F is None or F.shape != (3, 3) or mask.sum() < 15: continue
    inl = mask.ravel().astype(bool)
    if MODE == "E":
        Ee, me = cv2.findEssentialMat(pa, pb, Kc, cv2.RANSAC, 0.999, 2.0)
        if Ee is None or Ee.shape != (3, 3) or me.sum() < 15: continue
        cnt, Rr, t, _ = cv2.recoverPose(Ee, pa, pb, Kc, mask=me.copy())
        ein = me.ravel().astype(bool)
        if cnt < 0.75 * ein.sum(): continue
        Rr, _ = refine(Rr, t.ravel(), hom(pa[ein]), hom(pb[ein]))
        inl = ein
    else:
        E = Kc.T @ F @ Kc
        cnt, Rr, t, m2 = cv2.recoverPose(E, pa[inl], pb[inl], Kc)
        if cnt < 0.75 * inl.sum(): continue
        if MODE == "FLM": Rr, _ = refine(Rr, t.ravel(), hom(pa[inl]), hom(pb[inl]))
    tru = (a, b) in truepairs
    edges.append((int(a), int(b), Rr, int(inl.sum()), tru, rot_err(Rr, Rgt[b] @ Rgt[a].T)))
nt = sum(e[4] for e in edges)
print(f"{len(edges)} verified pairs: {nt} true, {len(edges) - nt} false; true-pair rel rot err median "
      f"{np.median([e[5] for e in edges if e[4]]):.2f} deg; false-pair median {np.median([e[5] for e in edges if not e[4]]):.1f} deg")
# Loop filter.
Em = {(e[0], e[1]): e[2] for e in edges}
def rel(i, j): return Em[(i, j)] if (i, j) in Em else Em[(j, i)].T
nb = {i: set() for i in range(n)}
for a, b in Em: nb[a].add(b); nb[b].add(a)
loop = [e for e in edges if any(rot_err(rel(c, e[0]) @ rel(e[1], c) @ e[2], np.eye(3)) < 5 for c in nb[e[0]] & nb[e[1]])]
print(f"loop-consistent {len(loop)}: {sum(e[4] for e in loop)} true, {sum(not e[4] for e in loop)} false")
conn, Rav, _ = average([(a, b, Rr, w) for a, b, Rr, w, *_ in loop])
print("averaged:", score(conn, Rav))
Q = proj_so3(sum(Rav[j].T @ Rgt[c] for j, c in enumerate(conn)))
Rabs = {c: Rav[j] for j, c in enumerate(conn)}
camerr = {c: rot_err(Rabs[c] @ Q, Rgt[c]) for c in conn}
bad = sorted([c for c in conn if camerr[c] > 5], key=lambda c: -camerr[c])
print("cameras > 5 deg off:", [(c, round(camerr[c], 1)) for c in bad])
kept = [e for e in edges if e[0] in Rabs and e[1] in Rabs and rot_err(Rabs[e[1]], e[2] @ Rabs[e[0]]) <= 10]
print(f"rotation-consistent {len(kept)}: {sum(e[4] for e in kept)} true, {sum(not e[4] for e in kept)} false")
for e in kept:
    if not e[4]: print(f"   FALSE kept: {e[0]}-{e[1]} inliers {e[3]} rel err {e[5]:.1f} deg (cams err {camerr[e[0]]:.1f}/{camerr[e[1]]:.1f})")
# Which cameras are in wrong components: the loop edges touching the bad cameras.
for c in bad:
    le = [e for e in loop if c in (e[0], e[1])]
    print(f"   cam {c} ({camerr[c]:.0f} deg): loop edges " + ", ".join(f"{e[0]}-{e[1]}{'' if e[4] else '(F)'}:{e[3]}" for e in le))
