# Translation averaging on DrJohnson from the five-point pairs (2026-10-01): are camera positions from pairwise
# translation DIRECTIONS (no tracks) close to COLMAP, where track-based positioning settles 64-77% off?
# Pipeline: F verify -> E-RANSAC + recoverPose + LM (dj_analyze "E" mode) -> loop filter -> rotation averaging ->
# rotation filter (10 deg) -> BATA on camera-camera directions (IRLS, Huber, d >= 0, random init, centroid gauge).
import os, sys
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "dj_analyze.py")).read()
# Reuse dj_analyze's setup + edge construction in E mode, without its printing tail.
setup = src[:src.index("nt = sum(e[4] for e in edges)")]
sys.argv = [sys.argv[0], "E"]
exec(setup.replace("edges.append((int(a), int(b), Rr, int(inl.sum()), tru, rot_err(Rr, Rgt[b] @ Rgt[a].T)))",
                   "edges.append((int(a), int(b), Rr, int(inl.sum()), tru, rot_err(Rr, Rgt[b] @ Rgt[a].T), t.ravel() / np.linalg.norm(t), hom(pa[inl]), hom(pb[inl])))"))
Em = {(e[0], e[1]): e[2] for e in edges}
def rel(i, j): return Em[(i, j)] if (i, j) in Em else Em[(j, i)].T
nb = {i: set() for i in range(n)}
for a, b in Em: nb[a].add(b); nb[b].add(a)
loop = [e for e in edges if any(rot_err(rel(c, e[0]) @ rel(e[1], c) @ e[2], np.eye(3)) < 5 for c in nb[e[0]] & nb[e[1]])]
conn, Rav, _ = average([(a, b, Rr, w) for a, b, Rr, w, *_ in loop])
Rabs = {c: Rav[j] for j, c in enumerate(conn)}
kept = [e for e in edges if e[0] in Rabs and e[1] in Rabs and rot_err(Rabs[e[1]], e[2] @ Rabs[e[0]]) <= 10]
print(f"{len(edges)} pairs, {len(loop)} loop-consistent, {len(conn)} cams, {len(kept)} rotation-consistent")

def direction_err(e, Rw):
    a, b, Rr, w, tru, rerr, t = e[:7]
    u = -Rw[b].T @ t              # world direction C_b - C_a (x_b = R x_a + t, t = R_b (C_a - C_b) s)
    g = Cgt[b] - Cgt[a]
    return np.degrees(np.arccos(np.clip(u @ g / np.linalg.norm(g), -1, 1)))
de = np.array([direction_err(e, Rgt) for e in kept])
print(f"pair translation direction vs COLMAP (with COLMAP rotations): median {np.median(de):.2f} p75 {np.percentile(de, 75):.2f} p90 {np.percentile(de, 90):.2f} deg")

def bata(edges_, Rw, cams, iters=200, huber=0.1, seed=1):
    idx = {c: k for k, c in enumerate(cams)}; m = len(cams)
    rng = np.random.default_rng(seed); C = rng.uniform(-1, 1, (m, 3)) * 0.1
    E_ = [(idx[e[0]], idx[e[1]], -Rw[e[1]].T @ e[6]) for e in edges_ if e[0] in idx and e[1] in idx]
    for it in range(iters):
        # scales at their optimum, then IRLS weights, then the linear LS for centres (gauge: centroid 0)
        rows, rhs, wts = [], [], []
        for i, j, u in E_:
            dlt = C[j] - C[i]; dd = dlt @ dlt
            d = max(1.0, (u @ dlt) / dd) if dd > 0 else 1.0   # BATA: d >= 1 (no collapse)
            r = np.linalg.norm(d * dlt - u)
            wt = 1.0 if r <= huber else huber / r
            rows.append((i, j, d)); rhs.append(u); wts.append(wt)
        A = np.zeros((len(rows) * 3 + 3, m * 3)); B = np.zeros(len(rows) * 3 + 3)
        for k, ((i, j, d), u, wt) in enumerate(zip(rows, rhs, wts)):
            s = np.sqrt(wt)
            for c3 in range(3):
                A[k * 3 + c3, j * 3 + c3] += s * d; A[k * 3 + c3, i * 3 + c3] -= s * d; B[k * 3 + c3] = s * u[c3]
        for c3 in range(3):
            A[-3 + c3, c3::3] = 1.0
        Cn = np.linalg.lstsq(A, B, rcond=None)[0].reshape(m, 3)
        if np.max(np.abs(Cn - C)) < 1e-9: C = Cn; break
        C = Cn
    return C

def pos_err(C, cams):
    G = np.array([Cgt[c] for c in cams]); X = C
    # similarity alignment (Umeyama)
    mx, mg = X.mean(0), G.mean(0); Xc, Gc = X - mx, G - mg
    U, S, Vt = np.linalg.svd(Gc.T @ Xc); D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    Rr = U @ D @ Vt; s = np.trace(np.diag(S) @ D) / (Xc ** 2).sum()
    Y = s * Xc @ Rr.T + mg
    spread = np.median(np.linalg.norm(G - np.median(G, 0), axis=1))
    e = np.linalg.norm(Y - G, axis=1) / spread
    return np.median(e), np.percentile(e, 90)
cams = sorted(Rabs)
Qal = proj_so3(sum(Rabs[c].T @ Rgt[c] for c in cams))
Rest = {c: Rabs[c] @ Qal for c in cams}
Rg = {c: Rgt[c] for c in cams}
for label, Rw in [("COLMAP rotations", Rg), ("estimated rotations", Rest)]:
    C = bata(kept, Rw, cams)
    med, p90 = pos_err(C, cams)
    print(f"translation averaging, {label}: {len(cams)} cams, position error median {med:.1%} p90 {p90:.1%} of spread")
# Exact COLMAP directions on the same graph: is the graph itself able to fix positions?
exact = []
for e in kept:
    a, b = e[0], e[1]
    g = Cgt[b] - Cgt[a]; g = g / np.linalg.norm(g)
    t = -Rgt[b] @ g      # inverse of u = -R_b^T t
    exact.append(e[:6] + (t,) + e[7:])
C = bata(exact, Rg, cams)
med, p90 = pos_err(C, cams)
print(f"translation averaging, EXACT directions on the kept graph: median {med:.1%} p90 {p90:.1%}")
# Same, but only the largest 2-connected / well-linked part? Report per-camera edge degree.
deg = {c: 0 for c in cams}
for e in kept: deg[e[0]] += 1; deg[e[1]] += 1
print("edges per camera:", sorted(deg.values()))

def parallax(e):
    Rr, ra, rb = e[2], e[7], e[8]
    A = (Rr @ ra.T).T; A /= np.linalg.norm(A, axis=1)[:, None]
    B = rb / np.linalg.norm(rb, axis=1)[:, None]
    return np.degrees(np.median(np.arccos(np.clip(np.sum(A * B, axis=1), -1, 1))))
px = np.array([parallax(e) for e in kept])
print(f"pair parallax (rotation-compensated, median per pair): median {np.median(px):.2f} p10 {np.percentile(px, 10):.2f} deg")
bad = de > 10
print(f"pairs with direction error > 10 deg: {bad.sum()} of {len(kept)}; their parallax median {np.median(px[bad]) if bad.any() else 0:.2f} vs others {np.median(px[~bad]):.2f}")
for thr in [1.0, 2.0, 3.0]:
    sub = [e for e, p in zip(kept, px) if p >= thr]
    cs = sorted({e[0] for e in sub} | {e[1] for e in sub})
    C = bata(sub, Rest, cs); med, p90 = pos_err(C, cs)
    print(f"  parallax >= {thr} deg: {len(sub)} pairs, {len(cs)} cams -> median {med:.1%} p90 {p90:.1%}")
def bata_robust(edges_, Rw, cams, rounds=4, maxdeg=5):
    cur = list(edges_)
    for r in range(rounds):
        C = bata(cur, Rw, cams)
        idx = {c: k for k, c in enumerate(cams)}
        keep = []
        for e in cur:
            u = -Rw[e[1]].T @ e[6]; d = C[idx[e[1]]] - C[idx[e[0]]]
            ang = np.degrees(np.arccos(np.clip(u @ d / max(np.linalg.norm(d), 1e-12), -1, 1)))
            if ang <= maxdeg * (3 if r == 0 else 1): keep.append(e)
        if len(keep) == len(cur): break
        cur = keep
    return C, cur
for thr in [0.0, 1.0, 2.0]:
    sub = [e for e, p in zip(kept, px) if p >= thr]
    cs = sorted({e[0] for e in sub} | {e[1] for e in sub})
    C, used = bata_robust(sub, Rest, cs); med, p90 = pos_err(C, cs)
    print(f"  robust rounds, parallax >= {thr}: {len(used)} of {len(sub)} pairs kept, {len(cs)} cams -> median {med:.1%} p90 {p90:.1%}")
