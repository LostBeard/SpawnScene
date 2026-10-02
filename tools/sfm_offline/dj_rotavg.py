# Can GLOMAP-style global rotation averaging recover DrJohnson's ABSOLUTE rotations from a better front end?
# Front end: SIFT + 5-point E-RANSAC with K (focal given), over ALL pairs (no GT used for pairing). Then robust chordal
# IRLS rotation averaging (spanning tree start), then GLOMAP-style view-graph filtering (drop edges whose residual > T,
# re-average). Score: absolute rotation error vs COLMAP after the best global alignment.
import json, struct, sys
import numpy as np, cv2

SRC = r"C:\Users\TJ\Downloads\tandt_db\db\drjohnson"
MAN = r"D:\users\tj\Projects\SpawnScene\SpawnScene\SpawnScene\wwwroot\datasets\DrJohnson\manifest.json"
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dj_matchers.py")).read().split('pairs = []')[0])
n = len(names)
import os
gray = {nm: cv2.imread(os.path.join(SRC, "images", nm), cv2.IMREAD_GRAYSCALE) for nm in names}
Rgt = [R[nm] for nm in names]

def rot_err(A, B):
    return np.degrees(np.arccos(np.clip((np.trace(A @ B.T) - 1) / 2, -1, 1)))

def proj_so3(M):
    U, _, Vt = np.linalg.svd(M); Rm = U @ Vt
    if np.linalg.det(Rm) < 0: U[:, -1] *= -1; Rm = U @ Vt
    return Rm

def edges_for(focal, nfeat=4000, ratio=0.8, min_in=15):
    K = np.array([[focal, 0, 666], [0, focal, 438], [0, 0, 1.0]])
    sift = cv2.SIFT_create(nfeat); feats = [sift.detectAndCompute(gray[nm], None) for nm in names]
    bf = cv2.BFMatcher(cv2.NORM_L2); E_ = []
    for a in range(n):
        for b in range(a + 1, n):
            ka, da = feats[a]; kb, db = feats[b]
            m = [x[0] for x in bf.knnMatch(da, db, k=2) if len(x) == 2 and x[0].distance < ratio * x[1].distance]
            if len(m) < min_in: continue
            pa = np.float64([ka[x.queryIdx].pt for x in m]); pb = np.float64([kb[x.trainIdx].pt for x in m])
            E, inl = cv2.findEssentialMat(pa, pb, K, cv2.RANSAC, 0.999, 2.0)
            if E is None or E.shape != (3, 3) or inl.sum() < min_in: continue
            cnt, Rr, t, _ = cv2.recoverPose(E, pa, pb, K, mask=inl.copy())
            if cnt < min_in: continue
            E_.append((a, b, Rr, int(cnt)))
    return E_

def average(edges, iters=30, delta=np.radians(5)):
    # spanning tree (max inliers) from the most connected root
    adj = {i: [] for i in range(n)}
    for k, (a, b, Rr, w) in enumerate(edges): adj[a].append(k); adj[b].append(k)
    root = max(range(n), key=lambda i: sum(edges[k][3] for k in adj[i]))
    Ri = [None] * n; Ri[root] = np.eye(3)
    import heapq; h = [(-edges[k][3], k) for k in adj[root]]; heapq.heapify(h)
    while h:
        _, k = heapq.heappop(h); a, b, Rr, w = edges[k]
        if Ri[a] is not None and Ri[b] is None: Ri[b] = Rr @ Ri[a]; nxt = b
        elif Ri[b] is not None and Ri[a] is None: Ri[a] = Rr.T @ Ri[b]; nxt = a
        else: continue
        for k2 in adj[nxt]: heapq.heappush(h, (-edges[k2][3], k2))
    conn = [i for i in range(n) if Ri[i] is not None]
    idx = {c: j for j, c in enumerate(conn)}; m = len(conn)
    es = [(idx[a], idx[b], Rr, w) for a, b, Rr, w in edges if a in idx and b in idx]
    R = [Ri[c] for c in conn]
    for it in range(iters):
        # weights from residuals
        rs = np.array([np.linalg.norm(R[b] - Rr @ R[a]) for a, b, Rr, w in es])
        wts = np.array([np.sqrt(w) for *_, w in es]) / np.sqrt(rs ** 2 + (2 * np.sin(delta / 2) * np.sqrt(2)) ** 2)
        # linear LS: R_b - Rr R_a = 0, root fixed; unknowns m x (3x3) -> solve per column block
        A = np.zeros((len(es) * 3, m * 3)); rootj = idx[root]
        Rn = [None] * m
        for col in range(3):
            rows = []; B = np.zeros(len(es) * 3)
            A[:] = 0
            for e, (a, b, Rr, w) in enumerate(es):
                s = wts[e]
                A[e*3:e*3+3, b*3:b*3+3] += s * np.eye(3)
                A[e*3:e*3+3, a*3:a*3+3] -= s * Rr
            # fix root: move its known column to RHS
            keep = [j for j in range(m) if j != rootj]
            cols = np.concatenate([np.arange(j*3, j*3+3) for j in keep])
            rootcol = R[rootj][:, col]
            B = -A[:, rootj*3:rootj*3+3] @ rootcol
            x, *_ = np.linalg.lstsq(A[:, cols], B, rcond=None)
            for t, j in enumerate(keep):
                if Rn[j] is None: Rn[j] = np.zeros((3, 3))
                Rn[j][:, col] = x[t*3:t*3+3]
        Rn[rootj] = R[rootj]
        R = [proj_so3(M) for M in Rn]
    return conn, R, es

def score(conn, R):
    # align: find Q minimising sum |Rgt_i - R_i Q|  -> Q = proj(sum R_i^T Rgt_i)
    Q = proj_so3(sum(R[j].T @ Rgt[c] for j, c in enumerate(conn)))
    e = np.sort([rot_err(R[j] @ Q, Rgt[c]) for j, c in enumerate(conn)])
    return f"{len(conn)}/{n} cams, abs rot median {np.median(e):.2f} p75 {e[len(e)*3//4]:.2f} p90 {e[len(e)*9//10]:.2f} deg"

truepairs = set()
for a in range(n):
    for b in range(a + 1, n):
        if len(P3[names[a]] & P3[names[b]]) >= 30: truepairs.add((a, b))
edges = edges_for(1035.5)
gt_edges = [(a, b, Rgt[b] @ Rgt[a].T, w) for a, b, Rr, w in edges]
c, Rr_, _ = average(gt_edges); print("VALIDATE same edges, GT rotations:", score(c, Rr_))
te = [e for e in edges if (e[0], e[1]) in truepairs]
c, Rr_, _ = average(te); print(f"only GT-true edges ({len(te)} of {len(edges)}), SIFT+E rotations:", score(c, Rr_))
fe = [e for e in edges if (e[0], e[1]) not in truepairs]
print("false edges:", len(fe), " their inlier counts median", int(np.median([e[3] for e in fe])) if fe else 0, " true edges median", int(np.median([e[3] for e in te])))
