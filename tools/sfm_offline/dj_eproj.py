# 8-point hypotheses on NORMALISED rays, each projected onto the essential manifold (singular values 1,1,0), scored by
# the projected E's Sampson error in pixels: an "E-RANSAC" with the 8-point minimal set instead of the 5-point solver.
exec(open(__file__.replace("dj_eproj.py", "dj_relpose.py")).read().split("errs = {")[0])
rng = np.random.default_rng(1)
def eight_point(ra, rb):
    A = np.c_[rb[:, :1] * ra, rb[:, 1:2] * ra, ra]   # x_b^T E x_a = 0 -> row (xb*xa, xb*ya, xb, yb*xa, yb*ya, yb, xa, ya, 1)
    _, _, Vt = np.linalg.svd(A); E = Vt[-1].reshape(3, 3)
    U, s, Vt2 = np.linalg.svd(E); return U @ np.diag([1, 1, 0]) @ Vt2
def sampson_px(E, ra, rb):
    Ex = ra @ E.T; Etx = rb @ E
    num = np.sum(rb * Ex, axis=1)
    return np.abs(num) / np.sqrt(Ex[:, 0]**2 + Ex[:, 1]**2 + Etx[:, 0]**2 + Etx[:, 1]**2) * f
def eransac(ra, rb, iters, thr=2.0, sample=8):
    best, bestE = -1, None
    for _ in range(iters):
        idx = rng.choice(len(ra), sample, replace=False)
        E = eight_point(ra[idx], rb[idx])
        c = np.sum(sampson_px(E, ra, rb) < thr)
        if c > best: best, bestE = c, E
    return bestE, sampson_px(bestE, ra, rb) < thr
def eight_rows(ra, rb):
    return np.c_[rb[:, 0:1] * ra, rb[:, 1:2] * ra, ra]
# fix eight_point row layout: x_b^T E x_a = sum_ij xb_i E_ij xa_j -> row = kron(xb, xa)
def eight_point(ra, rb):
    A = np.einsum("ni,nj->nij", rb, ra).reshape(len(ra), 9)
    _, _, Vt = np.linalg.svd(A); E = Vt[-1].reshape(3, 3)
    U, s, Vt2 = np.linalg.svd(E); return U @ np.diag([1, 1, 0]) @ Vt2
out = {k: [] for k in ["Eproj 8pt", "Eproj 8pt + LM", "E 5pt + LM"]}
for a, b, ia, ib in zip(z["a"], z["b"], z["ia"], z["ib"]):
    if (a, b) not in truepairs or len(ia) < 15: continue
    pa = kp[a][ia].astype(np.float64); pb = kp[b][ib].astype(np.float64)
    truth = Rgt[b] @ Rgt[a].T
    ra, rb = hom(pa), hom(pb)
    E, inl = eransac(ra, rb, 1000)
    if inl.sum() < 8: continue
    _, Rr, tr, _ = cv2.recoverPose(E, pa[inl], pb[inl], Kc)
    out["Eproj 8pt"].append(rot_err(Rr, truth))
    Rl, _ = refine(Rr, tr.ravel(), ra[inl], rb[inl]); out["Eproj 8pt + LM"].append(rot_err(Rl, truth))
    Ee, me = cv2.findEssentialMat(pa, pb, Kc, cv2.RANSAC, 0.999, 2.0)
    _, Re, te, _ = cv2.recoverPose(Ee, pa, pb, Kc, mask=me.copy()); ein = me.ravel().astype(bool)
    Rel, _ = refine(Re, te.ravel(), ra[ein], rb[ein]); out["E 5pt + LM"].append(rot_err(Rel, truth))
for k, v in out.items():
    v = np.sort(v)
    print(f"{k:15s}: {len(v)} pairs, median {np.median(v):.2f} p75 {v[len(v)*3//4]:.2f} p90 {v[len(v)*9//10]:.2f}, <5 deg {np.mean(v < 5):.0%}")
