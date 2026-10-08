import sys, numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
for f in sys.argv[1:]:
    d = np.load(f); out = []
    for k in ("zd", "zm"):
        z = d[k]; m = d["mask"]
        hi, lo = maximum_filter(z, 7), minimum_filter(z, 7)
        step = ((hi - lo) / lo > 0.25) & m
        t = (z - lo) / np.maximum(hi - lo, 1e-9)
        fly = step & (t > 0.2) & (t < 0.8)
        out.append(f"{k} step px {step.mean()*100:4.1f}%  flying {fly.mean()*100:5.2f}% ({fly.sum()/max(step.sum(),1)*100:4.1f}% of step zone)")
    print(f.split('_moge')[0].ljust(12), " | ".join(out))
