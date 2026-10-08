"""Delete stale fingerprinted copies in a Blazor publish's _framework folder: per 'name.<fingerprint>.ext' group keep the
newest file only (each publish into the same folder leaves the previous fingerprints behind).
usage: python prune_framework.py <publish dir> [--dry]"""
import collections
import os
import re
import sys

root = os.path.join(sys.argv[1], "wwwroot", "_framework")
dry = "--dry" in sys.argv
pat = re.compile(r"^(?P<stem>.+)\.(?P<fp>[a-z0-9]{10})\.(?P<ext>[A-Za-z0-9.]+)$")
groups = collections.defaultdict(list)
for f in os.listdir(root):
    m = pat.match(f)
    if m:
        groups[(m["stem"], m["ext"])].append(f)
freed = removed = 0
for key, files in groups.items():
    if len(files) < 2:
        continue
    files.sort(key=lambda f: os.path.getmtime(os.path.join(root, f)), reverse=True)
    for stale in files[1:]:
        p = os.path.join(root, stale)
        freed += os.path.getsize(p)
        removed += 1
        if not dry:
            os.remove(p)
print(f"{'would remove' if dry else 'removed'} {removed} stale files, {freed / 1e9:.2f} GB")
