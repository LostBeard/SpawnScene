"""Delete stale fingerprinted files from a Blazor publish's _framework folder: every publish into the same folder leaves
the previous fingerprints behind (3.8 GB over two harness folders, 2026-10-07).

Keeps exactly what the app LOADS: the files index.html names, closed over the names inside each kept .js/.json file
(dotnet.<fp>.js carries the boot manifest). A first version kept "the newest file of each name" - wrong when a folder
alternates interpreted and AOT publishes: an unchanged file keeps its old timestamp, a stale copy of another build was
newer, and the app's own Microsoft.AspNetCore.Components.<fp>.wasm was deleted (the f0 run could not boot).
Refuses to delete anything when the closure looks wrong (fewer than 20 files).

usage: python prune_framework.py <publish dir> [--dry]"""
import os
import re
import sys

wwwroot = os.path.join(sys.argv[1], "wwwroot")
fw = os.path.join(wwwroot, "_framework")
dry = "--dry" in sys.argv
present = set(os.listdir(fw))
fingerprinted = re.compile(r"^.+\.[a-z0-9]{10}\.[A-Za-z0-9.]+$")
name_pat = re.compile(r"[A-Za-z0-9_.\-]+\.[A-Za-z0-9]+")

with open(os.path.join(wwwroot, "index.html"), encoding="utf-8", errors="ignore") as f:
    seed = f.read()
keep = set()
todo = [n for n in set(name_pat.findall(seed)) if n in present]
while todo:
    n = todo.pop()
    if n in keep:
        continue
    keep.add(n)
    if n.endswith((".js", ".json", ".mjs")):
        with open(os.path.join(fw, n), encoding="utf-8", errors="ignore") as f:
            text = f.read()
        todo.extend(m for m in set(name_pat.findall(text)) if m in present and m not in keep)

# Names the kept files mention that are NOT on disk: a folder the app cannot boot from.
missing = set()
for n in keep:
    if n.endswith((".js", ".json", ".mjs")):
        with open(os.path.join(fw, n), encoding="utf-8", errors="ignore") as f:
            missing.update(m for m in name_pat.findall(f.read()) if fingerprinted.match(m) and m not in present
                           and m.endswith((".wasm", ".dat", ".js", ".pdb")))
if missing:
    print(f"WARNING: {len(missing)} referenced framework files are missing, e.g. {sorted(missing)[:3]}")
if len(keep) < 20:
    print(f"prune refused: only {len(keep)} referenced files found - the manifest was not parsed")
    sys.exit(0)
stale = [n for n in present if fingerprinted.match(n) and n not in keep]
freed = sum(os.path.getsize(os.path.join(fw, n)) for n in stale)
if not dry:
    for n in stale:
        os.remove(os.path.join(fw, n))
print(f"{'would remove' if dry else 'removed'} {len(stale)} stale files ({freed / 1e9:.2f} GB); kept {len(keep)} referenced")
