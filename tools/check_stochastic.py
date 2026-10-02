"""Stochastic-renderer gate: the converged stochastic image must match the sorted (alpha-blended) render.

    python tools/check_stochastic.py            (app on SPAWNSCENE_APP_PORT, harness Chrome on SPAWNSCENE_CDP_PORT)

Renders the single-photo Room sample (?autotest=generate-room) once with &render=sorted and once with &render=stochastic,
the camera still so the stochastic image converges, and compares the two captures: PSNR >= 28 dB and mean brightness
within 5%. Stochastic transparency converges to exactly the alpha-composited image, so a gap is a renderer bug.

MEASURED 2026-10-02 when written: before the fixes 14.70 dB, brightness 73 vs 109 (every splat over a pixel shared one
random number; 8-bit accumulation froze); after, 32.61 dB, 107 vs 109.
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
SHOTS = os.path.join(ROOT, '_shots')


def capture(mode):
    out = os.path.join(SHOTS, f'stochastic_gate_{mode}.png')
    env = dict(os.environ, MSYS_NO_PATHCONV='1')
    # Output to a FILE, not a pipe: the harness's Chrome inherits the child's handles and outlives it, so a pipe never
    # reaches EOF and subprocess.run hung until its timeout (2026-10-02).
    log_path = os.path.join(SHOTS, f'stochastic_gate_{mode}.log')
    with open(log_path, 'w', encoding='utf-8') as logf:
        subprocess.run(['node', os.path.join(ROOT, 'tools', '_cdp_page.js'),
                        f'/studio?autotest=generate-room&render={mode}', out, '1600x1000', '45000'],
                       cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT, timeout=600)
    log = open(log_path, encoding='utf-8', errors='replace').read()
    if '[Autotest] PASS' not in log:
        print(log[-2000:])
        sys.exit(f'FAIL: generate-room with render={mode} did not pass')
    return np.asarray(Image.open(out).convert('RGB')).astype(float)[60:900, :]  # below the HUD bar, above the stats


sorted_img = capture('sorted')
stochastic_img = capture('stochastic')
mse = ((sorted_img - stochastic_img) ** 2).mean()
psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-9))
bs, bt = sorted_img.mean(), stochastic_img.mean()
print(f'stochastic vs sorted: PSNR {psnr:.2f} dB, mean brightness {bt:.1f} vs {bs:.1f}')
if psnr < 28 or abs(bt - bs) > 0.05 * bs:
    sys.exit('FAIL: the converged stochastic image does not match the sorted render')
print('PASS')
