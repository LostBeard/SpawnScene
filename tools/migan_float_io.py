"""Make a float-I/O variant of MI-GAN's plain migan.onnx (uint8 image/mask in, uint8 out) for an engine whose tensors are
float: inputs typed FLOAT, every Cast(to=UINT8) becomes Cast(to=FLOAT), the output typed FLOAT. Values stay 0..255.
Then check against the original with onnxruntime on random data.
usage: python migan_float_io.py <migan.onnx> <out.onnx>"""
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src)
g = m.graph
for v in list(g.input) + list(g.output):
    v.type.tensor_type.elem_type = TensorProto.FLOAT
changed = 0
for n in g.node:
    if n.op_type == "Cast":
        for a in n.attribute:
            if a.name == "to" and a.i == TensorProto.UINT8:
                a.i = TensorProto.FLOAT
                changed += 1
onnx.checker.check_model(m)
onnx.save(m, dst)
print("casts retargeted:", changed)

rng = np.random.default_rng(0)
img = rng.integers(0, 256, (1, 3, 512, 512), dtype=np.uint8)
mask = np.full((1, 1, 512, 512), 255, np.uint8)
mask[:, :, 150:350, 180:380] = 0
a = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(None, {"image": img, "mask": mask})[0].astype(np.float32)
b = ort.InferenceSession(dst, providers=["CPUExecutionProvider"]).run(None, {"image": img.astype(np.float32), "mask": mask.astype(np.float32)})[0]
d = np.abs(a - np.round(b))
print("max |uint8 - round(float)|:", d.max(), "mean:", d.mean(), "float range", b.min(), b.max())
