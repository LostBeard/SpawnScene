# Third-party notices - SpawnScene

SpawnScene is MIT licensed (see LICENSE.txt). It downloads, at run time and through the SpawnDev hub's Hugging Face
proxy, model files that carry their own licences. They are listed here.

## Learned feature front end (RaCo-ALIKED + LightGlue+)

Used by `Services/LearnedFeatureMatcher.cs` (`&features=learned`). Files: `kornia/raco-aliked`
(`raco_aliked_extractor_k{1024,3072}.onnx`) and `kornia/lightglue` (`lightglue_matcher_k{1024,3072}.onnx`) on
Hugging Face. They are Kornia's split of fabio-sim's fused LightGlue-ONNX export. No weights were changed.

| Component | Source | Licence |
|---|---|---|
| RaCo detector + ranker | [cvg/RaCo](https://github.com/cvg/RaCo) | Apache-2.0 |
| ALIKED descriptors (`aliked-n16`) | [Shiaoming/ALIKED](https://github.com/Shiaoming/ALIKED) | BSD-3-Clause (notice below) |
| LightGlue matcher | [cvg/LightGlue](https://github.com/cvg/LightGlue) | Apache-2.0 |
| ONNX export | [fabio-sim/LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX) | Apache-2.0 |
| Extractor / matcher split | [kornia/vision-rt](https://github.com/kornia/vision-rt) | Apache-2.0 |

Credit belongs to the original authors:

- RaCo: Shenoi, Lindenberger, Sarlin, Pollefeys, "RaCo: Ranking and Covariance for Practical Learned Keypoints",
  3DV 2026, arXiv:2602.15755.
- ALIKED: Zhao et al., "ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable
  Transformation", IEEE TIM 2023.
- LightGlue: Lindenberger, Sarlin, Pollefeys, "LightGlue: Local Feature Matching at Light Speed", ICCV 2023.
- Export: Copyright 2023 ETH Zurich; Copyright 2023-2026 Fabio Milentiansen Sim.

### ALIKED - BSD-3-Clause

The ALIKED licence requires this notice to be reproduced with redistributions in binary form (an ONNX graph is one):

```
BSD 3-Clause License

Copyright (c) 2022, Zhao Xiaoming
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
