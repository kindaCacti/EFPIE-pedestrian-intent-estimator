# Trajectory extension attribution

Existing detector/core YAML modules are imported from the unchanged sibling
SOVA checkout. Its existing licenses and notices continue to apply.

The optional BiTraP adapter loads source from the authors' repository:
[BiTraP source](https://github.com/umautobots/bidirection-trajectory-predicter),
[paper](https://arxiv.org/abs/2007.14558),
[upstream license](https://github.com/umautobots/bidirection-trajectory-predicter/blob/main/LICENSE.md).
That source is licensed CC BY-NC-SA 4.0 and includes components attributed to
Trajectron++. Upstream headers/licenses are retained in the optional checkout.
Runtime compatibility transformations fix hard-coded CUDA devices,
distribution shape initialization, and GMM's hard-coded 2D/6D dimensions for
the configured 4D box schema. These changes are labelled in model metadata.

The class-aware tracker independently implements the two-stage motion/IoU
association method described in
[ByteTrack](https://github.com/FoundationVision/ByteTrack) and its
[paper](https://arxiv.org/abs/2110.06864). It does not vendor the YOLOX detector
or add an appearance model. No official MOT benchmark equivalence is claimed.
