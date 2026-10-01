# Design document: temporal pedestrian-path prediction extension

## Status

Proposed design for the next project increment. It extends the testing and training environment in `src/SOVA-signs-det-dev`; it does not change the existing object-detection contract.

## Context

`SOVA-signs-det-dev` already provides a configurable object-detection environment. Model families are isolated behind `DetectorAdapter`, task selection is made through `task`, and YAML composition (`__include__` plus CLI overrides) supplies one configuration mechanism. Detectors return canonical per-image detection dictionaries, so their implementation details do not leak into callers.

Pedestrian path prediction needs temporal input rather than a single image. For online/testing inference it must receive recent frames, detector outputs, timestamps, and stable pedestrian identities. For offline training it must turn labelled video sequences into observed history and future-path targets. The extension therefore introduces two reusable parts:

1. A bounded frame-and-detection cache used by live/test inference.
2. An offline dataset-preparation module that creates a detector-derived, COCO-compatible frame dataset.
3. A model-independent trajectory-training task with adapters analogous to the existing detector adapters.

## Goals

- Retain the last configurable number of frames and their bounding boxes.
- Provide a safe temporal snapshot API without exposing mutable cache internals.
- Associate pedestrian detections across cached frames and construct fixed-length histories.
- Produce a versioned offline dataset of per-frame pedestrian detections before trajectory training begins.
- Train, validate, checkpoint, and infer with path predictors through a common interface.
- Allow path-prediction model libraries to be added as adapters without changing the solver, dataset, cache, or CLI.
- Reuse the existing YAML configuration, output, device, and reproducibility conventions.

## Non-goals

- Replacing or retraining the object detector.
- Defining camera transport, a UI, a production video-ingestion service, or a long-term video archive.
- Mandating a tracker or a trajectory architecture.
- Estimating world coordinates without calibration. Version one uses image-space coordinates; a later transformer may add metres.

## Proposed package layout

```text
src/SOVA-signs-det-dev/
├── conf/
│   ├── base/temporal.yml
│   ├── dataset/pie_trajectory.yml
│   └── models/trajectory/<family>/{base.yml,<size>.yml}
├── engine/
│   ├── data_prep/
│   │   ├── coco_detection_export.py
│   │   ├── detection_writer.py
│   │   ├── manifest.py
│   │   └── sequence_index.py
│   ├── temporal/
│   │   ├── types.py
│   │   ├── frame_cache.py
│   │   ├── tracker.py
│   │   └── sequence_builder.py
│   ├── trajectory/
│   │   ├── base_adapter.py
│   │   ├── dataset.py
│   │   ├── collate.py
│   │   ├── metrics.py
│   │   └── <family>_adapter.py
│   └── solver/
│       ├── trajectory_solver.py
│       └── tasks.py
├── tools/run_temporal_inference.py
├── tools/prepare_trajectory_dataset.py
└── tests/test_frame_cache.py, test_sequence_builder.py,
    test_trajectory_adapter.py, test_trajectory_solver.py
```

The `data_prep` package is an offline producer: it invokes a configured detector once and writes an immutable dataset artifact. The `temporal` package has no PyTorch model dependency. The `trajectory` package consumes prepared artifacts and owns the adapter boundary. `TrajectorySolver` is the new task-level orchestrator. `tasks.py` registers it as `"trajectory"` alongside the current `"detection"` task.

## 1. Offline trajectory-dataset preparation

No trajectory-labelled dataset is assumed. The project must first prepare a reproducible pseudo-labelled dataset from a collection of input videos or ordered image sequences. Preparation is a separate, explicit command and is never run by `TrajectorySolver`; training must only read its completed output. This avoids detector inference during epochs, prevents labels changing between runs, and makes experiments reproducible.

### Input and output

The input is a manifest of immutable source sequences. Each entry identifies a `sequence_id`, an ordered video file or frame directory, an optional FPS/timestamp source, and the requested split (`train`, `val`, or `test`). Split assignment is performed at sequence level before detection and no source sequence may occur in more than one split.

The preparation command samples/reads every requested frame, runs one configured existing `DetectorAdapter`, retains configured pedestrian categories, then applies the configured tracker. It writes a self-contained output directory:

```text
data/trajectory/prepared/<dataset-version>/
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json
├── images/
│   ├── train/<sequence_id>/<frame_index>.jpg
│   ├── val/<sequence_id>/<frame_index>.jpg
│   └── test/<sequence_id>/<frame_index>.jpg
├── manifests/
│   ├── source_sequences.json
│   ├── prepared_dataset.json
│   └── rejected_frames.jsonl
└── resolved_config.yml
```

Frames are copied, hard-linked, or symlinked according to `image_mode`; `symlink` is the default where the source is already an image sequence. Video input is decoded to image files because standard COCO `file_name` values refer to individual images. The output directory is created atomically: preparation writes to a temporary sibling directory, validates it, then renames it into the versioned destination. An existing completed version is never overwritten without an explicit `--force` option.

### COCO-compatible annotation schema

Each split is valid COCO detection JSON: `images`, `annotations`, and `categories` are present; every annotation uses COCO's pixel `bbox: [x, y, width, height]`, `area`, `iscrowd: 0`, `image_id`, and `category_id`. Coordinates are clipped to the original image and boxes with non-positive width/height are rejected. The category list contains only the selected pedestrian categories, with the original detector category mapping recorded in the manifest.

COCO has no native video, time, or identity fields. The prepared dataset therefore uses documented, backward-compatible extension fields rather than altering the required COCO fields:

```json
{
  "images": [{
    "id": 1000007,
    "file_name": "train/camera_a/000007.jpg",
    "width": 1920, "height": 1080,
    "sequence_id": "camera_a",
    "frame_index": 7,
    "timestamp_s": 0.233333
  }],
  "annotations": [{
    "id": 42,
    "image_id": 1000007,
    "category_id": 1,
    "bbox": [812.5, 391.0, 97.0, 248.0],
    "area": 24056.0,
    "iscrowd": 0,
    "score": 0.91,
    "track_id": "camera_a:14"
  }]
}
```

Generic COCO readers can ignore the extension fields and still consume the detection annotations. `TrajectoryDataset` requires `sequence_id`, `frame_index`, `timestamp_s`, and `track_id`; it fails clearly if an ordinary COCO export lacking these fields is supplied. IDs must be globally unique within the JSON. `track_id` is namespaced by sequence, even if a tracker restarts its numeric counter.

### Preparation flow

```text
source manifest -> decode/order frames -> DetectorAdapter.predict(batch)
                -> pedestrian category and confidence filtering
                -> tracker.update per sequence -> COCO writer + frame files
                -> validation + immutable manifest
```

The tracker is reset at every sequence boundary. Frame order and timestamps are validated before inference. The writer streams annotations rather than retaining decoded images or all model outputs in memory; it may buffer JSON records per split before final serialization. Batching is a preparation performance setting only and cannot alter ordering or identifiers.

The preparation tool is conceptually invoked as:

```bash
uv run tools/prepare_trajectory_dataset.py \
  --source-manifest conf/dataset/source_sequences.yml \
  -c conf/models/rtdetrv4/s.yml \
  --output data/trajectory/prepared/camera-v1
```

It records the resolved detector YAML, checkpoint path and SHA-256, detector adapter/version, preprocessing settings, confidence/category filters, tracker configuration, source-file fingerprints, frame sampling rate, image mode, counts by split, and creation time in `prepared_dataset.json`. This manifest is the dataset provenance and is copied into each trajectory training run. A changed detector checkpoint, tracker configuration, source frame, or sampling rate requires a new dataset version.

### Preparation configuration

```yaml
dataset_preparation:
  source_manifest: conf/dataset/source_sequences.yml
  output_dir: data/trajectory/prepared/camera-v1
  image_mode: symlink              # symlink | hardlink | copy
  frame_stride: 1
  inference_batch_size: 8
  pedestrian_class_ids: [0]
  confidence_threshold: 0.30
  tracker:
    type: bytetrack
    high_confidence_threshold: 0.50
    low_confidence_threshold: 0.10
    match_iou_threshold: 0.30
    track_buffer_frames: 30
  validation:
    min_track_length_frames: 20
    reject_nonmonotonic_timestamps: true
```

Tracks shorter than the minimum future-plus-observation horizon are retained in COCO for auditability but excluded by `TrajectoryDataset`, with the exclusion count reported. `rejected_frames.jsonl` records unreadable frames and detection/annotation validation failures with their sequence and reason; a failure threshold prevents a silently incomplete dataset from being accepted.

### Nature and limitation of labels

The resulting paths are pseudo-labels: the future target is the continuation of detector boxes associated by the tracker, not human ground truth. They are suitable for building and comparing the pipeline where no labelled trajectory dataset exists, but report results as detector/tracker-derived trajectory prediction. The manifest and evaluation output must identify this provenance. Human annotations or a verified benchmark can later be imported through the same COCO-plus-temporal-extension schema, enabling like-for-like training and evaluation without changing path-model adapters.

## 2. Temporal cache

### Data model

`engine.temporal.types` defines immutable records. Detection boxes are always absolute `xyxy` pixel coordinates in the original image, matching the canonical output expected from `DetectorAdapter.predict`.

```python
@dataclass(frozen=True)
class Detection:
    box_xyxy: tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str | None = None
    track_id: int | None = None

@dataclass(frozen=True)
class FrameRecord:
    sequence_id: str
    frame_index: int
    timestamp_s: float
    image: np.ndarray | None          # optional, never persisted by default
    image_size: tuple[int, int]       # (width, height)
    detections: tuple[Detection, ...]

@dataclass(frozen=True)
class TemporalSnapshot:
    sequence_id: str
    frames: tuple[FrameRecord, ...]   # oldest to newest
```

The cache stores a lightweight frame reference or an image copy according to configuration. Bounding boxes, scores, labels, image size, source sequence, and timestamp are always retained. `track_id` is populated by the tracker after detector output is normalized. A cache entry is never modified after insertion; tracking creates a replacement `FrameRecord` before storage. This keeps snapshots reliable if a predictor runs concurrently with frame ingestion.

### API and behaviour

`FrameCache(max_frames)` wraps `collections.deque(maxlen=max_frames)` and exposes:

```python
append(frame: FrameRecord) -> None
snapshot(sequence_id: str | None = None) -> TemporalSnapshot
clear(sequence_id: str | None = None) -> None
__len__() -> int
```

- `append` requires monotonically increasing `frame_index` and `timestamp_s` within a sequence and rejects invalid boxes or image sizes.
- A changed `sequence_id` clears the active stream before adding the new frame. Histories from distinct videos/cameras therefore cannot be mixed.
- On overflow, the oldest frame is evicted automatically. There are no disk writes, background threads, or unbounded collections.
- `snapshot` returns an immutable, oldest-to-newest tuple. Consumers must not receive the internal deque or image arrays by reference.
- A lock protects mutation and snapshot construction. CUDA tensors and autograd graphs are forbidden from cache records.

`TemporalInferencePipeline` composes an existing detector, `FrameCache`, a `PedestrianTracker`, and a `TrajectoryAdapter`:

```text
frame + timestamp
  -> DetectorAdapter.predict([frame])
  -> retain configured pedestrian classes / normalize xyxy boxes
  -> tracker assigns track_id
  -> FrameCache.append(FrameRecord)
  -> SequenceBuilder.history_for(track_id)
  -> TrajectoryAdapter.predict_paths(batch) when enough history exists
```

It returns no path plus status `insufficient_history` until the observation horizon exists. It must never silently pad live histories by repeating an old frame; padding is opt-in configuration.

### Tracking and coordinate representation

The default tracker is **ByteTrack**, encapsulated as `ByteTrackPedestrianTracker`. It is used identically during offline preparation and live temporal inference so that the trajectory model observes the same track-identity behaviour in both paths. ByteTrack first associates high-confidence pedestrian detections to active tracks, then uses lower-confidence detections only to recover unmatched tracks. This is preferable to discarding every low-score pedestrian box, which commonly breaks tracks during partial occlusion.

The wrapper is class-aware and admits only configured pedestrian class IDs. It passes the detector's original `xyxy` boxes and scores to ByteTrack, assigns a sequence-namespaced `track_id`, and returns canonical `Detection` records. The wrapper must not add an appearance/re-identification model: identity association in the initial increment is motion and IoU based. Each source-sequence change resets the ByteTrack state; it is never shared between cameras, splits, or videos.

`ByteTrackPedestrianTracker` exposes `update(detections, timestamp_s) -> detections`. Its configuration contains a high-confidence threshold for initial matching, a lower threshold for the recovery association, an IoU match threshold, and a track buffer measured in frames. Validation requires `0 <= low_confidence_threshold <= high_confidence_threshold <= 1`, a positive buffer, and pedestrian-only class filtering. The implementation remains behind the `PedestrianTracker` interface, allowing an `IoUTracker` or a future appearance tracker to be selected without changing cache, exporter, dataset, or path-model adapters.

`SequenceBuilder` converts a track into a trajectory observation. The default point is normalized bounding-box bottom centre `(cx / width, y2 / height)`, a better ground-contact proxy than box centre. It also returns a validity mask and time deltas. This representation works across resolutions and detector architectures. Image-space targets are labelled as such in metadata. A future `CoordinateTransformer` can add calibrated world coordinates without changing adapters.

### Configuration

`conf/base/temporal.yml` is included by temporal-inference and trajectory configs. These are defaults, not hard-coded limits:

```yaml
temporal:
  cache:
    max_frames: 32
    store_images: false
    require_monotonic_timestamps: true
  tracker:
    type: bytetrack
    pedestrian_class_ids: [0]
    high_confidence_threshold: 0.50
    low_confidence_threshold: 0.10
    match_iou_threshold: 0.30
    track_buffer_frames: 30
  sequence:
    observation_frames: 8
    prediction_frames: 12
    allow_padded_history: false
    point: bottom_center_normalized
```

Validation rejects `max_frames < observation_frames`, non-positive horizons, and class IDs absent from detection dataset metadata. Cache capacity is intentionally independent from the observation horizon so debugging or a future model can inspect more context.

## 3. Model-independent trajectory training

### Dataset contract

`TrajectoryDataset` consumes the completed COCO-plus-temporal-extension export, never a detector or source video. It reads each per-track record as `(sequence_id, frame_index, timestamp_s, track_id, point, visible)` and slides an observation/prediction window over a single track without crossing sequence boundaries.

Each sample has this model-neutral form:

```python
{
  "sequence_id": str,
  "track_id": int,
  "observed_points": FloatTensor[T_obs, 2],
  "observed_time_deltas": FloatTensor[T_obs],
  "observed_mask": BoolTensor[T_obs],
  "future_points": FloatTensor[T_pred, 2],
  "future_mask": BoolTensor[T_pred],
  "metadata": {"coordinate_system": "image_normalized", ...},
}
```

The default dataset requires visible ground-truth pedestrian tracks at every future timestep. Observations may contain gaps only when configured; the collator pads variable-length observations and preserves masks. Splits are made by `sequence_id`, never random frame, to prevent adjacent frames of a person appearing in both train and validation data.

The first reader is the prepared COCO reader described above, which makes available video/image collections usable despite the absence of a ready-made trajectory dataset. Adding a labelled source such as PIE, JAAD, nuScenes, or custom annotations means adding an offline exporter into the same schema, not changing a path-model adapter.

### Adapter contract

`TrajectoryAdapter` mirrors the lifecycle and isolation principle of `DetectorAdapter`, but it deals only in canonical sequence batches. It neither imports a detector nor accesses `FrameCache`.

```python
class TrajectoryAdapter(ABC):
    name: str
    def __init__(self, cfg): ...
    @property
    def device(self) -> torch.device: ...
    def build(self) -> torch.nn.Module: ...
    @abstractmethod
    def build_model(self, device, checkpoint=None): ...
    @abstractmethod
    def prepare_batch(self, batch, training: bool): ...
    @abstractmethod
    def forward(self, prepared_batch): ...
    @abstractmethod
    def compute_loss(self, outputs, prepared_batch): ...
    @abstractmethod
    def decode(self, outputs, prepared_batch) -> TrajectoryPrediction: ...
    def predict_paths(self, batch) -> list[TrajectoryPrediction]: ...
    def save(self, path): ...
    def load(self, path): ...
```

`TrajectoryPrediction` contains `track_id`, a `[T_pred, 2]` path, optional confidence/uncertainty, a coordinate-system label, and source sequence/frame identifiers. Deterministic models return one path; probabilistic models may return `K` paths and probabilities. `decode` must represent that distinction explicitly rather than overload a detection-box structure.

An adapter declares whether it uses the framework-native loop or an external trainer, following the established detector pattern. Native adapters implement the hooks above. External adapters override `train_external` and `validate_external`, but still accept and emit canonical structures. A dedicated `TRAJECTORY_ADAPTERS` registry belongs in `engine/trajectory/__init__.py`; the detector registry is not changed.

Checkpoints use a portable envelope:

```python
{
  "format_version": 1,
  "task": "trajectory",
  "adapter": "constant_velocity",
  "model": state_dict,
  "optimizer": optimizer_state_or_null,
  "epoch": epoch,
  "config": resolved_yaml,
  "coordinate_system": "image_normalized",
}
```

The first reference implementations should be a constant-velocity extrapolator (parameter-free) and a compact GRU/MLP baseline. They validate data, metrics, and lifecycle before a research model is integrated; neither mandates the final architecture.

### ASTRA adapter: reduced trajectory-only profile

ASTRA is an optional `TrajectoryAdapter` implementation. The first supported profile is a **reduced trajectory-only ASTRA** configuration: it consumes SOVA detector boxes associated by ByteTrack and does not enable ASTRA's pretrained U-Net scene-encoder branch. SOVA therefore supplies the tracked pedestrian measurements; it does not replace or provide weights for ASTRA's U-Net.

```text
prepared COCO boxes -> ByteTrack track history -> ASTRAAdapter
                                           -> ASTRA trajectory prediction
```

For this adapter, `TrajectoryDataset` must retain the normalized box sequence in addition to the generic two-dimensional point sequence. `ASTRAAdapter.prepare_batch` converts each detection to `[cx, cy, width, height]` and supplies the observed and future box sequences expected by ASTRA. The model's returned future boxes are decoded to the common trajectory result using their bottom-centre points; raw predicted boxes remain available in result metadata for rendering and evaluation.

```yaml
task: trajectory
model:
  adapter: astra
astra:
  profile: trajectory_only
  use_pretrained_unet: false
  use_social: false             # enabled only after scene-window grouping is implemented
  use_vae: false                # deterministic baseline
  coordinate_representation: normalized_cxcywh
```

`use_pretrained_unet: false` means no ASTRA U-Net checkpoint, observation-frame image tensors, or scene-embedding module is loaded. This makes the reduced profile compatible with the offline SOVA-plus-ByteTrack COCO export and live cache. It is not the full scene-aware ASTRA architecture described in the paper, so experiments and result tables must label it `ASTRA trajectory-only` rather than `ASTRA scene-aware`.

The downloaded ASTRA trajectory checkpoint must not be loaded blindly into this profile: checkpoints trained with a scene-U-Net branch have a different architecture and input schema. The adapter either loads a checkpoint trained with the same `trajectory_only` profile or trains a new one from the prepared dataset. A later `scene_aware` profile may enable the U-Net only after the sequence pipeline supplies aligned observation frames and the adapter implements ASTRA's frozen U-Net embedding path. Likewise, social ASTRA support requires a scene-window builder that groups neighbouring ByteTrack tracks at each timestamp; it is intentionally out of scope for the first reduced profile.

### BiTraP adapter: goal-conditioned multi-modal baseline

BiTraP is an optional trajectory-only baseline well matched to the prepared SOVA-plus-ByteTrack export. `BiTraPAdapter` consumes an observed sequence of one pedestrian's bounding boxes and predicts future bounding boxes; it does not require scene images, semantic maps, social-track grouping, or ego-motion as part of its core input contract. The adapter converts the COCO boxes to the configured `xywh` or normalized `cxcywh` representation at its boundary and preserves the original image dimensions required to render or score box predictions.

BiTraP first predicts one or more future end-point goals, then applies its bidirectional decoder to generate a complete path from the current observation towards each goal. The adapter exposes three profiles:

```yaml
task: trajectory
model:
  adapter: bitrap
bitrap:
  variant: deterministic     # deterministic | np | gmm
  coordinate_representation: normalized_cxcywh
  observation_frames: 15
  prediction_frames: 45
  num_samples: 20            # required for np/gmm validation
```

- `deterministic` is the `BiTraP-D` ablation and emits one future box path.
- `np` is BiTraP-NP: a CVAE with a continuous Gaussian latent variable; it emits `num_samples` plausible paths.
- `gmm` is BiTraP-GMM: a categorical/Gaussian-mixture formulation that emits component probabilities with its paths.

For the multi-modal profiles, `TrajectoryPrediction` contains all `K` sampled box paths and their probabilities when available. Evaluation reports the existing single-path metrics for the deterministic profile and additionally reports `minADE`, `minFDE`, and sample count for multi-modal profiles. To compare with the BiTraP first-person protocol, the metrics module also provides box ADE, centre ADE (CADE), centre FDE (CFDE), and optional KDE-NLL. Reports must state the number of samples used for every minimum-of-`K` result.

The prepared dataset must retain observed and future boxes, not only bottom-centre points, for this adapter. A valid BiTraP training window contains a continuous track with the configured observation plus future length; gaps are rejected by default because the original model's velocity/residual construction assumes consecutive timesteps. The reference benchmark uses 15 observed frames (0.5 s) and 45 future frames (1.5 s) at 30 Hz; values may be changed only together with the dataset sampling rate and clearly recorded in resolved configuration.

Published BiTraP checkpoints are not assumed compatible with detector-derived pseudo-labels: they were trained with the source dataset's splits, annotation format, normalization, and horizons. They may be used only after `BiTraPAdapter` validates the complete input schema; otherwise the model is trained from the prepared COCO export. The upstream implementation uses an older PyTorch stack, so it is vendored or isolated as an optional adapter dependency and must not constrain the core SOVA environment.

### Solver, metrics, and CLI

`TrajectorySolver` follows `DetSolver`: resolve `model.adapter`, delegate external trainers, otherwise run the generic epoch/optimizer/checkpoint loop. `train.py` remains the entry point; `task: trajectory` selects the solver.

Validation reports masked displacement metrics in the configured coordinate space:

- ADE: average displacement error over valid predicted timesteps.
- FDE: final displacement error at each track's last valid timestep.
- minADE/minFDE for multi-modal predictions.
- Valid sample and point counts, so scores cannot hide dropped trajectories.

For image-normalized points metrics are dimensionless and must be labelled accordingly. Pixel/metre reporting is allowed only when a coordinate transformer records the resolution/calibration used.

```yaml
# conf/models/trajectory/gru/base.yml
__include__: ["../../../base/runtime.yml", "../../../base/temporal.yml"]
task: trajectory
model:
  adapter: gru_trajectory
trajectory:
  hidden_size: 128
  learning_rate: 0.001
```

```bash
uv run train.py -c conf/models/trajectory/gru/base.yml \
  --dataset-config conf/dataset/pie_trajectory.yml \
  -u temporal.sequence.observation_frames=8 temporal.sequence.prediction_frames=12
```

The present dataset-override helper may need a small generalization: it must merge trajectory fields without assuming detection-only fields such as `num_classes`, `ann_format`, or `sparse_ids`.

## Integration boundaries

Preparation is offline, so training does not invoke the detector or mutate labels. Initially, both training inputs and live inputs are detector-plus-tracker trajectories, sourced from the prepared COCO export and live cache respectively. If labelled trajectories later become available, evaluation supports two modes:

- `ground_truth_history`: labelled observed points, measuring predictor quality.
- `detected_history`: configured detector/tracker observations, measuring the end-to-end pipeline for tracks matched to ground truth.

Both use the same adapter and metrics. Reports identify detector checkpoint, tracker configuration, and mode.

## Failure handling and observability

- Missing/late detections create a masked observation or no prediction as configured; they never acquire another pedestrian's history.
- A sequence change flushes cache and tracker.
- Invalid frames, non-monotonic timestamps, invalid boxes, incompatible coordinate systems, and short caches fail with actionable errors.
- Log cache occupancy, evictions, active tracks, history-ready tracks, prediction latency, and drops by reason. Save these with resolved YAML for an inference run.
- Optional rendering may show track IDs, history, and paths, but rendering stays outside core cache/model APIs.

## Tests and acceptance criteria

Preparation tests cover deterministic image/annotation IDs, standard COCO validity, `xyxy`-to-COCO-`xywh` conversion, clipping/rejection of invalid boxes, sequence-level split isolation, tracker reset, extension-field presence, and manifest/checkpoint provenance. They use a mock detector and tiny image fixture, so no model download is required. Cache tests cover ordering/eviction, immutability, sequence reset, timestamp validation, and no cross-sequence snapshot. ByteTrack tests cover high-confidence association, low-confidence recovery of an unmatched track, expiry after `track_buffer_frames`, class filtering, sequence reset, and unmatched detections. Sequence tests cover bottom-centre conversion, normalization, masks, and insufficient history.

Dataset tests verify window counts, no `sequence_id` split leakage, aligned targets, and padding. Adapter-contract tests run a tiny native baseline through build, one train step, save/load, and `predict_paths`. Solver smoke tests run CPU-only on a synthetic two-track dataset and assert a checkpoint and ADE/FDE output. An end-to-end test feeds mock detector outputs through cache, tracker, sequence builder, and constant velocity.

The increment is accepted when the configured cache retains exactly the last `max_frames` records with canonical boxes; no prediction occurs before valid history; `task: trajectory` trains and validates a baseline from YAML; and a second adapter can be added without edits to `TrajectorySolver`, `TrajectoryDataset`, or temporal-cache code.

## Delivery sequence

1. Add the source manifest, offline COCO writer, provenance manifest, and preparation tests using a mock detector.
2. Add ByteTrack pedestrian tracking to preparation and temporal records, YAML validation, `FrameCache`, and unit tests.
3. Add `SequenceBuilder`, the prepared-COCO dataset/collator, sequence-level split validation, and ADE/FDE metrics using a synthetic export.
4. Add `TrajectoryAdapter`, registry, solver/task registration, checkpoint envelope, and constant-velocity smoke baseline.
5. Implement the GRU baseline against a prepared dataset.
6. Add temporal inference CLI, optional labelled-data exporter/evaluation, visualization, and experiment metadata.

This order establishes data and interface contracts before binding the project to a trajectory model or dataset implementation.
