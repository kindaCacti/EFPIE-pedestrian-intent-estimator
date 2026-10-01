# Pedestrian trajectory extension

All new code lives directly in `src/`. The sibling `SOVA-signs-det-dev/`
checkout is unchanged. `engine` extends its import path to reuse that project's
YAML composition and detector adapters read-only. ASTRA is excluded.

## Setup

From the repository root, initialize the read-only detector dependency before
running the extension:

```bash
git submodule update --init --recursive
```

Run commands from this directory:

```bash
cd src
uv sync --group dev
```

Core training, tracking, cache and image-sequence preparation use PyTorch,
NumPy, SciPy, Pillow and YAML. Video decoding is optional:
`uv sync --extra temporal-video`. Importing the temporal package does not
import PyTorch. To use detector adapters, also install their existing optional
dependencies: `uv sync --extra detector --extra rtdetrv4` (or another model
extra). Detector configs are read from `SOVA-signs-det-dev/conf/`. Optional
detector sources can live under this directory's `third_party/`; running an
existing pull script **from `src/`** places its sources there:

```bash
bash SOVA-signs-det-dev/tools/pull_rtdetrv4.sh
```

For the CPU environment, downloaded PIE videos, and exact end-to-end run
commands in this workspace, see [RUN_RESULTS.md](RUN_RESULTS.md). Use
`.venv/bin/python` with that environment; a plain `uv run` or `uv sync` can
replace the installed CPU PyTorch wheels with the lockfile's CUDA build.

## Prepare immutable pseudo-labels

Copy/edit `conf/dataset/source_sequences.yml`. Sources are relative to the
manifest. Each sequence must have a unique ID and source, exactly one split,
and a positive FPS or strictly increasing timestamps. Image directories use
natural filename order; `frames: ["000001.png", ...]` specifies an explicit
order. `timestamps_s` accepts a numeric list, a JSON array path, or a path to
newline-separated seconds. Video FPS can be obtained from container metadata.
Optional `start_frame` (inclusive) and `end_frame` (exclusive) select a bounded
section while preserving original source frame indices and timestamps.

```bash
uv run tools/prepare_trajectory_dataset.py \
  -c SOVA-signs-det-dev/conf/models/rtdetrv4/s.yml \
  -d SOVA-signs-det-dev/conf/dataset/coco.yml \
  --source-manifest conf/dataset/source_sequences.yml \
  --output data/trajectory/prepared/camera-v1 \
  -u dataset_preparation.image_mode=copy
```

`conf/dataset/preparation.yml` documents all preparation settings; pass its
fields through `-u` or include it from a detector YAML stored outside the
sibling checkout. Use `-r` for an existing SOVA checkpoint. Detector labels are
contiguous indices (`person` is `0` with the COCO config), even when original
COCO category IDs are sparse. The provenance manifest records both mappings.

Each artifact has COCO annotations for all three splits, frame files,
`source_sequences.json`, `prepared_dataset.json`, `rejected_frames.jsonl`, and
`resolved_config.yml`. Image annotations use clipped pixel `xywh` boxes;
image records add sequence/index/timestamp and annotations add sequence-scoped
track IDs. Unmatched detections may have `track_id: null` and do not form
training windows. Image sequence extensions are preserved. Videos are decoded
to JPEG frames.

`symlink` is the default image mode. Source images must remain available and
immutable with that mode; choose `copy` for a portable independent artifact.
Hardlinks share underlying image storage as well. Preparation checks source
SHA-256 before and after inference and annotation checksums are checked by the
reader. A detector checkpoint SHA-256 is recorded. If an adapter obtains its
weights internally, its exact built checkpoint is saved with the artifact.

Preparation uses bounded batches, resets tracking between sequences and
publishes only after validating a temporary sibling directory. Existing
versions require `--force`; the previous prepared version is retained in a
named backup. Failure thresholds default to at most 100 bad frames and 5% of
attempted frames. Short/untracked windows are counted and excluded by the
dataset. Training reads completed artifacts and never runs detection.

## Train and evaluate

Create a dataset YAML referencing the prepared artifact:

```yaml
task: trajectory
train_dataloader:
  dataset:
    prepared_dir: data/trajectory/prepared/camera-v1
val_dataloader:
  dataset:
    prepared_dir: data/trajectory/prepared/camera-v1
```

`conf/dataset/pie_trajectory.yml` is an example for previously prepared PIE
sequences; it does not download or assume human labels.

```bash
uv run train.py -c conf/models/trajectory/gru/base.yml \
  --dataset-config conf/dataset/pie_trajectory.yml --device cpu \
  -u temporal.sequence.observation_frames=8 temporal.sequence.prediction_frames=12

uv run train.py -c conf/models/trajectory/constant_velocity/base.yml \
  --dataset-config conf/dataset/pie_trajectory.yml --device cpu

uv run train.py -c conf/models/trajectory/gru/base.yml \
  --dataset-config conf/dataset/pie_trajectory.yml --test-only \
  -r outputs/trajectory/gru/checkpoint_last.pth
```

The native solver seeds RNGs, batches canonical samples, trains through adapter
hooks, validates each epoch and saves portable versioned envelopes containing
model/optimizer/epoch/config/coordinate and input-schema information. Resume
also restores scheduler and AMP state. `--tuning` loads model weights only.
Parameter-free constant velocity follows the same lifecycle. External adapters
can implement `train_external` and `validate_external`.

Results include ADE/FDE, minADE/minFDE for sampled predictions, valid sample
and point counts, actual sample count, dataset/label/detector/tracker provenance,
and evaluation mode. ADE is averaged per valid track window; FDE uses its last
valid target. Single-path scores for multi-modal models use the highest
probability path, or the first sample if per-path probabilities are unavailable.
All current point/box metrics are in normalized image space and dimensionless.
Box ADE is Euclidean displacement of normalized `cxcywh`; CADE/CFDE measure box
centre displacement. These are not published pixel benchmark scores. Optional
`evaluation.kde_bandwidth` enables point KDE-NLL with isotropic Gaussian kernels.
Pixel/metre metrics require a future explicit coordinate transformer.

## BiTraP

The optional adapter executes the authors' goal predictor and bidirectional
decoder in a private import namespace. It supports deterministic BiTraP-D,
Gaussian-latent NP, and categorical/GMM profiles. Fetch upstream source only:

```bash
bash tools/pull_bitrap.sh               # optionally pass a commit as argument
uv run train.py -c conf/models/trajectory/bitrap/np.yml \
  --dataset-config conf/dataset/pie_trajectory.yml
```

Use `bitrap.upstream_root` to point at another checkout. The adapter does not
install upstream's old PyTorch/training dependencies. Small AST-based runtime
corrections support the configured device and four-dimensional boxes without
editing that checkout. Source SHA-256 and the compatibility profile are stored
in checkpoints. Profiles have a common box schema; continuous visible windows
are required. Defaults are 15 observations, 45 future frames, 30 Hz and 20
samples. Horizons and sampling FPS must agree with the prepared data.

`bitrap.coordinate_representation` accepts `normalized_cxcywh` or pixel `xywh`;
the adapter retains image sizes and converts outputs back to canonical
normalized bottom-centre paths. Predictions expose sampled paths and raw box
paths. GMM outputs additionally expose component-mean paths with their actual
mixture probabilities; no artificial probabilities are assigned to samples.
Generic profiles reject published raw checkpoint files because they lack an
input schema. For the verified official PIE NP weights, use the dedicated
importer and pretrained profile below. See source/license details in
`THIRD_PARTY_NOTICES.md`.

### Pretrained PIE BiTraP-NP

The installed `models/bitrap/pie_np_k20.pth` contains the authors' official PIE
K=20 weights, converted into a schema-checked trajectory envelope. No training
is required. From `src/`, launch the pretrained viewer:

```bash
QT_QPA_PLATFORM=xcb uv run --no-sync python tools/view_trajectories.py \
  -c conf/models/trajectory/bitrap/pretrained_pie.yml --loop
```

It uses 15 observed frames and displays **20 possible futures, each 45 frames
(1.5 seconds) ahead at 30 FPS**. These sampled alternatives have no individual
probabilities; they are not guaranteed paths. CPU playback can be slow.
The original GRU viewer command still works unchanged.

To reproduce the import on another installation, fetch the upstream source
with `bash tools/pull_bitrap.sh`, then download and convert the weights:

```bash
mkdir -p models/bitrap/PIE
curl --fail --location \
  'https://drive.google.com/uc?export=download&id=1jLkwi1YSwCRfixAxL6K5cNAvzs3NqtVJ' \
  --output models/bitrap/PIE/bitrap_np_K_20.pth
uv run --no-sync python tools/import_bitrap_checkpoint.py \
  --source models/bitrap/PIE/bitrap_np_K_20.pth
```

The importer checks SHA-256 against the audited official PIE K=1/K=20 files,
uses tensor-only loading, loads all model keys strictly, records provenance,
and refuses to overwrite an existing converted checkpoint. K=20 SHA-256 is
`858ce863fd5d34766323798db928e85297d0fb363e822f9be07cbb3de418decf`.
The profile sets `decoder_with_z: false` to match the published architecture;
generic/self-trained BiTraP profiles retain their previous defaults.

This profile requires original **1920x1080 images, consecutive frames at 30
FPS**, normalized centre/width/height boxes, and 15/45-frame horizons. It rejects
other image sizes rather than silently changing the authors' fixed-reference
normalization. Using detector-derived tracks is not an official PIE benchmark.
The source is licensed CC BY-NC-SA 4.0; retain its notices and review upstream
terms before redistributing or using commercially.

## Live/test inference

To compute predictions while watching the validation video in a desktop
window, use the configured CPU environment from `src/`:

```bash
uv run --no-sync python tools/view_trajectories.py --loop
```

This loads the trained GRU checkpoint and RT-DETRv4 weights, processes the
validation section, and overlays tracked pedestrian boxes (green), observed
histories (yellow), and predicted futures (cyan). Space pauses/resumes;
Q, Esc, or closing the window exits. `--loop` resets tracking/history on each
replay. Without it the viewer exits after the selected section. Use
`--source-manifest conf/dataset/pie_sample_sources.yml` for the complete
validation video, or `--split train` / `--split all` for other sequences.
The viewer computes predictions now; it does not play saved overlays.

The GUI requires a desktop session and GUI-enabled OpenCV. On CPU, playback
slows to the inference speed: no frames are skipped, and original video
timestamps and the trained temporal sampling are preserved. `--no-sync`
keeps the installed CPU PyTorch wheels. This viewer writes no output files.

For offline export instead:

```bash
uv run tools/run_temporal_inference.py \
  --detector-config SOVA-signs-det-dev/conf/models/rtdetrv4/s.yml \
  --dataset-config SOVA-signs-det-dev/conf/dataset/coco.yml \
  -c conf/models/trajectory/gru/base.yml \
  -r outputs/trajectory/gru/checkpoint_last.pth \
  --source-manifest conf/dataset/source_sequences.yml \
  --output outputs/temporal --render
```

`TemporalInferencePipeline.process(image, sequence_id, frame_index, timestamp_s)`
returns detections, per-track readiness, canonical predictions and telemetry.
It retains exactly the last configured number of frames; a sequence change
resets both tracker and cache. Snapshots never expose mutable cache images.
Default point histories use normalized box bottom-centres. Missing detections
never acquire another track's history. There are no predictions before the
configured observation horizon; padding is explicit and masks missing values
with zeros rather than repeating observations. `allow_observation_gaps` is
separate from padding. Dropped ingestion frames cannot silently compress time.

ByteTrack associates high-confidence boxes first and uses lower confidence
boxes to recover unmatched active tracks. The same Kalman/IoU implementation is
used offline and live; `type: iou` selects the no-motion baseline. Thresholds,
class filtering and lifetime are configurable. CUDA tensors are detached at
the detector boundary and never stored in cache records.

The CLI writes `predictions.jsonl`, rejected frames, resolved detector/trajectory
YAML and experiment metadata. Per-frame telemetry contains cache occupancy,
evictions, active tracks, history-ready tracks, latency and drops by reason.
Optional Pillow rendering draws track IDs, observed histories and future paths.

## Verified human annotations

Export PIE/JAAD/custom labelled sources to standard COCO with the temporal
extension fields described above, then import them without a detector:

```bash
uv run tools/import_trajectory_annotations.py \
  --train annotations/train.json --val annotations/val.json \
  --image-root labelled_images --output data/trajectory/prepared/human-v1 \
  --source-name verified_custom --fps 30
```

`evaluation.mode: ground_truth_history` measures prediction from labelled
histories and rejects pseudo-labels. For end-to-end `detected_history`, set
`val_dataloader.dataset.prepared_dir` to the human artifact and
`val_dataloader.dataset.detected_history_dir` to the corresponding detector
artifact. The reader matches identities one-to-one by observed endpoint IoU
(`history_match_iou`, default 0.3), validates aligned sequence/frame/timestamps,
uses detector observations with human future targets, and counts unmatched
human windows. Matching requires only observed detections. Both evaluation
modes use the same predictors and metrics; provenance for both artifacts is
saved with the run.

## Verification and adapter extension

```bash
uv run pytest
uv run ruff check engine tests tools train.py
# Exercise optional real BiTraP without downloading model weights:
BITRAP_TEST_ROOT=/path/to/upstream uv run pytest tests/test_trajectory_adapter.py
```

Tests cover preparation validity/atomicity/provenance, clipping, split leakage,
cache immutability/concurrency, tracker association/recovery/expiry/reset,
history masks, dataset windows, labelled matching, CPU training/resume,
checkpoint schema, displacement metrics and a live mock-detector pipeline.
Additional adapters subclass `TrajectoryAdapter` and register with
`register_trajectory_adapter`; solver, dataset and cache require no changes.
