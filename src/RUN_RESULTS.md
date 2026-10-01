# Local CPU run

All commands below run from `src/`. The original `SOVA-signs-det-dev` checkout
is reused read-only, and ASTRA is not included.

## Completed training and evaluation

The prepared artifact contains 180 training frames with 783 pedestrian
annotations (11 track IDs) and 180 validation frames with 3,173 annotations
(37 track IDs). No frames were rejected. After excluding untracked detections,
short tracks and discontinuous windows, there are 473 training windows and
1,644 validation windows. The test split is empty.

The GRU completed 30 epochs. Reloading `checkpoint_last.pth` reproduces its
final validation metrics exactly:

| Model | Validation ADE | Validation FDE |
| --- | ---: | ---: |
| GRU, epoch 30 | 0.00519943 | 0.00845584 |
| Constant velocity | 0.00943657 | 0.01726474 |

These are normalized-image distances against pseudo-labels, not an official
human-labelled PIE benchmark; see the interpretation section below.

Saved training/evaluation artifacts:

- [Trained GRU checkpoint](outputs/trajectory/gru/checkpoint_last.pth)
- [GRU training metrics](outputs/trajectory/gru/training_metrics.jsonl)
- [GRU validation metrics](outputs/trajectory/gru/trajectory_metrics.json)
- [Checkpoint reload evaluation](outputs/trajectory/gru-evaluation/trajectory_metrics.json)
- [Constant-velocity metrics](outputs/trajectory/constant_velocity/trajectory_metrics.json)

The configured project environment passes all 37 extension tests (including
the three optional upstream BiTraP cases), and Ruff reports no errors. The
BiTraP tests use the pre-existing checkout at
`/home/cacteyy/studia/bidirection-trajectory-predicter` read-only.

## Completed temporal inference

The final run processed and rendered all 360 selected frames: 180 per
sequence. Predictions are present on 173 frames per sequence after the
eight-frame history warm-up, producing 2,650 twelve-frame future paths in
total. There were no rejected input frames. The final bounded cache contains
32 frames. Preparation and inference detector checkpoint SHA-256 values
match exactly.

- [Validation preview](outputs/temporal-gru/validation_preview.mp4)
- [Training preview](outputs/temporal-gru/training_preview.mp4)
- [Per-frame detections and predictions](outputs/temporal-gru/predictions.jsonl)
- [Inference telemetry](outputs/temporal-gru/inference_statistics.json)
- [Inference provenance](outputs/temporal-gru/experiment_metadata.json)

Both MP4 previews contain 180 frames at 30 FPS, with a 960x540 resolution and
a six-second duration. Green boxes include low-confidence recovery candidates;
boxes labelled `None` are untracked and receive no trajectory prediction.
Yellow lines are observed histories and cyan lines are predicted paths.

An earlier rendering process lost its monitoring connection and was stopped;
its partial files remain in `outputs/temporal-gru-interrupted/`. Use only
`outputs/temporal-gru/` for the completed, verified run.

## Environment and downloaded inputs

This workspace has a Python 3.12 interpreter in `.python/` and a configured
`.venv/` with CPU PyTorch, detector dependencies and OpenCV. Use
`.venv/bin/python`, not a plain `uv run`: automatic synchronization can replace
the CPU PyTorch installation with the CUDA build in the generic lockfile.
There is no working CUDA device in this environment.

Downloaded inputs:

- `models/rtdetrv4/rtdetrv4_s.pth`: the official RT-DETRv4-S detector checkpoint.
- `third_party/rtdetrv4/`: detector source at commit
  `55fefaaed7efe2a5f72d0a18fd4e05965e35c292`.
- `data/trajectory/raw/pie/video_0001.mp4` and `video_0002.mp4`: complete PIE
  set01 videos, each 18,000 frames at 30 FPS (ten minutes).

The source URLs used were:

```text
https://drive.google.com/uc?id=1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z
https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set01/video_0001.mp4
https://data.nvision2.eecs.yorku.ca/PIE_dataset/PIE_clips/set01/video_0002.mp4
```

Only a pedestrian-containing six-second section from each video is used for
the CPU demonstration. `conf/dataset/pie_sample_segments.yml` selects original
frames `[360, 540)` for training and `[12060, 12240)` for validation. Sequence
IDs, original frame indices and timestamps are preserved. The full videos
remain available for longer runs through `pie_sample_sources.yml`.

## Commands used

The section-selection step is already complete. To select different sections,
write to a new manifest rather than overwriting the one used for provenance:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -u tools/find_pedestrian_segments.py \
  --source-manifest conf/dataset/pie_sample_sources.yml \
  --detector-config conf/rtdetrv4_pedestrian.yml \
  --output conf/dataset/pie_sample_segments_new.yml \
  --segment-frames 180 --scan-frames 20
```

Prepare the immutable detector/ByteTrack pseudo-label artifact. Once this
version exists, use a new output version for another preparation and update
the trajectory dataset YAML accordingly; do not overwrite it in place.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -u tools/prepare_trajectory_dataset.py \
  -c conf/rtdetrv4_pedestrian.yml \
  --source-manifest conf/dataset/pie_sample_segments.yml \
  --output data/trajectory/prepared/pie-sample-v1 --device cpu \
  -u dataset_preparation.image_mode=copy dataset_preparation.inference_batch_size=4
```

Train the GRU for 30 epochs and evaluate the constant-velocity baseline:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -u train.py \
  -c conf/models/trajectory/gru/base.yml \
  --dataset-config conf/dataset/pie_sample_trajectory.yml --device cpu

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -u train.py \
  -c conf/models/trajectory/constant_velocity/base.yml \
  --dataset-config conf/dataset/pie_sample_trajectory.yml --device cpu
```

Use a different `--output-dir` for additional training experiments to keep
their logs and checkpoints separate. Re-evaluate the saved GRU without training:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python train.py \
  -c conf/models/trajectory/gru/base.yml \
  --dataset-config conf/dataset/pie_sample_trajectory.yml --device cpu \
  --test-only -r outputs/trajectory/gru/checkpoint_last.pth \
  --output-dir outputs/trajectory/gru-evaluation
```

Run detection, tracking, history construction and trained GRU inference,
with green boxes, yellow observed histories and cyan predicted paths:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -u tools/run_temporal_inference.py \
  --detector-config conf/rtdetrv4_pedestrian.yml \
  -c conf/models/trajectory/gru/base.yml \
  -r outputs/trajectory/gru/checkpoint_last.pth \
  --source-manifest conf/dataset/pie_sample_segments.yml \
  --device cpu --output outputs/temporal-gru --render
```

For subsequent inference runs, choose a fresh `--output` directory to preserve
previous predictions and provenance.

Convert each rendered section into a six-second, 960x540 MP4 preview (FFmpeg
is installed on this machine). `-n` prevents overwriting existing previews:

```bash
ffmpeg -hide_banner -loglevel error -nostdin -n -framerate 30 -start_number 360 \
  -i outputs/temporal-gru/rendered/pie_set01_video0001/%06d.jpg -frames:v 180 \
  -vf scale=960:-2 -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p \
  -threads 2 -an outputs/temporal-gru/training_preview.mp4

ffmpeg -hide_banner -loglevel error -nostdin -n -framerate 30 -start_number 12060 \
  -i outputs/temporal-gru/rendered/pie_set01_video0002/%06d.jpg -frames:v 180 \
  -vf scale=960:-2 -c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p \
  -threads 2 -an outputs/temporal-gru/validation_preview.mp4
```

## Interpretation

This run uses **detector/tracker pseudo-labels**, not human PIE annotations.
ADE/FDE are dimensionless normalized-image-space distances over overlapping
trajectory windows, not published PIE pixel benchmark metrics. Observation and
prediction horizons are 8 and 12 frames (about 0.27 and 0.40 seconds at 30 FPS).
A two-section demonstration validates the workflow; it does not establish
generalization or production-quality pedestrian intent prediction.

Only RT-DETRv4 detector weights need downloading for this GRU workflow. The
GRU checkpoint is created locally by training, and constant velocity has no
learned parameters. Optional BiTraP experiments use separate configurations
and are not part of this sample training run.

## Pretrained BiTraP-NP viewer integration

The official PIE NP K=20 raw checkpoint was verified against the upstream
download (SHA-256
`858ce863fd5d34766323798db928e85297d0fb363e822f9be07cbb3de418decf`)
and imported into `models/bitrap/pie_np_k20.pth`. The upstream package and its
license are installed under `third_party/bitrap`; the original SOVA checkout
was not changed.

The new `conf/models/trajectory/bitrap/pretrained_pie.yml` uses the published
decoder architecture (`decoder_with_z: false`), 15 observed frames, 45 future
frames, 30 FPS, and 20 sampled futures. Original 1920x1080 input is required.
Existing self-trained BiTraP and GRU profiles retain their defaults.

Verification: 47 extension tests passed, including the real pretrained
checkpoint import/inference/save/load round trip and legacy D/NP/GMM tests.
Ruff passed on changed Python files. A smoke test sampled 24 of the 686 valid
15/45-frame validation histories: every prediction was finite, shaped
`[20,45,2]`; batch inference took about 1.08 seconds with four CPU threads.
This is an integration check, not an accuracy benchmark.

Launch from `src/` in a desktop session:

```bash
QT_QPA_PLATFORM=xcb uv run --no-sync python tools/view_trajectories.py \
  -c conf/models/trajectory/bitrap/pretrained_pie.yml --loop
```

Green boxes are tracks, yellow lines are observed history, cyan lines are
20 possible 1.5-second futures. Space pauses; Q/Esc exits. All source frames
are processed, so playback runs slower than real time on this CPU.
