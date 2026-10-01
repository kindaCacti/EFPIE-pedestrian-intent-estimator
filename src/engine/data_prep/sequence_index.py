"""Validate sequence-level splits and read ordered images/videos lazily."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

from PIL import Image
import yaml

from .manifest import fingerprint


def natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


@dataclass(frozen=True)
class SourceSequence:
    sequence_id: str
    split: str
    source: Path
    frames: tuple[Path, ...]
    fps: float
    timestamps: tuple[float, ...] | None
    video: bool
    start_frame: int = 0
    end_frame: int | None = None

    def provenance(self):
        sources = [self.source] if self.video else list(self.frames)
        return {"sequence_id": self.sequence_id, "split": self.split,
                "source": str(self.source), "fps": self.fps,
                "timestamps_s": self.timestamps,
                "start_frame": self.start_frame, "end_frame": self.end_frame,
                "files": [fingerprint(p) for p in sources]}


def load_source_manifest(path):
    path = Path(path).resolve()
    with path.open() as stream:
        data = yaml.safe_load(stream)
    entries = data.get("sequences", []) if isinstance(data, dict) else data
    if not entries:
        raise ValueError("Source manifest must contain a nonempty sequences list")
    ids, sources, seen_frames, sequences = set(), set(), set(), []
    for entry in entries:
        sid = str(entry["sequence_id"])
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sid) or sid in ids:
            raise ValueError(f"Invalid or duplicate sequence_id {sid!r}; use a safe unique directory name")
        if entry.get("split") not in {"train", "val", "test"}:
            raise ValueError(f"Sequence {sid}: split must be train, val or test")
        raw = entry.get("video") or entry.get("frame_dir") or entry.get("source")
        if not raw:
            raise ValueError(f"Sequence {sid}: specify video or frame_dir")
        source = (path.parent / raw).resolve(strict=True)
        if source in sources:
            raise ValueError(f"Source {source} appears in multiple sequences/splits")
        video = source.is_file()
        frames = ()
        fps = float(entry.get("fps", 0))
        if video:
            try:
                import cv2
            except ImportError as exc:
                raise ImportError("Video decoding requires uv sync --extra temporal-video") from exc
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise ValueError(f"Cannot open video {source}")
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = fps or float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
        else:
            if entry.get("frames"):
                frames = tuple((source / p).resolve(strict=True) for p in entry["frames"])
            else:
                frames = tuple(sorted((p.resolve() for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}), key=natural_key))
            if not frames or len(set(frames)) != len(frames):
                raise ValueError(f"Sequence {sid}: no frames or duplicate frame paths")
            if seen_frames.intersection(frames):
                raise ValueError("Source frames are shared across sequences/splits")
            seen_frames.update(frames)
            count = len(frames)
        timestamps = entry.get("timestamps_s", entry.get("timestamps"))
        if isinstance(timestamps, str):
            timestamp_path = (path.parent / timestamps).resolve(strict=True)
            with timestamp_path.open() as stream:
                timestamps = json.load(stream) if timestamp_path.suffix == ".json" else [float(line) for line in stream if line.strip()]
        if timestamps is not None:
            timestamps = tuple(map(float, timestamps))
            if len(timestamps) != count or any(not math.isfinite(t) for t in timestamps):
                raise ValueError(f"Sequence {sid}: timestamps must align with all source frames and be finite")
            if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
                raise ValueError(f"Sequence {sid}: nonmonotonic timestamps")
            if not fps and len(timestamps) > 1:
                fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"Sequence {sid}: a positive fps or timestamp source is required")
        start, end = entry.get("start_frame", 0), entry.get("end_frame", count)
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= count:
            raise ValueError(f"Sequence {sid}: require integer 0 <= start_frame < end_frame <= {count}")
        ids.add(sid)
        sources.add(source)
        sequences.append(SourceSequence(sid, entry["split"], source, frames, fps, timestamps, video, start, end))
    return sequences


def iter_frames(sequence, stride=1):
    """Yield (source index, seconds, RGB PIL image or None, source path, error)."""
    if stride <= 0:
        raise ValueError("frame_stride must be positive")
    if sequence.video:
        import cv2
        cap = cv2.VideoCapture(str(sequence.source))
        index = sequence.start_frame
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if sequence.end_frame is not None:
                count = min(count, sequence.end_frame)
            if sequence.start_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, sequence.start_frame)
            while index < count:
                ok, frame = cap.read()
                if (index - sequence.start_frame) % stride == 0:
                    stamp = sequence.timestamps[index] if sequence.timestamps else index / sequence.fps
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ok else None
                    yield index, stamp, image, None, None if ok else "unreadable video frame"
                index += 1
        finally:
            cap.release()
    else:
        end = min(len(sequence.frames), sequence.end_frame) if sequence.end_frame is not None else len(sequence.frames)
        for index in range(sequence.start_frame, end, stride):
            source = sequence.frames[index]
            stamp = sequence.timestamps[index] if sequence.timestamps else index / sequence.fps
            try:
                with Image.open(source) as raw:
                    image = raw.convert("RGB")
                yield index, stamp, image, source, None
            except (OSError, ValueError) as exc:
                yield index, stamp, None, source, str(exc)
