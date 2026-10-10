"""Run the swimmer pipeline over all selected video cuts and aggregate results.

Default selection is ``data/raw/*_c*.mp4``. Each video gets an isolated output
folder and one row in ``batch_summary.csv``/``batch_summary.json``.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.run_pipeline import SwimmerTrackingPipeline
from src.utils.config import PipelineConfig

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
_CUT_PATTERN = re.compile(r"_c\d+(?:\.[^.]+)?$", re.IGNORECASE)


def discover_videos(input_dir: str | Path, pattern: str = "*_c*.mp4") -> list[Path]:
    directory = Path(input_dir)
    videos = [
        path for path in directory.glob(pattern)
        if path.is_file() and path.suffix.lower() in _VIDEO_EXTENSIONS
        and _CUT_PATTERN.search(path.stem + path.suffix)
    ]
    return sorted(videos, key=lambda path: path.name.lower())


def output_name(video_path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", video_path.stem).strip("._")


def summarize_result(video_path: Path, results: dict[str, Any], elapsed_wall: float) -> dict[str, Any]:
    tracking = results.get("tracking", {})
    stroke_counts = results.get("stroke_counts", {})
    total_strokes = sum(
        float(counts.get("estimated_total_strokes", 0)) for counts in stroke_counts.values()
    )
    frames = int(results.get("frames_processed", 0))
    observations = int(tracking.get("track_observations", 0))
    return {
        "video": video_path.name,
        "path": str(video_path),
        "status": "ok",
        "frames_processed": frames,
        "video_fps": results.get("video", {}).get("fps"),
        "processing_fps": results.get("fps_processing"),
        "elapsed_seconds": results.get("elapsed_seconds", round(elapsed_wall, 2)),
        "detections_before_filters": tracking.get("detections_before_filters", 0),
        "detections_after_filters": tracking.get("detections_after_filters", 0),
        "track_observations": observations,
        "unique_track_ids": tracking.get("unique_track_ids", 0),
        "track_observations_per_frame": round(observations / frames, 4) if frames else 0.0,
        "track_coverage_percent": round(100 * observations / frames, 4) if frames else 0.0,
        "persistent_reassociations": tracking.get("persistent_reassociations", 0),
        "new_logical_identities": tracking.get("new_logical_identities", 0),
        "identity_fragmentation_ratio": round(
            tracking.get("new_logical_identities", 0)
            / max(1, tracking.get("persistent_reassociations", 0)), 4
        ),
        "tracks_with_strokes": sum(
            float(counts.get("estimated_total_strokes", 0)) > 0
            for counts in stroke_counts.values()
        ),
        "total_estimated_strokes": round(total_strokes, 2),
        "wall_seconds": round(elapsed_wall, 2),
    }


def failed_summary(video_path: Path, error: Exception, elapsed_wall: float) -> dict[str, Any]:
    return {
        "video": video_path.name,
        "path": str(video_path),
        "status": "error",
        "error": f"{type(error).__name__}: {error}",
        "wall_seconds": round(elapsed_wall, 2),
    }


def write_summary(rows: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "batch_summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fields = sorted({key for row in rows for key in row})
    with open(output_dir / "batch_summary.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_config(
    base: PipelineConfig,
    video: Path,
    output_dir: Path,
    save_video: bool,
    roi_mode: str,
) -> PipelineConfig:
    output = dataclasses.replace(
        base.output,
        dir=str(output_dir),
        save_annotated_video=save_video,
        save_metrics_json=True,
        save_stroke_counts_csv=True,
        save_mot_predictions=True,
    )
    video_config = dataclasses.replace(base.video, source=str(video))
    if roi_mode in {"full", "none"}:
        roi = dataclasses.replace(base.roi, enabled=False, points=[], lanes=[])
    else:
        roi = base.roi
    return dataclasses.replace(base, video=video_config, output=output, roi=roi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o pipeline em cortes de vídeos.")
    parser.add_argument("--config", default="configs/pipeline_real_video_natacao.yaml")
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--pattern", default="*_c*.mp4")
    parser.add_argument("--output-dir", default="outputs_batch")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--save-video", action="store_true",
                        help="Salva vídeo anotado de cada corte; desligado por padrão para economizar disco.")
    parser.add_argument("--roi-mode", choices=("full", "config", "none"), default="full",
                        help="ROI usada no lote; full/none desativam a ROI específica da config.")
    parser.add_argument("--lane-mode", choices=("config", "auto", "disabled"), default="config",
                        help="Modo de calibração das raias no lote.")
    args = parser.parse_args()

    base_config = PipelineConfig.load(args.config)
    videos = discover_videos(args.input_dir, args.pattern)
    if args.limit is not None:
        videos = videos[:args.limit]
    if not videos:
        raise SystemExit(f"Nenhum vídeo encontrado em {args.input_dir!r} com padrão {args.pattern!r}.")

    batch_dir = Path(args.output_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    print(f"Encontrados {len(videos)} vídeos cortados.")

    for index, video in enumerate(videos, start=1):
        video_output = batch_dir / output_name(video)
        result_file = video_output / "results.json"
        if args.skip_existing and result_file.exists():
            print(f"[{index}/{len(videos)}] pulando {video.name}: results.json já existe")
            try:
                result = json.loads(result_file.read_text(encoding="utf-8"))
                rows.append(summarize_result(video, result, 0.0))
            except (OSError, json.JSONDecodeError) as error:
                rows.append(failed_summary(video, error, 0.0))
            continue

        print(f"[{index}/{len(videos)}] processando {video.name}")
        started = time.perf_counter()
        try:
            config = build_config(
                base_config, video, video_output, args.save_video, args.roi_mode,
            )
            if args.lane_mode == "auto":
                config = dataclasses.replace(
                    config,
                    lanes=dataclasses.replace(config.lanes, enabled=True, mode="auto"),
                )
            elif args.lane_mode == "disabled":
                config = dataclasses.replace(
                    config,
                    lanes=dataclasses.replace(config.lanes, enabled=False, mode="disabled"),
                )
            config.validate_inputs()
            result = SwimmerTrackingPipeline(config).run(max_frames=args.max_frames)
            row = summarize_result(video, result, time.perf_counter() - started)
            print(
                f"  frames={row['frames_processed']} tracks={row['unique_track_ids']} "
                f"observacoes={row['track_observations']} fps={row['processing_fps']}"
            )
        except Exception as error:  # keep the batch going and report the failed cut
            row = failed_summary(video, error, time.perf_counter() - started)
            print(f"  ERRO: {row['error']}")
        rows.append(row)
        write_summary(rows, batch_dir)

    write_summary(rows, batch_dir)
    ok = sum(row["status"] == "ok" for row in rows)
    print(f"Concluído: {ok}/{len(rows)} vídeos OK")
    print(f"Resumo: {batch_dir / 'batch_summary.csv'}")


if __name__ == "__main__":
    main()
