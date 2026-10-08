"""
Ponto de entrada único do pipeline.

Uso:
    python scripts/run_demo.py                          # usa configs/pipeline.yaml
    python scripts/run_demo.py --config configs/outro.yaml
    python scripts/run_demo.py --regenerate-synthetic    # recria o vídeo/GT sintéticos antes

Ao final, imprime um resumo e salva em outputs/:
    - annotated_output.mp4  (vídeo com overlay de tracking/pose/braçadas)
    - results.json          (métricas + contagens)
    - stroke_counts.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.run_pipeline import SwimmerTrackingPipeline
from src.utils.config import PipelineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de tracking de nadadores (TCC).")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--regenerate-synthetic", action="store_true",
                         help="Recria o vídeo e o GT sintéticos antes de rodar.")
    parser.add_argument("--max-frames", type=int, default=None,
                         help="Processa só os N primeiros frames (útil para testar rápido em máquina lenta).")
    args = parser.parse_args()

    if args.regenerate_synthetic:
        from scripts.generate_synthetic_data import generate
        generate()

    config = PipelineConfig.load(args.config)
    if config.mode == "synthetic" and not Path(config.video.source).exists():
        print("[INFO] Vídeo sintético não encontrado, gerando automaticamente...")
        from scripts.generate_synthetic_data import generate
        generate(
            output_video=config.video.source,
            output_gt=config.evaluation.gt_path,
            width=config.video.width, height=config.video.height, fps=config.video.fps,
        )

    pipeline = SwimmerTrackingPipeline(config)
    results = pipeline.run(max_frames=args.max_frames)

    print("\n=== RESUMO DA EXECUÇÃO ===")
    print(f"Frames processados: {results['frames_processed']}")
    print(f"Tempo total: {results['elapsed_seconds']}s "
          f"({results['fps_processing']} fps de processamento)")

    if "metrics" in results:
        print("\n--- Métricas (vs. ground-truth) ---")
        for k, v in results["metrics"].items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

    print("\n--- Braçadas estimadas por nadador ---")
    for tid, counts in sorted(results["stroke_counts"].items()):
        print(f"  Nadador #{tid}: {counts['estimated_total_strokes']} "
              f"(esq={counts.get('left_wrist')}, dir={counts.get('right_wrist')})")

    print(f"\nSaídas salvas em: {config.output.dir}/")


if __name__ == "__main__":
    main()
