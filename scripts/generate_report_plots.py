"""
Gera gráficos a partir de outputs/results.json, prontos para colar no capítulo
de resultados do TCC (formato .png, alta resolução).

Uso:
    python scripts/generate_report_plots.py
    python scripts/generate_report_plots.py --results outputs/results.json --out outputs/plots
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def plot_metrics_bar(metrics: dict, out_path: Path) -> None:
    keys = ["mAP@0.5", "mAP@0.5:0.95", "MOTA", "IDF1", "HOTA"]
    values = [metrics.get(k, 0.0) for k in keys]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(keys, values, color="#4477AA")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Métricas de detecção e tracking (vs. ground-truth)")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_ap_curve(metrics: dict, out_path: Path) -> None:
    thresholds, values = [], []
    for k, v in metrics.items():
        if k.startswith("AP@"):
            thresholds.append(float(k.replace("AP@", "")))
            values.append(v)
    if not thresholds:
        return
    order = sorted(range(len(thresholds)), key=lambda i: thresholds[i])
    thresholds = [thresholds[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, values, marker="o", color="#CC6677")
    ax.set_xlabel("Limiar de IoU")
    ax.set_ylabel("AP")
    ax.set_title("AP em função do limiar de IoU")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_stroke_counts(stroke_counts: dict, out_path: Path) -> None:
    ids = sorted(stroke_counts.keys(), key=lambda x: int(x))
    totals = [stroke_counts[tid]["estimated_total_strokes"] for tid in ids]
    left = [stroke_counts[tid]["left_wrist"] for tid in ids]
    right = [stroke_counts[tid]["right_wrist"] for tid in ids]

    x = range(len(ids))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - width / 2 for i in x], left, width, label="Punho esquerdo", color="#4477AA")
    ax.bar([i + width / 2 for i in x], right, width, label="Punho direito", color="#DDCC77")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"#{tid}" for tid in ids])
    ax.set_ylabel("Braçadas contadas")
    ax.set_title("Contagem de braçadas por nadador")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera gráficos de resultados para o TCC.")
    parser.add_argument("--results", default="outputs/results.json")
    parser.add_argument("--out", default="outputs/plots")
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "metrics" in results:
        plot_metrics_bar(results["metrics"], out_dir / "metrics_bar.png")
        plot_ap_curve(results["metrics"], out_dir / "ap_curve.png")
        print(f"[OK] {out_dir / 'metrics_bar.png'}")
        print(f"[OK] {out_dir / 'ap_curve.png'}")

    plot_stroke_counts(results["stroke_counts"], out_dir / "stroke_counts.png")
    print(f"[OK] {out_dir / 'stroke_counts.png'}")


if __name__ == "__main__":
    main()
