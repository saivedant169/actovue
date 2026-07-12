"""The honest benchmark: score a probe across datasets, in and out of distribution.

For each dataset directory (holding acts.pt [N, hidden] and labels.pt [N] captured
from the probe's base model at the probe's layer), this computes AUROC, ECE,
Brier, and recall at a fixed false-positive rate, marks it in-distribution or
out-of-distribution, and prints a markdown table plus the headline number: the
gap between mean in-distribution and mean out-of-distribution AUROC. That gap,
reported rather than optimized, is the point of the exercise.

Run:
    python bench/eval_matrix.py --probe probes/qwen3-14b-halu-probe-v1 \
        --data-root runs/eval --id ragtruth --fpr 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from actovue.metrics import auroc, brier, ece, recall_at_fpr
from actovue.probe import Probe

COLUMNS = ["dataset", "split", "n", "auroc", "ece", "brier", "recall@fpr"]


def evaluate(probe: Probe, acts: torch.Tensor, labels: torch.Tensor, target_fpr: float) -> dict:
    scores = probe.reference(acts.to(torch.float32)).tolist()
    y = labels.tolist()
    return {
        "auroc": auroc(y, scores),
        "ece": ece(y, scores),
        "brier": brier(y, scores),
        "recall@fpr": recall_at_fpr(y, scores, target_fpr=target_fpr),
    }


def format_table(rows: list[dict]) -> str:
    header = "| " + " | ".join(COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in COLUMNS) + " |"
    lines = [header, sep]
    for r in rows:
        cells = [
            str(r["dataset"]),
            r["split"],
            str(r["n"]),
            f"{r['auroc']:.3f}",
            f"{r['ece']:.3f}",
            f"{r['brier']:.3f}",
            f"{r['recall@fpr']:.3f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def id_ood_gap(rows: list[dict]) -> float | None:
    """Mean in-distribution AUROC minus mean out-of-distribution AUROC."""
    id_aurocs = [r["auroc"] for r in rows if r["split"] == "ID"]
    ood_aurocs = [r["auroc"] for r in rows if r["split"] == "OOD"]
    if not id_aurocs or not ood_aurocs:
        return None
    return sum(id_aurocs) / len(id_aurocs) - sum(ood_aurocs) / len(ood_aurocs)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score a probe across datasets, in and out of distribution"
    )
    ap.add_argument("--probe", required=True, help="probe directory or hub id")
    ap.add_argument("--data-root", required=True, help="root with one subdir per dataset")
    ap.add_argument("--id", nargs="*", default=[], help="dataset names to mark in-distribution")
    ap.add_argument("--fpr", type=float, default=0.1, help="false-positive-rate budget for recall")
    args = ap.parse_args()

    probe = Probe.from_pretrained(args.probe)
    id_names = set(args.id)

    rows: list[dict] = []
    for sub in sorted(Path(args.data_root).iterdir()):
        if not (sub / "acts.pt").exists():
            continue
        acts = torch.load(sub / "acts.pt")
        labels = torch.load(sub / "labels.pt")
        metrics = evaluate(probe, acts, labels, args.fpr)
        rows.append(
            {
                "dataset": sub.name,
                "split": "ID" if sub.name in id_names else "OOD",
                "n": acts.shape[0],
                **metrics,
            }
        )

    if not rows:
        raise SystemExit(f"no datasets with acts.pt found under {args.data_root}")

    print(f"# actovue probe benchmark: {probe.probe_id}\n")
    print(format_table(rows))
    gap = id_ood_gap(rows)
    if gap is not None:
        print(f"\nID minus OOD mean AUROC gap: {gap:.3f} (higher means worse OOD transfer)")


if __name__ == "__main__":
    main()
