#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot STT benchmark results from a JSONL produced by evaluate.py.

Creates:
  graphs/asr_wer_latest.png
  graphs/asr_cer_latest.png
  graphs/asr_rtf_latest.png
  graphs/asr_cpu_avg_latest.png
  graphs/asr_cpu_max_latest.png
  graphs/asr_rss_avg_latest.png
  graphs/asr_rss_max_latest.png
  graphs/asr_gpu_util_latest.png   (if GPU data available)
  graphs/asr_gpu_mem_latest.png    (if GPU data available)
  graphs/asr_speed_accuracy_latest.png  (WER% vs RTF scatter)
  graphs/asr_summary_latest.json
Also writes/updates:
  content/asr_eval_summary.csv

Usage:
  python plot_results.py --input out/results.jsonl --outdir graphs --labels "Vosk,FW base.en,Wav2Vec2,S2T"
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _nan_to_none(x):
    try:
        return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)
    except Exception:
        return None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # tolerate trailing commas/newlines
                continue
    return rows


def choose_labels(rows: List[Dict[str, Any]], cli_labels: List[str] = None) -> List[str]:
    labels = []
    for i, r in enumerate(rows):
        label = (
            r.get("label")
            or r.get("backend")
            or r.get("model")
            or r.get("fw_model")
            or r.get("w2v_model")
        )
        if not label:
            label = f"run_{i+1}"
        labels.append(str(label))
    if cli_labels:
        # override sequentially if user provided a comma-separated list
        for i, name in enumerate(cli_labels):
            if i < len(labels):
                labels[i] = name.strip()
    return labels


def ensure_outdir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


def bar_chart(values: List[float], labels: List[str], title: str, ylabel: str, outfile: Path, yfmt=None):
    # Remove None values (and corresponding labels)
    filt = [(v, l) for v, l in zip(values, labels) if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not filt:
        return
    vals, labs = zip(*filt)
    x = np.arange(len(vals))

    plt.figure(figsize=(10, 5))
    bars = plt.bar(x, vals)
    plt.xticks(x, labs, rotation=20, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    # annotate
    for rect, v in zip(bars, vals):
        try:
            txt = f"{yfmt(v) if yfmt else v:.3f}"
        except Exception:
            txt = f"{v:.3f}" if isinstance(v, float) else str(v)
        plt.text(rect.get_x() + rect.get_width()/2.0, rect.get_height(), txt,
                 ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()


def scatter_speed_accuracy(wer_pct: List[float], rtf: List[float], labels: List[str], outfile: Path):
    pts = [(w, r, lbl) for w, r, lbl in zip(wer_pct, rtf, labels) if w is not None and r is not None and not math.isnan(w) and not math.isnan(r)]
    if not pts:
        return
    wlist, rlist, labs = zip(*pts)

    plt.figure(figsize=(8, 6))
    plt.scatter(wlist, rlist)
    for w, r, lbl in pts:
        plt.annotate(lbl, (w, r), textcoords="offset points", xytext=(5, 5), fontsize=9)
    plt.xlabel("WER (%)")
    plt.ylabel("Real-Time Factor (RTF)")
    plt.title("Speed vs Accuracy (lower-left = better)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=150)
    plt.close()


def pareto_front(wer_pct: List[float], rtf: List[float], labels: List[str]) -> List[str]:
    """Return labels on the Pareto front (minimize both WER and RTF)."""
    pts = [(wer_pct[i], rtf[i], labels[i]) for i in range(len(labels))
           if wer_pct[i] is not None and rtf[i] is not None and not math.isnan(wer_pct[i]) and not math.isnan(rtf[i])]
    front = []
    for i, (wi, ri, li) in enumerate(pts):
        dominated = False
        for j, (wj, rj, lj) in enumerate(pts):
            if j == i:
                continue
            if (wj <= wi and rj <= ri) and (wj < wi or rj < ri):
                dominated = True
                break
        if not dominated:
            front.append(li)
    return front


def write_csv(rows: List[Dict[str, Any]], csv_path: Path):
    keys = [
        "label", "text",
        "wall_time_sec", "audio_sec", "real_time_factor",
        "cpu_percent_avg", "cpu_percent_max",
        "rss_mb_avg", "rss_mb_max",
        "gpu_mem_mb_max", "gpu_util_percent_max",
        "wer", "cer"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            r = r.copy()
            # ensure label exists
            if "label" not in r:
                r["label"] = ""
            w.writerow({k: r.get(k, "") for k in keys})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out/results.jsonl", help="Path to results JSONL")
    ap.add_argument("--outdir", default="graphs", help="Directory to write charts")
    ap.add_argument("--labels", default=None, help="Comma-separated labels to use in order (optional)")
    args = ap.parse_args()

    inpath = Path(args.input)
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    rows = read_jsonl(inpath)
    if not rows:
        raise SystemExit(f"No rows found in {inpath}")

    cli_labels = [s.strip() for s in args.labels.split(",")] if args.labels else None
    labels = choose_labels(rows, cli_labels)

    # Normalize metrics
    wer = [_nan_to_none(r.get("wer")) for r in rows]
    cer = [_nan_to_none(r.get("cer")) for r in rows]
    rtf = [_nan_to_none(r.get("real_time_factor")) for r in rows]
    cpu_avg = [_nan_to_none(r.get("cpu_percent_avg")) for r in rows]
    cpu_max = [_nan_to_none(r.get("cpu_percent_max")) for r in rows]
    rss_avg = [_nan_to_none(r.get("rss_mb_avg")) for r in rows]
    rss_max = [_nan_to_none(r.get("rss_mb_max")) for r in rows]
    gpu_util = [_nan_to_none(r.get("gpu_util_percent_max")) for r in rows]
    gpu_mem = [_nan_to_none(r.get("gpu_mem_mb_max")) for r in rows]

    # Charts
    bar_chart([w * 100 if (w is not None and not math.isnan(w)) else None for w in wer], labels,
              "Word Error Rate", "WER (%)", outdir / "asr_wer_latest.png",
              yfmt=lambda v: f"{v:.1f}%")
    bar_chart([c * 100 if (c is not None and not math.isnan(c)) else None for c in cer], labels,
              "Character Error Rate", "CER (%)", outdir / "asr_cer_latest.png",
              yfmt=lambda v: f"{v:.1f}%")
    bar_chart(rtf, labels, "Real-Time Factor (RTF)", "RTF (× real time)", outdir / "asr_rtf_latest.png",
              yfmt=lambda v: f"{v:.2f}")

    bar_chart(cpu_avg, labels, "CPU Usage (avg)", "CPU %", outdir / "asr_cpu_avg_latest.png",
              yfmt=lambda v: f"{v:.1f}%")
    bar_chart(cpu_max, labels, "CPU Usage (max)", "CPU %", outdir / "asr_cpu_max_latest.png",
              yfmt=lambda v: f"{v:.1f}%")

    bar_chart(rss_avg, labels, "Memory RSS (avg)", "MB", outdir / "asr_rss_avg_latest.png",
              yfmt=lambda v: f"{v:.0f}")
    bar_chart(rss_max, labels, "Memory RSS (max)", "MB", outdir / "asr_rss_max_latest.png",
              yfmt=lambda v: f"{v:.0f}")

    if any(g is not None for g in gpu_util):
        bar_chart(gpu_util, labels, "GPU Utilization (max)", "%", outdir / "asr_gpu_util_latest.png",
                  yfmt=lambda v: f"{v:.0f}%")
    if any(m is not None for m in gpu_mem):
        bar_chart(gpu_mem, labels, "GPU Memory (max)", "MB", outdir / "asr_gpu_mem_latest.png",
                  yfmt=lambda v: f"{v:.0f}")

    scatter_speed_accuracy(
        [w * 100 if (w is not None and not math.isnan(w)) else None for w in wer],
        rtf, labels, outdir / "asr_speed_accuracy_latest.png"
    )

    # Summary
    def idx_min(vals: List[float]) -> int:
        filt = [(i, v) for i, v in enumerate(vals) if v is not None and not math.isnan(v)]
        return min(filt, key=lambda t: t[1])[0] if filt else -1

    best_wer_i = idx_min([w if w is None else (w * 100) for w in wer])  # compare in percent or raw same ordering
    best_rtf_i = idx_min(rtf)

    summary = {
        "labels": labels,
        "best_wer": {
            "label": labels[best_wer_i] if best_wer_i >= 0 else None,
            "wer_percent": (wer[best_wer_i] * 100) if best_wer_i >= 0 else None
        },
        "best_rtf": {
            "label": labels[best_rtf_i] if best_rtf_i >= 0 else None,
            "rtf": rtf[best_rtf_i] if best_rtf_i >= 0 else None
        },
        "pareto_front": pareto_front(
            [w * 100 if (w is not None and not math.isnan(w)) else None for w in wer],
            rtf, labels
        ),
        "count": len(labels),
    }

    (outdir / "asr_summary_latest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Also write/update a CSV in content/ for convenience
    # attach labels into rows before writing
    labeled_rows = []
    for lab, r in zip(labels, rows):
        rr = r.copy()
        rr["label"] = lab
        labeled_rows.append(rr)
    write_csv(labeled_rows, Path("content/asr_eval_summary.csv"))

    print("Wrote charts to", str(outdir))
    print("Summary:", json.dumps(summary, indent=2))
    print("CSV:", "content/asr_eval_summary.csv")


if __name__ == "__main__":
    main()
