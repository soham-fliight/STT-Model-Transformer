from __future__ import annotations
import json
from pathlib import Path
import subprocess


def read_last_json(path: Path) -> dict:
    last = None
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    if last is None:
        raise RuntimeError(f"No valid JSON lines in {path}")
    return last


def main():
    base = Path(".")
    vosk_in = base / "out/vosk.jsonl"
    fw_in = base / "out/fw.jsonl"
    out_merge = base / "out/compare_vosk_fw.jsonl"
    out_graphs = base / "graphs/vosk_vs_fw"
    report = base / "reports/vosk_vs_whisper.md"

    vosk_row = read_last_json(vosk_in)
    fw_row = read_last_json(fw_in)

    # Attach labels for plotting/reporting
    vosk_row = {**vosk_row, "label": "Vosk"}
    fw_row = {**fw_row, "label": "Whisper (FW)"}

    out_merge.parent.mkdir(parents=True, exist_ok=True)
    with out_merge.open("w", encoding="utf-8") as f:
        f.write(json.dumps(vosk_row) + "\n")
        f.write(json.dumps(fw_row) + "\n")

    # Generate charts using existing plotter
    out_graphs.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "python", "plot_results.py",
        "--input", str(out_merge),
        "--outdir", str(out_graphs),
        "--labels", "Vosk,Whisper (FW)",
    ], check=True)

    # Build a concise Markdown report for ClickUp
    def fmt_row(r: dict) -> str:
        return (
            f"- Text: {r.get('label')}\n"
            f"- WER: {r.get('wer'):.3f}  CER: {r.get('cer'):.3f}\n"
            f"- RTF: {r.get('real_time_factor'):.3f}  Wall: {r.get('wall_time_sec'):.2f}s  Audio: {r.get('audio_sec'):.2f}s\n"
            f"- CPU avg/max: {r.get('cpu_percent_avg'):.1f}%/{r.get('cpu_percent_max'):.1f}%  RSS avg/max: {r.get('rss_mb_avg'):.0f}/{r.get('rss_mb_max'):.0f} MB\n"
        )

    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", encoding="utf-8") as f:
        f.write("# Vosk vs Whisper (faster-whisper)\n\n")
        f.write("This report compares the latest Vosk and faster-whisper runs.\n\n")
        f.write("## Metrics\n\n")
        f.write("### Vosk\n" + fmt_row(vosk_row) + "\n")
        f.write("### Whisper (FW)\n" + fmt_row(fw_row) + "\n")
        f.write("## Charts\n\n")
        # List the key generated charts
        charts = [
            "asr_wer_latest.png",
            "asr_cer_latest.png",
            "asr_rtf_latest.png",
            "asr_speed_accuracy_latest.png",
            "asr_cpu_avg_latest.png",
            "asr_rss_avg_latest.png",
        ]
        for ch in charts:
            f.write(f"![{ch}]({(out_graphs / ch).as_posix()})\n\n")

    print(f"Wrote merged JSONL: {out_merge}")
    print(f"Wrote graphs to: {out_graphs}")
    print(f"Wrote report: {report}")


if __name__ == "__main__":
    main()

