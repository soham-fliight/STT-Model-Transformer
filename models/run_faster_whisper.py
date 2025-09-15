from __future__ import annotations
import time, argparse, json
from pathlib import Path
from sttbench import preloader  # noqa: F401
from sttbench.audio_utils import load_audio_pcm16_mono_16k
from sttbench.resource_monitor import ResourceMonitor
from sttbench.metrics import compute_err_rates, EvalResult
from sttbench.io_utils import read_text_file
from faster_whisper import WhisperModel
from typing import List
import numpy as np


def run_fw(model_size_or_path: str, audio_path: str | Path, device: str = "auto", compute_type: str = "default", ref_text: str | None = None) -> EvalResult:
    pcm = load_audio_pcm16_mono_16k(audio_path)
    audio = pcm.to_numpy_float32()

    t0 = time.time()
    with ResourceMonitor(interval=0.2) as mon:
        # Pick a supported compute_type with graceful fallbacks
        candidates: List[str]
        if compute_type and compute_type != "default":
            candidates = [compute_type]
        else:
            # Prefer float16 on GPU; otherwise int8 on CPU. Fall back to float32.
            if device == "cuda":
                candidates = ["float16", "int8", "float32"]
            elif device == "cpu":
                candidates = ["int8", "float32"]
            else:  # auto or anything else
                candidates = ["float16", "int8", "float32"]

        last_err: Exception | None = None
        model = None
        for ct in candidates:
            try:
                model = WhisperModel(model_size_or_path, device=device, compute_type=ct)
                break
            except Exception as e:  # ValueError for unsupported types; keep trying
                last_err = e
                continue
        if model is None:
            raise RuntimeError(f"Failed to initialize WhisperModel with compute_type from {candidates}: {last_err}")
        segments, info = model.transcribe(audio, language="en", vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments)
    wall = time.time() - t0
    audio_sec = len(audio) / 16000.0
    rtf = wall / max(audio_sec, 1e-6)

    if ref_text is not None and Path(ref_text).exists():
        ref = read_text_file(ref_text)
    else:
        ref = ref_text or ""

    w, c = compute_err_rates(text, ref) if ref else (float("nan"), float("nan"))

    return EvalResult(
        text=text,
        wall_time_sec=wall,
        audio_sec=audio_sec,
        real_time_factor=rtf,
        cpu_percent_avg=mon.stats.cpu_avg_percent,
        cpu_percent_max=mon.stats.cpu_max_percent,
        rss_mb_avg=mon.stats.rss_avg_mb,
        rss_mb_max=mon.stats.rss_max_mb,
        gpu_mem_mb_max=mon.stats.gpu_mem_max_mb,
        gpu_util_percent_max=mon.stats.gpu_util_max_percent,
        wer=w,
        cer=c,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=str, help="Path to WAV file (PCM)")
    ap.add_argument("--model", type=str, required=True, help="faster-whisper model id or local path (e.g., 'base.en')")
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    ap.add_argument("--compute_type", type=str, default="default", help="float16|int8_float16|int8|default")
    ap.add_argument("--ref", type=str, default=None, help="Reference transcript text file for WER/CER")
    ap.add_argument("--out", type=str, default=None, help="Write JSON result here")
    args = ap.parse_args()

    res = run_fw(args.model, args.audio, args.device, args.compute_type, args.ref)
    out = res.to_dict()
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    else:
        print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
