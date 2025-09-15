#!/usr/bin/env python3
from __future__ import annotations

# --- keep this first to avoid libgomp/TLS issues on Jetson ---
from sttbench import preloader  # noqa: F401

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import Speech2TextForConditionalGeneration, Speech2TextProcessor

from sttbench.audio_utils import load_audio_pcm16_mono_16k
from sttbench.io_utils import read_text_file
from sttbench.metrics import EvalResult, compute_err_rates
from sttbench.resource_monitor import ResourceMonitor


def run_s2t(
    model_id: str,
    audio_path: str | Path,
    device: str = "auto",
    ref_text: str | None = None,
    num_beams: int = 5,
    max_new_tokens: int = 256,
    fp16: bool | None = None,
) -> EvalResult:
    """
    Run facebook/s2t-small-librispeech-asr (or any Speech2Text*) on a 16 kHz mono waveform.
    """
    pcm = load_audio_pcm16_mono_16k(audio_path)
    audio: np.ndarray = pcm.to_numpy_float32()  # [-1,1], 16 kHz

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    # Heuristic: use fp16 on CUDA if not explicitly set
    if fp16 is None:
        fp16 = (device == "cuda")

    t0 = time.time()
    with ResourceMonitor(interval=0.2) as mon:
        processor = Speech2TextProcessor.from_pretrained(model_id)
        model = Speech2TextForConditionalGeneration.from_pretrained(model_id)
        model.to(torch_device).eval()

        inputs = processor(
            audio, sampling_rate=16000, return_tensors="pt"
        )

        # Move to device
        input_features = inputs["input_features"].to(torch_device)

        gen_kwargs = {
            "num_beams": num_beams,
            "max_new_tokens": max_new_tokens,
        }

        # Mixed precision on GPU if available/requested
        autocast = torch.cuda.amp.autocast if (fp16 and device == "cuda") else torch.no_grad
        with autocast():
            generated_ids = model.generate(input_features=input_features, **gen_kwargs)

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    wall = time.time() - t0
    audio_sec = len(audio) / 16000.0
    rtf = wall / max(audio_sec, 1e-6)

    # Reference handling
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
    ap.add_argument("audio", type=str, help="Path to WAV (PCM) file")
    ap.add_argument("--model", type=str, default="facebook/s2t-small-librispeech-asr",
                    help="HF model id or local path")
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    ap.add_argument("--ref", type=str, default=None, help="Optional reference transcript .txt for WER/CER")
    ap.add_argument("--out", type=str, default=None, help="Write JSON result here")
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--no-fp16", action="store_true", help="Disable fp16 autocast on CUDA")
    args = ap.parse_args()

    res = run_s2t(
        model_id=args.model,
        audio_path=args.audio,
        device=args.device,
        ref_text=args.ref,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        fp16=False if args.no_fp16 else None,
    )
    out = res.to_dict()
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
