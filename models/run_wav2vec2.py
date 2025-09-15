from __future__ import annotations
import time, argparse, json
from pathlib import Path
from sttbench import preloader  # noqa: F401
from sttbench.audio_utils import load_audio_pcm16_mono_16k
from sttbench.resource_monitor import ResourceMonitor
from sttbench.metrics import compute_err_rates, EvalResult
from sttbench.io_utils import read_text_file
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


def run_w2v(model_id: str, audio_path: str | Path, device: str = "auto", ref_text: str | None = None, quantize_cpu: bool = True) -> EvalResult:
    pcm = load_audio_pcm16_mono_16k(audio_path)
    audio = pcm.to_numpy_float32()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_index = 0 if device == "cuda" else -1

    t0 = time.time()
    with ResourceMonitor(interval=0.2) as mon:
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model = Wav2Vec2ForCTC.from_pretrained(model_id)
        if device == "cpu" and quantize_cpu:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        model.to(device).eval()

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        pred_ids = torch.argmax(logits, dim=-1)
        text = processor.batch_decode(pred_ids)[0]
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
    ap.add_argument("--model", type=str, default="facebook/wav2vec2-base-960h")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--ref", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no-quant", action="store_true", help="Disable CPU dynamic quantization")
    args = ap.parse_args()

    res = run_w2v(args.model, args.audio, args.device, None if args.ref is None else args.ref, quantize_cpu=not args.no_quant)
    out = res.to_dict()
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    else:
        print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()