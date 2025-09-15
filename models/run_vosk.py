from __future__ import annotations
import time, argparse
from pathlib import Path
from sttbench import preloader  # noqa: F401 (ensures preload side-effects)
from sttbench.audio_utils import load_audio_pcm16_mono_16k
from sttbench.resource_monitor import ResourceMonitor
from sttbench.metrics import compute_err_rates, EvalResult
from sttbench.io_utils import read_text_file
from vosk import Model, KaldiRecognizer
import json
import math
import numpy as np
import re


def _maybe_normalize_pcm(pcm_bytes: bytes, target_rms: float | None) -> bytes:
    """Optionally loudness-normalize PCM16 to a target RMS in [-1, 1].
    target_rms is linear (e.g., 0.05 ~ -26 dBFS). If None, return input.
    """
    if not target_rms:
        return pcm_bytes
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    # Compute RMS excluding absolute silence to avoid blowing up on quiet tails
    mask = np.abs(arr) > 1e-4
    if not np.any(mask):
        return pcm_bytes
    rms = math.sqrt(float(np.mean(arr[mask] ** 2)))
    if rms <= 1e-8:
        return pcm_bytes
    gain = target_rms / rms
    # Limit gain to avoid clipping/over-amplification
    gain = float(np.clip(gain, 0.25, 8.0))
    out = np.clip(arr * gain, -1.0, 1.0)
    return (out * 32767.0).astype(np.int16).tobytes()


def _load_grammar(grammar_path: str | Path | None) -> str | None:
    if not grammar_path:
        return None
    p = Path(grammar_path)
    if not p.exists():
        raise FileNotFoundError(f"Grammar file not found: {p}")
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return None
    # Allow either JSON array as-is, or newline-separated phrases to convert to JSON array
    if txt.lstrip().startswith("["):
        return txt
    phrases = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    return json.dumps(phrases, ensure_ascii=False)


def _load_post_corrections(path: str | Path) -> list[tuple[re.Pattern, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Corrections file not found: {p}")
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return []
    rules: list[tuple[re.Pattern, str]] = []
    # JSON dict format
    if txt.lstrip().startswith("{"):
        data = json.loads(txt)
        for src, dst in data.items():
            pat = re.compile(rf"\b{re.escape(src)}\b", flags=re.IGNORECASE)
            rules.append((pat, dst))
        return rules
    # Line-based: src=>dst or src<TAB>dst
    for line in txt.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=>" in line:
            src, dst = [x.strip() for x in line.split("=>", 1)]
        elif "\t" in line:
            src, dst = [x.strip() for x in line.split("\t", 1)]
        else:
            # Single token normalization (src -> dst identical), skip
            parts = line.split()
            if len(parts) >= 2:
                src = parts[0]
                dst = " ".join(parts[1:])
            else:
                continue
        pat = re.compile(rf"\b{re.escape(src)}\b", flags=re.IGNORECASE)
        rules.append((pat, dst))
    return rules


def _apply_post_corrections(text: str, path: str | Path) -> str:
    rules = _load_post_corrections(path)
    out = text
    for pat, repl in rules:
        out = pat.sub(repl, out)
    return out


def run_vosk(
    model_dir: str | Path,
    audio_path: str | Path,
    ref_text: str | None = None,
    grammar_path: str | Path | None = None,
    chunk_ms: int = 30,
    normalize_rms: float | None = None,
    post_corrections_path: str | Path | None = None,
) -> EvalResult:
    pcm = load_audio_pcm16_mono_16k(audio_path)
    model = Model(str(model_dir))
    grammar = _load_grammar(grammar_path)
    rec = KaldiRecognizer(model, 16000, grammar) if grammar else KaldiRecognizer(model, 16000)
    rec.SetWords(True)

    # Optional loudness normalization can help robustness for very quiet inputs
    pcm_bytes = _maybe_normalize_pcm(pcm.pcm_bytes, normalize_rms)

    step = int(16000 * 2 * (chunk_ms / 1000.0))  # chunk size in bytes
    t0 = time.time()
    with ResourceMonitor(interval=0.2) as mon:
        for i in range(0, len(pcm_bytes), step):
            rec.AcceptWaveform(pcm_bytes[i:i+step])
        j = json.loads(rec.FinalResult())
        text = j.get("text", "")
    wall = time.time() - t0
    rtf = wall / max(len(pcm_bytes) / (2 * 16000), 1e-6)

    if ref_text is not None and Path(ref_text).exists():
        ref = read_text_file(ref_text)
    else:
        ref = ref_text or ""

    if post_corrections_path:
        text = _apply_post_corrections(text, post_corrections_path)
    w, c = compute_err_rates(text, ref) if ref else (float("nan"), float("nan"))

    return EvalResult(
        text=text,
        wall_time_sec=wall,
        audio_sec=len(pcm_bytes) / (2 * 16000),
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
    ap.add_argument("audio", type=str, help="Path to WAV file")
    ap.add_argument("--model_dir", type=str, required=True, help="Path to Vosk model directory")
    ap.add_argument("--ref", type=str, default=None, help="Optional reference transcript text file for WER/CER")
    ap.add_argument("--grammar", type=str, default=None, help="Optional grammar file (JSON array or newline phrases)")
    ap.add_argument("--chunk_ms", type=int, default=30, help="Chunk size in ms (default 30)")
    ap.add_argument("--normalize_rms", type=float, default=None, help="Optional target RMS in linear scale (e.g., 0.05 ~ -26 dBFS)")
    ap.add_argument("--post_corrections", type=str, default=None, help="Optional corrections file (JSON dict or lines: src=>dst or src<TAB>dst)")
    ap.add_argument("--out", type=str, default=None, help="Write JSON result here")
    args = ap.parse_args()
    res = run_vosk(
        args.model_dir,
        args.audio,
        args.ref,
        grammar_path=args.grammar,
        chunk_ms=args.chunk_ms,
        normalize_rms=args.normalize_rms,
        post_corrections_path=args.post_corrections,
    )
    out = res.to_dict()
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        # Append a single JSON line for JSONL workflows
        with outp.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")
    else:
        print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
