from __future__ import annotations
import argparse, json
from pathlib import Path
from sttbench import preloader  # noqa: F401

"""Orchestrator: run one or more backends and collect metrics to JSONL/CSV.
Usage examples:
  python evaluate.py --audio sample.wav --ref ref.txt --backend vosk --vosk-model content/vosk-model-small-en-us-0.15
  python evaluate.py --audio sample.wav --ref ref.txt --backend fw --fw-model base.en --device auto
  python evaluate.py --audio sample.wav --ref ref.txt --backend w2v --w2v-model facebook/wav2vec2-base-960h
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--backend", choices=["vosk", "fw", "w2v"], required=True)
    ap.add_argument("--out", default="results.jsonl")

    # Backend-specific
    ap.add_argument("--vosk-model", default=None)
    ap.add_argument("--vosk-grammar", default=None)
    ap.add_argument("--vosk-chunk-ms", type=int, default=30)
    ap.add_argument("--vosk-normalize-rms", type=float, default=None)
    ap.add_argument("--vosk-post-corrections", default=None)
    ap.add_argument("--fw-model", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compute_type", default="default")
    ap.add_argument("--w2v-model", default="facebook/wav2vec2-base-960h")

    args = ap.parse_args()
    audio = args.audio
    ref = args.ref

    if args.backend == "vosk":
        from models.run_vosk import run_vosk
        if not args.vosk_model:
            raise SystemExit("--vosk-model is required for backend=vosk")
        res = run_vosk(
            args.vosk_model,
            audio,
            ref,
            grammar_path=args.vosk_grammar,
            chunk_ms=args.vosk_chunk_ms,
            normalize_rms=args.vosk_normalize_rms,
            post_corrections_path=args.vosk_post_corrections,
        )
    elif args.backend == "fw":
        from models.run_faster_whisper import run_fw
        if not args.fw_model:
            raise SystemExit("--fw-model is required for backend=fw")
        res = run_fw(args.fw_model, audio, device=args.device, compute_type=args.compute_type, ref_text=ref)
    elif args.backend == "w2v":
        from models.run_wav2vec2 import run_w2v
        res = run_w2v(args.w2v_model, audio, device=args.device, ref_text=ref)
    else:
        raise SystemExit("Unknown backend")

    # Append JSON line
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("a", encoding="utf-8") as f:
        f.write(json.dumps(res.to_dict()) + "\n")
    print(json.dumps(res.to_dict(), indent=2))

if __name__ == "__main__":
    main()
