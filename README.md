# STT Benchmark Kit (Jetson-safe)

This kit lets you run and compare **Vosk**, **faster-whisper (CT2)**, and **Wav2Vec2-CTC** on the same WAV input,
with per-process resource metering (CPU%, RSS, optional GPU util/mem via NVML) and standard WER/CER scoring.

## 1) Install

```bash
git clone <this>
cd stt_bench
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

*Jetson note*: if you ever hit scikit-learn/libgomp TLS issues, the entrypoints import `sttbench.preloader` first, which preloads `libgomp.so.1` and caps thread counts.

## 2) Inputs
- **Audio**: WAV (any PCM), converted in-memory to **16 kHz mono PCM16** using stdlib only (no ffmpeg dependency).
- **Reference**: optional plain text file for scoring. If omitted, WER/CER are set to NaN.

## 3) Run examples

### Vosk
```bash
python -m models.run_vosk content/ref_pcm.wav \
  --model_dir content/vosk-model-small-en-us-0.15 \
  --ref content/ref.txt \
  --out out/results.jsonl

# Optional tuning knobs:
#  - Provide a grammar (JSON array or newline phrases) to bias decoding
#  - Normalize audio RMS to improve robustness on quiet inputs
#  - Adjust chunk size
python -m models.run_vosk content/ref_pcm.wav \
  --model_dir content/vosk-model-en-us-0.22 \
  --grammar content/atc_grammar.txt \
  --normalize_rms 0.05 \
  --chunk_ms 50 \
  --ref content/ref.txt \
  --out out/results.jsonl

```

#### Vosk Parameters
- `audio`: WAV path (positional). PCM WAV recommended; loader converts to 16 kHz mono PCM16 in‑memory.
- `--model_dir`: Path to unzipped Vosk model directory (required).
- `--ref`: Optional reference transcript text file for WER/CER.
- `--out`: Output file; appends one compact JSON line per run (JSONL).
- `--grammar`: Optional grammar file to bias decoding. Accepts JSON array or newline‑separated phrases (see `content/atc_grammar.txt`).
- `--chunk_ms`: Audio chunk size in milliseconds for feeding the recognizer. Default: `30`.
- `--normalize_rms`: Target RMS in linear scale (e.g., `0.05` ≈ −26 dBFS) to normalize quiet audio before decoding.
- `--post_corrections`: Optional corrections file to fix common confusions after decoding. Supports JSON dict or `src=>dst` lines.


### faster-whisper (CT2 backend)
```bash
python evaluate.py --audio content/ref_pcm.wav --ref content/ref.txt --backend fw --fw-model base.en --device auto --compute_type default --out out/results.jsonl

```

### Wav2Vec2-CTC (Transformers)
```bash
python evaluate.py --audio content/ref_pcm.wav --ref content/ref.txt --backend w2v --w2v-model facebook/wav2vec2-base-960h --out out/results.jsonl
```

### LibriSpeech S2T
```bash
python -m models.run_librispeech content/ref_pcm.wav --model facebook/s2t-small-librispeech-asr --device auto --ref content/ref.txt --out out/results.jsonl
```

### Unified runner & JSONL log
```bash
python -m models.run_vosk content/ref_pcm.wav --model_dir content/vosk-model-small-en-us-0.15 --ref content/ref.txt --out out/results.jsonl
python evaluate.py --audio content/ref_pcm.wav --ref content/ref.txt --backend fw --fw-model base.en --device auto --compute_type default --out out/results.jsonl
python evaluate.py --audio content/ref.wav --ref content/ref.txt --backend w2v  --w2v-model facebook/wav2vec2-base-960h --out out/results.jsonl
python -m models.run_librispeech content/ref_pcm.wav --model facebook/s2t-small-librispeech-asr --device auto --ref content/ref.txt --out out/results.jsonl


```

#### Vosk via Orchestrator (evaluate.py)
- `--backend vosk` Use the Vosk backend.
- `--vosk-model` Path to Vosk model directory (required for Vosk runs).
- `--vosk-grammar` Optional grammar file (JSON array or newline phrases).
- `--vosk-chunk-ms` Audio chunk size in ms (default `30`).
- `--vosk-normalize-rms` Target RMS (linear) for audio normalization.
- `--vosk-post-corrections` Corrections file (same formats as `--post_corrections`).

## 3.1) Vosk quality tips
- Prefer a larger acoustic model for English when possible (e.g., `vosk-model-en-us-0.22`) over `-small` for markedly better WER.
- Use `--grammar` with a domain phrase list to constrain/guide decoding (newline file or JSON array). This can drastically reduce substitutions in narrow domains (e.g., ATC).
- If inputs are very quiet, enable `--normalize_rms 0.05` (roughly -26 dBFS target) to improve recognition stability.
- Increase `--chunk_ms` (e.g., 50–80 ms) to reduce Python overhead on long files; WER is typically unaffected.

## 4) Output fields
Each run prints and/or appends a JSON object like:
```json
{
  "text": "hello world",
  "wall_time_sec": 0.92,
  "audio_sec": 1.00,
  "real_time_factor": 0.92,
  "cpu_percent_avg": 84.1,
  "cpu_percent_max": 186.4,
  "rss_mb_avg": 412.7,
  "rss_mb_max": 635.9,
  "gpu_mem_mb_max": 512.0,
  "gpu_util_percent_max": 47.0,
  "wer": 0.12,
  "cer": 0.08
}
```

## 5) Notes
- GPU stats require `pynvml` and an NVIDIA driver; otherwise GPU fields are `null`.
- For repeatable numbers, run with a warm cache or discard the first run.
- On CPU, dynamic quantization for Wav2Vec2 is enabled by default (disable with `--no-quant`).
- The resource monitor samples **only the current process**—exactly what you asked: no system-wide mixing.
