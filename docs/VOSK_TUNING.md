Vosk Tuning and Comparison

Summary of improvements and how to compare against faster-whisper.

What changed
- Grammar biasing: `--grammar` accepts a JSON array or newline-separated phrases (see `content/atc_grammar.txt`).
- Audio normalization: `--normalize_rms` applies loudness normalization on PCM (e.g., `0.05` ≈ −26 dBFS).
- Post-corrections: `--post_corrections` applies case-insensitive word/phrase fixes after decoding.
- Chunk size control: `--chunk_ms` controls feeder chunk size; larger reduces Python overhead.
- JSONL append: `--out` now appends compact JSON lines.

Run examples
- Baseline Vosk:
  - `python -m models.run_vosk content/ref_pcm.wav --model_dir content/vosk-model-small-en-us-0.15 --ref content/ref.txt --out out/vosk.jsonl`
- Improved Vosk (recommended):
  - `python -m models.run_vosk content/ref_pcm.wav --model_dir content/vosk-model-en-us-0.22 --grammar content/atc_grammar.txt --post_corrections content/post_corrections.json --normalize_rms 0.05 --chunk_ms 50 --ref content/ref.txt --out out/vosk.jsonl`
- Via orchestrator:
  - `python evaluate.py --audio content/ref_pcm.wav --ref content/ref.txt --backend vosk --vosk-model content/vosk-model-en-us-0.22 --vosk-grammar content/atc_grammar.txt --vosk-post-corrections content/post_corrections.json --vosk-normalize-rms 0.05 --vosk-chunk-ms 50 --out out/results.jsonl`

Corrections file formats
- JSON dict: `{\"defending\": \"descending\", ...}`
- Line-based: `src=>dst` or `src<TAB>dst` (case-insensitive whole-word replacements)

Generate Vosk vs Whisper comparison
- Ensure `out/vosk.jsonl` and `out/fw.jsonl` exist (each with at least one run).
- Run: `python scripts/compare_vosk_fw.py`
- Outputs:
  - `out/compare_vosk_fw.jsonl` (merged latest entries)
  - `graphs/vosk_vs_fw/*` (charts)
  - `reports/vosk_vs_whisper.md` (Markdown report for ClickUp)

Utility
- If a JSONL was previously pretty-printed, fix to compact JSONL with: `python scripts/fix_jsonl.py out/vosk.jsonl`

