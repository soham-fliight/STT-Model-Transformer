"""Audio helpers with no ffmpeg dependency.
Ensures 16 kHz mono signed 16-bit little-endian PCM in-memory bytes.
"""
from __future__ import annotations
from pathlib import Path
import wave, audioop
import numpy as np

class PCM16Mono:
    def __init__(self, pcm_bytes: bytes, sample_rate: int = 16000):
        self.pcm_bytes = pcm_bytes
        self.sample_rate = sample_rate

    def to_numpy_float32(self) -> np.ndarray:
        # Convert 16-bit PCM to float32 in [-1, 1]
        arr = np.frombuffer(self.pcm_bytes, dtype=np.int16).astype(np.float32)
        return arr / 32768.0


def load_audio_pcm16_mono_16k(path: str | Path) -> PCM16Mono:
    """Load any WAV and convert to PCM16 mono 16k using stdlib only.
    Supports PCM WAV inputs (most STT test refs). For other formats, convert to WAV first.
    """
    path = Path(path)
    try:
        with wave.open(str(path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            fr = wf.getframerate()
            frames = wf.getnframes()
            pcm = wf.readframes(frames)
    except wave.Error as e:
        # Provide an actionable hint for non-PCM WAV inputs (e.g., ADPCM)
        suggested = path.with_suffix(".pcm16_16k_mono.wav")
        raise RuntimeError(
            f"Unsupported WAV encoding for '{path}': {e}. "
            f"Please convert to PCM 16-bit mono 16k. Example:\n"
            f"  ffmpeg -y -i '{path}' -ac 1 -ar 16000 -c:a pcm_s16le '{suggested}'"
        )

    # Ensure 16-bit
    if sample_width != 2:
        pcm = audioop.lin2lin(pcm, sample_width, 2)

    # Ensure mono
    if n_channels != 1:
        pcm = audioop.tomono(pcm, 2, 0.5, 0.5)

    # Resample to 16k if needed
    if fr != 16000:
        pcm, _ = audioop.ratecv(pcm, 2, 1, fr, 16000, None)
        fr = 16000

    return PCM16Mono(pcm_bytes=pcm, sample_rate=fr)
