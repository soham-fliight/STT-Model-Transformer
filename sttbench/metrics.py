from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any
from jiwer import wer, cer, Compose, RemovePunctuation, ToLowerCase, RemoveMultipleSpaces, Strip

# A conservative, traditional normalization chain
_normalize = Compose([
    ToLowerCase(),
    RemovePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
])

@dataclass
class EvalResult:
    text: str
    wall_time_sec: float
    audio_sec: float
    real_time_factor: float
    cpu_percent_avg: float
    cpu_percent_max: float
    rss_mb_avg: float
    rss_mb_max: float
    gpu_mem_mb_max: float | None
    gpu_util_percent_max: float | None
    wer: float
    cer: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_err_rates(hyp: str, ref: str) -> tuple[float, float]:
    h = _normalize(hyp)
    r = _normalize(ref)
    return wer(r, h), cer(r, h)