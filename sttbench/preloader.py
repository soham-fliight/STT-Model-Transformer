"""Jetson-safe preloader to avoid libgomp TLS errors and cap threads.
Import this at the top of entrypoints BEFORE heavy libs (sklearn/torch/transformers).
"""
import os, ctypes

# Keep threads modest on embedded boards
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Preload system OpenMP early so it gets a static TLS slot
try:
    ctypes.CDLL("libgomp.so.1", mode=getattr(ctypes, "RTLD_GLOBAL", 0x100))
except OSError:
    # Safe to continue; only impacts sklearn-like stacks. We avoid importing sklearn here.
    pass