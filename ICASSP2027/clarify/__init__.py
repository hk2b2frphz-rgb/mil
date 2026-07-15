"""ICASSP2027 clarification-under-uncertainty experiment package.

Modules here are import-light: heavy dependencies (torch, moshi, TTS,
faster-whisper) are imported lazily inside the functions that need them, so
the benchmark-construction logic, corruption DSP, detection rules, and
metrics are unit-testable on a CPU-only machine.
"""

__version__ = "0.1.0"
