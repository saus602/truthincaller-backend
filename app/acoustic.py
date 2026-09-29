"""
Acoustic-stress feature extraction.

This is the "left axis" of the Correlation Grid: a windowed DSP pipeline that
turns raw call audio into a rolling 0..1 acoustic-stress index, built from
pitch (F0) instability, jitter/shimmer-style frame-to-frame variation,
energy, and zero-crossing rate.

Deliberately dependency-light: only numpy/scipy, so it runs anywhere this
service runs without pulling in a heavy DSP stack. Pitch tracking is a plain
autocorrelation method restricted to the typical speech F0 band, which is
sufficient for a relative, windowed stress index (we are not trying to do
clinical-grade voice analysis — see the product spec's framing: this is a
decision-support signal, not a verdict).
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from typing import List

import numpy as np

# Typical human speech F0 range (Hz). Widened slightly past typical male/female
# ranges so we don't clip legitimate voices.
F0_MIN_HZ = 70.0
F0_MAX_HZ = 400.0

WINDOW_S = 1.0     # analysis window presented to the caller
HOP_S = 0.5        # hop between windows (50% overlap)
SUBFRAME_S = 0.032  # sub-frame size used inside a window for jitter/shimmer
SUBFRAME_HOP_S = 0.016


@dataclass
class AcousticWindow:
    t_start: float
    t_end: float
    f0_mean: float          # Hz, 0 if unvoiced
    f0_std: float           # Hz
    jitter: float           # mean abs relative F0 change between sub-frames, 0..~1
    shimmer: float          # mean abs relative energy change between sub-frames, 0..~1
    rms_energy: float       # 0..1, relative to the loudest window seen so far
    zcr: float              # zero-crossing rate, Hz-ish
    voiced_ratio: float      # fraction of sub-frames with detectable pitch
    stress_index: float = 0.0  # filled in by normalize_stress()


def _read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV file and return (mono float32 samples in [-1,1], sample_rate).

    Upstream callers are responsible for converting other formats (mp3, m4a, ...)
    to WAV first (see main.py's convert_to_wav, which already does this via pydub).
    """
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sample_width} bytes")

    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    return data, sr


def _autocorr_f0(frame: np.ndarray, sr: int) -> float:
    """Estimate F0 (Hz) of a single frame via autocorrelation. Returns 0.0 if unvoiced."""
    frame = frame - frame.mean()
    energy = np.sqrt(np.mean(frame ** 2))
    if energy < 1e-4:
        return 0.0  # silence

    min_lag = int(sr / F0_MAX_HZ)
    max_lag = int(sr / F0_MIN_HZ)
    if max_lag >= len(frame):
        return 0.0

    windowed = frame * np.hanning(len(frame))
    corr = np.correlate(windowed, windowed, mode="full")
    mid = len(corr) // 2
    corr = corr[mid:]  # non-negative lags only
    corr = corr[: max_lag + 1]

    if len(corr) <= min_lag:
        return 0.0

    search = corr[min_lag:]
    if len(search) == 0 or corr[0] <= 0:
        return 0.0

    peak_lag_rel = int(np.argmax(search))
    peak_lag = peak_lag_rel + min_lag
    peak_val = corr[peak_lag]

    # Voicing check: a real pitch period should show strong self-similarity
    # relative to the zero-lag autocorrelation (energy).
    if peak_val / (corr[0] + 1e-9) < 0.30:
        return 0.0

    return sr / float(peak_lag)


def extract_windows(wav_path: str) -> List[AcousticWindow]:
    """Slide a WINDOW_S window (hop HOP_S) over the file, sub-framing each window
    for jitter/shimmer, and return one AcousticWindow per hop. stress_index is
    left at 0.0 here; call normalize_stress() on the returned list to fill it in
    relative to the session's own baseline.
    """
    samples, sr = _read_wav_mono(wav_path)
    duration = len(samples) / sr

    win_len = int(WINDOW_S * sr)
    hop_len = int(HOP_S * sr)
    sub_len = max(1, int(SUBFRAME_S * sr))
    sub_hop = max(1, int(SUBFRAME_HOP_S * sr))

    windows: List[AcousticWindow] = []
    t = 0.0
    while t < duration:
        start = int(t * sr)
        end = min(start + win_len, len(samples))
        chunk = samples[start:end]
        if len(chunk) < sub_len:
            break

        f0s: List[float] = []
        rmss: List[float] = []
        zcrs: List[float] = []
        i = 0
        while i + sub_len <= len(chunk):
            sub = chunk[i : i + sub_len]
            f0 = _autocorr_f0(sub, sr)
            rms = float(np.sqrt(np.mean(sub ** 2)))
            zc = float(np.mean(np.abs(np.diff(np.sign(sub))) > 0)) * sr / 2.0
            f0s.append(f0)
            rmss.append(rms)
            zcrs.append(zc)
            i += sub_hop

        voiced = [f for f in f0s if f > 0]
        voiced_ratio = len(voiced) / len(f0s) if f0s else 0.0
        f0_mean = float(np.mean(voiced)) if voiced else 0.0
        f0_std = float(np.std(voiced)) if len(voiced) > 1 else 0.0

        # jitter: relative frame-to-frame pitch change, voiced sub-frames only
        jitter = 0.0
        if len(voiced) > 1:
            diffs = np.abs(np.diff(voiced))
            denom = np.mean(voiced)
            jitter = float(np.mean(diffs) / denom) if denom > 1e-6 else 0.0

        # shimmer: relative frame-to-frame energy change, all sub-frames
        shimmer = 0.0
        if len(rmss) > 1:
            arr = np.array(rmss)
            diffs = np.abs(np.diff(arr))
            denom = np.mean(arr)
            shimmer = float(np.mean(diffs) / denom) if denom > 1e-6 else 0.0

        windows.append(
            AcousticWindow(
                t_start=round(t, 3),
                t_end=round(min(t + WINDOW_S, duration), 3),
                f0_mean=f0_mean,
                f0_std=f0_std,
                jitter=jitter,
                shimmer=shimmer,
                rms_energy=float(np.mean(rmss)) if rmss else 0.0,
                zcr=float(np.mean(zcrs)) if zcrs else 0.0,
                voiced_ratio=voiced_ratio,
            )
        )
        t += HOP_S

    return windows


def _sigmoid(x: float, midpoint: float, scale: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(x - midpoint) / max(scale, 1e-6))))


def normalize_stress(windows: List[AcousticWindow], baseline_s: float = 5.0) -> List[AcousticWindow]:
    """Fill in stress_index for each window, 0..1, using the session's own opening
    seconds as a personal baseline (everyone's voice sits at a different natural
    pitch/energy, so we score movement *relative to this speaker*, not an absolute
    scale). This mirrors the spec's point that these are relative decision-support
    signals, not a universal truth-o-meter.
    """
    if not windows:
        return windows

    baseline_windows = [w for w in windows if w.t_start < baseline_s and w.voiced_ratio > 0.2]
    if not baseline_windows:
        baseline_windows = windows[: max(1, len(windows) // 4)]

    base_jitter = np.mean([w.jitter for w in baseline_windows]) or 1e-3
    base_shimmer = np.mean([w.shimmer for w in baseline_windows]) or 1e-3
    base_f0std = np.mean([w.f0_std for w in baseline_windows]) or 1.0
    base_energy = np.mean([w.rms_energy for w in baseline_windows]) or 1e-3

    for w in windows:
        jitter_ratio = w.jitter / base_jitter
        shimmer_ratio = w.shimmer / base_shimmer
        pitch_var_ratio = w.f0_std / base_f0std if base_f0std > 0 else 1.0
        energy_ratio = w.rms_energy / base_energy if base_energy > 0 else 1.0

        # combine: pitch instability and jitter/shimmer carry most of the weight,
        # energy is a smaller supporting signal (loud speech isn't inherently
        # "stressed" the way erratic pitch/jitter tends to be).
        combined = (
            0.35 * jitter_ratio
            + 0.30 * shimmer_ratio
            + 0.25 * pitch_var_ratio
            + 0.10 * energy_ratio
        )
        w.stress_index = round(_sigmoid(combined, midpoint=1.3, scale=0.9), 4)
        if w.voiced_ratio < 0.15:
            # mostly silence/unvoiced — don't report a confident stress reading
            w.stress_index = round(w.stress_index * w.voiced_ratio / 0.15, 4)

    return windows


def analyze(wav_path: str) -> List[AcousticWindow]:
    """Convenience entry point: extract + normalize in one call."""
    return normalize_stress(extract_windows(wav_path))
