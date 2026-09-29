"""
Sanity check for the acoustic/linguistic/correlate pipeline using synthetic
audio and a hand-written transcript — no network call, no OpenAI key needed.

Run: python3 tests/test_pipeline_synthetic.py
"""

import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import acoustic, linguistic, correlate  # noqa: E402


def _write_wav(path: str, samples: np.ndarray, sr: int = 16000):
    samples = np.clip(samples, -1.0, 1.0)
    ints = (samples * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(ints.tobytes())


def make_synthetic_call(sr: int = 16000, seconds: float = 12.0) -> np.ndarray:
    """Builds a toy 'call': a calm steady-pitch stretch for the first half,
    then a shakier, louder, pitch-wobbling stretch for the second half — a
    stand-in for a calm answer followed by a stressed one.
    """
    t = np.arange(int(sr * seconds)) / sr
    out = np.zeros_like(t)

    half = len(t) // 2

    # calm half: steady 140 Hz tone with light natural vibrato
    calm_t = t[:half]
    f0_calm = 140 + 2 * np.sin(2 * np.pi * 0.3 * calm_t)
    phase_calm = 2 * np.pi * np.cumsum(f0_calm) / sr
    out[:half] = 0.15 * np.sin(phase_calm)

    # stressed half: wobbling pitch (180-260 Hz), louder, more jitter/shimmer
    stressed_t = t[half:] - t[half]
    f0_stressed = 220 + 40 * np.sin(2 * np.pi * 3.0 * stressed_t) + np.random.normal(0, 8, size=stressed_t.shape)
    phase_stressed = 2 * np.pi * np.cumsum(f0_stressed) / sr
    amp_wobble = 0.28 + 0.08 * np.sin(2 * np.pi * 5.0 * stressed_t)
    out[half:] = amp_wobble * np.sin(phase_stressed)

    out += np.random.normal(0, 0.01, size=out.shape)  # noise floor
    return out


def main():
    tmp_path = "/tmp/synthetic_call.wav" if os.path.isdir("/tmp") else "synthetic_call.wav"
    audio = make_synthetic_call()
    _write_wav(tmp_path, audio)

    windows = acoustic.analyze(tmp_path)
    print(f"[acoustic] {len(windows)} windows extracted")
    first_half = [w for w in windows if w.t_end <= 6.0]
    second_half = [w for w in windows if w.t_start >= 6.0]
    avg_stress_calm = np.mean([w.stress_index for w in first_half]) if first_half else 0
    avg_stress_tense = np.mean([w.stress_index for w in second_half]) if second_half else 0
    print(f"[acoustic] avg stress_index calm half={avg_stress_calm:.3f} tense half={avg_stress_tense:.3f}")
    assert avg_stress_tense > avg_stress_calm, "expected the wobblier half to score higher stress"

    segments = [
        {"start": 0.0, "end": 3.0, "text": "I was home all evening, watching a movie with my brother."},
        {"start": 3.0, "end": 6.0, "text": "We ordered pizza around eight and just relaxed after that."},
        {"start": 6.0, "end": 9.0, "text": "I mean, I guess maybe someone stopped by, I don't really remember."},
        {"start": 9.0, "end": 12.0, "text": "It's kind of a blur, honestly, I don't know, it was a while ago."},
    ]
    ling_features = linguistic.analyze_segments(segments)
    for seg, feat in zip(segments, ling_features):
        print(f"[linguistic] [{seg['start']:.0f}-{seg['end']:.0f}s] "
              f"simplification_index={feat.simplification_index:.3f} hedge_rate={feat.hedge_rate:.2f} "
              f"'{seg['text'][:40]}...'")

    assert ling_features[-1].simplification_index > ling_features[0].simplification_index, (
        "expected the hedging segment to score higher than the plain narrative segment"
    )

    samples = correlate.build_samples(windows, segments, ling_features)
    transcript = correlate.build_transcript(segments, ling_features, windows)
    summary = correlate.summarize(samples)
    print(f"[correlate] summary={summary}")
    flagged = [t for t in transcript if t.flag]
    print(f"[correlate] {len(flagged)}/{len(transcript)} transcript lines flagged for divergence")
    for t in transcript:
        marker = "**FLAG**" if t.flag else ""
        print(f"  [{t.t_start:>5.1f}s] acoustic={t.acoustic_index:.2f} ling={t.linguistic_index:.2f} {marker} {t.text[:50]}")

    print("\nOK — pipeline runs end to end and the stress/hedging signal moves in the expected direction.")


if __name__ == "__main__":
    main()
