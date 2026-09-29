"""
Correlates the acoustic-stress axis with the linguistic-simplification axis
into the sample series the Correlation Grid UI plots, plus a transcript with
per-line divergence flags.

Divergence = both axes elevated at the same time. That is the whole point of
the two-axis design per the product spec: neither signal alone is meaningful
enough to act on, but a rise in vocal stress *together with* a shift toward
hedging/simplified language is the pattern worth a follow-up question from
the professional using the tool. This module never emits a truth/lie verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Sequence

from .acoustic import AcousticWindow
from .linguistic import LinguisticFeatures

DIVERGENCE_THRESHOLD = 0.55


@dataclass
class Sample:
    t: float
    stress: float
    ling: float
    divergence: bool


@dataclass
class TranscriptLine:
    t_start: float
    t_end: float
    text: str
    linguistic_index: float
    acoustic_index: float
    flag: bool


def _ling_index_at(t: float, segments: Sequence[dict], features: Sequence[LinguisticFeatures]) -> float:
    """Carry-forward the most recent segment's linguistic index at time t.
    Language lags the audio slightly (you have to finish the sentence to
    score it), so this deliberately holds the last known value rather than
    interpolating forward.
    """
    idx = 0.0
    for seg, feat in zip(segments, features):
        if seg.get("start", 0.0) <= t:
            idx = feat.simplification_index
        else:
            break
    return idx


def build_samples(windows: Sequence[AcousticWindow], segments: Sequence[dict],
                   features: Sequence[LinguisticFeatures]) -> List[Sample]:
    samples: List[Sample] = []
    for w in windows:
        ling_idx = _ling_index_at(w.t_start, segments, features)
        divergence = w.stress_index >= DIVERGENCE_THRESHOLD and ling_idx >= DIVERGENCE_THRESHOLD
        samples.append(Sample(t=w.t_start, stress=w.stress_index, ling=ling_idx, divergence=divergence))
    return samples


def build_transcript(segments: Sequence[dict], features: Sequence[LinguisticFeatures],
                      windows: Sequence[AcousticWindow]) -> List[TranscriptLine]:
    lines: List[TranscriptLine] = []
    for seg, feat in zip(segments, features):
        t_start = seg.get("start", 0.0)
        t_end = seg.get("end", t_start)
        nearby = [w.stress_index for w in windows if t_start - 0.5 <= w.t_start <= t_end + 0.5]
        acoustic_idx = max(nearby) if nearby else 0.0
        flag = acoustic_idx >= DIVERGENCE_THRESHOLD and feat.simplification_index >= DIVERGENCE_THRESHOLD
        lines.append(
            TranscriptLine(
                t_start=round(t_start, 2),
                t_end=round(t_end, 2),
                text=seg.get("text", "").strip(),
                linguistic_index=feat.simplification_index,
                acoustic_index=acoustic_idx,
                flag=flag,
            )
        )
    return lines


def summarize(samples: Sequence[Sample]) -> dict:
    if not samples:
        return {"sample_count": 0, "divergence_windows": 0, "divergence_ratio": 0.0,
                "peak_stress": 0.0, "peak_linguistic": 0.0}
    div_count = sum(1 for s in samples if s.divergence)
    return {
        "sample_count": len(samples),
        "divergence_windows": div_count,
        "divergence_ratio": round(div_count / len(samples), 3),
        "peak_stress": round(max(s.stress for s in samples), 3),
        "peak_linguistic": round(max(s.ling for s in samples), 3),
    }


def to_json(samples: Sequence[Sample], transcript: Sequence[TranscriptLine]) -> dict:
    return {
        "samples": [asdict(s) for s in samples],
        "transcript": [asdict(t) for t in transcript],
        "summary": summarize(samples),
        "disclaimer": (
            "This output surfaces relative shifts in acoustic stress and linguistic "
            "structure for a trained professional to weigh alongside other context. "
            "It is a decision-support signal, not a determination of truthfulness."
        ),
    }
