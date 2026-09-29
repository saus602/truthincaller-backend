"""
Linguistic-structure feature extraction.

This is the "right axis" of the Correlation Grid: given a transcript segment
(a sentence or a few seconds of speech), score hedging, distancing language,
and sentence-structure simplification into a 0..1 index.

Pure-Python, no NLP model download required — a curated cue list plus basic
tokenization. This is intentionally a lightweight structural signal, not a
sentiment or deception classifier: the product spec frames this as one axis
of a decision-support display, meant to be read alongside the acoustic axis
by a trained professional, never as a verdict on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

# Cue lists are intentionally broad categories of *hedging / distancing /
# qualifying* language, not "lie words" — the same phrase is often just how
# someone talks. The signal is the *rate* of these forms, and whether it
# shifts, not any single occurrence.
HEDGE_WORDS = {
    "maybe", "probably", "possibly", "perhaps", "guess", "suppose",
    "kind of", "sort of", "i think", "i guess", "i believe", "not sure",
    "i mean", "you know", "basically", "honestly", "actually", "literally",
    "to be honest", "if i'm being honest", "i don't know", "i don't recall",
    "i don't remember", "as far as i know", "i can't be sure",
}

DISTANCING_WORDS = {
    "that person", "that guy", "that individual", "those people",
    "someone", "some guy", "some person",
}

QUALIFIERS = {
    "a while", "a bit", "kind of", "sort of", "somewhat", "roughly",
    "approximately", "around", "ish", "or something", "or whatever",
    "not really", "not exactly",
}

SENTENCE_SPLIT_RE = re.compile(r"[.!?]+\s*")
WORD_RE = re.compile(r"[a-zA-Z']+")


@dataclass
class LinguisticFeatures:
    text: str
    word_count: int
    sentence_count: int
    avg_sentence_len: float     # words per sentence
    hedge_rate: float           # hedge phrases per 20 words
    distancing_rate: float      # distancing phrases per 20 words
    qualifier_rate: float       # qualifier phrases per 20 words
    type_token_ratio: float     # unique words / total words (lower => more repetitive/simplified)
    simplification_index: float = 0.0  # filled in by score(); 0..1


def _count_phrase_hits(text_lower: str, phrases: Sequence[str]) -> int:
    hits = 0
    for phrase in phrases:
        hits += len(re.findall(r"\b" + re.escape(phrase) + r"\b", text_lower))
    return hits


def extract_features(text: str) -> LinguisticFeatures:
    text = (text or "").strip()
    text_lower = text.lower()

    words = WORD_RE.findall(text_lower)
    word_count = len(words)
    sentences = [s for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    sentence_count = max(1, len(sentences))
    avg_sentence_len = word_count / sentence_count

    hedge_hits = _count_phrase_hits(text_lower, HEDGE_WORDS)
    distancing_hits = _count_phrase_hits(text_lower, DISTANCING_WORDS)
    qualifier_hits = _count_phrase_hits(text_lower, QUALIFIERS)

    norm = max(word_count, 1) / 20.0
    hedge_rate = hedge_hits / norm
    distancing_rate = distancing_hits / norm
    qualifier_rate = qualifier_hits / norm

    unique = len(set(words))
    type_token_ratio = unique / word_count if word_count else 1.0

    return LinguisticFeatures(
        text=text,
        word_count=word_count,
        sentence_count=sentence_count,
        avg_sentence_len=round(avg_sentence_len, 2),
        hedge_rate=round(hedge_rate, 3),
        distancing_rate=round(distancing_rate, 3),
        qualifier_rate=round(qualifier_rate, 3),
        type_token_ratio=round(type_token_ratio, 3),
    )


def _sigmoid(x: float, midpoint: float, scale: float) -> float:
    import numpy as np
    return float(1.0 / (1.0 + np.exp(-(x - midpoint) / max(scale, 1e-6))))


def score(features: LinguisticFeatures, baseline_avg_sentence_len: float | None = None) -> LinguisticFeatures:
    """Fill in simplification_index, 0..1.

    Higher = more hedging/qualifying/distancing language combined with shorter,
    more repetitive sentence structure relative to the speaker's own baseline
    (when a baseline is supplied) or a generic conversational baseline otherwise.
    """
    baseline_len = baseline_avg_sentence_len or 12.0
    # shorter-than-baseline sentences contribute to "simplification"
    brevity = max(0.0, (baseline_len - features.avg_sentence_len) / baseline_len)

    combined = (
        0.40 * min(features.hedge_rate / 3.0, 1.5)
        + 0.20 * min(features.qualifier_rate / 3.0, 1.5)
        + 0.15 * min(features.distancing_rate / 2.0, 1.5)
        + 0.15 * brevity
        + 0.10 * (1.0 - min(features.type_token_ratio, 1.0))
    )
    features.simplification_index = round(_sigmoid(combined, midpoint=0.55, scale=0.35), 4)
    return features


def analyze_segments(segments: Sequence[dict]) -> List[LinguisticFeatures]:
    """segments: list of {"start": float, "end": float, "text": str} (e.g. from a
    transcription API's segment output). Returns per-segment LinguisticFeatures
    with simplification_index scored against the session's own average sentence
    length, so a naturally terse speaker isn't flagged just for being terse.
    """
    all_features = [extract_features(seg.get("text", "")) for seg in segments]
    lens = [f.avg_sentence_len for f in all_features if f.word_count > 0]
    baseline = sum(lens) / len(lens) if lens else 12.0
    return [score(f, baseline_avg_sentence_len=baseline) for f in all_features]
