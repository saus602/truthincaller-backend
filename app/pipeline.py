"""
Session pipeline: wav file in -> Correlation Grid JSON out.

Split from main.py so the actual analysis logic can be tested without a
running FastAPI process or a network call to OpenAI (see tests/).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from . import acoustic, linguistic, correlate


def analyze_session(wav_path: str, segments: Sequence[dict]) -> dict:
    """segments: transcript segments, e.g. [{"start": 0.0, "end": 2.4, "text": "..."}, ...]
    Typically produced by main.py from OpenAI's Whisper transcription
    (response_format="verbose_json" gives segment-level timestamps directly).
    """
    windows = acoustic.analyze(wav_path)
    ling_features = linguistic.analyze_segments(segments)
    samples = correlate.build_samples(windows, segments, ling_features)
    transcript = correlate.build_transcript(segments, ling_features, windows)
    return correlate.to_json(samples, transcript)


def segments_from_whisper_verbose(whisper_response) -> List[dict]:
    """Adapt an OpenAI verbose_json transcription response into the plain
    {"start","end","text"} dicts the pipeline expects, so main.py doesn't need
    to know about this module's internal segment shape.
    """
    segments = getattr(whisper_response, "segments", None) or []
    out = []
    for seg in segments:
        # openai SDK returns objects with attribute access; support dicts too
        get = seg.get if isinstance(seg, dict) else lambda k, d=None: getattr(seg, k, d)
        out.append({
            "start": float(get("start", 0.0)),
            "end": float(get("end", 0.0)),
            "text": str(get("text", "")),
        })
    return out
