from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydub import AudioSegment
import os
import tempfile

from app.pipeline import analyze_session, segments_from_whisper_verbose

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI(
    title="Correlation Grid API",
    description=(
        "Backend for the Correlation Grid decision-support instrument: turns call "
        "audio into an acoustic-stress series and a linguistic-simplification series, "
        "aligned side by side, for a trained professional to review. Not a "
        "truthfulness verdict — see /session/analyze's `disclaimer` field."
    ),
)

# -------------------------
# Enable CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Health check
# -------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

# -------------------------
# OpenAI client (used for transcription only — the two-axis analysis below is
# our own DSP + linguistic pipeline, not an LLM call)
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
_client: OpenAI | None = None
if OPENAI_API_KEY:
    if OPENAI_API_KEY.startswith("sk-proj-"):
        print("WARNING: using a project key (sk-proj-...); a standard key (sk-...) is expected.")
    _client = OpenAI(api_key=OPENAI_API_KEY)
else:
    print("WARNING: OPENAI_API_KEY not set — /session/analyze will fail until it is configured.")


def _get_client() -> OpenAI:
    if _client is None:
        raise HTTPException(
            status_code=503,
            detail="Transcription is unavailable: OPENAI_API_KEY is not configured on this server.",
        )
    return _client


# -------------------------
# Convert file to WAV (mono 16-bit PCM, as app.acoustic expects)
# -------------------------
def convert_to_wav(file_path: str) -> str:
    wav_path = file_path.rsplit(".", 1)[0] + ".converted.wav"
    audio = AudioSegment.from_file(file_path).set_channels(1).set_sample_width(2)
    audio.export(wav_path, format="wav")
    return wav_path


# -------------------------
# Session analysis endpoint — replaces the old single-score /analyze.
#
# The previous version of this endpoint returned a "truth_score" / "deception_risk"
# verdict. That framing was dropped along with the "Truth in Caller" name: the
# product now presents two raw signal axes (acoustic stress, linguistic
# simplification) plus where they diverge together, and leaves the read to the
# professional using it — see the product spec's "Repositioning & Naming" and
# "UX Concept: The Live Correlation Grid" sections.
# -------------------------
@app.post("/session/analyze")
async def session_analyze(file: UploadFile):
    client = _get_client()
    audio_path = None
    wav_path = None
    try:
        suffix = file.filename.split(".")[-1] if "." in file.filename else "audio"
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(await file.read())
            audio_path = tmp.name

        wav_path = convert_to_wav(audio_path)

        with open(wav_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
            )

        segments = segments_from_whisper_verbose(transcription)
        if not segments:
            # fall back to a single segment spanning the whole clip so the
            # linguistic axis still has something to score
            segments = [{"start": 0.0, "end": 0.0, "text": getattr(transcription, "text", "")}]

        result = analyze_session(wav_path, segments)
        result["raw_transcript"] = getattr(transcription, "text", "")
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        for p in (audio_path, wav_path):
            if p and os.path.exists(p):
                os.remove(p)
