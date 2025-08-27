from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pyAudioAnalysis import audioBasicIO, ShortTermFeatures
from pydub import AudioSegment
import numpy as np
import tempfile, os, json

# -------------------------
# FastAPI app
# -------------------------
app = FastAPI()

# -------------------------
# Enable CORS (for Rocket frontend or any client)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to ["http://127.0.0.1:8001"] later
    allow_credentials=True,
    allow_methods=["*"],   # GET, POST, OPTIONS, etc.
    allow_headers=["*"],
)

# -------------------------
# Health check endpoint
# -------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

# -------------------------
# OpenAI client (hardcoded key for now)
# -------------------------
client = OpenAI(api_key="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")  # <-- replace with your real key

# -------------------------
# Convert uploaded file to WAV
# -------------------------
def convert_to_wav(file_path: str) -> str:
    wav_path = file_path.rsplit(".", 1)[0] + ".wav"
    audio = AudioSegment.from_file(file_path)
    audio.export(wav_path, format="wav")
    return wav_path

# -------------------------
# Voice Analysis
# -------------------------
def analyze_voice(file_path: str):
    try:
        [Fs, x] = audioBasicIO.read_audio_file(file_path)
        x = audioBasicIO.stereo_to_mono(x)

        F, f_names = ShortTermFeatures.feature_extraction(
            x, Fs, 0.050*Fs, 0.025*Fs
        )

        pitch = np.mean(F[0])
        energy = np.mean(F[1])
        entropy = np.mean(F[2])
        flux = np.mean(F[3])

        stress_level = "High" if entropy > 5 else "Moderate"
        deception_risk = "Possible" if flux > 0.2 else "Low"
        voice_score = int(100 - (entropy * 10))

        return {
            "avg_pitch": float(pitch),
            "avg_energy": float(energy),
            "avg_entropy": float(entropy),
            "avg_flux": float(flux),
            "stress_level": stress_level,
            "deception_risk": deception_risk,
            "voice_score": max(0, min(voice_score, 100))
        }
    except Exception as e:
        return {"error": str(e)}

# -------------------------
# Analyze Endpoint
# -------------------------
@app.post("/analyze")
async def analyze(file: UploadFile):
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp:
            tmp.write(await file.read())
            audio_path = tmp.name

        # Convert to WAV
        wav_path = convert_to_wav(audio_path)

        # Step 1: Transcription
        with open(wav_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        # Step 2: GPT JSON credibility analysis
        analysis = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": "You are a credibility analysis assistant. Always return only valid JSON."},
                {"role": "user", "content": f"Analyze this transcript:\n{transcript.text}\n\nReturn strictly JSON with keys: summary, red_flags, truth_score (0-100)."}
            ],
            response_format={"type": "json_object"}  # <--- forces JSON output
        )

        text_report = json.loads(analysis.choices[0].message.content)
        text_score = int(text_report.get("truth_score", 70))

        # Step 3: Voice analysis
        voice_report = analyze_voice(wav_path)
        voice_score = voice_report.get("voice_score", 60) if isinstance(voice_report, dict) else 60

        # Step 4: Final Truth Meter
        credibility_score = int((text_score * 0.6) + (voice_score * 0.4))

        # Cleanup temp files
        os.remove(audio_path)
        os.remove(wav_path)

        return {
            "transcript": transcript.text,
            "text_report": text_report,       # parsed JSON object
            "voice_report": voice_report,
            "final_truth_meter": credibility_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
