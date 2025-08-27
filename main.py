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
# OpenAI client with key check
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY environment variable not set!")

# Warn if it's the wrong type of key
if OPENAI_API_KEY.startswith("sk-proj-"):
    print("⚠️ WARNING: You are using a project key (sk-proj-...). Use a standard key (sk-...) instead!")
elif OPENAI_API_KEY.startswith("sk-"):
    print("✅ Correct OpenAI API key type detected.")

client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# Convert file to WAV
# -------------------------
def convert_to_wav(file_path: str) -> str:
    wav_path = file_path.rsplit(".", 1)[0] + ".wav"
    audio = AudioSegment.from_file(file_path)
    audio.export(wav_path, format="wav")
    return wav_path

# -------------------------
# Voice analysis
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
# Analyze endpoint
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
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a credibility analysis assistant. Always return only valid JSON."},
                {"role": "user", "content": f"Analyze this transcript:\n{transcript.text}\n\nReturn strictly JSON with keys: summary, red_flags, truth_score (0-100)."}
            ],
            response_format={"type": "json_object"}
        )

        text_report = json.loads(analysis.choices[0].message.content)
        text_score = int(text_report.get("truth_score", 70))

        # Step 3: Voice analysis
        voice_report = analyze_voice(wav_path)
        voice_score = voice_report.get("voice_score", 60) if isinstance(voice_report, dict) else 60

        # Step 4: Final Truth Meter
        credibility_score = int((text_score * 0.6) + (voice_score * 0.4))

        # Cleanup
        os.remove(audio_path)
        os.remove(wav_path)

        return {
            "transcript": transcript.text,
            "text_report": text_report,
            "voice_report": voice_report,
            "final_truth_meter": credibility_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
