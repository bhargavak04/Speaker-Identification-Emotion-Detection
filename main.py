import os
import torch
import numpy as np
import torchaudio
import librosa
import webrtcvad
from pydub import AudioSegment
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn

# ======================
# CLOUD CONFIGURATION
# ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = "./model_cache"  # For Render persistent storage
os.makedirs(CACHE_DIR, exist_ok=True)

# ======================
# DATABASE SETUP
# ======================
DATABASE_URL = "sqlite:///./speaker_database.db"
Base = declarative_base()

class SpeakerModel(Base):
    __tablename__ = 'speakers'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    embedding = Column(LargeBinary)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)

# ======================
# MODEL INITIALIZATION
# ======================
print("Initializing models (will auto-download if missing)...")

# 1. Speaker Recognition Model
from speechbrain.pretrained import SpeakerRecognition
spkrec_model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=os.path.join(CACHE_DIR, "ecapa_model"),
    run_opts={"device": DEVICE}
).to(DEVICE)

# 2. Whisper Speech-to-Text
from transformers import WhisperForConditionalGeneration, WhisperProcessor
whisper_processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small",
    cache_dir=CACHE_DIR
)
whisper_model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small",
    cache_dir=CACHE_DIR
).to(DEVICE)

# 3. Emotion Detection
from transformers import pipeline
emotion_classifier = pipeline(
    "audio-classification",
    model="superb/wav2vec2-base-superb-er",
    device=0 if DEVICE.type == "cuda" else -1,
    cache_dir=CACHE_DIR
)

# ======================
# CORE FUNCTIONALITY
# ======================
ENROLLMENT_DIR = Path("enrolled_speakers")
ENROLLMENT_DIR.mkdir(exist_ok=True)
EMBEDDING_DIM = 192

def convert_to_wav(file_path):
    """Universal audio to WAV converter"""
    try:
        if isinstance(file_path, str) and file_path.lower().endswith(".wav"):
            return file_path

        if hasattr(file_path, 'file'):
            temp_path = f"temp_{file_path.filename}"
            with open(temp_path, "wb") as buffer:
                buffer.write(file_path.file.read())
            file_path = temp_path

        output_path = file_path.rsplit(".", 1)[0] + ".wav"
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"Conversion error: {str(e)}")
        return None

def apply_vad(audio_path, aggressiveness=3):
    """Voice activity detection with WebRTC"""
    try:
        signal, sr = librosa.load(audio_path, sr=16000, mono=True)
        signal = (signal * 32767).astype(np.int16)
        vad = webrtcvad.Vad(aggressiveness)
        frame_length = int(16000 * 0.03)
        signal = signal[:len(signal) - (len(signal) % frame_length)]
        frames = np.array_split(signal, len(signal) // frame_length)
        voiced_frames = [frame for frame in frames if vad.is_speech(frame.tobytes(), 16000)]
        return np.concatenate(voiced_frames) if voiced_frames else None
    except Exception as e:
        print(f"VAD error: {str(e)}")
        return None

def extract_embedding(audio_path):
    """ECAPA-TDNN embedding extraction"""
    try:
        signal, sr = torchaudio.load(audio_path)
        if sr != 16000:
            signal = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(signal)
        signal = signal.mean(dim=0, keepdim=True) if signal.shape[0] > 1 else signal
        if signal.shape[1] < 16000:
            return None
            
        signal = signal.to(DEVICE)
        with torch.no_grad():
            embedding = spkrec_model.encode_batch(signal).squeeze().cpu().numpy()

        # Normalize embedding dimensions
        if len(embedding) > EMBEDDING_DIM:
            embedding = embedding[:EMBEDDING_DIM]
        elif len(embedding) < EMBEDDING_DIM:
            embedding = np.pad(embedding, (0, EMBEDDING_DIM - len(embedding)))
        return embedding
    except Exception as e:
        print(f"Embedding error: {str(e)}")
        return None

def database_operation(name=None, embedding=None, mode="save"):
    """Unified database handler"""
    session = SessionLocal()
    try:
        if mode == "save":
            embedding_bytes = embedding.tobytes()
            existing = session.query(SpeakerModel).filter_by(name=name).first()
            if existing:
                existing.embedding = embedding_bytes
            else:
                session.add(SpeakerModel(name=name, embedding=embedding_bytes))
            session.commit()
            return True
            
        elif mode == "identify":
            speakers = session.query(SpeakerModel).all()
            if not speakers:
                return "No speakers enrolled"
                
            scores = []
            for speaker in speakers:
                db_embed = np.frombuffer(speaker.embedding, dtype=np.float32)
                score = np.dot(embedding, db_embed) / (np.linalg.norm(embedding) * np.linalg.norm(db_embed))
                scores.append((speaker.name, score))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            best_match, confidence = scores[0]
            return f"{best_match} (confidence: {confidence:.2f})" if confidence > 0.5 else "Unknown"
            
    except Exception as e:
        session.rollback()
        print(f"Database error: {str(e)}")
        return False
    finally:
        session.close()

# ======================
# FASTAPI ENDPOINTS
# ======================
app = FastAPI(title="Cloud Speaker Recognition API")

@app.post("/enroll")
async def enroll_speaker(name: str, audio: UploadFile = File(...)):
    """Speaker enrollment endpoint"""
    wav_path = convert_to_wav(audio)
    if not wav_path:
        raise HTTPException(400, "Audio conversion failed")
    
    vad_audio = apply_vad(wav_path)
    if vad_audio is None:
        os.remove(wav_path)
        raise HTTPException(400, "No speech detected")
    
    embedding = extract_embedding(wav_path)
    if embedding is None:
        os.remove(wav_path)
        raise HTTPException(400, "Embedding extraction failed")
    
    if database_operation(name=name, embedding=embedding, mode="save"):
        os.remove(wav_path)
        return {"status": f"Speaker {name} enrolled successfully"}
    raise HTTPException(500, "Database operation failed")

@app.post("/recognize")
async def recognize_speaker(audio: UploadFile = File(...)):
    """Speaker recognition endpoint"""
    wav_path = convert_to_wav(audio)
    if not wav_path:
        raise HTTPException(400, "Audio conversion failed")
    
    embedding = extract_embedding(wav_path)
    if embedding is None:
        os.remove(wav_path)
        raise HTTPException(400, "Embedding extraction failed")
    
    speaker_id = database_operation(embedding=embedding, mode="identify")
    
    # Additional processing
    transcription = transcribe_audio(wav_path)
    emotion = detect_emotion(wav_path)
    os.remove(wav_path)
    
    return {
        "speaker": speaker_id,
        "transcription": transcription,
        "emotion": emotion
    }

def transcribe_audio(audio_path):
    """Whisper transcription with error handling"""
    try:
        audio, _ = librosa.load(audio_path, sr=16000)
        inputs = whisper_processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE)
        with torch.no_grad():
            predicted_ids = whisper_model.generate(inputs)
        return whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    except Exception as e:
        return f"Transcription error: {str(e)}"

def detect_emotion(audio_path):
    """Emotion classification with error handling"""
    try:
        result = emotion_classifier(audio_path)
        return result[0]["label"]
    except Exception as e:
        return f"Emotion detection error: {str(e)}"

# ======================
# STARTUP CONFIG
# ======================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
