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
import gc

# Device Setup - Force CPU to save memory
device = torch.device("cpu")
os.environ["HF_HUB_LOCAL_STRATEGY"] = "copy"
# Set lower precision for reduced memory usage
torch.set_default_dtype(torch.float32)

# Database Setup - Using SQLite for simplicity
Base = declarative_base()
engine = create_engine('sqlite:///speaker_database.db')
SessionLocal = sessionmaker(bind=engine)

class SpeakerModel(Base):
    __tablename__ = 'speakers'
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    embedding = Column(LargeBinary)

# Create database tables
Base.metadata.create_all(bind=engine)

# Load models only when needed (lazy loading)
spkrec = None
whisper_processor = None
whisper_model = None
emotion_classifier = None

def load_speaker_recognition_model():
    global spkrec
    if spkrec is None:
        from speechbrain.pretrained import SpeakerRecognition
        print("Loading speaker recognition model...")
        spkrec = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_ecapa_tdnn",
            run_opts={"device": device}
        )
    return spkrec

def load_whisper_model():
    global whisper_processor, whisper_model
    if whisper_processor is None or whisper_model is None:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        print("Loading Whisper model...")
        # Use tiny model to reduce memory footprint
        whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    return whisper_processor, whisper_model

def load_emotion_classifier():
    global emotion_classifier
    if emotion_classifier is None:
        from transformers import pipeline
        print("Loading emotion classifier...")
        # Use a simpler emotion model or consider removing if memory is critical
        emotion_classifier = pipeline(
            "audio-classification", 
            model="MIT/ast-finetuned-speech-commands-v2",
            device=device
        )
    return emotion_classifier

# Constants
ENROLLMENT_DIR = Path("enrolled_speakers")
ENROLLMENT_DIR.mkdir(exist_ok=True)
EMBEDDING_DIM = 192

def convert_to_wav(file_path):
    """Converts audio to WAV if it's not already in WAV format."""
    if isinstance(file_path, str) and file_path.lower().endswith(".wav"):
        return file_path  

    try:
        # If file is an UploadFile from FastAPI
        if hasattr(file_path, 'file'):
            temp_path = f"temp_{file_path.filename}"
            with open(temp_path, "wb") as buffer:
                buffer.write(file_path.file.read())
            file_path = temp_path

        new_file_path = file_path.rsplit(".", 1)[0] + ".wav"
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)  # Ensure 16kHz mono
        audio.export(new_file_path, format="wav")
        
        # Clean up original file if it was temporary
        if file_path.startswith('temp_') and os.path.exists(file_path):
            os.remove(file_path)
            
        return new_file_path
    except Exception as e:
        print(f"Error converting {file_path} to WAV: {str(e)}")
        return None

def apply_vad(audio_path, aggressiveness=3):
    """Removes silence & noise using WebRTC VAD."""
    try:
        # Load audio with lower quality to save memory
        signal, sr = librosa.load(audio_path, sr=16000, mono=True, res_type='kaiser_fast')
        signal = (signal * 32767).astype(np.int16)
        vad = webrtcvad.Vad(aggressiveness)
        frame_length = int(16000 * 0.03)
        signal = signal[:len(signal) - (len(signal) % frame_length)]
        frames = np.array_split(signal, len(signal) // frame_length)
        voiced_frames = [frame for frame in frames if vad.is_speech(frame.tobytes(), 16000)]
        
        if not voiced_frames:
            return None
            
        return np.concatenate(voiced_frames)
    except Exception as e:
        print(f"Error in VAD processing: {str(e)}")
        return None
    finally:
        # Force garbage collection
        gc.collect()

def extract_speaker_embedding(audio_path):
    """Extracts speaker embedding using ECAPA-TDNN model."""
    try:
        # Load model on demand
        spkrec = load_speaker_recognition_model()
        
        # Process audio with minimal memory usage
        signal, sr = torchaudio.load(audio_path)
        if sr != 16000:
            signal = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(signal)
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
        if signal.shape[1] < 16000:  # Require at least 1 second of audio
            return None
            
        signal = signal.to(device)
        
        # Extract embedding in smaller chunks if audio is long
        max_chunk_length = 10 * 16000  # 10 seconds
        if signal.shape[1] > max_chunk_length:
            chunks = torch.split(signal, max_chunk_length, dim=1)
            embeddings = []
            for chunk in chunks:
                with torch.no_grad():
                    emb = spkrec.encode_batch(chunk).squeeze().detach().cpu().numpy()
                    embeddings.append(emb)
            embedding = np.mean(embeddings, axis=0)
        else:
            with torch.no_grad():
                embedding = spkrec.encode_batch(signal).squeeze().detach().cpu().numpy()

        if len(embedding) > EMBEDDING_DIM:
            embedding = embedding[:EMBEDDING_DIM]
        elif len(embedding) < EMBEDDING_DIM:
            embedding = np.pad(embedding, (0, EMBEDDING_DIM - len(embedding)), 'constant')

        return embedding
    except Exception as e:
        print(f"Error extracting embedding: {str(e)}")
        return None
    finally:
        # Clean up memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

def save_speaker_to_database(speaker_name, embedding):
    """Save speaker embedding to the database."""
    session = SessionLocal()
    try:
        # Convert numpy array to bytes for storage
        embedding_bytes = embedding.tobytes()
        
        # Check if speaker already exists
        existing_speaker = session.query(SpeakerModel).filter_by(name=speaker_name).first()
        
        if existing_speaker:
            # Update existing speaker
            existing_speaker.embedding = embedding_bytes
        else:
            # Create new speaker entry
            new_speaker = SpeakerModel(name=speaker_name, embedding=embedding_bytes)
            session.add(new_speaker)
        
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"Database error: {str(e)}")
        return False
    finally:
        session.close()

def identify_speaker_from_database(test_embedding):
    """Identify speaker from database using cosine similarity."""
    session = SessionLocal()
    try:
        speakers = session.query(SpeakerModel).all()
        
        scores = []
        for speaker in speakers:
            # Convert bytes back to numpy array
            db_embedding = np.frombuffer(speaker.embedding, dtype=np.float32)
            
            # Compute cosine similarity
            score = np.dot(test_embedding, db_embedding) / (np.linalg.norm(test_embedding) * np.linalg.norm(db_embedding))
            scores.append((speaker.name, score))
        
        # Sort scores in descending order
        scores.sort(key=lambda x: x[1], reverse=True)
        
        if not scores:
            return "No enrolled speakers found"
        
        best_speaker, best_score = scores[0]
        second_best_score = scores[1][1] if len(scores) > 1 else 0
        
        if best_score > 0.5 and (best_score - second_best_score) > 0.05:
            return f"{best_speaker} (confidence: {best_score:.2f})"
        return "Unknown Speaker"
    
    except Exception as e:
        print(f"Database identification error: {str(e)}")
        return "Error in speaker identification"
    finally:
        session.close()

def transcribe_audio(audio_file):
    """Transcribe speech using OpenAI Whisper."""
    try:
        # Load models on demand
        whisper_processor, whisper_model = load_whisper_model()
        
        # Process audio with minimal memory
        audio, _ = librosa.load(audio_file, sr=16000, res_type='kaiser_fast')
        
        # Process in chunks if audio is long
        max_length = 30 * 16000  # 30 seconds
        if len(audio) > max_length:
            chunks = [audio[i:i+max_length] for i in range(0, len(audio), max_length)]
            transcriptions = []
            
            for chunk in chunks:
                input_features = whisper_processor(chunk, sampling_rate=16000, return_tensors="pt").input_features
                with torch.no_grad():
                    predicted_ids = whisper_model.generate(input_features)
                transcriptions.append(whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])
            
            return " ".join(transcriptions)
        else:
            input_features = whisper_processor(audio, sampling_rate=16000, return_tensors="pt").input_features
            with torch.no_grad():
                predicted_ids = whisper_model.generate(input_features)
            return whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"
    finally:
        # Clean up memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

def detect_emotion(audio_file):
    """Detect emotion from the audio file."""
    try:
        # Load emotion classifier on demand
        classifier = load_emotion_classifier()
        
        # Process small segment for emotion detection to save memory
        audio, sr = librosa.load(audio_file, sr=16000, duration=5, res_type='kaiser_fast')  # Just use 5 seconds
        result = classifier(audio_file)
        return result[0]["label"]
    except Exception as e:
        return f"Error detecting emotion: {str(e)}"
    finally:
        # Clean up memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

# Create memory-efficient FastAPI app
app = FastAPI(title="Speaker Recognition API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add endpoints to unload models to free memory when needed
@app.post("/unload-models/")
async def unload_models():
    """Endpoint to unload models and free memory."""
    global spkrec, whisper_processor, whisper_model, emotion_classifier
    spkrec = None
    whisper_processor = None
    whisper_model = None
    emotion_classifier = None
    
    # Force garbage collection
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    return {"status": "Models unloaded successfully"}

@app.post("/enroll-speaker/")
async def enroll_speaker_endpoint(
    name: str, 
    audio: UploadFile = File(...)
):
    """Endpoint to enroll a new speaker."""
    try:
        # Save uploaded file and convert to WAV
        wav_file = convert_to_wav(audio)
        if not wav_file:
            raise HTTPException(status_code=400, detail="Failed to convert audio")
        
        vad_audio = apply_vad(wav_file)
        if vad_audio is None:
            raise HTTPException(status_code=400, detail="No valid speech detected")
        
        embedding = extract_speaker_embedding(wav_file)
        if embedding is None:
            raise HTTPException(status_code=400, detail="Failed to extract speaker embedding")
        
        # Save to database
        if save_speaker_to_database(name, embedding):
            # Clean up temporary file if created
            if wav_file.startswith('temp_') and os.path.exists(wav_file):
                os.remove(wav_file)
            return {"status": "Speaker enrolled successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to save speaker to database")
    
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))
    finally:
        # Free memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

@app.post("/process-audio/")
async def process_audio_endpoint(
    audio: UploadFile = File(...),
    do_transcription: bool = True,
    do_emotion: bool = True
):
    """Endpoint to process audio and return speaker, transcription, and emotion."""
    try:
        # Save uploaded file and convert to WAV
        wav_file = convert_to_wav(audio)
        if not wav_file:
            raise HTTPException(status_code=400, detail="Failed to convert audio")
        
        # Extract embedding for speaker identification
        test_embedding = extract_speaker_embedding(wav_file)
        if test_embedding is None:
            raise HTTPException(status_code=400, detail="Failed to extract speaker embedding")
        
        # Identify speaker from database
        speaker = identify_speaker_from_database(test_embedding)
        
        result = {"speaker": speaker}
        
        # Only do transcription if requested
        if do_transcription:
            result["transcription"] = transcribe_audio(wav_file)
        
        # Only do emotion detection if requested
        if do_emotion:
            result["emotion"] = detect_emotion(wav_file)
        
        # Clean up temporary file if created
        if wav_file.startswith('temp_') and os.path.exists(wav_file):
            os.remove(wav_file)
        
        return result
    
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))
    finally:
        # Free memory
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

# Add health check endpoint
@app.get("/health/")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}

# Add a memory status endpoint for monitoring
@app.get("/memory-status/")
async def memory_status():
    """Check memory usage."""
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        "memory_used_mb": memory_info.rss / 1024 / 1024,
        "memory_percent": process.memory_percent(),
        "models_loaded": {
            "speaker_recognition": spkrec is not None,
            "whisper": whisper_model is not None,
            "emotion_classifier": emotion_classifier is not None
        }
    }

if __name__ == "__main__":
    # Set low-memory worker settings
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.environ.get("PORT", 8000)),
        workers=1,  # Use just one worker to conserve memory
        loop="asyncio"
    )
