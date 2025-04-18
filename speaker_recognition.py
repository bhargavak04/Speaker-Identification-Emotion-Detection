import os
import torch
import numpy as np
import torchaudio
import librosa
import webrtcvad
from pydub import AudioSegment
from pathlib import Path
import soundfile as sf
import tempfile

# Import model loading functions
from model_loader import (
    get_speaker_recognition_model,
    get_whisper_model_and_processor,
    get_emotion_classifier,
    get_tts_model_and_tokenizers
)

# Import database functions
from db_setup import store_speaker_embedding, get_all_speaker_embeddings, log_recognition_result

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["HF_HUB_LOCAL_STRATEGY"] = "copy"

# Load models
print("Loading models...")
spkrec = get_speaker_recognition_model()
whisper_model, whisper_processor = get_whisper_model_and_processor()
emotion_classifier = get_emotion_classifier()
tts_model, tts_tokenizer, description_tokenizer = get_tts_model_and_tokenizers()

# Constants
EMBEDDING_DIM = 192
TEMP_DIR = Path(tempfile.gettempdir()) / "speaker_recognition"
TEMP_DIR.mkdir(exist_ok=True)

def convert_to_wav(file_path):
    """Converts audio to WAV if it's not already in WAV format."""
    if file_path.lower().endswith(".wav"):
        return file_path  

    try:
        file_name = Path(file_path).name
        new_file_path = str(TEMP_DIR / f"{file_name.rsplit('.', 1)[0]}.wav")
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)  # Ensure 16kHz mono
        audio.export(new_file_path, format="wav")
        return new_file_path
    except Exception as e:
        print(f"Error converting {file_path} to WAV: {str(e)}")
        return None

def apply_vad(audio_path, aggressiveness=3):
    """Removes silence & noise using WebRTC VAD."""
    try:
        signal, sr = librosa.load(audio_path, sr=16000, mono=True)
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
    
def extract_speaker_embedding(audio_path, spkrec_model=spkrec):
    """Extracts speaker embedding using ECAPA-TDNN model."""
    try:
        signal, sr = torchaudio.load(audio_path)
        if sr != 16000:
            signal = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(signal)
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
        if signal.shape[1] < 16000:
            return None
            
        # Device handling
        spkrec_model = spkrec_model.to(device)
        signal = signal.to(device)
        
        with torch.no_grad():
            embedding = spkrec_model.encode_batch(signal).squeeze().cpu().numpy()

        # Ensure consistent embedding dimension
        embedding = embedding[:EMBEDDING_DIM] if len(embedding) > EMBEDDING_DIM else np.pad(
            embedding, (0, EMBEDDING_DIM - len(embedding)), 'constant'
        )

        return embedding
    except Exception as e:
        print(f"Error extracting embedding: {str(e)}")
        return None
        
def enroll_speaker(speaker_name, audio_files):
    """Extract and store speaker embeddings from given audio files."""
    embeddings = []
    
    for audio_file in audio_files:
        wav_file = convert_to_wav(audio_file)
        if not wav_file:
            continue

        vad_audio = apply_vad(wav_file)
        if vad_audio is None:
            continue

        embedding = extract_speaker_embedding(wav_file)
        if embedding is not None:
            embeddings.append(embedding)

    if not embeddings:
        return False, "No valid embeddings extracted"

    speaker_embedding = np.mean(np.array(embeddings), axis=0)
    
    # Store in PostgreSQL database
    success = store_speaker_embedding(speaker_name, speaker_embedding)
    if success:
        return True, f"Enrolled {speaker_name} successfully!"
    return False, f"Failed to enroll {speaker_name} in database"

def identify_speaker(audio_file):
    """Identify the speaker of a given audio file using cosine similarity."""
    wav_file = convert_to_wav(audio_file)
    if not wav_file:
        return "Unknown Speaker", 0.0, "Failed to convert audio to WAV format"

    vad_audio = apply_vad(wav_file)
    if vad_audio is None:
        return "Unknown Speaker", 0.0, "No valid speech detected in audio"

    test_embedding = extract_speaker_embedding(wav_file)
    if test_embedding is None:
        return "Unknown Speaker", 0.0, "Failed to extract speaker embedding"

    # Get all enrolled speakers from database
    enrolled_speakers = get_all_speaker_embeddings()

    if not enrolled_speakers:
        return "Unknown Speaker", 0.0, "No enrolled speakers found. Please enroll speakers first."

    scores = [(name, np.dot(test_embedding, emb) / (np.linalg.norm(test_embedding) * np.linalg.norm(emb)))
              for name, emb in enrolled_speakers.items()]
    scores.sort(key=lambda x: x[1], reverse=True)

    best_speaker, best_score = scores[0]
    second_best_score = scores[1][1] if len(scores) > 1 else 0

    if best_score > 0.5 and (best_score - second_best_score) > 0.05:
        return best_speaker, best_score, "Speaker identified successfully"
    return "Unknown Speaker", best_score, "Speaker confidence too low"

def transcribe_audio(audio_file):
    """Transcribe speech using Whisper."""
    try:
        audio_file = convert_to_wav(audio_file)
        audio, _ = librosa.load(audio_file, sr=16000)
        input_features = whisper_processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        
        # Move to device
        input_features = input_features.to(device)
        whisper_model.to(device)
        
        predicted_ids = whisper_model.generate(input_features)
        return whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

def detect_emotion(audio_file):
    """Detect emotion from the audio file."""
    try:
        audio_file = convert_to_wav(audio_file)
        result = emotion_classifier(audio_file)
        return result[0]["label"]
    except Exception as e:
        return f"Error detecting emotion: {str(e)}"
    
def tts(prompt, output_path=None):
    """Text to speech conversion."""
    try:
        description = "Divya's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."
        description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
        prompt_input_ids = tts_tokenizer(prompt, return_tensors="pt").to(device)

        generation = tts_model.generate(
            input_ids=description_input_ids.input_ids, 
            attention_mask=description_input_ids.attention_mask, 
            prompt_input_ids=prompt_input_ids.input_ids, 
            prompt_attention_mask=prompt_input_ids.attention_mask
        )
        
        audio_arr = generation.cpu().numpy().squeeze()
        
        if output_path is None:
            output_path = str(TEMP_DIR / "indic_tts_output.wav")
            
        sf.write(output_path, audio_arr, tts_model.config.sampling_rate)
        return output_path
    except Exception as e:
        print(f"Error in TTS: {str(e)}")
        return None

def process_audio(audio_file):
    """Process the audio file to identify speaker, transcribe, and detect emotion."""
    speaker, confidence, message = identify_speaker(audio_file)
    transcription = transcribe_audio(audio_file)
    emotion = detect_emotion(audio_file)   
    tts_output_path = tts(prompt=transcription)
    
    # Log results to database
    log_recognition_result(audio_file, speaker, confidence, transcription, emotion)
    
    results = {
        "speaker": speaker,
        "confidence": round(float(confidence), 2) if isinstance(confidence, (float, int)) else 0.0,
        "transcription": transcription,
        "emotion": emotion,
        "tts_output_path": tts_output_path,
        "message": message
    }
    
    return results