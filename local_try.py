import os
import torch
import numpy as np
import torchaudio
import librosa
import webrtcvad
from pydub import AudioSegment
from pathlib import Path
from speechbrain.pretrained import SpeakerRecognition
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

# Set device to GPU if available
device = torch.device("cpu")
os.environ["HF_HUB_LOCAL_STRATEGY"] = "copy"


print("Loading models...")
spkrec = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_ecapa_tdnn"
).to(device)

whisper_processor = WhisperProcessor.from_pretrained(r"C:\Users\bharg\Downloads\SID&ED\ModelOpen\openw\whisper-small")
whisper_model = WhisperForConditionalGeneration.from_pretrained(r"C:\Users\bharg\Downloads\SID&ED\ModelOpen\openw\whisper-small")

emotion_classifier = pipeline("audio-classification", model=r"C:\Users\bharg\Downloads\SID&ED\ModelOpen\wavv2vec")

tts_model = ParlerTTSForConditionalGeneration.from_pretrained(r"C:\Users\bharg\Downloads\SID&ED\indic_parler").to(device)
tts_tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\bharg\Downloads\SID&ED\indic_parler")
description_tokenizer = AutoTokenizer.from_pretrained(tts_model.config.text_encoder._name_or_path) # AutoTokenizer.from_pretrained(tts_model.config.text_encoder._name_or_path)

# Constants
ENROLLMENT_DIR = Path("enrolled_speakers")
ENROLLMENT_DIR.mkdir(exist_ok=True)
EMBEDDING_DIM = 192

def convert_to_wav(file_path):
    #Converts audio to WAV if it's not already in WAV format.
    if file_path.lower().endswith(".wav"):
        return file_path  

    try:
        new_file_path = file_path.rsplit(".", 1)[0] + ".wav"
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
        print(f"Failed to enroll {speaker_name}: No valid embeddings extracted")
        return False

    speaker_embedding = np.mean(np.array(embeddings), axis=0)
    np.save(ENROLLMENT_DIR / f"{speaker_name}.npy", speaker_embedding)
    print(f"Enrolled {speaker_name} successfully!")
    return True

def identify_speaker(audio_file):
    """Identify the speaker of a given audio file using cosine similarity."""
    wav_file = convert_to_wav(audio_file)
    if not wav_file:
        return "Failed to convert audio to WAV format"

    vad_audio = apply_vad(wav_file)
    if vad_audio is None:
        return "No valid speech detected in audio"

    test_embedding = extract_speaker_embedding(wav_file)
    if test_embedding is None:
        return "Failed to extract speaker embedding"

    enrolled_speakers = {f.stem: np.load(f) for f in ENROLLMENT_DIR.glob("*.npy")}

    if not enrolled_speakers:
        return "No enrolled speakers found. Please enroll speakers first."

    scores = [(name, np.dot(test_embedding, emb) / (np.linalg.norm(test_embedding) * np.linalg.norm(emb)))
              for name, emb in enrolled_speakers.items()]
    scores.sort(key=lambda x: x[1], reverse=True)

    best_speaker, best_score = scores[0]
    second_best_score = scores[1][1] if len(scores) > 1 else 0

    if best_score > 0.5 and (best_score - second_best_score) > 0.05:
        return f"{best_speaker} (confidence: {best_score:.2f})"
    return "Unknown Speaker"

def transcribe_audio(audio_file):
    """Transcribe speech using OpenAI Whisper."""
    try:
        audio_file = convert_to_wav(audio_file)
        audio, _ = librosa.load(audio_file, sr=16000)
        input_features = whisper_processor(audio, sampling_rate=16000, return_tensors="pt").input_features
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
    
def tts(prompt):
    description = "Divya's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."
    description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
    prompt_input_ids = tts_tokenizer(prompt, return_tensors="pt").to(device)

    generation = tts_model.generate(input_ids=description_input_ids.input_ids, attention_mask=description_input_ids.attention_mask, prompt_input_ids=prompt_input_ids.input_ids, prompt_attention_mask=prompt_input_ids.attention_mask)
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write("indic_tts_out_hindib.wav", audio_arr, tts_model.config.sampling_rate)
    return "indic_tts_out_hindib.wav created successfully"

def process_audio(audio_file):
    """Process the audio file to identify speaker, transcribe, and detect emotion."""
    speaker = identify_speaker(audio_file)
    transcription = transcribe_audio(audio_file)
    emotion = detect_emotion(audio_file)   
    tts(prompt=transcription)
    
    print("\nResults:")
    print(f"Speaker: {speaker}")
    print(f"Transcription: {transcription}")
    print(f"Emotion: {emotion}")
    print(f"TTS: {tts(prompt=transcription)}")

# Main Execution
if __name__ == "__main__":
    audio_directory = "Audio_Formatted"
    print("Enrolling speakers...")
    for speaker in os.listdir(audio_directory):
        enroll_speaker(speaker, [os.path.join(audio_directory, speaker, f) for f in os.listdir(os.path.join(audio_directory, speaker))])

    test_audio = r"C:\Users\bharg\Downloads\SID&ED\Audio_Formatted\Prudhvi\Prudhvi_kumar_Surprise.wav"
    process_audio(test_audio)
