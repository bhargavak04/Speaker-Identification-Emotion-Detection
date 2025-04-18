import os
import torch
from speechbrain.pretrained import SpeakerRecognition
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model paths (you can modify these to use environment variables)
WHISPER_MODEL_PATH = os.getenv('WHISPER_MODEL_PATH', r"C:\Users\bharg\Downloads\SID&ED\ModelOpen\openw\whisper-small")
EMOTION_MODEL_PATH = os.getenv('EMOTION_MODEL_PATH', r"C:\Users\bharg\Downloads\SID&ED\ModelOpen\wavv2vec")
TTS_MODEL_PATH = os.getenv('TTS_MODEL_PATH', r"C:\Users\bharg\Downloads\SID&ED\indic_parler")

def get_speaker_recognition_model():
    """Load and return the speaker recognition model"""
    return SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_ecapa_tdnn"
    ).to(device)

def get_whisper_model_and_processor():
    """Load and return the Whisper model and processor"""
    processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_PATH).to(device)
    return model, processor

def get_emotion_classifier():
    """Load and return the emotion classification model"""
    return pipeline("audio-classification", model=EMOTION_MODEL_PATH, device=0 if torch.cuda.is_available() else -1)

def get_tts_model_and_tokenizers():
    """Load and return the TTS model and tokenizers"""
    model = ParlerTTSForConditionalGeneration.from_pretrained(TTS_MODEL_PATH).to(device)
    tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_PATH)
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    
    return model, tokenizer, description_tokenizer