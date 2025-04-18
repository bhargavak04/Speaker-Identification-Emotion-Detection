# Speaker Recognition and Audio Processing API

A comprehensive system for speaker recognition, speech transcription, emotion detection, and text-to-speech conversion with a modern REST API interface.

## Features

- **Speaker Recognition**: Enroll and identify speakers based on voice characteristics using ECAPA-TDNN embeddings
- **Speech Transcription**: Convert speech to text using Whisper models
- **Emotion Detection**: Detect emotions in speech
- **Text-to-Speech**: Generate natural-sounding speech from text with Indian language support
- **PostgreSQL Integration**: Store speaker embeddings and recognition logs in a PostgreSQL database
- **RESTful API**: Easy-to-use API for all functionalities
- **Optional Docker Support**: Containerized deployment option with Docker and Docker Compose

## Models Used

This project leverages several state-of-the-art deep learning models:

- **Speaker Recognition**: [speechbrain/spkrec-ecapa-voxceleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) - ECAPA-TDNN model trained on VoxCeleb
- **Speech Transcription**: [openai/whisper-small](https://huggingface.co/openai/whisper-small) - Whisper automatic speech recognition model
- **Emotion Detection**: [wav2vec2 model for emotion recognition](https://huggingface.co/ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition)
- **Text-to-Speech**: [Parler TTS for Indian languages](https://huggingface.co/ai4bharat/indic-parler-tts)

## Architecture

The project is structured into several modules:

1. **db_setup.py**: Database initialization and operations
2. **model_loader.py**: Model loading and management
3. **speaker_recognition.py**: Core recognition and processing functions
4. **api.py**: FastAPI server and API endpoints

## API Endpoints

- `POST /enroll-speaker/`: Enroll a new speaker with audio samples
- `POST /identify-speaker/`: Identify a speaker in an audio file
- `POST /transcribe/`: Transcribe speech in an audio file
- `POST /detect-emotion/`: Detect emotion in an audio file
- `POST /text-to-speech/`: Convert text to speech
- `POST /process-audio/`: Complete processing (identification, transcription, emotion detection)
- `GET /download-tts/{filename}`: Download generated TTS files

## Setup Instructions

### Prerequisites

- Python 3.9+
- PostgreSQL
- CUDA-compatible GPU (recommended but not required)

### Standard Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bhargavak04/Speaker-Identification-Emotion-Detection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your configuration:
   ```
   # Database Configuration
   DB_HOST=localhost
   DB_NAME=speaker_recognition
   DB_USER=postgres
   DB_PASSWORD=your_password
   DB_PORT=5432

   # Model Paths
   WHISPER_MODEL_PATH=/path/to/whisper-small
   EMOTION_MODEL_PATH=/path/to/emotion-model
   TTS_MODEL_PATH=/path/to/tts-model
   ```

5. Initialize the database:
   ```bash
   python db_setup.py
   ```

6. Start the API server:
   ```bash
   python api.py
   ```

7. Access the API at http://localhost:8000

### Docker Installation (Optional)

If you prefer containerized deployment, you can use Docker:

1. Make sure Docker and Docker Compose are installed on your system.

2. Create a `.env` file with your configuration (as above).

3. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

4. Access the API at http://localhost:8000

## Usage Examples

### Enrolling a Speaker

```bash
curl -X POST "http://localhost:8000/enroll-speaker/" \
  -F "speaker_name=John" \
  -F "audio_files=@sample1.wav" \
  -F "audio_files=@sample2.wav"
```

### Identifying a Speaker

```bash
curl -X POST "http://localhost:8000/identify-speaker/" \
  -F "audio_file=@unknown_speaker.wav"
```

### Processing an Audio File

```bash
curl -X POST "http://localhost:8000/process-audio/" \
  -F "audio_file=@sample.wav"
```

