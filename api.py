import os
import uuid
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import shutil
import tempfile

# Import our modules
from db_setup import initialize_database
from speaker_recognition import (
    enroll_speaker, 
    identify_speaker, 
    transcribe_audio, 
    detect_emotion, 
    tts, 
    process_audio
)

# Initialize database
initialize_database()

# Create uploads directory for temporary storage
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Speaker Recognition API", description="API for speaker recognition, transcription, and emotion detection")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify exact domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to the Speaker Recognition API"}

@app.post("/enroll-speaker/")
async def api_enroll_speaker(
    speaker_name: str = Form(...),
    audio_files: List[UploadFile] = File(...)
):
    """Enroll a new speaker with one or more audio samples"""
    if not audio_files:
        raise HTTPException(status_code=400, detail="No audio files provided")
    
    # Create temporary directory for this enrollment
    temp_dir = UPLOAD_DIR / f"enroll_{uuid.uuid4()}"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        saved_files = []
        
        # Save uploaded files
        for audio_file in audio_files:
            file_path = temp_dir / audio_file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(audio_file.file, buffer)
            saved_files.append(str(file_path))
        
        # Enroll the speaker
        success, message = enroll_speaker(speaker_name, saved_files)
        
        if success:
            return {"status": "success", "message": message}
        else:
            raise HTTPException(status_code=400, detail=message)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enrolling speaker: {str(e)}")
    
    finally:
        # Clean up temporary files
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

@app.post("/identify-speaker/")
async def api_identify_speaker(audio_file: UploadFile = File(...)):
    """Identify the speaker in an audio file"""
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Save uploaded file
    temp_file = UPLOAD_DIR / f"identify_{uuid.uuid4()}_{audio_file.filename}"
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Identify the speaker
        speaker, confidence, message = identify_speaker(str(temp_file))
        
        return {
            "speaker": speaker,
            "confidence": round(float(confidence), 2),
            "message": message
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error identifying speaker: {str(e)}")
    
    finally:
        # Clean up
        if temp_file.exists():
            temp_file.unlink()

@app.post("/transcribe/")
async def api_transcribe(audio_file: UploadFile = File(...)):
    """Transcribe speech in an audio file"""
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Save uploaded file
    temp_file = UPLOAD_DIR / f"transcribe_{uuid.uuid4()}_{audio_file.filename}"
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Transcribe the audio
        transcription = transcribe_audio(str(temp_file))
        
        return {"transcription": transcription}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")
    
    finally:
        # Clean up
        if temp_file.exists():
            temp_file.unlink()

@app.post("/detect-emotion/")
async def api_detect_emotion(audio_file: UploadFile = File(...)):
    """Detect emotion in an audio file"""
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Save uploaded file
    temp_file = UPLOAD_DIR / f"emotion_{uuid.uuid4()}_{audio_file.filename}"
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Detect emotion
        emotion = detect_emotion(str(temp_file))
        
        return {"emotion": emotion}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting emotion: {str(e)}")
    
    finally:
        # Clean up
        if temp_file.exists():
            temp_file.unlink()

@app.post("/text-to-speech/")
async def api_text_to_speech(text: str = Form(...)):
    """Convert text to speech"""
    try:
        # Generate a unique output path
        output_file = UPLOAD_DIR / f"tts_{uuid.uuid4()}.wav"
        
        # Generate speech
        output_path = tts(text, str(output_file))
        
        if output_path:
            # Return the audio file
            return FileResponse(
                output_path, 
                media_type="audio/wav", 
                filename=os.path.basename(output_path)
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to generate speech")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@app.post("/process-audio/")
async def api_process_audio(audio_file: UploadFile = File(...)):
    """Process audio for speaker identification, transcription, and emotion detection"""
    if not audio_file:
        raise HTTPException(status_code=400, detail="No audio file provided")
    
    # Save uploaded file
    temp_file = UPLOAD_DIR / f"process_{uuid.uuid4()}_{audio_file.filename}"
    tts_output = None
    
    try:
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        # Process the audio
        results = process_audio(str(temp_file))
        tts_output = results.get('tts_output_path')
        
        # If TTS output exists, modify the result to include a URL
        if tts_output and os.path.exists(tts_output):
            results['tts_output_url'] = f"/download-tts/{os.path.basename(tts_output)}"
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")
    
    finally:
        # Clean up the input file but keep the TTS output
        if temp_file.exists():
            temp_file.unlink()

@app.get("/download-tts/{filename}")
async def download_tts(filename: str):
    """Download a generated TTS file"""
    file_path = Path(tempfile.gettempdir()) / "speaker_recognition" / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        str(file_path), 
        media_type="audio/wav", 
        filename=filename
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)