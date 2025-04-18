import requests
import os
import time

# Base URL of your deployed service
BASE_URL = "https://speaker-identification-emotion-detection.onrender.com"

def test_health(max_retries=3, delay=5):
    """Test the health check endpoint with retries"""
    print("\nTesting health check endpoint...")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health/", timeout=30)
            print(f"Health check status: {response.status_code}")
            
            if response.status_code == 200:
                print(response.json())
                return True
            elif response.status_code == 502:
                print(f"Service is starting up (attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
            else:
                print(f"Unexpected status code: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to service: {e}")
            print(f"Retrying in {delay} seconds... (attempt {attempt+1}/{max_retries})")
            time.sleep(delay)
    
    print("Failed to connect after multiple attempts")
    return False

def enroll_speaker(name, audio_file_path):
    """Enroll a new speaker"""
    print(f"\nEnrolling speaker '{name}' with audio file: {audio_file_path}")
    
    if not os.path.exists(audio_file_path):
        print(f"Error: File not found: {audio_file_path}")
        return
        
    with open(audio_file_path, 'rb') as f:
        # The API expects name as a query parameter, not as form data
        files = {'audio': (os.path.basename(audio_file_path), f, 'audio/wav')}
        
        # Name is now a query parameter
        url = f"{BASE_URL}/enroll-speaker/?name={name}"
        
        print("Sending enrollment request...")
        response = requests.post(url, files=files)
        
        print(f"Enrollment status: {response.status_code}")
        try:
            print(response.json())
        except:
            print(response.text)

def process_audio(audio_file_path, do_transcription=True, do_emotion=True):
    """Process audio for speaker recognition, transcription, and emotion"""
    print(f"\nProcessing audio file: {audio_file_path}")
    
    if not os.path.exists(audio_file_path):
        print(f"Error: File not found: {audio_file_path}")
        return
        
    with open(audio_file_path, 'rb') as f:
        files = {'audio': (os.path.basename(audio_file_path), f, 'audio/wav')}
        params = {
            'do_transcription': str(do_transcription).lower(),
            'do_emotion': str(do_emotion).lower()
        }
        
        print("Sending processing request...")
        response = requests.post(f"{BASE_URL}/process-audio/", files=files, params=params)
        
        print(f"Processing status: {response.status_code}")
        try:
            print(response.json())
        except:
            print(response.text)

def debug_audio_file(audio_file_path):
    """Print debug information about the audio file"""
    print(f"\nDebugging audio file: {audio_file_path}")
    
    if not os.path.exists(audio_file_path):
        print(f"Error: File not found: {audio_file_path}")
        return
    
    file_size = os.path.getsize(audio_file_path)
    print(f"File size: {file_size} bytes")
    
    # You can add more audio file checks if you have libraries like pydub or librosa installed
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_file_path)
        print(f"Audio duration: {len(audio)/1000:.2f} seconds")
        print(f"Channels: {audio.channels}")
        print(f"Sample rate: {audio.frame_rate} Hz")
        print(f"Sample width: {audio.sample_width} bytes")
    except ImportError:
        print("Pydub not installed, skipping detailed audio analysis")

if __name__ == "__main__":
    # Test health endpoint
    test_health()
    
    # Debug the audio file
    audio_path = r"C:\Users\bharg\Downloads\SID&ED\test_yaswanth_neu.wav"
    debug_audio_file(audio_path)
    
    # Example: Enroll a speaker
    enroll_speaker("TestUser", audio_path)
    
    # Example: Process the same audio
    process_audio(audio_path)