import requests
from pathlib import Path

def process_audio(file_path):
    """Equivalent Python implementation of the PowerShell audio processing"""
    url = "http://localhost:8000/process-audio/"
    file_name = Path(file_path).name
    
    try:
        # Read the file in binary mode
        with open(file_path, 'rb') as audio_file:
            files = {'audio': (file_name, audio_file, 'audio/wav')}
            
            # Make the request
            response = requests.post(url, files=files)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error {response.status_code}: {response.text}")
                return None
                
    except Exception as e:
        print(f"Failed to process audio: {str(e)}")
        return None

if __name__ == "__main__":
    # Example usage
    audio_path = r"C:\Users\bharg\Downloads\SID&ED\test_yaswanth_neu.wav"
    result = process_audio(audio_path)
    
    if result:
        print("Processing Results:")
        print(f"Speaker: {result.get('speaker', 'Unknown')}")
        print(f"Transcription: {result.get('transcription', '')}")
        print(f"Emotion: {result.get('emotion', '')}")