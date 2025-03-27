# **Speaker Identification & Emotion Detection (SID&ED)**  
**By Bhargav Akshit**  

---

## **📌 Project Overview**  
This project is a **Speaker Identification and Emotion Detection (SID&ED)** system that:  
✅ Identifies speakers using **ECAPA-TDNN embeddings**  
✅ Transcribes speech using **OpenAI's Whisper**  
✅ Detects emotions from audio using **Wav2Vec2**  

Built with **Python, FastAPI, and PyTorch**, it provides a REST API for:  
- **Enrolling speakers** (`/enroll-speaker/`)  
- **Processing audio** (`/process-audio/`)  

---

## **⚙️ Setup & Installation**  

### **1. Clone the Repository**  
```bash
git clone (https://github.com/bhargavak04/Speaker-Identification-Emotion-Detection).git
cd SID-ED
```

### **2. Create & Activate Virtual Environment**  
```bash
python -m venv env
source env/bin/activate  # Linux/Mac
.\env\Scripts\activate   # Windows
```

### **3. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **4. Download Pretrained Models**  
Replace these paths in `main.py` with **your local model directories**:  
```python
# Replace these paths with your actual model locations!
whisper_processor = WhisperProcessor.from_pretrained(r"**YOUR_WHISPER_MODEL_PATH**")
whisper_model = WhisperForConditionalGeneration.from_pretrained(r"**YOUR_WHISPER_MODEL_PATH**")
emotion_classifier = pipeline("audio-classification", model=r"**YOUR_WAV2VEC2_MODEL_PATH**")
```

---

## **🚀 Running the API**  
Start the FastAPI server:  
```bash
uvicorn main:app --reload
```
- Access API docs: [http://localhost:8000/docs](http://localhost:8000/docs)  

---

## **📂 Project Structure**  
```
SID-ED/  
├── main.py                # FastAPI server & core logic  
├── requirements.txt       # Python dependencies  
├── README.md              
├── .gitignore            # Excludes env/, __pycache__/, etc.  
├── enrolled_speakers/    # Stores enrolled speaker audios  
└── Audio_Formatted/      # Sample audio dataset (optional)  
```

---

## **🔧 API Endpoints**  

### **1. Enroll a Speaker**  
**Endpoint:** `POST /enroll-speaker/`  
**Input:**  
- `name` (str): Speaker name  
- `audio` (file): Audio file (MP3/WAV)  

**PowerShell Example:**  
```powershell
curl.exe -X POST -F "name=Bhargav" -F "audio=@sample.wav" http://localhost:8000/enroll-speaker/
```

### **2. Process Audio**  
**Endpoint:** `POST /process-audio/`  
**Output:**  
```json
{
  "speaker": "Bhargav (confidence: 0.92)",
  "transcription": "Hello world!",
  "emotion": "happy"
}
```

**Python Example:**  
```python
import requests
response = requests.post("http://localhost:8000/process-audio/", files={"audio": open("test.wav", "rb")})
print(response.json())
```

---

## **📌 Key Notes**  
🔹 **Do NOT commit `env/` or model files** (use `.gitignore`).  
🔹 **Replace model paths** in `main.py` before running.  
🔹 For large audio files, consider **chunking/VAD** (included in code).  

---

## **📜 License**  
MIT © **Bhargav Akshit**  

--- 

**🌟 Star this repo if you find it useful!**  
**🐛 Report issues [here](https://github.com/bhargavak04/Speaker-Identification-Emotion-Detection/issues).**  

--- 

### **Happy Coding!** 🚀
