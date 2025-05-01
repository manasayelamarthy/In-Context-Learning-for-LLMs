import sounddevice as sd
from scipy.io.wavfile import write
import whisper

def record_voice(filename="input.wav", duration=5, fs=44100):
    print("Recording for", duration, "seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)
    print("Recording saved as", filename)

def transcribe_audio(file_path="input.wav"):
    model = whisper.load_model("base")  
    result = model.transcribe(file_path)
    return result["text"]

def get_input(user_input = "text"):
    if user_input == "text":
        return input('Enter your Query':)
    
    elif user_input == "voice":
        return transcribe_audio()
    
    elif user_input == "pdf":
        


