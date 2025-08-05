import requests
import sounddevice as sd
import soundfile as sf
import tempfile
import time
import os
import webbrowser
import numpy as np
import scipy.signal

# CONFIGURATION
DURATION = 30  # seconds
SAMPLE_RATE = 16000
SERVER_URL = "https://kannada-stt.onrender.com/transcribe"

def record_audio(duration=DURATION, samplerate=SAMPLE_RATE):
    print(f"🎙️  Recording {duration} seconds of audio... Speak now!")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()

    # Optional: Basic denoising using a simple high-pass filter
    audio = apply_highpass_filter(audio, samplerate)

    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_wav.name, audio, samplerate)
    print(f"✅ Recording saved to {temp_wav.name}")
    return temp_wav.name

def apply_highpass_filter(audio, samplerate, cutoff=100.0):
    """Apply a basic high-pass filter to reduce low-frequency noise."""
    b, a = scipy.signal.butter(1, cutoff / (0.5 * samplerate), btype='high', analog=False)
    filtered_audio = scipy.signal.filtfilt(b, a, audio[:, 0])
    return np.expand_dims(filtered_audio, axis=1)

def send_to_server(audio_path):
    print("📡 Sending to server...")
    try:
        with open(audio_path, "rb") as f:
            files = {"audio": f}
            response = requests.post(SERVER_URL, files=files)

        if response.ok:
            result = response.json()
            transcript = result.get("transcript", "").strip()
            if transcript:
                print("\n✅ Transcription:")
                print(transcript)

                # Save to text file
                transcript_file = "transcript.txt"
                with open(transcript_file, "w", encoding="utf-8") as tf:
                    tf.write(transcript)
                print(f"📝 Transcript saved to {transcript_file}")
                webbrowser.open(f"file://{os.path.abspath(transcript_file)}")
            else:
                print("⚠️ No transcript returned.")
        else:
            print(f"❌ Server error [{response.status_code}]: {response.text}")
    except Exception as e:
        print("❌ Error sending request:", e)

if __name__ == "__main__":
    audio_path = record_audio()
    time.sleep(1)
    send_to_server(audio_path)
