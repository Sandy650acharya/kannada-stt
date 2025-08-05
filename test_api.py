import requests
import sounddevice as sd
import soundfile as sf
import tempfile
import time
import os
import webbrowser

# CONFIGURATION
DURATION = 10  # seconds
SAMPLE_RATE = 16000
SERVER_URL = "https://kannada-stt.onrender.com/transcribe"

def record_audio(duration=DURATION, samplerate=SAMPLE_RATE):
    print("🎙️  Recording 60 seconds of audio... Speak now!")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()

    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_wav.name, audio, samplerate)
    print(f"✅ Recording saved to {temp_wav.name}")
    return temp_wav.name

def send_to_server(audio_path):
    print("📡 Sending to server...")
    try:
        with open(audio_path, "rb") as f:
            files = {"audio": f}  # ✅ Important fix
            response = requests.post(SERVER_URL, files=files)

        if response.ok:
            result = response.json()
            transcript = result.get("transcript", "").strip()
            print("✅ Transcription:")
            print(transcript)

            # Save to file
            transcript_file = "transcript.txt"
            with open(transcript_file, "w", encoding="utf-8") as tf:
                tf.write(transcript)
            print(f"📝 Transcript saved to {transcript_file}")
            webbrowser.open(f"file://{os.path.abspath(transcript_file)}")
        else:
            print(f"❌ Server error: {response.status_code}", response.text)
    except Exception as e:
        print("❌ Error sending request:", e)

if __name__ == "__main__":
    path = record_audio()
    time.sleep(1)
    send_to_server(path)
