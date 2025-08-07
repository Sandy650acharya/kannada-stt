import requests
import sounddevice as sd
import soundfile as sf
import tempfile
import time
import os
import webbrowser
import numpy as np
import scipy.signal
from pydub import AudioSegment, silence

# CONFIGURATION
TOTAL_DURATION = 150        # Total recording duration (seconds)
SAMPLE_RATE = 16000
SERVER_URL = "https://kannada-stt.onrender.com/transcribe"
AUTO_SAVE_TRANSCRIPT = True

def choose_language():
    print("\n🌐 Choose Language for Transcription:")
    print("1. Kannada (Default)")
    print("2. English")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "2":
        return "en-IN"
    else:
        return "kn-IN"

def apply_highpass_filter(audio, samplerate, cutoff=100.0):
    """Denoise using high-pass filter to remove low-frequency hums."""
    b, a = scipy.signal.butter(1, cutoff / (0.5 * samplerate), btype='high', analog=False)
    filtered_audio = scipy.signal.filtfilt(b, a, audio[:, 0])
    return np.expand_dims(filtered_audio, axis=1)

def record_full_audio(duration=TOTAL_DURATION, samplerate=SAMPLE_RATE):
    print(f"\n🎙️ Recording full session for {duration} seconds... Speak continuously.")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()

    audio = apply_highpass_filter(audio, samplerate)

    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(temp_wav.name, audio, samplerate)
    print(f"✅ Full audio saved to {temp_wav.name}")
    return temp_wav.name

def chunk_audio_by_silence(audio_path):
    print("\n🔍 Splitting audio based on silence...")
    audio = AudioSegment.from_wav(audio_path).set_channels(1).set_frame_rate(16000)

    primary_chunks = silence.split_on_silence(
        audio,
        min_silence_len=1000,           # ~1 sec silence
        silence_thresh=audio.dBFS - 14, # Adaptive threshold
        keep_silence=300                # Padding for context
    )

    valid_chunks = []
    current = AudioSegment.silent(duration=0)

    for chunk in primary_chunks:
        if len(chunk) < 1000:
            continue

        if len(current) + len(chunk) <= 30000:
            current += chunk
        else:
            if len(current) > 1000:
                valid_chunks.append(current)
            current = chunk

    if len(current) > 1000:
        valid_chunks.append(current)

    print(f"✅ Total valid chunks: {len(valid_chunks)}")
    return valid_chunks

def send_chunk_to_server(chunk_audio, chunk_index, language):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as chunk_file:
        chunk_audio.export(chunk_file.name, format="wav")
        chunk_path = chunk_file.name

    print(f"📡 Sending chunk {chunk_index} to server...")

    try:
        with open(chunk_path, "rb") as f:
            response = requests.post(
                f"{SERVER_URL}?lang={language}", 
                files={"audio": f}
            )
        os.remove(chunk_path)

        if response.ok:
            result = response.json()
            transcript = result.get("transcript", "").strip()
            print(f"✅ Chunk {chunk_index} transcript:\n{transcript}\n")
            return transcript
        else:
            print(f"❌ Server error [{response.status_code}]: {response.text}")
            return ""
    except Exception as e:
        print(f"❌ Error sending chunk {chunk_index}: {e}")
        return ""

def save_transcript(text, filename="full_transcript.txt"):
    with open(filename, "w", encoding="utf-8") as tf:
        tf.write(text)
    print(f"📝 Transcript saved to {filename}")
    webbrowser.open(f"file://{os.path.abspath(filename)}")

if __name__ == "__main__":
    selected_language = choose_language()
    audio_path = record_full_audio()
    chunks = chunk_audio_by_silence(audio_path)

    all_transcripts = []
    for idx, chunk in enumerate(chunks, 1):
        transcript = send_chunk_to_server(chunk, idx, selected_language)
        if transcript:
            all_transcripts.append(transcript)

    full_transcript = "\n".join(all_transcripts)

    if full_transcript:
        print("\n📄 Final Combined Transcript:\n")
        print(full_transcript)
        if AUTO_SAVE_TRANSCRIPT:
            save_transcript(full_transcript)
    else:
        print("⚠️ No valid transcript received.")
